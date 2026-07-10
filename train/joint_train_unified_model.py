import functools
import os
import wandb
import yaml
from copy import deepcopy
from dataclasses import dataclass, field
from time import time, strftime
from train.fsdp_utils import rank0_print
import gc

import torch
import torch.distributed as dist
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from torch.utils.data import DataLoader
from transformers import HfArgumentParser, set_seed
from transformers.optimization import (
    get_constant_schedule_with_warmup,
    get_cosine_with_min_lr_schedule_with_warmup,
)

from data.dataset_base import DataConfig, PackedDataset, collate_wrapper
from data.dataset_base_periter import PackedDatasetPerIter
from data.data_utils import add_special_tokens
from g2vlm_utils import save_ply_visualization

from modeling.g2vlm import (
    G2VLMConfig, 
    G2VLM, 
    Qwen2VLConfig,
    Qwen2VLForCausalLM,
    Dinov2WithRegistersConfig, Dinov2WithRegistersModel
)
from modeling.qwen2vl.modeling_qwen2_vl import Qwen2VisionTransformerPretrainedModel
from modeling.qwen2vl.configuration_qwen2_vl import Qwen2VLVisionConfig

from modeling.qwen2 import Qwen2Tokenizer
from train.train_utils import create_logger, get_latest_ckpt

from train.fsdp_utils import (
    FSDPCheckpoint, FSDPConfig, grad_checkpoint_check_fn, fsdp_wrapper, debug_fsdp_memory_table, print_fsdp_memory_summary, 
    construct_dummy_batch, init_fsdp_log_file, warmup_fsdp_memory, mark_step_start, fsdp_ema_setup, fsdp_ema_update, save_latest_checkpoints
)

NUMS_GPUS=64
# MAX_STEPS_PER_EPOCH = max(10504501 // (8 * NUMS_GPUS ), 1)


def load_dinov3_classes():
    from modeling.dinov3.configuration_dinov3_vit import DINOv3ViTConfig
    from modeling.dinov3.dinov3_model import DINOv3ViTModel

    return DINOv3ViTConfig, DINOv3ViTModel


@dataclass
class ModelArguments:
    model_path: str = field(
        default="InternRobotics/G2VLM-Qwen2-VL-2B",
        metadata={"help": "Path of the pretrained G2VLM model."}
    ) 
    llm_path: str = field(
        default="InternRobotics/G2VLM-Qwen2-VL-2B",
        metadata={"help": "Path or HuggingFace repo ID of the pretrained Qwen2-style language model."}
    )
    llm_qk_norm: bool = field(
        default=True,
        metadata={"help": "Enable QK LayerNorm (qk_norm) inside the attention blocks."}
    )
    tie_word_embeddings: bool = field(
        default=False,
        metadata={"help": "Share input and output word embeddings (tied embeddings)."}
    )
    layer_module: str = field(
        default="Qwen2VLMoTDecoderLayer",
        metadata={"help": "Python class name of the decoder layer to instantiate."}
    )
    vit_path: str = field(
        default="InternRobotics/G2VLM-Qwen2-VL-2B",
        metadata={"help": "Path or repo ID of the Qwen Vision Transformer used for image understanding."}
    )
    dino_path: str = field(
        default="facebook/dinov2-with-registers-large",
        metadata={"help": "Path or repo ID of the Dino Vision Transformer used for image recon."}
    )
    vit_patch_size: int = field(
        default=14,
        metadata={"help": "Patch size (pixels) for the Vision Transformer encoder."}
    )
    dino_patch_size: int = field(
        default=14,
        metadata={"help": "Patch size (pixels) for the Vision Transformer encoder."}
    )
    vit_max_num_patch_per_side: int = field(
        default=70,
        metadata={"help": "Maximum number of ViT patches along one image side after cropping / resize."}
    )
    dino_max_num_patch_per_side: int = field(
        default=37,
        metadata={"help": "Maximum number of ViT patches along one image side after cropping / resize."}
    )
    interpolate_pos: bool = field(
        default=False,
        metadata={"help": "Interpolate positional embeddings when image resolution differs from pre-training."}
    )

@dataclass
class DataArguments:
    dataset_config_file: str = field(
        default="data/configs/example.yaml",
        metadata={"help": "YAML file specifying dataset groups, weights, and preprocessing rules."}
    )
    prefetch_factor: int = field(
        default=2,
        metadata={"help": "How many batches each DataLoader worker pre-loads in advance."}
    )
    num_workers: int = field(
        default=4,
        metadata={"help": "Number of background workers for the PyTorch DataLoader."}
    )
    max_num_tokens_per_sample: int = field(
        default=16384,
        metadata={"help": "Maximum tokens allowed in one raw sample; longer samples are skipped."}
    )
    max_num_tokens: int = field(
        default=36864,
        metadata={"help": "Hard limit on tokens in a packed batch; flush if adding a sample would exceed it."}
    )
    prefer_buffer_before: int = field(
        default=16384,
        metadata={"help": "While batch length is below this, pop from the overflow buffer before new sampling."}
    )
    max_buffer_size: int = field(
        default=50,
        metadata={"help": "Maximum number of oversized samples kept in the overflow buffer."}
    )
    data_seed: int = field(
        default=42,
        metadata={"help": "Seed used when shuffling / sampling data shards to ensure reproducibility."}
    )


@dataclass
class TrainingArguments:
    visual_und: bool = field(
        default=True,
        metadata={"help": "Train image understanding branch."}
    )
    visual_recon: bool = field(
        default=True,
        metadata={"help": "Train recon branch."}
    )
    joint_train_recon: bool = field(
        default=False,
        metadata={"help": "Train recon then und branch with recon."}
    )
    pretrain_train_recon: bool = field(
        default=False,
        metadata={"help": "Train recon then und branch with recon."}
    )
    train_conf_pi3: bool = field(
        default=False,
        metadata={"help": "Train recon branch."}
    )
    enable_ema_model: bool = field(
        default=True,
        metadata={"help": "Train recon branch."}
    )
    use_dinov3: bool = field(
        default=False,
        metadata={"help": "Train recon branch."}
    )

    # --- bookkeeping & logging ---
    results_dir: str = field(
        default="results",
        metadata={"help": "Root directory for logs."}
    )
    checkpoint_dir: str = field(
        default="results/checkpoints",
        metadata={"help": "Root directory for model checkpoints."}
    )
    wandb_project: str = field(
        default="g2vlm",
        metadata={"help": "Weights & Biases project name."}
    )
    wandb_name: str = field(
        default="run",
        metadata={"help": "Name shown in the Weights & Biases UI for this run."}
    )
    wandb_runid: str = field(
        default="0",
        metadata={"help": "Unique identifier to resume a previous W&B run, if desired."}
    )
    wandb_resume: str = field(
        default="allow",
        metadata={"help": "W&B resume mode: 'allow', 'must', or 'never'."}
    )
    wandb_offline: bool = field(
        default=False,
        metadata={"help": "Run W&B in offline mode (logs locally, sync later)."}
    )

    # --- reproducibility & resume ---
    global_seed: int = field(
        default=4396,
        metadata={"help": "Base random seed; actual seed is offset by rank for DDP."}
    )
    auto_resume: bool = field(
        default=False,
        metadata={"help": "Automatically pick up the latest checkpoint found in checkpoint_dir."}
    )
    resume_from: str = field(
        default=None,
        metadata={"help": "Explicit checkpoint path to resume from (overrides auto_resume)." }
    )
    resume_model_only: bool = field(
        default=False,
        metadata={"help": "Load only model weights, ignoring optimizer/scheduler states."}
    )
    finetune_from_ema: bool = field(
        default=False,
        metadata={"help": "When resume_model_only=True, load the EMA (exponential moving average) weights instead of raw weights."}
    )
    finetune_from_hf: bool = field(
        default=False,
        metadata={"help": "Whether finetune from HugginFace model."}
    )
    finetune_dino_from_hf: bool = field(
        default=False,
        metadata={"help": "Whether finetune from HugginFace model."}
    )

    # --- reporting frequency ---
    log_every: int = field(
        default=10,
        metadata={"help": "Print / log every N training steps."}
    )
    save_every: int = field(
        default=2000,
        metadata={"help": "Save a checkpoint every N training steps."}
    )
    total_steps: int = field(
        default=500_000,
        metadata={"help": "Total number of optimizer steps to train for."}
    )

    # --- optimization & scheduler ---
    warmup_steps: int = field(
        default=2000,
        metadata={"help": "Linear warm-up steps before applying the main LR schedule."}
    )
    lr_scheduler: str = field(
        default="constant",
        metadata={"help": "Type of LR schedule: 'constant' or 'cosine'."}
    )
    lr: float = field(
        default=1e-4,
        metadata={"help": "Peak learning rate after warm-up."}
    )
    min_lr: float = field(
        default=1e-7,
        metadata={"help": "Minimum learning rate for cosine schedule (ignored for constant)."}
    )
    beta1: float = field(
        default=0.9,
        metadata={"help": "AdamW β₁ coefficient."}
    )
    beta2: float = field(
        default=0.95,
        metadata={"help": "AdamW β₂ coefficient."}
    )
    eps: float = field(
        default=1e-15,
        metadata={"help": "AdamW ε for numerical stability."}
    )
    ema: float = field(
        default=0.9999,
        metadata={"help": "Decay rate for the exponential moving average of model weights."}
    )
    max_grad_norm: int = field(
        default=1.0,
        metadata={"help": "Gradient clipping threshold (L2 norm)."}
    )
    ce_weight: float = field(
        default=1.0,
        metadata={"help": "Scaling factor for the language cross-entropy loss term."}
    )
    vg_weight: float = field(
        default=1.0,
        metadata={"help": "Scaling factor for the visual geometry loss term."}
    )
    ce_loss_reweighting: bool = field(
        default=False,
        metadata={"help": "Reweight CE loss by token importance (provided via ce_loss_weights)."}
    )
    expected_num_tokens: int = field(
        default=32768,
        metadata={"help": "Soft target token count; yield the batch once it reaches or exceeds this size."}
    )
    ce_loss_dino: bool = field(
        default=False,
        metadata={"help": "CE loss on dino tokens."}
    )

    # --- distributed training / FSDP ---
    num_replicate: int = field(
        default=1,
        metadata={"help": "Number of model replicas per GPU rank for tensor parallelism."}
    )
    num_shard: int = field(
        default=8,
        metadata={"help": "Number of parameter shards when using FSDP HYBRID_SHARD."}
    )
    sharding_strategy: str = field(
        default="HYBRID_SHARD",
        metadata={"help": "FSDP sharding strategy: FULL_SHARD, SHARD_GRAD_OP, HYBRID_SHARD, etc."}
    )
    backward_prefetch: str = field(
        default="BACKWARD_PRE",
        metadata={"help": "FSDP backward prefetch strategy (BACKWARD_PRE or NO_PREFETCH)."}
    )
    cpu_offload: bool = field(
        default=False,
        metadata={"help": "Enable FSDP parameter offload to CPU."}
    )

    # --- module freezing ---
    freeze_llm: bool = field(
        default=False,
        metadata={"help": "Keep language-model weights fixed (no gradient updates)."}
    )
    freeze_vit: bool = field(
        default=False,
        metadata={"help": "Keep ViT weights fixed during training."}
    )
    freeze_dino: bool = field(
        default=False,
        metadata={"help": "Keep Dino weights fixed during training."}
    )
    freeze_und: bool = field(
        default=False,
        metadata={"help": "Freeze the visual understanding connector layers."}
    )
    freeze_recon: bool = field(
        default=False,
        metadata={"help": "Freeze the visual reconstruction connector layers."}
    )
    copy_init_moe: bool = field(
        default=True,
        metadata={"help": "Duplicate initial MoE experts so each has identical initialisation."}
    )
    use_flex: bool = field(
        default=False,
        metadata={"help": "Enable FLEX (flash-ext friendly) packing algorithm for sequence data."}
    )


def main():
    assert torch.cuda.is_available()
    dist.init_process_group("nccl")
    device = dist.get_rank() % torch.cuda.device_count()
    torch.cuda.set_device(device)
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging:
    if dist.get_rank() == 0:
        os.makedirs(training_args.results_dir, exist_ok=True)
        os.makedirs(training_args.checkpoint_dir, exist_ok=True)
        logger = create_logger(training_args.results_dir, dist.get_rank())
        wandb.init(
            project=training_args.wandb_project, 
            id=f"{training_args.wandb_name}-run{training_args.wandb_runid}", 
            name=training_args.wandb_name, 
            resume=training_args.wandb_resume,
            mode="offline" if training_args.wandb_offline else "online"
            # settings=wandb.Settings(verify_ssl=False) # edit: disable ssl, tnot allowed
        )
        wandb.config.update(training_args)
        wandb.config.update(model_args)
        wandb.config.update(data_args)
    else:
        logger = create_logger(None, dist.get_rank())
    dist.barrier()
    logger.info(f'Training arguments {training_args}')
    logger.info(f'Model arguments {model_args}')
    logger.info(f'Data arguments {data_args}')

    # prepare auto resume logic:
    if training_args.auto_resume:
        resume_from = get_latest_ckpt(training_args.checkpoint_dir)
        if resume_from is None:
            resume_from = training_args.resume_from
            resume_model_only = training_args.resume_model_only
            if resume_model_only:
                finetune_from_ema = training_args.finetune_from_ema
            else:
                finetune_from_ema = False
        else:
            resume_model_only = False
            finetune_from_ema = False
    else:
        resume_from = training_args.resume_from
        resume_model_only = training_args.resume_model_only
        if resume_model_only:
            finetune_from_ema = training_args.finetune_from_ema
        else:
            finetune_from_ema = False

    seed = training_args.global_seed * dist.get_world_size() + dist.get_rank()
    set_seed(seed)

    if training_args.finetune_from_hf:
        llm_config = Qwen2VLConfig.from_json_file(os.path.join(model_args.model_path, "text_config.json"))
    else:
        llm_config = Qwen2VLConfig.from_pretrained(model_args.llm_path)
    llm_config.layer_module = model_args.layer_module
    llm_config.qk_norm = model_args.llm_qk_norm
    llm_config.tie_word_embeddings = model_args.tie_word_embeddings
    llm_config.freeze_und = training_args.freeze_und
    llm_config.freeze_recon = training_args.freeze_recon
    if training_args.finetune_from_hf:
        language_model = Qwen2VLForCausalLM(llm_config)
    else:
        language_model = Qwen2VLForCausalLM.from_pretrained(model_args.llm_path, config=llm_config)
    if training_args.copy_init_moe:
        language_model.init_moe()

    if training_args.visual_und:  
        if training_args.finetune_from_hf:
            vit_config = Qwen2VLVisionConfig.from_json_file(os.path.join(model_args.model_path, "vit_config.json"))
            vit_config.patch_size =14
        else:
            vit_config = Qwen2VLVisionConfig.from_pretrained(model_args.vit_path)
            vit_config.patch_size =14
        if training_args.finetune_from_hf:
            vit_model = Qwen2VisionTransformerPretrainedModel(vit_config)
        else:
            vit_model = Qwen2VisionTransformerPretrainedModel.from_pretrained(model_args.vit_path, config=vit_config)
    
    if training_args.visual_recon:  
        if training_args.use_dinov3:
            DINOv3ViTConfig, DINOv3ViTModel = load_dinov3_classes()
        if training_args.finetune_dino_from_hf:
            if training_args.use_dinov3:
                dino_config = DINOv3ViTConfig.from_json_file(os.path.join(model_args.dino_path, "config.json"))
            else:
                dino_config = Dinov2WithRegistersConfig.from_json_file(os.path.join(model_args.model_path, "dino_config.json"))
        else:
            if training_args.use_dinov3:
                dino_config = DINOv3ViTConfig.from_pretrained(model_args.dino_path)
            else:
                dino_config = Dinov2WithRegistersConfig.from_pretrained(model_args.dino_path)

        dino_config.rope = False
        if training_args.finetune_dino_from_hf:
            if training_args.use_dinov3:
                dino_model = DINOv3ViTModel(dino_config)
            else:
                dino_model = Dinov2WithRegistersModel(dino_config)
        else:
            if training_args.use_dinov3:
                dino_model = DINOv3ViTModel.from_pretrained(model_args.dino_path, config=dino_config)
            else:
                dino_model = Dinov2WithRegistersModel.from_pretrained(model_args.dino_path, config=dino_config)

        if training_args.use_dinov3:
            dino_model = dino_model.to(torch.bfloat16)

    config = G2VLMConfig(
        visual_und=training_args.visual_und,
        visual_recon=training_args.visual_recon,
        joint_train_recon=training_args.joint_train_recon,
        pretrain_train_recon=training_args.pretrain_train_recon,
        use_dinov3=training_args.use_dinov3,
        ce_loss_dino=training_args.ce_loss_dino,
        train_conf_pi3=training_args.train_conf_pi3,
        llm_config=llm_config, 
        dino_config=dino_config if training_args.visual_recon else None,
        vit_config=vit_config if training_args.visual_und else None,
        vit_max_num_patch_per_side=model_args.vit_max_num_patch_per_side,
        dino_max_num_patch_per_side=model_args.dino_max_num_patch_per_side,
        interpolate_pos=model_args.interpolate_pos,
    )
    model = G2VLM(
        language_model, 
        vit_model if training_args.visual_und else None, 
        dino_model if training_args.visual_recon else None, 
        config
    )

    # Setup tokenizer for model:
    tokenizer = Qwen2Tokenizer.from_pretrained(model_args.model_path if training_args.finetune_from_hf else model_args.llm_path)
    tokenizer, new_token_ids, num_new_tokens = add_special_tokens(tokenizer)
    if num_new_tokens > 0:
        model.language_model.resize_token_embeddings(len(tokenizer))
        model.config.llm_config.vocab_size = len(tokenizer)
        model.language_model.config.vocab_size = len(tokenizer)

    if training_args.visual_recon: 
        if hasattr(model.dino_model.embeddings, "mask_token"):
            model.dino_model.embeddings.mask_token.requires_grad_(False)
        else:
            print('error, should freeze mask_token when finetuning dino vit ')

    # maybe freeze something:
    if training_args.freeze_llm:
        model.language_model.eval()
        for param in model.language_model.parameters():
            param.requires_grad = False
    if training_args.freeze_vit and training_args.visual_und:
        model.vit_model.eval()
        for param in model.vit_model.parameters():
            param.requires_grad = False
    if training_args.freeze_dino and training_args.visual_recon:
        model.dino_model.eval()
        for param in model.dino_model.parameters():
            param.requires_grad = False

    # Setup FSDP and load pretrained model:
    fsdp_config = FSDPConfig(
        sharding_strategy=training_args.sharding_strategy,
        backward_prefetch=training_args.backward_prefetch,
        cpu_offload=training_args.cpu_offload,
        num_replicate=training_args.num_replicate,
        num_shard=training_args.num_shard,
    )

    if training_args.enable_ema_model:
        ema_model = deepcopy(model)
    else:
        ema_model = None 


    ignored_modules = []
    ema_ignored_modules = []

    if training_args.visual_recon: 
        if not training_args.visual_und or training_args.joint_train_recon:
            ignored_modules.append(model.camera_head)
            ignored_modules.append(model.point_head)

            if model.global_point_head is not None:
                ignored_modules.append(model.global_point_head)
            if model.conf_head is not None:
                ignored_modules.append(model.conf_head)
                ignored_modules.append(model.conf_decoder)
                ignored_modules.append(model.Pi3Loss.point_loss.segformer)
                
            if training_args.enable_ema_model:
                ema_ignored_modules.append(ema_model.camera_head)
                ema_ignored_modules.append(ema_model.point_head)
                if model.conf_head is not None:
                    ema_ignored_modules.append(ema_model.conf_head)
                    ema_ignored_modules.append(ema_model.conf_decoder)
                    ema_ignored_modules.append(ema_model.Pi3Loss.point_loss.segformer)

                if model.global_point_head is not None:
                    ema_ignored_modules.append(ema_model.global_point_head)

    if training_args.enable_ema_model:
        ema_model = fsdp_ema_setup(ema_model, fsdp_config, ignored_modules=ema_ignored_modules)
    fsdp_model = fsdp_wrapper(model, fsdp_config, ignored_modules=ignored_modules)
  
    fsdp_model, ema_model = FSDPCheckpoint.try_load_fsdp_ckpt( 
        resume_from, logger, fsdp_model, ema_model, resume_from_ema=finetune_from_ema
    ) 

    batch_ref = {}  
    job_id = os.environ.get("SLURM_JOB_ID") 
    nan_hooks = attach_first_nan_observer(fsdp_model.module, batch_ref=batch_ref, job_id=job_id)

    if dist.is_initialized():
        world_size = dist.get_world_size()
    else:
        world_size = 1
    log_path = init_fsdp_log_file(log_dir="./logs", world_size=world_size)
    
    local_params = sum(p.numel() for p in fsdp_model.parameters())
    print(f"[Rank {dist.get_rank()}] After FSDP: local_params={local_params/1e6:.2f}M")
    print(f"[Rank {dist.get_rank()}] Before checkpointing: params={sum(p.numel() for p in fsdp_model.parameters())/1e6:.2f}M")

    apply_activation_checkpointing(
        fsdp_model, 
        checkpoint_wrapper_fn=functools.partial(
            checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT
        ), 
        check_fn=grad_checkpoint_check_fn
    )

    print(f"[Rank {dist.get_rank()}] After checkpointing: params={sum(p.numel() for p in fsdp_model.parameters())/1e6:.2f}M")

    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(
        fsdp_model.parameters(), 
        lr=training_args.lr, 
        betas=(training_args.beta1, training_args.beta2), 
        eps=training_args.eps, 
        weight_decay=0
    )
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    if training_args.lr_scheduler == 'cosine':
        scheduler = get_cosine_with_min_lr_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=training_args.warmup_steps,
            num_training_steps=training_args.total_steps,
            min_lr=training_args.min_lr,
        )
    elif training_args.lr_scheduler == 'constant':
        scheduler = get_constant_schedule_with_warmup(
            optimizer=optimizer, num_warmup_steps=training_args.warmup_steps
        )
    else:
        raise ValueError

    # maybe resume optimizer, scheduler, and train_steps
    if resume_model_only:
        train_step = 0
        data_status = None
    else:
        optimizer, scaler, scheduler, train_step, data_status = FSDPCheckpoint.try_load_train_state(
            resume_from, optimizer, scaler, scheduler, fsdp_config, 
        )

    # Setup packed dataloader
    with open(data_args.dataset_config_file, "r") as stream:
        dataset_meta = yaml.safe_load(stream)

    dataset_config = DataConfig(grouped_datasets=dataset_meta)
    if training_args.visual_und:
        dataset_config.vit_patch_size = model_args.vit_patch_size
        dataset_config.vit_max_num_patch_per_side = model_args.vit_max_num_patch_per_side
    if training_args.visual_recon:
        dataset_config.dino_patch_size = model_args.dino_patch_size
        dataset_config.dino_max_num_patch_per_side = model_args.dino_max_num_patch_per_side

    # train_dataset = PackedDataset(
    train_dataset = PackedDatasetPerIter(
        dataset_config,
        tokenizer=tokenizer,
        special_tokens=new_token_ids,
        local_rank=dist.get_rank(),
        world_size=dist.get_world_size(),
        num_workers=data_args.num_workers,
        expected_num_tokens=training_args.expected_num_tokens,
        max_num_tokens_per_sample=data_args.max_num_tokens_per_sample,
        max_num_tokens=data_args.max_num_tokens,
        max_buffer_size=data_args.max_buffer_size,
        prefer_buffer_before=data_args.prefer_buffer_before,
        interpolate_pos=model_args.interpolate_pos,
        use_flex=training_args.use_flex,
        data_status=data_status,
    )

    # mapped ignored modules to it's device . 
    if len(ignored_modules) > 0:
        if training_args.visual_recon: 
            model.camera_head.to(device)
            model.point_head.to(device)
        else:
            if model.camera_head is not None:
                model.camera_head.to(device)
            if model.point_head is not None:
                model.point_head.to(device)
            if model.global_point_head is not None:
                model.global_point_head.to(device)
            if model.conf_head is not None:
                model.conf_head.to(device)
                model.conf_decoder.to(device)
                model.Pi3Loss.point_loss.segformer.to(device)

    fsdp_model.train()
    if training_args.enable_ema_model:
        ema_model.eval()

    # train loop
    start_time = time()
    logger.info(f"Training for {training_args.total_steps} steps, starting at {train_step}...")

    global_step = 0
    data_seed = data_args.data_seed
    global_epoch = 0


    last_step_local_pts_loss = 0 
    last_step_global_pts_loss = 0

    while True: 
        train_dataset.set_epoch(data_seed+global_epoch * 100) #MAX_STEPS_PER_EPOCH)
        train_loader = DataLoader(
            train_dataset,
            batch_size=1, # batch size is 1 packed dataset
            num_workers=data_args.num_workers,
            pin_memory=True,
            collate_fn=collate_wrapper(),
            drop_last=True,
            prefetch_factor=data_args.prefetch_factor,
        )

        epoch_step = 0
        batch_ref = {}

        for curr_step_in_loop, data in enumerate(train_loader, start=train_step):
            curr_step = curr_step_in_loop + global_step
            mark_step_start(curr_step)   
            # for debugging 
            data_dict_test = data.to_dict()
            for kk,vv in data_dict_test.items():
                if isinstance(vv, torch.Tensor):
                    rank0_print(kk, vv.shape)
                else: 
                    rank0_print(kk, vv)
            del data_dict_test

            data = data.cuda(device).to_dict()
            data_indexes = data.pop('batch_data_indexes', None)
            ce_loss_weights = data.pop('ce_loss_weights', None)
                    
            with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16): 
                loss_dict, predictions = fsdp_model(**data)

            loss = 0
            # This joint fine-tune path does not use an MSE token objective, but
            # the shared logging schema expects the field to exist.
            total_mse_tokens = torch.tensor(0, device=device)
            ce = loss_dict["ce"]
            if ce is not None:
                if training_args.ce_loss_dino:
                    total_ce_tokens = torch.tensor(len(data['ce_loss_indexes']) + len(data['packed_dino_token_indexes']), device=device)
                    dist.all_reduce(total_ce_tokens, op=dist.ReduceOp.SUM)
                else:
                    total_ce_tokens = torch.tensor(len(data['ce_loss_indexes']), device=device)
                    dist.all_reduce(total_ce_tokens, op=dist.ReduceOp.SUM)

                if training_args.ce_loss_reweighting:
                    ce = ce * ce_loss_weights
                    total_ce_loss_weights = ce_loss_weights.sum()
                    dist.all_reduce(total_ce_loss_weights, op=dist.ReduceOp.SUM)
                    ce = ce.sum() * dist.get_world_size() / total_ce_loss_weights
                else:
                    ce = ce.sum() * dist.get_world_size() / total_ce_tokens
     
                loss_dict["ce"] = ce.detach()
                loss = loss + ce * training_args.ce_weight
            else:
                assert not training_args.visual_und
                loss_dict["ce"] = torch.tensor(0, device=device)
                total_ce_tokens = torch.tensor(0, device=device)
                
            ###########################debug
            # if (training_args.visual_recon and training_args.joint_train_recon) or training_args.pretrain_train_recon:
            #     if dist.get_rank() == 0 and curr_step <= 1: 
            #         print('SAVING point clouds VIS')
            #         save_ply_visualization_pi3(predictions, curr_step, training_args.wandb_name + '_debug', debug=True)
            ###########################

            if (training_args.visual_recon and training_args.joint_train_recon) or training_args.pretrain_train_recon:
                vg = loss_dict["vg"]
                total_vg_tokens = torch.tensor(len(data['packed_dino_token_indexes']), device=device)
                dist.all_reduce(total_vg_tokens, op=dist.ReduceOp.SUM)
                if training_args.joint_train_recon or training_args.pretrain_train_recon:
                    dist.all_reduce(vg, op=dist.ReduceOp.SUM)
                    vg = vg / dist.get_world_size()
                else:
                    loss_dict["vg"] = torch.tensor(0, device=device)
                    vg = loss_dict["vg"]
                
                if 'global_pts_loss' in loss_dict:
                    global_pts_loss = torch.tensor(loss_dict["global_pts_loss"].item(), device=device) 
                    dist.all_reduce(global_pts_loss, op=dist.ReduceOp.SUM)
                    global_pts_loss = global_pts_loss.item() / dist.get_world_size()
                if 'local_pts_loss' in loss_dict:
                    local_pts_loss = torch.tensor(loss_dict["local_pts_loss"].item(), device=device) 
                    dist.all_reduce(local_pts_loss, op=dist.ReduceOp.SUM)
                    local_pts_loss = local_pts_loss.item() / dist.get_world_size()
                    
                # vg_clip_loss = 10 
                # spike_loss = 0
                # if vg > vg_clip_loss:
                #     spike_loss = vg.clone()
                #     vg = vg * 0.0
                # elif vg < 0: #must be non zero    
                #     spike_loss = vg.clone()
                #     vg = vg * 0.0
        
                if training_args.joint_train_recon or training_args.pretrain_train_recon:
                    loss_dict["vg"] = vg.detach()

                loss = loss + vg * training_args.vg_weight

                # loss_value = vg.item()
                # if not math.isfinite(loss_value):
                #     rank = dist.get_rank()
                #     print(
                #         f"Rank {rank}: Loss is {loss_value}, stopping training at global step{curr_step} global epoch {global_epoch})."
                #     )
                #     nan_dir = 'debug_output/'
                #     os.makedirs(nan_dir, exist_ok=True)
                #     nan_dir = os.path.join(nan_dir, str(curr_step)+'lossNan_Rank:'+str(rank) + training_args.wandb_name)
                #     os.makedirs(nan_dir, exist_ok=True)

                #     batch_save_path = os.path.join(nan_dir, f'rank_{rank}_batch.pt')
                #     output_save_path = os.path.join(nan_dir, f'rank_{rank}_batch_output.pt')
                #     torch.save(data, batch_save_path)
                #     torch.save(predictions, output_save_path)
                #     print(f"Rank {rank}: Saved the NaN-causing batch and output to: {nan_dir}")
                #     cpu_loss_details = {
                #         key: value.detach().cpu() for key, value in loss_dict.items()
                #     }
                #     torch.save(cpu_loss_details, os.path.join(nan_dir, f'rank_{rank}_info.pt'))
                #     save_ply_visualization_pi3(predictions, str(curr_step)+'lossNan_Rank:'+str(rank), training_args.wandb_name, all_batch=True, gt_only=True)

                #     sys.exit(1)
      
            else:
           
                loss_dict["vg"] = torch.tensor(0, device=device)
                total_vg_tokens = torch.tensor(0, device=device)
     
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer) 

            total_norm = fsdp_model.clip_grad_norm_(training_args.max_grad_norm) 

            scaler.step(optimizer)
            scaler.update()

            scheduler.step()
            if training_args.enable_ema_model:
                fsdp_ema_update(ema_model, fsdp_model, decay=training_args.ema)
            
            # current_visualization_freq = 500 #1000 ##harcode !!!! 
            # if (training_args.visual_recon and training_args.joint_train_recon) or training_args.pretrain_train_recon:
            #     if curr_step % current_visualization_freq == 0 and dist.get_rank() == 0: 
            #         print('SAVING point clouds VIS')
            #         save_ply_visualization_pi3(predictions, curr_step, training_args.wandb_name)

            # Log loss values:
            if curr_step % training_args.log_every == 0:
                total_samples = torch.tensor(len(data['sample_lens']), device=device)
                dist.all_reduce(total_samples, op=dist.ReduceOp.SUM)

                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = training_args.log_every / (end_time - start_time)
                message = f"(step={curr_step:07d}) "
                wandb_log = {}
                for key, value in loss_dict.items():
                    avg_loss = torch.tensor(value.item(), device=device)
                    dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                    avg_loss = avg_loss.item() / dist.get_world_size()
                    message += f"Train Loss {key}: {avg_loss:.4f}, "
                    wandb_log[key] = avg_loss

                message += f"Train Steps/Sec: {steps_per_sec:.2f}, "
                message += f"Train loss vg: {loss_dict['vg'].item():.4f}, "

                logger.info(message)

                wandb_log['lr'] = optimizer.param_groups[0]['lr']
                wandb_log['total_mse_tokens'] = total_mse_tokens.item()
                wandb_log['total_vg_tokens'] = total_vg_tokens.item()
                wandb_log['total_ce_tokens'] = total_ce_tokens.item()
                wandb_log['total_norm'] = total_norm.item()
                wandb_log['total_samples'] = total_samples.item()
                wandb_log['vg'] = loss_dict["vg"].item()

                mem_allocated = torch.tensor(torch.cuda.max_memory_allocated() / 1024**2, device=device)
                dist.all_reduce(mem_allocated, op=dist.ReduceOp.MAX)
                wandb_log['mem_allocated'] = mem_allocated
                mem_cache = torch.tensor(torch.cuda.max_memory_reserved() / 1024**2, device=device)
                dist.all_reduce(mem_cache, op=dist.ReduceOp.MAX)
                wandb_log['mem_cache'] = mem_cache

                if dist.get_rank() == 0:
                    wandb.log(wandb_log, step=curr_step)
                start_time = time()

            if data_status is None:
                data_status = {}
            for item in data_indexes:
                if item['dataset_name'] not in data_status.keys():
                    data_status[item['dataset_name']] = {}
                data_status[item['dataset_name']][item['worker_id']] = item['data_indexes']

            if curr_step > 0 and curr_step % training_args.save_every == 0:
                if dist.get_rank() == 0:
                    gather_list = [None] * dist.get_world_size()
                else:
                    gather_list = None
                dist.gather_object(data_status, gather_list, dst=0)

                FSDPCheckpoint.fsdp_save_ckpt(
                    ckpt_dir=training_args.checkpoint_dir, 
                    train_steps=curr_step, 
                    model=fsdp_model, 
                    ema_model=ema_model, 
                    optimizer=optimizer, 
                    scaler=scaler,
                    scheduler=scheduler, 
                    logger=logger,
                    fsdp_config=fsdp_config,
                    data_status=gather_list
                )

                save_latest_checkpoints(training_args.checkpoint_dir, keep_latest=2)

            epoch_step += 1
              
            if curr_step >= training_args.total_steps:
                logger.info("Done!")
                if dist.get_rank() == 0:
                    wandb.finish()
                dist.destroy_process_group()
                return

if __name__ == "__main__":
    main()
