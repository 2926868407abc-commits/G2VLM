#!/bin/bash

set -x -e
export NCCL_DEBUG=INFO
export NCCL_TIMEOUT=18000000
export NNODES=4
export num_gpus=8
export CPUS_PER_TASK=16


MASTER_ADDR=`scontrol show hostname $SLURM_JOB_NODELIST | head -n1`
MASTER_PORT=$((RANDOM % 101 + 25199))
export MASTER_ADDR=$MASTER_ADDR
export MASTER_PORT=$MASTER_PORT
echo $MASTER_ADDR
echo $MASTER_PORT

job_id=${SLURM_JOB_ID}

name="g2vlm_pretrain_${WORLD_SIZE}g_${job_id}"
export MODEL_PATH="InternRobotics/G2VLM-Qwen2-VL-2B" 
export output_dir="./checkpoints/${name}/"
mkdir -p ${output_dir}
export checkpoint_dir="./checkpoints/${name}"
mkdir -p ${checkpoint_dir}


# export WANDB_MODE=offline 
# export WANDB_API_KEY="your key"

export current_time=$(date +%Y%m%d_%H%M%S)
export wandb_name=$name

torchrun \
    --nnodes $NNODES \
    --nproc_per_node 8 \
    --node_rank="${SLURM_NODEID}" \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    train/pretrain_unified_model.py \
    --dataset_config_file data/configs/pretrain.yaml \
    --layer_module Qwen2VLMoTDecoderLayer \
    --vit_path ${MODEL_PATH} \
    --dino_path facebook/dinov2-with-registers-large \
    --llm_path ${MODEL_PATH} \
    --model_path ${MODEL_PATH} \
    --use_flex True \
    --expected_num_tokens 25600 \
    --max_num_tokens 25600 \
    --max_num_tokens_per_sample 25600 \
    --wandb_project G2VLM \
    --wandb_name ${wandb_name} \
    --wandb_offline True \
    --wandb_resume allow \
    --checkpoint_dir ${checkpoint_dir} \
    --llm_qk_norm True \
    --finetune_from_hf True \
    --auto_resume False \
    --resume-model-only True \
    --finetune-from-ema True \
    --resume_from ${MODEL_PATH} \
    --finetune_dino_from_hf False \
    --copy_init_moe False \
    --visual_und False \
    --visual_recon True \
    --pretrain_train_recon True \
    --enable_ema_model False \
    --freeze_dino True \
    --freeze_vit True \
    --freeze_und True \
    --results_dir $output_dir \
    --save_every 2000 \
    --total_steps 100000 \
    --warmup_steps 5000 \
    --log_every 1 \
    --num_shard 8 \
    --num_replicate 4 \
    --lr 2e-4 \
    --lr_scheduler cosine \
    --num_workers 4
    
