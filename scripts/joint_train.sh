#!/bin/bash

set -x -e

export NCCL_DEBUG=INFO
export NCCL_TIMEOUT=18000000
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'
export MKL_THREADING_LAYER='GNU'
export HYDRA_FULL_ERROR='1'
export NCCL_ASYNC_ERROR_HANDLING='1'

NNODES=8  
GPUS_PER_NODE=8     
CPUS_PER_TASK=16      

WORLD_SIZE=$((NNODES * GPUS_PER_NODE))

NUM_REPLICATE=$NNODES
NUM_SHARD=$GPUS_PER_NODE

MASTER_ADDR=$(scontrol show hostname $SLURM_JOB_NODELIST | head -n1)
MASTER_PORT=$((RANDOM % 101 + 25199))

echo "MASTER_ADDR=$MASTER_ADDR"
echo "MASTER_PORT=$MASTER_PORT"
echo "NNODES=$NNODES, GPUS_PER_NODE=$GPUS_PER_NODE, WORLD_SIZE=$WORLD_SIZE"

job_id=${SLURM_JOB_ID}

name="g2vlm_joint_train_${WORLD_SIZE}g_${job_id}"

export MODEL_PATH="InternRobotics/G2VLM-Qwen2-VL-2B" 
export PRETRAINED_CHECKPOINT="InternRobotics/G2VLM-2B-MoT" 
export output_dir="./checkpoints/${name}/"
mkdir -p ${output_dir}
export checkpoint_dir="./checkpoints/${name}"
mkdir -p ${checkpoint_dir}

# export WANDB_MODE=offline 
# export WANDB_API_KEY="your key"
# export CUDA_LAUNCH_BLOCKING=1

export current_time=$(date +%Y%m%d_%H%M%S)
export wandb_name=$name
# export PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.6,max_split_size_mb:128,expandable_segments:True

torchrun \
    --nnodes=${NNODES} \
    --nproc_per_node=${GPUS_PER_NODE} \
    --node_rank=\$SLURM_NODEID \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    train/joint_train_unified_model.py \
    --dataset_config_file data/configs/joint_train.yaml \
    --layer_module Qwen2VLMoTDecoderLayer \
    --vit_path ${MODEL_PATH} \
    --dino_path facebook/dinov2-with-registers-large \
    --llm_path ${MODEL_PATH} \
    --model_path ${MODEL_PATH} \
    --use_flex True \
    --expected_num_tokens 40960 \
    --max_num_tokens 40960 \
    --max_num_tokens_per_sample 40960 \
    --wandb_project G2VLM \
    --wandb_name ${wandb_name} \
    --wandb_offline True \
    --wandb_resume allow \
    --checkpoint_dir ${checkpoint_dir} \
    --llm_qk_norm True \
    --finetune_from_hf True \
    --auto_resume False \
    --resume-model-only True \
    --finetune-from-ema False \
    --enable_ema_model False \
    --resume_from ${PRETRAINED_CHECKPOINT} \
    --finetune_dino_from_hf False \
    --copy_init_moe False \
    --visual_und True \
    --visual_recon True \
    --freeze_dino True \
    --freeze_vit True \
    --freeze_und False \
    --freeze_recon False \
    --joint_train_recon False \
    --pretrain_train_recon False \
    --results_dir ${output_dir} \
    --save_every 500 \
    --total_steps 12000 \
    --warmup_steps 400 \
    --log_every 1 \
    --sharding_strategy HYBRID_SHARD \
    --cpu_offload True \
    --num_replicate=${NUM_REPLICATE} \
    --lr 2e-5 \
    --lr_scheduler cosine \
    --num_shard=${NUM_SHARD} \
    --num_workers 4"