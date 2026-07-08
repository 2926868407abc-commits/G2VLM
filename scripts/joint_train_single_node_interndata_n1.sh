#!/bin/bash

set -x -e

export NCCL_DEBUG=${NCCL_DEBUG:-INFO}
export NCCL_TIMEOUT=${NCCL_TIMEOUT:-18000000}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export MKL_THREADING_LAYER=${MKL_THREADING_LAYER:-GNU}
export HYDRA_FULL_ERROR=${HYDRA_FULL_ERROR:-1}
export NCCL_ASYNC_ERROR_HANDLING=${NCCL_ASYNC_ERROR_HANDLING:-1}

if [ -z "${GPUS_PER_NODE:-}" ]; then
    GPUS_PER_NODE=$(nvidia-smi -L | wc -l)
fi

MASTER_PORT=${MASTER_PORT:-29501}
MODEL_PATH=${MODEL_PATH:-InternRobotics/G2VLM-Qwen2-VL-2B}
PRETRAINED_CHECKPOINT=${PRETRAINED_CHECKPOINT:-InternRobotics/G2VLM-2B-MoT}
TOTAL_STEPS=${TOTAL_STEPS:-1000}
SAVE_EVERY=${SAVE_EVERY:-200}
WARMUP_STEPS=${WARMUP_STEPS:-50}
NUM_WORKERS=${NUM_WORKERS:-2}

name="g2vlm_interndata_n1_replica_${GPUS_PER_NODE}g_$(date +%Y%m%d_%H%M%S)"
output_dir="./checkpoints/${name}/"
checkpoint_dir="./checkpoints/${name}"
mkdir -p "${output_dir}" "${checkpoint_dir}"

torchrun \
    --nnodes=1 \
    --nproc_per_node=${GPUS_PER_NODE} \
    --master_port=${MASTER_PORT} \
    train/joint_train_unified_model.py \
    --dataset_config_file data/configs/joint_train_interndata_n1_replica.yaml \
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
    --wandb_name ${name} \
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
    --save_every ${SAVE_EVERY} \
    --total_steps ${TOTAL_STEPS} \
    --warmup_steps ${WARMUP_STEPS} \
    --log_every 1 \
    --sharding_strategy FULL_SHARD \
    --cpu_offload True \
    --num_replicate=1 \
    --num_shard=${GPUS_PER_NODE} \
    --lr 2e-5 \
    --lr_scheduler cosine \
    --num_workers ${NUM_WORKERS}
