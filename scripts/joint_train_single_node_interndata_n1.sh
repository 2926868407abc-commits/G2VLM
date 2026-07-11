#!/bin/bash

set -x -e

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "${REPO_ROOT}"

export NCCL_DEBUG=${NCCL_DEBUG:-INFO}
export NCCL_TIMEOUT=${NCCL_TIMEOUT:-18000000}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export MKL_THREADING_LAYER=${MKL_THREADING_LAYER:-GNU}
export HYDRA_FULL_ERROR=${HYDRA_FULL_ERROR:-1}
export NCCL_ASYNC_ERROR_HANDLING=${NCCL_ASYNC_ERROR_HANDLING:-1}
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/modeling:${PYTHONPATH:-}"
export DATA_ROOT=${DATA_ROOT:-/mnt/data/wangqq/G2VLM/data}
export HF_HOME=${HF_HOME:-${DATA_ROOT}/.cache/huggingface}
export HF_HUB_ETAG_TIMEOUT=${HF_HUB_ETAG_TIMEOUT:-120}
export HF_HUB_DOWNLOAD_TIMEOUT=${HF_HUB_DOWNLOAD_TIMEOUT:-120}

GPUS_PER_NODE=${GPUS_PER_NODE:-4}

MASTER_PORT=${MASTER_PORT:-29501}
MODEL_PATH=${MODEL_PATH:-InternRobotics/G2VLM-Qwen2-VL-2B}
PRETRAINED_CHECKPOINT=${PRETRAINED_CHECKPOINT:-InternRobotics/G2VLM-2B-MoT}
TOTAL_STEPS=${TOTAL_STEPS:-1000}
SAVE_EVERY=${SAVE_EVERY:-200}
WARMUP_STEPS=${WARMUP_STEPS:-50}
NUM_WORKERS=${NUM_WORKERS:-2}
EXPECTED_NUM_TOKENS=${EXPECTED_NUM_TOKENS:-40960}
MAX_NUM_TOKENS=${MAX_NUM_TOKENS:-40960}
MAX_NUM_TOKENS_PER_SAMPLE=${MAX_NUM_TOKENS_PER_SAMPLE:-40960}
USE_FLEX=${USE_FLEX:-True}

MODEL_PATH=$(python scripts/resolve_hf_repo.py \
    --repo-or-path "${MODEL_PATH}" \
    --local-root "${DATA_ROOT}/models" \
    --mode config \
    --required text_config.json vit_config.json dino_config.json)
PRETRAINED_CHECKPOINT=$(python scripts/resolve_hf_repo.py \
    --repo-or-path "${PRETRAINED_CHECKPOINT}" \
    --local-root "${DATA_ROOT}/models" \
    --mode full \
    --required model.safetensors)
echo "[train] MODEL_PATH=${MODEL_PATH}"
echo "[train] PRETRAINED_CHECKPOINT=${PRETRAINED_CHECKPOINT}"

name="g2vlm_interndata_n1_replica_${GPUS_PER_NODE}g_$(date +%Y%m%d_%H%M%S)"
output_dir="./checkpoints/${name}/"
checkpoint_dir="./checkpoints/${name}"
mkdir -p "${output_dir}" "${checkpoint_dir}"

TRAIN_ARGS=(
    --dataset_config_file data/configs/joint_train_interndata_n1_replica.yaml
    --layer_module Qwen2VLMoTDecoderLayer
    --vit_path "${MODEL_PATH}"
    --dino_path facebook/dinov2-with-registers-large
    --llm_path "${MODEL_PATH}"
    --model_path "${MODEL_PATH}"
    --use_flex "${USE_FLEX}"
    --expected_num_tokens "${EXPECTED_NUM_TOKENS}"
    --max_num_tokens "${MAX_NUM_TOKENS}"
    --max_num_tokens_per_sample "${MAX_NUM_TOKENS_PER_SAMPLE}"
    --wandb_project G2VLM
    --wandb_name "${name}"
    --wandb_offline True
    --wandb_resume allow
    --checkpoint_dir "${checkpoint_dir}"
    --llm_qk_norm True
    --finetune_from_hf True
    --auto_resume False
    --resume-model-only True
    --finetune-from-ema False
    --enable_ema_model False
    --resume_from "${PRETRAINED_CHECKPOINT}"
    --finetune_dino_from_hf False
    --copy_init_moe False
    --visual_und True
    --visual_recon True
    --freeze_dino True
    --freeze_vit True
    --freeze_und False
    --freeze_recon False
    --joint_train_recon False
    --pretrain_train_recon False
    --results_dir "${output_dir}"
    --save_every "${SAVE_EVERY}"
    --total_steps "${TOTAL_STEPS}"
    --warmup_steps "${WARMUP_STEPS}"
    --log_every 1
    --sharding_strategy FULL_SHARD
    --cpu_offload True
    --num_replicate=1
    --num_shard="${GPUS_PER_NODE}"
    --lr 2e-5
    --lr_scheduler cosine
    --num_workers "${NUM_WORKERS}"
)

torchrun \
    --nnodes=1 \
    --nproc_per_node="${GPUS_PER_NODE}" \
    --master_port="${MASTER_PORT}" \
    train/joint_train_unified_model.py \
    "${TRAIN_ARGS[@]}"
