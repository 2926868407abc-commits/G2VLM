#!/bin/bash

set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "${REPO_ROOT}"

if [ -z "${VIRTUAL_ENV:-}" ] && [ -f "${REPO_ROOT}/envs/g2vlm/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "${REPO_ROOT}/envs/g2vlm/bin/activate"
fi

export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/modeling:${PYTHONPATH:-}"
export DATA_ROOT=${DATA_ROOT:-/data/wqq/G2VLM/data}
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export HF_HOME=${HF_HOME:-${DATA_ROOT}/.cache/huggingface}
export HF_HUB_ETAG_TIMEOUT=${HF_HUB_ETAG_TIMEOUT:-120}
export HF_HUB_DOWNLOAD_TIMEOUT=${HF_HUB_DOWNLOAD_TIMEOUT:-120}

TAR_BASE=${TAR_BASE:-${DATA_ROOT}/InternData-N1/vln_n1/traj_data}
EXTRACT_BASE=${EXTRACT_BASE:-${DATA_ROOT}/InternData-N1-extracted-chunks/vln_n1/traj_data}
CONVERT_BASE=${CONVERT_BASE:-${DATA_ROOT}/g2vlm_interndata_n1/chunks}
SUBSETS=${SUBSETS:-"3dfront_d435i gibson_d435i hm3d_d435i"}
TARS_PER_CHUNK=${TARS_PER_CHUNK:-1}
MAX_CHUNKS=${MAX_CHUNKS:-0}
FRAMES_PER_SAMPLE=${FRAMES_PER_SAMPLE:-8}
MAX_SAMPLES_PER_CHUNK=${MAX_SAMPLES_PER_CHUNK:-0}
SCENE_NAME=${SCENE_NAME:-replica}
DATASET_NAME=${DATASET_NAME:-spar_interndata_n1_chunk}
RUN_NAME=${RUN_NAME:-g2vlm_interndata_n1_chunked}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-./checkpoints/${RUN_NAME}}
OUTPUT_DIR=${OUTPUT_DIR:-./checkpoints/${RUN_NAME}}
STEPS_PER_CHUNK=${STEPS_PER_CHUNK:-200}
SAVE_EVERY=${SAVE_EVERY:-${STEPS_PER_CHUNK}}
WARMUP_STEPS=${WARMUP_STEPS:-50}
DELETE_EXTRACTED=${DELETE_EXTRACTED:-1}
STATE_DIR=${STATE_DIR:-${DATA_ROOT}/g2vlm_interndata_n1/chunk_state/${RUN_NAME}}
JOINT_TRAIN_RECON=${JOINT_TRAIN_RECON:-True}
PRETRAIN_TRAIN_RECON=${PRETRAIN_TRAIN_RECON:-False}
PI3_POINT_WEIGHT=${PI3_POINT_WEIGHT:-1.0}
PI3_DEPTH_WEIGHT=${PI3_DEPTH_WEIGHT:-0.5}
PI3_CAMERA_WEIGHT=${PI3_CAMERA_WEIGHT:-0.2}

mkdir -p "${EXTRACT_BASE}" "${CONVERT_BASE}" "${CHECKPOINT_DIR}" "${OUTPUT_DIR}" "${STATE_DIR}/done"

latest_step() {
    local max_step=-1
    local path base
    shopt -s nullglob
    for path in "${CHECKPOINT_DIR}"/[0-9]*; do
        [ -d "${path}" ] || continue
        base=$(basename "${path}")
        [[ "${base}" =~ ^[0-9]+$ ]] || continue
        if [ "${base}" -gt "${max_step}" ]; then
            max_step=${base}
        fi
    done
    shopt -u nullglob
    echo "${max_step}"
}

safe_remove_extract_root() {
    local target=$1
    local resolved_target resolved_base
    resolved_target=$(realpath -m "${target}")
    resolved_base=$(realpath -m "${EXTRACT_BASE}")
    case "${resolved_target}" in
        "${resolved_base}"/*)
            rm -rf "${resolved_target}"
            ;;
        *)
            echo "Refusing to delete outside EXTRACT_BASE: ${resolved_target}" >&2
            exit 1
            ;;
    esac
}

chunk_count=0
for subset in ${SUBSETS}; do
    tar_dir="${TAR_BASE}/${subset}"
    if [ ! -d "${tar_dir}" ]; then
        echo "[chunk] skip missing subset dir: ${tar_dir}"
        continue
    fi

    mapfile -t tar_files < <(find "${tar_dir}" -maxdepth 1 -type f -name "*.tar.gz" | sort)
    if [ "${#tar_files[@]}" -eq 0 ]; then
        echo "[chunk] skip empty subset: ${subset}"
        continue
    fi

    echo "[chunk] subset=${subset} tar_files=${#tar_files[@]}"
    start=0
    chunk_index=0
    while [ "${start}" -lt "${#tar_files[@]}" ]; do
        end=$((start + TARS_PER_CHUNK))
        if [ "${end}" -gt "${#tar_files[@]}" ]; then
            end=${#tar_files[@]}
        fi

        chunk_id="${subset}_$(printf "%05d" "${chunk_index}")"
        done_file="${STATE_DIR}/done/${chunk_id}.done"
        if [ -f "${done_file}" ]; then
            echo "[chunk] skip done ${chunk_id}"
            start=${end}
            chunk_index=$((chunk_index + 1))
            continue
        fi

        if [ "${MAX_CHUNKS}" != "0" ] && [ "${chunk_count}" -ge "${MAX_CHUNKS}" ]; then
            echo "[chunk] reached MAX_CHUNKS=${MAX_CHUNKS}"
            exit 0
        fi

        extract_root="${EXTRACT_BASE}/${subset}/${chunk_id}"
        output_root="${CONVERT_BASE}/${chunk_id}"
        mkdir -p "${extract_root}" "${output_root}"

        echo "[chunk] extracting ${chunk_id}: indexes ${start}..$((end - 1))"
        for ((i=start; i<end; i++)); do
            echo "[chunk] tar: ${tar_files[$i]}"
            tar -xzf "${tar_files[$i]}" -C "${extract_root}"
        done

        convert_args=(
            --input-root "${extract_root}"
            --output-root "${output_root}"
            --frames-per-sample "${FRAMES_PER_SAMPLE}"
            --scene-name "${SCENE_NAME}"
            --dataset-name "${DATASET_NAME}"
            --output-name "interndata_n1_${chunk_id}"
        )
        if [ "${MAX_SAMPLES_PER_CHUNK}" != "0" ]; then
            convert_args+=(--max-samples "${MAX_SAMPLES_PER_CHUNK}")
        fi

        echo "[chunk] converting ${chunk_id}"
        python data/preprocessing/convert_interndata_n1_replica_to_g2vlm.py "${convert_args[@]}"

        current_step=$(latest_step)
        if [ "${current_step}" -lt 0 ]; then
            target_step=${STEPS_PER_CHUNK}
            auto_resume=False
            resume_model_only=True
        else
            target_step=$((current_step + STEPS_PER_CHUNK))
            auto_resume=True
            resume_model_only=False
        fi

        echo "[chunk] training ${chunk_id}: current_step=${current_step} target_step=${target_step}"
        G2VLM_INTERNDATA_N1_REPLICA_ROOT="${output_root}" \
        RUN_NAME="${RUN_NAME}" \
        CHECKPOINT_DIR="${CHECKPOINT_DIR}" \
        OUTPUT_DIR="${OUTPUT_DIR}" \
        AUTO_RESUME="${auto_resume}" \
        RESUME_MODEL_ONLY="${resume_model_only}" \
        TOTAL_STEPS="${target_step}" \
        SAVE_EVERY="${SAVE_EVERY}" \
        WARMUP_STEPS="${WARMUP_STEPS}" \
        JOINT_TRAIN_RECON="${JOINT_TRAIN_RECON}" \
        PRETRAIN_TRAIN_RECON="${PRETRAIN_TRAIN_RECON}" \
        PI3_POINT_WEIGHT="${PI3_POINT_WEIGHT}" \
        PI3_DEPTH_WEIGHT="${PI3_DEPTH_WEIGHT}" \
        PI3_CAMERA_WEIGHT="${PI3_CAMERA_WEIGHT}" \
        bash scripts/joint_train_single_node_interndata_n1.sh

        date > "${done_file}"
        printf "subset=%s\nchunk_id=%s\nstart=%s\nend=%s\noutput_root=%s\n" \
            "${subset}" "${chunk_id}" "${start}" "$((end - 1))" "${output_root}" >> "${done_file}"

        if [ "${DELETE_EXTRACTED}" = "1" ]; then
            echo "[chunk] deleting extracted chunk: ${extract_root}"
            safe_remove_extract_root "${extract_root}"
        else
            echo "[chunk] keeping extracted chunk: ${extract_root}"
        fi

        chunk_count=$((chunk_count + 1))
        start=${end}
        chunk_index=$((chunk_index + 1))
    done
done

echo "[chunk] all available chunks finished"
