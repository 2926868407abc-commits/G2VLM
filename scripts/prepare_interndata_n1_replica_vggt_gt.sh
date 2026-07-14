#!/bin/bash

set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "${REPO_ROOT}"

if [ -z "${VIRTUAL_ENV:-}" ] && [ -f "${REPO_ROOT}/envs/g2vlm/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "${REPO_ROOT}/envs/g2vlm/bin/activate"
fi

DATA_ROOT=${DATA_ROOT:-/mnt/wqq/G2VLM/data}
TAR_ROOT=${TAR_ROOT:-${DATA_ROOT}/InternData-N1/vln_n1/traj_data/replica_d435i}
EXTRACT_ROOT=${EXTRACT_ROOT:-${DATA_ROOT}/InternData-N1-extracted/vln_n1/traj_data/replica_d435i}
OUTPUT_ROOT=${OUTPUT_ROOT:-${DATA_ROOT}/g2vlm_interndata_n1/replica_d435i_vggt_gt}
FRAMES_PER_SAMPLE=${FRAMES_PER_SAMPLE:-8}
MAX_SAMPLES=${MAX_SAMPLES:-0}
VGGT_MODEL=${VGGT_MODEL:-facebook/VGGT-1B}
DEVICE=${DEVICE:-cuda}
LOAD_IMG_SIZE=${LOAD_IMG_SIZE:-518}
ALIGN_DEPTH_TO_SENSOR=${ALIGN_DEPTH_TO_SENSOR:-median}
SCENES=${SCENES:-}

mkdir -p "${EXTRACT_ROOT}" "${OUTPUT_ROOT}"

if [ ! -d "${TAR_ROOT}" ]; then
    echo "Missing tar directory: ${TAR_ROOT}" >&2
    echo "Download replica_d435i tar files before running this script." >&2
    exit 1
fi

shopt -s nullglob
tar_files=("${TAR_ROOT}"/*.tar.gz)
if [ ${#tar_files[@]} -eq 0 ]; then
    echo "No .tar.gz files found in ${TAR_ROOT}" >&2
    exit 1
fi

for tar_file in "${tar_files[@]}"; do
    scene=$(basename "${tar_file}" .tar.gz)
    if [ -n "${SCENES}" ] && [[ " ${SCENES} " != *" ${scene} "* ]]; then
        continue
    fi
    if [ -f "${EXTRACT_ROOT}/${scene}/meta/episodes.jsonl" ]; then
        echo "[prepare-vggt] skip extracted scene: ${scene}"
        continue
    fi
    echo "[prepare-vggt] extracting ${tar_file}"
    tar -xzf "${tar_file}" -C "${EXTRACT_ROOT}"
done

convert_args=(
    --input-root "${EXTRACT_ROOT}"
    --output-root "${OUTPUT_ROOT}"
    --frames-per-sample "${FRAMES_PER_SAMPLE}"
    --vggt-model "${VGGT_MODEL}"
    --device "${DEVICE}"
    --load-img-size "${LOAD_IMG_SIZE}"
    --align-depth-to-sensor "${ALIGN_DEPTH_TO_SENSOR}"
)

if [ "${MAX_SAMPLES}" != "0" ]; then
    convert_args+=(--max-samples "${MAX_SAMPLES}")
fi

if [ -n "${SCENES}" ]; then
    read -r -a scene_args <<< "${SCENES}"
    convert_args+=(--scenes "${scene_args[@]}")
fi

python data/preprocessing/convert_interndata_n1_replica_with_vggt_gt.py "${convert_args[@]}"

echo "[prepare-vggt] done"
echo "[prepare-vggt] parquet: ${OUTPUT_ROOT}/parquets/interndata_n1_replica_d435i_vggt_gt.parquet"
echo "[prepare-vggt] parquet_info: ${OUTPUT_ROOT}/parquet_info.json"
echo "[prepare-vggt] train with: export G2VLM_INTERNDATA_N1_REPLICA_ROOT=${OUTPUT_ROOT}"
