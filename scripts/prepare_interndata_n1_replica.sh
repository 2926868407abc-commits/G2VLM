#!/bin/bash

set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "${REPO_ROOT}"

if [ -z "${VIRTUAL_ENV:-}" ] && [ -f "${REPO_ROOT}/envs/g2vlm/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "${REPO_ROOT}/envs/g2vlm/bin/activate"
fi

DATA_ROOT=${DATA_ROOT:-/mnt/data/wangqq/G2VLM/data}
TAR_ROOT=${TAR_ROOT:-${DATA_ROOT}/InternData-N1/vln_n1/traj_data/replica_d435i}
EXTRACT_ROOT=${EXTRACT_ROOT:-${DATA_ROOT}/InternData-N1-extracted/vln_n1/traj_data/replica_d435i}
OUTPUT_ROOT=${OUTPUT_ROOT:-${DATA_ROOT}/g2vlm_interndata_n1/replica_d435i}
export G2VLM_INTERNDATA_N1_REPLICA_ROOT=${G2VLM_INTERNDATA_N1_REPLICA_ROOT:-${OUTPUT_ROOT}}
FRAMES_PER_SAMPLE=${FRAMES_PER_SAMPLE:-8}
MAX_SAMPLES=${MAX_SAMPLES:-0}

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
    if [ -f "${EXTRACT_ROOT}/${scene}/meta/episodes.jsonl" ]; then
        echo "[prepare] skip extracted scene: ${scene}"
        continue
    fi
    echo "[prepare] extracting ${tar_file}"
    tar -xzf "${tar_file}" -C "${EXTRACT_ROOT}"
done

convert_args=(
    --input-root "${EXTRACT_ROOT}"
    --output-root "${OUTPUT_ROOT}"
    --frames-per-sample "${FRAMES_PER_SAMPLE}"
)

if [ "${MAX_SAMPLES}" != "0" ]; then
    convert_args+=(--max-samples "${MAX_SAMPLES}")
fi

python data/preprocessing/convert_interndata_n1_replica_to_g2vlm.py "${convert_args[@]}"

echo "[prepare] done"
echo "[prepare] parquet: ${OUTPUT_ROOT}/parquets/interndata_n1_replica_d435i.parquet"
echo "[prepare] parquet_info: ${OUTPUT_ROOT}/parquet_info.json"
