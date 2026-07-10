#!/bin/bash

set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "${REPO_ROOT}"

if [ -z "${VIRTUAL_ENV:-}" ] && [ -f "${REPO_ROOT}/envs/g2vlm/bin/activate" ]; then
    # Convenience for the server layout used in this project.
    # shellcheck disable=SC1091
    source "${REPO_ROOT}/envs/g2vlm/bin/activate"
fi

export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export DATA_ROOT=${DATA_ROOT:-/mnt/data/wangqq/G2VLM/data}
export HF_HOME=${HF_HOME:-${DATA_ROOT}/.cache/huggingface}
export HF_HUB_ETAG_TIMEOUT=${HF_HUB_ETAG_TIMEOUT:-120}
export HF_HUB_DOWNLOAD_TIMEOUT=${HF_HUB_DOWNLOAD_TIMEOUT:-120}
export G2VLM_INTERNDATA_N1_REPLICA_ROOT=${G2VLM_INTERNDATA_N1_REPLICA_ROOT:-${DATA_ROOT}/g2vlm_interndata_n1/replica_d435i}
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/modeling:${PYTHONPATH:-}"

python - <<'PY'
import json
import os
from pathlib import Path

import huggingface_hub
import pyarrow.parquet as pq
import torch
from torch.nn.attention.flex_attention import and_masks, or_masks

repo = Path.cwd()
converted_root = Path(os.environ["G2VLM_INTERNDATA_N1_REPLICA_ROOT"])
parquet_path = converted_root / "parquets" / "interndata_n1_replica_d435i.parquet"
parquet_info = converted_root / "parquet_info.json"
required_repo_files = [
    repo / "train" / "joint_train_unified_model.py",
    repo / "data" / "dataset_info.py",
    repo / "data" / "configs" / "joint_train_interndata_n1_replica.yaml",
    repo / "data" / "interleave_datasets" / "recon_then_und_dataset.py",
    repo / "data" / "interleave_datasets" / "draw_marker.py",
]

print("python:", __import__("sys").executable)
print("torch:", torch.__version__, "cuda:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu0:", torch.cuda.get_device_name(0))
print("torch flex_attention import: ok", and_masks.__name__, or_masks.__name__)
print("huggingface_hub:", huggingface_hub.__version__)

major = int(huggingface_hub.__version__.split(".", 1)[0])
if major >= 1:
    raise SystemExit(
        "huggingface_hub must be <1.0 for transformers==4.49.0. "
        "Run: python -m pip install --force-reinstall 'huggingface_hub==0.29.1'"
    )

import transformers
from easydict import EasyDict
from pi3.utils.geometry import depthmap_to_absolute_camera_coordinates

print("transformers:", transformers.__version__)
print("easydict import: ok", EasyDict.__name__)
print("pi3 geometry import: ok", depthmap_to_absolute_camera_coordinates.__name__)

for path in [*required_repo_files, parquet_path, parquet_info]:
    if not path.exists():
        raise SystemExit(f"missing required file: {path}")

dataset_info_text = (repo / "data" / "dataset_info.py").read_text()
if "intern_n1_replica_d435i" not in dataset_info_text:
    raise SystemExit("dataset_info.py is missing intern_n1_replica_d435i")

loader_text = (repo / "data" / "interleave_datasets" / "recon_then_und_dataset.py").read_text()
if "from .interleave_dataset import" not in loader_text:
    raise SystemExit("recon_then_und_dataset.py should import from .interleave_dataset; sync the latest file")
if "scene_name'] == 'replica'" not in loader_text and 'scene_name"] == "replica"' not in loader_text:
    raise SystemExit("recon_then_und_dataset.py is missing replica depth support")
if 'metadata[\'type\'] == "nav"' not in loader_text and 'metadata["type"] == "nav"' not in loader_text:
    raise SystemExit("recon_then_und_dataset.py is missing raw-RGB nav handling")

draw_marker_text = (repo / "data" / "interleave_datasets" / "draw_marker.py").read_text()
if "draw_goal_on_input" not in draw_marker_text:
    raise SystemExit("draw_marker.py is missing the nav goal-marker leakage guard")

train_text = (repo / "train" / "joint_train_unified_model.py").read_text()
if "total_mse_tokens = torch.tensor(0" not in train_text:
    raise SystemExit("joint_train_unified_model.py is missing the total_mse_tokens smoke-run fix")

launch_text = (repo / "scripts" / "joint_train_single_node_interndata_n1.sh").read_text()
for needle in ["PYTHONPATH", "TRAIN_ARGS=(", "resolve_hf_repo.py"]:
    if needle not in launch_text:
        raise SystemExit(f"joint_train_single_node_interndata_n1.sh is missing {needle}; sync the latest script")

table = pq.read_table(parquet_path)
print("converted rows:", table.num_rows)
print("parquet_info:", json.loads(parquet_info.read_text()))
if table.num_rows == 0:
    raise SystemExit("converted parquet is empty")

row = table.slice(0, 1).to_pylist()[0]
metadata = row.get("metadata", "")
if "goal_pixel" not in str(metadata):
    raise SystemExit("converted parquet metadata is missing goal_pixel; rerun the converter")
PY

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
GPUS_PER_NODE=${GPUS_PER_NODE:-1} \
TOTAL_STEPS=${TOTAL_STEPS:-20} \
SAVE_EVERY=${SAVE_EVERY:-20} \
WARMUP_STEPS=${WARMUP_STEPS:-2} \
NUM_WORKERS=${NUM_WORKERS:-1} \
bash scripts/joint_train_single_node_interndata_n1.sh
