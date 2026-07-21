# InternData-N1 Replica Training Flow

This note records the current smoke-test path for adapting G2VLM to InternData-N1
`vln_n1/traj_data/replica_d435i` navigation data.

## 1. Activate Environment

```bash
cd /mnt/data/wangqq/G2VLM
source /mnt/data/wangqq/G2VLM/envs/g2vlm/bin/activate
```

The scripts default to this data root:

```bash
export DATA_ROOT=/mnt/data/wangqq/G2VLM/data
export G2VLM_INTERNDATA_N1_REPLICA_ROOT=${DATA_ROOT}/g2vlm_interndata_n1/replica_d435i
```

You only need to override these if the dataset is moved.

Keep `huggingface_hub` compatible with `transformers==4.49.0`:

```bash
python -m pip install --force-reinstall "huggingface_hub==0.29.1"
python -m pip install easydict==1.13 h5py imageio numpy-quaternion omegaconf plyfile prettytable timm
```

## 2. Required Files To Sync

Sync these files from the working tree to the server before running the flow:

```text
data/preprocessing/convert_interndata_n1_replica_to_g2vlm.py
data/preprocessing/preview_interndata_n1_goal.py
data/configs/joint_train_interndata_n1_replica.yaml
data/dataset_info.py
data/interleave_datasets/__init__.py
data/interleave_datasets/recon_then_und_dataset.py
data/interleave_datasets/draw_marker.py
data/draw_marker.py
scripts/joint_train_single_node_interndata_n1.sh
scripts/chunk_train_interndata_n1.sh
scripts/check_interndata_n1_tar_format.py
scripts/estimate_interndata_n1_space_local.py
scripts/doctor_interndata_n1_replica.py
scripts/export_interndata_n1_checkpoint.py
scripts/infer_interndata_n1_replica_sample.py
scripts/list_interndata_n1_sync_files.py
scripts/smoke_train_interndata_n1.sh
scripts/prepare_interndata_n1_replica.sh
scripts/resolve_hf_repo.py
modeling/g2vlm/g2vlm.py
modeling/g2vlm/qwen2vl.py
modeling/pi3/models/pi3_loss.py
train/joint_train_unified_model.py
train/pretrain_unified_model.py
train/fsdp_utils.py
g2vlm_utils.py
requirements.txt
docs/interndata_n1_replica_training.md
```

You can print and verify this list locally with:

```bash
python scripts/list_interndata_n1_sync_files.py --check
```

## 3. Check Current Status

Before converting or training, inspect what is already on disk:

```bash
python scripts/doctor_interndata_n1_replica.py
```

This prints downloaded tar count, extracted scenes, converted parquet rows,
goal-pixel coverage, free disk space, and a suggested next step.

## 4. Convert A Small Smoke-Test Set

This converts 20 samples from `apartment_2` and adds pixel-goal supervision:

```bash
python data/preprocessing/convert_interndata_n1_replica_to_g2vlm.py \
  --input-root /mnt/data/wangqq/G2VLM/data/InternData-N1-extracted/vln_n1/traj_data/replica_d435i \
  --output-root /mnt/data/wangqq/G2VLM/data/g2vlm_interndata_n1/replica_d435i \
  --scenes apartment_2 \
  --frames-per-sample 8 \
  --max-samples 20
```

Expected output includes:

```text
Converted rows: 20
Parquet: /mnt/data/wangqq/G2VLM/data/g2vlm_interndata_n1/replica_d435i/parquets/interndata_n1_replica_d435i.parquet
Parquet info: /mnt/data/wangqq/G2VLM/data/g2vlm_interndata_n1/replica_d435i/parquet_info.json
```

## 5. Preview Pixel Goal

Generate a contact-sheet preview for one converted row:

```bash
python data/preprocessing/preview_interndata_n1_goal.py --row 0
```

The script validates image/depth paths, pose shapes, intrinsics, and goal-pixel
metadata. It saves a preview under:

```text
/mnt/data/wangqq/G2VLM/data/g2vlm_interndata_n1/replica_d435i/previews/
```

The green marker is the current pixel-goal supervision target.
It is only drawn by the preview script for inspection. Training inputs use the
original RGB frames and do not draw the goal marker, so the model has to infer
the target from the trajectory and instruction.

## 6. Export One Training/Validation Sample

Before loading the model, export one converted row as JSON:

```bash
python scripts/infer_interndata_n1_replica_sample.py --row 0 --dry-run
```

This verifies the 8 RGB/depth paths, prints the exact question and label, and
saves a JSON record under:

```text
/mnt/data/wangqq/G2VLM/data/g2vlm_interndata_n1/replica_d435i/inference_samples/
```

After smoke training, the same script can run a model-style sanity check with an
HF-format checkpoint path:

```bash
python scripts/infer_interndata_n1_replica_sample.py \
  --row 0 \
  --model-path InternRobotics/G2VLM-2B-MoT
```

For `--dry-run`, no model files are loaded or downloaded. For real inference,
passing a HF repo id downloads the full model weights under `${DATA_ROOT}/models`.
The training script saves step checkpoints, so export them first before using a
trained checkpoint as `--model-path`.

## 7. Single-GPU Smoke Test

Run the checked smoke path:

```bash
bash scripts/smoke_train_interndata_n1.sh
```

This script checks environment versions, verifies the converted parquet, and runs:

```bash
CUDA_VISIBLE_DEVICES=0 GPUS_PER_NODE=1 TOTAL_STEPS=20 SAVE_EVERY=20 WARMUP_STEPS=2 NUM_WORKERS=1 \
bash scripts/joint_train_single_node_interndata_n1.sh
```

Success means the process enters the training loop and prints loss.
When `TOTAL_STEPS` is reached, `train/joint_train_unified_model.py` should exit
with status 0 after logging `Done!`.
On the first run, `scripts/joint_train_single_node_interndata_n1.sh` resolves HF
repo ids into local directories under `${DATA_ROOT}/models`. It downloads small
config/tokenizer files for `G2VLM-Qwen2-VL-2B` and the pretrained
`G2VLM-2B-MoT` weights before starting `torchrun`.

The dataset config keeps `num_used_data: 1` because this codebase interprets it
as the number of parquet files to sample. The converter writes all converted
rows into one parquet file, so `1` still lets training iterate through all rows
inside that file.

## 8. Chunked Large Training

For large runs, do not extract all InternData-N1 tar files at once. The converted
parquet stores RGB/depth file paths, so training needs the extracted images to
exist while that batch is training. Use the chunked launcher to extract a small
set of tar files, convert them, train to the next checkpoint, then delete only
that temporary extracted chunk:

```bash
cd /data/wqq/G2VLM
source /data/wqq/G2VLM/env.sh

tmux new -s g2vlm_chunk_train
```

Inside tmux:

```bash
cd /data/wqq/G2VLM
source /data/wqq/G2VLM/env.sh

CUDA_VISIBLE_DEVICES=0 \
GPUS_PER_NODE=1 \
SUBSETS="3dfront_d435i gibson_d435i hm3d_d435i" \
TARS_PER_CHUNK=1 \
STEPS_PER_CHUNK=200 \
FRAMES_PER_SAMPLE=8 \
NUM_WORKERS=1 \
bash scripts/chunk_train_interndata_n1.sh
```

The script keeps original `.tar.gz` files, deletes only temporary extracted
chunks under `${DATA_ROOT}/InternData-N1-extracted-chunks`, and resumes from the
same checkpoint directory:

```text
./checkpoints/g2vlm_interndata_n1_chunked
```

For a quick trial, add `MAX_CHUNKS=1 STEPS_PER_CHUNK=20`. If the GPU reports a
FlexAttention/shared-memory error, retry with:

```bash
USE_FLEX=False \
EXPECTED_NUM_TOKENS=18000 \
MAX_NUM_TOKENS=20000 \
MAX_NUM_TOKENS_PER_SAMPLE=20000
```

## 9. Enable Camera Pose And Depth Loss

The default smoke path keeps `joint_train_recon=False`, so it only checks the
text-answer CE loss. To train the geometry heads as well, enable recon loss:

```bash
CUDA_VISIBLE_DEVICES=0 \
GPUS_PER_NODE=1 \
TOTAL_STEPS=2 \
SAVE_EVERY=2 \
WARMUP_STEPS=1 \
NUM_WORKERS=1 \
EXPECTED_NUM_TOKENS=18000 \
MAX_NUM_TOKENS=20000 \
MAX_NUM_TOKENS_PER_SAMPLE=20000 \
USE_FLEX=False \
JOINT_TRAIN_RECON=True \
PI3_DEPTH_WEIGHT=0.5 \
PI3_CAMERA_WEIGHT=0.2 \
bash scripts/smoke_train_interndata_n1.sh
```

With `JOINT_TRAIN_RECON=True`, the total geometry loss `dl` includes:

```text
local_pts_loss   xyz point/depth supervision from depth maps
depth_loss       extra z-channel depth supervision
global_pts_loss  global point supervision, when available
normal_loss      surface-normal consistency from neighboring depth pixels
camera_loss      weighted relative camera pose loss
trans_loss       camera relative translation term
rot_loss         camera relative rotation angle term
```

`PI3_DEPTH_WEIGHT` controls only the extra z-channel depth term. The original
`local_pts_loss` already supervises depth through the z coordinate, so increasing
`PI3_DEPTH_WEIGHT` makes depth matter more. If the run OOMs, keep
`JOINT_TRAIN_RECON=True` but reduce token limits or the number of frames first.

## 10. Export A Smoke-Test Checkpoint

After the smoke run saves a step directory, create an inference-ready directory
by combining the trained `model.safetensors` with base model config/tokenizer
files:

```bash
python scripts/export_interndata_n1_checkpoint.py \
  --checkpoint checkpoints/<run_name>/0000020
```

By default, the script uses `InternRobotics/G2VLM-2B-MoT` and downloads only
small config/tokenizer files, not the base model weights. If you already have a
local HF-format base model directory, pass it with `--base-model-path` or set
`G2VLM_BASE_MODEL_PATH`.

The export script prints a follow-up command like:

```bash
python scripts/infer_interndata_n1_replica_sample.py \
  --row 0 \
  --model-path checkpoints/<run_name>/hf_export_0000020
```

By default the large `model.safetensors` is symlinked to save disk space. Add
`--copy-weights` if the exported directory must be self-contained.

## 10. Prepare All Downloaded Replica Scenes

After the single-GPU smoke test passes, convert all downloaded
`replica_d435i/*.tar.gz` scenes:

```bash
bash scripts/prepare_interndata_n1_replica.sh
```

For a capped conversion:

```bash
MAX_SAMPLES=1000 bash scripts/prepare_interndata_n1_replica.sh
```

## 11. Four-GPU Training

After the smoke test and larger conversion pass:

```bash
GPUS_PER_NODE=4 TOTAL_STEPS=1000 SAVE_EVERY=200 NUM_WORKERS=2 JOINT_TRAIN_RECON=True PI3_DEPTH_WEIGHT=0.5 PI3_CAMERA_WEIGHT=0.2 \
bash scripts/joint_train_single_node_interndata_n1.sh
```

## Current Limitation

The current pixel-goal label is derived from the final pose/action projected into
an observed frame when possible, otherwise it falls back to the final-frame center.
This is a practical smoke-test supervision signal. For final task quality, replace
or refine it with the dataset's most authoritative goal annotation if available.
