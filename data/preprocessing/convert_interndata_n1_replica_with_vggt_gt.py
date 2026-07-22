#!/usr/bin/env python3
"""Build G2VLM parquet rows using VGGT-predicted depth and camera pseudo-GT.

This script reuses the InternData-N1 Replica conversion logic, then runs VGGT on
each sampled RGB sequence. The resulting parquet keeps the existing G2VLM schema:

* ``image_list`` points to the original RGB images.
* ``depth_list`` points to VGGT depth maps saved as uint16 millimeter PNG files.
* ``poses`` contains VGGT camera-to-world matrices, flattened as 4x4 values.
* ``intrinsic`` and ``depth_intrinsic`` contain VGGT intrinsics in original image
  pixel coordinates, stored as 4x4 matrices for the current loader.

Original sensor depth and pose paths are preserved in ``metadata`` for audit.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
RECONS_ROOT = REPO_ROOT / "eval_code" / "recons"
for path in (REPO_ROOT, RECONS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from data.preprocessing.convert_interndata_n1_replica_to_g2vlm import (  # noqa: E402
    convert_scene,
    discover_scene_dirs,
)


def parse_metadata(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if value is None:
        return {}
    try:
        parsed = ast.literal_eval(str(value))
    except (SyntaxError, ValueError):
        return {"raw_metadata": str(value)}
    return parsed if isinstance(parsed, dict) else {"raw_metadata": parsed}


def matrix4(values: Any) -> np.ndarray:
    return np.asarray(values, dtype=np.float32).reshape(4, 4)


def matrix3_to_4x4(values: np.ndarray) -> list[float]:
    out = np.eye(4, dtype=np.float32)
    out[:3, :3] = np.asarray(values, dtype=np.float32).reshape(3, 3)
    return out.reshape(-1).tolist()


def sanitize_id(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    return value.strip("_") or "sample"


def load_sensor_depth_meters(path: str, target_hw: tuple[int, int]) -> np.ndarray:
    with Image.open(path) as image:
        depth = np.asarray(image).astype(np.float32)
    depth[~np.isfinite(depth)] = 0.0
    valid = depth > 0
    if valid.any() and float(np.nanmax(depth[valid])) > 100.0:
        depth = depth / 1000.0
    if depth.shape != target_hw:
        depth = np.asarray(
            Image.fromarray(depth.astype(np.float32), mode="F").resize(
                (target_hw[1], target_hw[0]),
                Image.Resampling.NEAREST,
            )
        )
    depth[~np.isfinite(depth)] = 0.0
    return depth


def median_depth_scale(
    predicted_depths: Iterable[np.ndarray],
    sensor_depth_paths: Iterable[str],
    min_valid_pixels: int,
) -> float:
    scales: list[float] = []
    for pred_depth, sensor_path in zip(predicted_depths, sensor_depth_paths):
        sensor_depth = load_sensor_depth_meters(sensor_path, pred_depth.shape)
        valid = (
            np.isfinite(pred_depth)
            & np.isfinite(sensor_depth)
            & (pred_depth > 1e-6)
            & (sensor_depth > 1e-6)
        )
        if int(valid.sum()) < min_valid_pixels:
            continue
        ratio = sensor_depth[valid] / pred_depth[valid]
        ratio = ratio[np.isfinite(ratio) & (ratio > 1e-6)]
        if ratio.size:
            scales.append(float(np.median(ratio)))
    if not scales:
        return 1.0
    scale = float(np.median(scales))
    if not np.isfinite(scale) or scale <= 0:
        return 1.0
    return scale


def save_depth_png(depth_meters: np.ndarray, path: Path, max_depth_m: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    depth_meters = np.asarray(depth_meters, dtype=np.float32)
    depth_meters[~np.isfinite(depth_meters)] = 0.0
    depth_mm = np.clip(depth_meters, 0.0, max_depth_m) * 1000.0
    Image.fromarray(depth_mm.round().astype(np.uint16)).save(path)


def preprocess_geometries(image_paths: list[str], load_img_size: int) -> list[dict[str, float]]:
    """Mirror VGGT load_fn geometry so predictions can be mapped back to originals."""

    items: list[dict[str, float]] = []
    for image_path in image_paths:
        with Image.open(image_path) as image:
            orig_w, orig_h = image.size
        resized_w = load_img_size
        resized_h = round(orig_h * (load_img_size / orig_w) / 14) * 14
        crop_top = 0
        proc_h = resized_h
        proc_w = resized_w
        if resized_h > load_img_size:
            crop_top = (resized_h - load_img_size) // 2
            proc_h = load_img_size
        items.append(
            {
                "orig_w": float(orig_w),
                "orig_h": float(orig_h),
                "resized_w": float(resized_w),
                "resized_h": float(resized_h),
                "proc_w": float(proc_w),
                "proc_h": float(proc_h),
                "crop_top": float(crop_top),
            }
        )

    final_h = max(int(item["proc_h"]) for item in items)
    final_w = max(int(item["proc_w"]) for item in items)
    for item in items:
        item["pad_top"] = float((final_h - int(item["proc_h"])) // 2)
        item["pad_left"] = float((final_w - int(item["proc_w"])) // 2)
        item["final_h"] = float(final_h)
        item["final_w"] = float(final_w)
    return items


def map_intrinsic_to_original(intrinsic: np.ndarray, geom: dict[str, float]) -> np.ndarray:
    intr = np.asarray(intrinsic, dtype=np.float32).copy()
    orig_w, orig_h = geom["orig_w"], geom["orig_h"]
    resized_w, resized_h = geom["resized_w"], geom["resized_h"]
    crop_top, pad_top, pad_left = geom["crop_top"], geom["pad_top"], geom["pad_left"]

    out = np.eye(3, dtype=np.float32)
    out[0, 0] = intr[0, 0] * orig_w / resized_w
    out[1, 1] = intr[1, 1] * orig_h / resized_h
    out[0, 2] = (intr[0, 2] - pad_left) * orig_w / resized_w
    out[1, 2] = (intr[1, 2] - pad_top + crop_top) * orig_h / resized_h
    return out


def map_depth_to_original(depth: np.ndarray, geom: dict[str, float]) -> np.ndarray:
    depth = np.asarray(depth, dtype=np.float32)
    proc_h, proc_w = int(geom["proc_h"]), int(geom["proc_w"])
    pad_top, pad_left = int(geom["pad_top"]), int(geom["pad_left"])
    crop_top = int(geom["crop_top"])
    resized_h, resized_w = int(geom["resized_h"]), int(geom["resized_w"])
    orig_h, orig_w = int(geom["orig_h"]), int(geom["orig_w"])

    unpadded = depth[pad_top : pad_top + proc_h, pad_left : pad_left + proc_w]
    resized_depth = np.zeros((resized_h, resized_w), dtype=np.float32)
    resized_depth[crop_top : crop_top + proc_h, :proc_w] = unpadded
    original = Image.fromarray(resized_depth, mode="F").resize(
        (orig_w, orig_h),
        Image.Resampling.BILINEAR,
    )
    original_depth = np.asarray(original).astype(np.float32)
    original_depth[~np.isfinite(original_depth)] = 0.0
    return original_depth


def load_vggt_model(model_name_or_path: str, device: str):
    import torch
    from models.fastmodel import VGGT

    torch_device = torch.device(device)
    if torch_device.type == "cuda" and torch_device.index is not None:
        torch.cuda.set_device(torch_device.index)
    model = VGGT(pretrained_model_name_or_path=model_name_or_path)
    model.to(torch_device)
    model.eval()
    return model


def infer_vggt(
    model,
    image_paths: list[str],
    device: str,
    load_img_size: int,
) -> tuple[list[np.ndarray], np.ndarray, list[np.ndarray], list[np.ndarray], tuple[int, int]]:
    import torch
    from models.vggt.utils.load_fn import load_and_preprocess_images
    from models.vggt.utils.pose_enc import pose_encoding_to_extri_intri

    torch_device = torch.device(device)
    device_type = torch_device.type
    images = load_and_preprocess_images(image_paths, new_width=load_img_size).to(torch_device)
    image_size_hw = tuple(int(x) for x in images.shape[-2:])

    if device_type == "cuda":
        if torch_device.index is None:
            capability = torch.cuda.get_device_capability()
        else:
            capability = torch.cuda.get_device_capability(torch_device.index)
        dtype = torch.bfloat16 if capability[0] >= 8 else torch.float16
        autocast_context = torch.amp.autocast(device_type="cuda", dtype=dtype)
    else:
        autocast_context = nullcontext()

    with torch.no_grad():
        with autocast_context:
            predictions = model(images)

    depth = predictions["depth"].squeeze(0).squeeze(-1).float().cpu().numpy()
    depth_conf = predictions.get("depth_conf")
    if depth_conf is not None:
        depth_conf_np = depth_conf.squeeze(0).float().cpu().numpy()
    else:
        depth_conf_np = np.zeros(depth.shape, dtype=np.float32)

    extrinsics_w2c, intrinsics = pose_encoding_to_extri_intri(
        predictions["pose_enc"].float(),
        image_size_hw=image_size_hw,
    )
    extrinsics_w2c_4x4 = torch.eye(4, dtype=torch.float32, device=torch_device)[None, None].repeat(
        1,
        extrinsics_w2c.shape[1],
        1,
        1,
    )
    extrinsics_w2c_4x4[:, :, :3, :] = extrinsics_w2c.float()
    poses_c2w = torch.linalg.inv(extrinsics_w2c_4x4[0]).cpu().numpy().astype(np.float32)
    intrinsics_np = intrinsics[0].float().cpu().numpy().astype(np.float32)

    geoms = preprocess_geometries(image_paths, load_img_size)
    original_depths = [map_depth_to_original(frame_depth, geom) for frame_depth, geom in zip(depth, geoms)]
    original_intrinsics = [
        map_intrinsic_to_original(frame_intrinsic, geom)
        for frame_intrinsic, geom in zip(intrinsics_np, geoms)
    ]
    return original_depths, depth_conf_np, list(poses_c2w), original_intrinsics, image_size_hw


def rows_from_scenes(
    input_root: Path,
    scenes: list[str] | None,
    frames_per_sample: int,
    max_samples: int,
    scene_name: str,
    dataset_name: str,
) -> list[dict[str, Any]]:
    scene_dirs = discover_scene_dirs(input_root, scenes)
    print(f"[vggt-gt] discovered scene dirs: {len(scene_dirs)}")
    for scene_dir in scene_dirs:
        print(f"[vggt-gt] will process: {scene_dir}")

    rows: list[dict[str, Any]] = []
    for scene_dir in scene_dirs:
        if not (scene_dir / "meta" / "episodes.jsonl").exists():
            continue
        before = len(rows)
        rows.extend(convert_scene(scene_dir, frames_per_sample, scene_name, dataset_name))
        print(f"[vggt-gt] base rows from {scene_dir.name}: {len(rows) - before}")
        if max_samples and len(rows) >= max_samples:
            rows = rows[:max_samples]
            break
    return rows


def write_outputs(rows: list[dict[str, Any]], output_root: Path, row_group_size: int, output_name: str) -> None:
    parquet_dir = output_root / "parquets"
    parquet_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = (parquet_dir / f"{output_name}.parquet").resolve()
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, parquet_path, row_group_size=row_group_size)

    num_row_groups = pq.ParquetFile(parquet_path).num_row_groups
    parquet_info = {str(parquet_path): {"num_row_groups": num_row_groups}}
    parquet_info_path = output_root / "parquet_info.json"
    with parquet_info_path.open("w", encoding="utf-8") as f:
        json.dump(parquet_info, f, indent=2)

    dataset_info = {
        "data_dir": str(parquet_dir.resolve()),
        "num_files": 1,
        "num_total_samples": len(rows),
        "parquet_info_path": str(parquet_info_path.resolve()),
    }
    with (output_root / "dataset_info_snippet.json").open("w", encoding="utf-8") as f:
        json.dump(dataset_info, f, indent=2)

    print(f"[vggt-gt] Converted rows: {len(rows)}")
    print(f"[vggt-gt] Parquet: {parquet_path}")
    print(f"[vggt-gt] Parquet info: {parquet_info_path.resolve()}")
    print("[vggt-gt] DATASET_INFO snippet:")
    print(json.dumps(dataset_info, indent=2))


def main() -> None:
    data_root = Path(os.environ.get("DATA_ROOT", "/mnt/wqq/G2VLM/data"))
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-root",
        default=str(data_root / "InternData-N1-extracted/vln_n1/traj_data/replica_d435i"),
        help="Extracted replica_d435i root containing scene folders such as apartment_2.",
    )
    parser.add_argument(
        "--output-root",
        default=str(data_root / "g2vlm_interndata_n1/replica_d435i_vggt_gt"),
        help="Output root for VGGT pseudo-GT parquet files.",
    )
    parser.add_argument("--scenes", nargs="*", default=None)
    parser.add_argument("--frames-per-sample", type=int, default=8)
    parser.add_argument("--row-group-size", type=int, default=32)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--vggt-model", default="facebook/VGGT-1B")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--load-img-size", type=int, default=518)
    parser.add_argument("--align-depth-to-sensor", choices=("median", "none"), default="median")
    parser.add_argument("--min-valid-depth-pixels", type=int, default=256)
    parser.add_argument("--max-depth-m", type=float, default=65.0)
    parser.add_argument("--scene-name", default="replica")
    parser.add_argument("--dataset-name", default="spar_interndata_n1_replica_d435i_vggt_gt")
    parser.add_argument("--output-name", default="interndata_n1_replica_d435i_vggt_gt")
    args = parser.parse_args()

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    vggt_depth_root = output_root / "vggt_gt" / "depth_png"
    output_root.mkdir(parents=True, exist_ok=True)

    base_rows = rows_from_scenes(
        input_root,
        args.scenes,
        args.frames_per_sample,
        args.max_samples,
        args.scene_name,
        args.dataset_name,
    )
    if not base_rows:
        raise RuntimeError(f"No base rows converted from {input_root}")

    model = load_vggt_model(args.vggt_model, args.device)
    converted_rows: list[dict[str, Any]] = []

    for row_idx, row in enumerate(base_rows):
        metadata = parse_metadata(row.get("metadata"))
        row_id = sanitize_id(str(metadata.get("id", f"row_{row_idx:06d}")))
        print(f"[vggt-gt] [{row_idx + 1}/{len(base_rows)}] infer {row_id}", flush=True)

        image_list = [str(path) for path in row["image_list"]]
        sensor_depth_list = [str(path) for path in row["depth_list"]]
        sensor_poses = [matrix4(pose).reshape(-1).tolist() for pose in row["poses"]]
        sensor_intrinsic = list(row["intrinsic"])
        sensor_depth_intrinsic = list(row["depth_intrinsic"])

        vggt_depths, depth_conf, poses_c2w, intrinsics, processed_hw = infer_vggt(
            model,
            image_list,
            args.device,
            args.load_img_size,
        )

        depth_scale = 1.0
        if args.align_depth_to_sensor == "median":
            depth_scale = median_depth_scale(vggt_depths, sensor_depth_list, args.min_valid_depth_pixels)
            vggt_depths = [depth * depth_scale for depth in vggt_depths]
            for pose in poses_c2w:
                pose[:3, 3] *= depth_scale

        depth_paths: list[str] = []
        for frame_idx, depth in enumerate(vggt_depths):
            depth_path = (vggt_depth_root / row_id / f"frame_{frame_idx:03d}.png").resolve()
            save_depth_png(depth, depth_path, args.max_depth_m)
            depth_paths.append(str(depth_path))

        metadata.update(
            {
                "pseudo_gt_source": "vggt",
                "vggt_model": args.vggt_model,
                "vggt_processed_hw": list(processed_hw),
                "vggt_depth_scale_to_sensor": depth_scale,
                "vggt_pose_convention": "camera_to_world",
                "vggt_depth_unit": "meters_saved_as_uint16_millimeters_png",
                "sensor_depth_list": sensor_depth_list,
                "sensor_poses": sensor_poses,
                "sensor_intrinsic": sensor_intrinsic,
                "sensor_depth_intrinsic": sensor_depth_intrinsic,
                "vggt_intrinsics_per_frame": [intr.reshape(-1).tolist() for intr in intrinsics],
                "vggt_depth_conf_shape": list(depth_conf.shape),
            }
        )

        new_row = dict(row)
        new_row["depth_list"] = depth_paths
        new_row["poses"] = [pose.reshape(-1).tolist() for pose in poses_c2w]
        new_row["intrinsic"] = matrix3_to_4x4(intrinsics[0])
        new_row["depth_intrinsic"] = matrix3_to_4x4(intrinsics[0])
        new_row["metadata"] = repr(metadata)
        converted_rows.append(new_row)

    write_outputs(converted_rows, output_root, args.row_group_size, args.output_name)


if __name__ == "__main__":
    main()
