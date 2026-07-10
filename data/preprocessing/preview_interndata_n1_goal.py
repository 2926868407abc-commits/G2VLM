#!/usr/bin/env python3
"""Preview converted InternData-N1 G2VLM rows with the navigation pixel goal."""

import argparse
import ast
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image, ImageDraw

try:
    RESAMPLE_BICUBIC = Image.Resampling.BICUBIC
except AttributeError:
    RESAMPLE_BICUBIC = Image.BICUBIC


def parse_metadata(value):
    if isinstance(value, dict):
        return value
    return ast.literal_eval(value)


def draw_goal(image, goal_pixel):
    image = image.convert("RGB")
    draw = ImageDraw.Draw(image)
    width, height = image.size
    x = int(goal_pixel[0] / 1000.0 * width)
    y = int(goal_pixel[1] / 1000.0 * height)
    radius = max(8, min(width, height) // 35)
    draw.ellipse([x - radius - 4, y - radius - 4, x + radius + 4, y + radius + 4], fill="white")
    draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill="lime")
    draw.line([x - radius * 2, y, x + radius * 2, y], fill="black", width=max(2, radius // 4))
    draw.line([x, y - radius * 2, x, y + radius * 2], fill="black", width=max(2, radius // 4))
    return image


def make_contact_sheet(images, labels, thumb_width):
    thumbs = []
    label_height = 26
    for image, label in zip(images, labels):
        image = image.convert("RGB")
        scale = thumb_width / image.width
        thumb_height = max(1, int(image.height * scale))
        image = image.resize((thumb_width, thumb_height), RESAMPLE_BICUBIC)
        canvas = Image.new("RGB", (thumb_width, thumb_height + label_height), "white")
        canvas.paste(image, (0, label_height))
        draw = ImageDraw.Draw(canvas)
        draw.text((6, 6), label, fill="black")
        thumbs.append(canvas)

    sheet = Image.new("RGB", (thumb_width * len(thumbs), max(t.height for t in thumbs)), "white")
    x = 0
    for thumb in thumbs:
        sheet.paste(thumb, (x, 0))
        x += thumb_width
    return sheet


def validate_row(row, metadata):
    errors = []
    image_list = row.get("image_list") or []
    depth_list = row.get("depth_list") or []
    poses = row.get("poses") or []
    goal_pixel = metadata.get("goal_pixel", [[None, None]])[0]
    goal_img_idx = metadata.get("goal_pixel_img_idx", [None])[0]

    if not image_list:
        errors.append("image_list is empty")
    if len(image_list) != len(depth_list):
        errors.append(f"image/depth length mismatch: {len(image_list)} vs {len(depth_list)}")
    if len(image_list) != len(poses):
        errors.append(f"image/pose length mismatch: {len(image_list)} vs {len(poses)}")

    for label, paths in [("image", image_list), ("depth", depth_list)]:
        for path in paths:
            if not Path(path).exists():
                errors.append(f"missing {label}: {path}")

    for idx, pose in enumerate(poses):
        if np.asarray(pose).size != 16:
            errors.append(f"pose[{idx}] size is {np.asarray(pose).size}, expected 16")

    for key in ["intrinsic", "depth_intrinsic"]:
        if np.asarray(row.get(key)).size != 16:
            errors.append(f"{key} size is {np.asarray(row.get(key)).size}, expected 16")

    if goal_img_idx is None or not (0 <= int(goal_img_idx) < max(1, len(image_list))):
        errors.append(f"goal_pixel_img_idx out of range: {goal_img_idx}")
    if len(goal_pixel) != 2 or not all(0 <= float(v) <= 1000 for v in goal_pixel):
        errors.append(f"goal_pixel out of 0-1000 range: {goal_pixel}")

    if errors:
        raise SystemExit("Invalid converted row:\n- " + "\n- ".join(errors))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--parquet",
        default="/mnt/data/wangqq/G2VLM/data/g2vlm_interndata_n1/replica_d435i/parquets/interndata_n1_replica_d435i.parquet",
    )
    parser.add_argument("--row", type=int, default=0)
    parser.add_argument("--out", default="/mnt/data/wangqq/G2VLM/data/g2vlm_interndata_n1/replica_d435i/previews")
    parser.add_argument("--thumb-width", type=int, default=220)
    args = parser.parse_args()

    table = pq.read_table(args.parquet)
    if args.row < 0 or args.row >= table.num_rows:
        raise SystemExit(f"row index out of range: {args.row}, total rows: {table.num_rows}")

    row = table.slice(args.row, 1).to_pylist()[0]
    metadata = parse_metadata(row["metadata"])
    validate_row(row, metadata)
    goal_pixel = metadata["goal_pixel"][0]
    goal_img_idx = int(metadata["goal_pixel_img_idx"][0])
    images = []
    labels = []
    for idx, image_path in enumerate(row["image_list"]):
        image = Image.open(image_path)
        if idx == goal_img_idx:
            image = draw_goal(image, goal_pixel)
            labels.append(f"{idx}: GOAL {goal_pixel}")
        else:
            labels.append(str(idx))
        images.append(image)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"row_{args.row:05d}_goal_preview.jpg"
    make_contact_sheet(images, labels, args.thumb_width).save(out_path, quality=95)

    print(f"saved: {out_path}")
    print(f"goal_pixel: {goal_pixel}")
    print(f"goal_pixel_img_idx: {goal_img_idx}")
    print(f"goal_pixel_source: {metadata.get('goal_pixel_source', 'unknown')}")
    print("question:", row["question"])
    print("answer:", row["answer"])


if __name__ == "__main__":
    main()
