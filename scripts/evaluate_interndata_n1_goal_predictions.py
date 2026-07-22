#!/usr/bin/env python3
"""Evaluate G2VLM InternData-N1 pixel-goal predictions saved as JSON."""

import argparse
import ast
import json
import math
import re
from pathlib import Path


IMAGE_INDEX_RE = re.compile(r"(?:image[_ -]?index|image|idx)\s*[=:]\s*(-?\d+)", re.IGNORECASE)
XY_PAIR_RE = re.compile(r"\b(?:x\s*[=:]\s*)?(-?\d+(?:\.\d+)?)\s*[,;]\s*(?:y\s*[=:]\s*)?(-?\d+(?:\.\d+)?)\b", re.IGNORECASE)
X_RE = re.compile(r"\bx\s*[=:]\s*(-?\d+(?:\.\d+)?)", re.IGNORECASE)
Y_RE = re.compile(r"\by\s*[=:]\s*(-?\d+(?:\.\d+)?)", re.IGNORECASE)


def parse_metadata(value):
    if isinstance(value, dict):
        return value
    return ast.literal_eval(value)


def parse_prediction(text):
    if not text:
        return None
    text = str(text)

    image_index = None
    match = IMAGE_INDEX_RE.search(text)
    if match:
        image_index = int(match.group(1))

    x = y = None
    x_match = X_RE.search(text)
    y_match = Y_RE.search(text)
    if x_match and y_match:
        x = float(x_match.group(1))
        y = float(y_match.group(1))
    else:
        pair_match = XY_PAIR_RE.search(text)
        if pair_match:
            x = float(pair_match.group(1))
            y = float(pair_match.group(2))

    if image_index is None or x is None or y is None:
        return None
    return {"image_index": image_index, "x": x, "y": y}


def load_prediction_files(paths):
    files = []
    for path in paths:
        p = Path(path)
        if p.is_dir():
            files.extend(sorted(p.rglob("*.json")))
        else:
            files.append(p)
    return files


def l2(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def percentile(values, q):
    if not values:
        return float("nan")
    values = sorted(values)
    idx = (len(values) - 1) * q
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return values[lo]
    return values[lo] * (hi - idx) + values[hi] * (idx - lo)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", help="Prediction JSON file(s) or directory/directories.")
    parser.add_argument("--success-thresholds", default="50,100,150", help="Normalized 0-1000 pixel L2 thresholds.")
    parser.add_argument("--show-bad", type=int, default=5)
    args = parser.parse_args()

    thresholds = [float(item) for item in args.success_thresholds.split(",") if item.strip()]
    files = load_prediction_files(args.paths)
    if not files:
        raise SystemExit("No prediction JSON files found.")

    rows = []
    bad = []
    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        metadata = parse_metadata(payload["metadata"])
        gt_pixels = metadata.get("goal_pixel") or []
        gt_indices = metadata.get("goal_pixel_img_idx") or []
        prediction = parse_prediction(payload.get("prediction"))
        if not gt_pixels or not gt_indices or prediction is None:
            bad.append((path, payload.get("prediction")))
            continue
        gt_pixel = [float(gt_pixels[0][0]), float(gt_pixels[0][1])]
        gt_index = int(gt_indices[0])
        pred_pixel = [prediction["x"], prediction["y"]]
        error = l2(pred_pixel, gt_pixel)
        rows.append(
            {
                "path": path,
                "gt_index": gt_index,
                "pred_index": prediction["image_index"],
                "gt_pixel": gt_pixel,
                "pred_pixel": pred_pixel,
                "pixel_l2": error,
                "index_ok": int(prediction["image_index"] == gt_index),
            }
        )

    total = len(files)
    parsed = len(rows)
    print(f"files: {total}")
    print(f"parsed predictions: {parsed} ({parsed / total * 100:.2f}%)")
    print(f"unparsed/missing: {len(bad)} ({len(bad) / total * 100:.2f}%)")
    if not rows:
        raise SystemExit("No valid predictions to score.")

    errors = [row["pixel_l2"] for row in rows]
    index_acc = sum(row["index_ok"] for row in rows) / len(rows)
    print(f"image_index_acc: {index_acc * 100:.2f}%")
    print(f"pixel_l2_mean: {sum(errors) / len(errors):.3f}")
    print(f"pixel_l2_median: {percentile(errors, 0.5):.3f}")
    print(f"pixel_l2_p90: {percentile(errors, 0.9):.3f}")
    for threshold in thresholds:
        rate = sum(error <= threshold for error in errors) / len(errors)
        print(f"success@{threshold:g}: {rate * 100:.2f}%")

    worst = sorted(rows, key=lambda row: row["pixel_l2"], reverse=True)[: args.show_bad]
    if worst:
        print("\nworst predictions:")
        for row in worst:
            print(
                f"{row['path']}: err={row['pixel_l2']:.2f}, "
                f"idx pred/gt={row['pred_index']}/{row['gt_index']}, "
                f"xy pred/gt={row['pred_pixel']}/{row['gt_pixel']}"
            )

    if bad and args.show_bad:
        print("\nunparsed examples:")
        for path, prediction in bad[: args.show_bad]:
            print(f"{path}: {prediction!r}")


if __name__ == "__main__":
    main()
