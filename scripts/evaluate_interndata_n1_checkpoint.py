#!/usr/bin/env python3
"""Run held-out InternData-N1 pixel-goal evaluation for a G2VLM checkpoint."""

import argparse
import ast
import json
import math
import os
import re
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image


IMAGE_INDEX_RE = re.compile(r"(?:image[_ -]?index|image|idx)\s*[=:]\s*(-?\d+)", re.IGNORECASE)
X_RE = re.compile(r"\bx\s*[=:]\s*(-?\d+(?:\.\d+)?)", re.IGNORECASE)
Y_RE = re.compile(r"\by\s*[=:]\s*(-?\d+(?:\.\d+)?)", re.IGNORECASE)
XY_PAIR_RE = re.compile(r"\b(-?\d+(?:\.\d+)?)\s*[,;]\s*(-?\d+(?:\.\d+)?)\b")


def default_parquet_path():
    root = Path(
        os.environ.get(
            "G2VLM_INTERNDATA_N1_REPLICA_ROOT",
            "/data/wqq/G2VLM/data/g2vlm_interndata_n1/replica_d435i",
        )
    )
    files = sorted((root / "parquets").glob("*.parquet"))
    return str(files[0]) if files else str(root / "parquets" / "interndata_n1_replica_d435i.parquet")


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


def looks_like_repo_id(value):
    path = Path(value).expanduser()
    if path.exists() or path.is_absolute():
        return False
    if value.startswith((".", "~")):
        return False
    if "\\" in value or ":" in value:
        return False
    first_part = value.split("/", 1)[0]
    if first_part in {"checkpoints", "data", "docs", "models", "scripts", "mnt", "home", "root", "tmp"}:
        return False
    return "/" in value


def resolve_model_path(model_path):
    if not looks_like_repo_id(model_path):
        resolved = Path(model_path).expanduser().resolve()
        if not resolved.exists():
            raise SystemExit(f"model path does not exist: {resolved}")
        return str(resolved)

    from huggingface_hub import snapshot_download

    data_root = Path(os.environ.get("DATA_ROOT", "/data/wqq/G2VLM/data"))
    local_dir = data_root / "models" / model_path.replace("/", "__")
    resolved = snapshot_download(
        repo_id=model_path,
        local_dir=str(local_dir),
        allow_patterns=[
            "*.json",
            "*.txt",
            "*.model",
            "*.safetensors",
            "tokenizer*",
            "vocab*",
            "merges*",
            "special_tokens_map.json",
            "added_tokens.json",
            "preprocessor_config.json",
            "generation_config.json",
        ],
        resume_download=True,
    )
    return str(Path(resolved).resolve())


def l2(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def percentile(values, q):
    if not values:
        return None
    values = sorted(values)
    idx = (len(values) - 1) * q
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return values[lo]
    return values[lo] * (hi - idx) + values[hi] * (idx - lo)


def score_rows(rows, thresholds):
    parsed = [row for row in rows if row.get("parsed_prediction")]
    errors = [row["pixel_l2"] for row in parsed]
    summary = {
        "num_samples": len(rows),
        "num_parsed": len(parsed),
        "parse_rate": len(parsed) / len(rows) if rows else 0.0,
        "image_index_acc": sum(row["image_index_ok"] for row in parsed) / len(parsed) if parsed else 0.0,
        "pixel_l2_mean": sum(errors) / len(errors) if errors else None,
        "pixel_l2_median": percentile(errors, 0.5),
        "pixel_l2_p90": percentile(errors, 0.9),
    }
    for threshold in thresholds:
        summary[f"success@{threshold:g}"] = (
            sum(error <= threshold for error in errors) / len(errors) if errors else 0.0
        )
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", default=default_parquet_path())
    parser.add_argument("--model-path", required=True, help="Exported HF-format checkpoint or base model path.")
    parser.add_argument("--start-row", type=int, default=0)
    parser.add_argument("--max-rows", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=120)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--success-thresholds", default="50,100,150")
    args = parser.parse_args()

    parquet_path = Path(args.parquet)
    table = pq.read_table(parquet_path)
    if table.num_rows == 0:
        raise SystemExit(f"empty parquet: {parquet_path}")

    end_row = min(table.num_rows, args.start_row + args.max_rows)
    selected = table.slice(args.start_row, end_row - args.start_row).to_pylist()
    out_dir = Path(args.out_dir) if args.out_dir else parquet_path.parents[1] / "eval_predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    thresholds = [float(item) for item in args.success_thresholds.split(",") if item.strip()]

    from g2vlm_utils import build_transform, load_model_and_tokenizer, process_conversation

    args.model_path = resolve_model_path(args.model_path)
    print(f"parquet: {parquet_path}")
    print(f"rows: {args.start_row}..{end_row - 1} / {table.num_rows}")
    print(f"model_path: {args.model_path}")
    print(f"out_dir: {out_dir}")

    model, tokenizer, new_token_ids, _, dino_transform = load_model_and_tokenizer(args)
    image_transform = build_transform(pixel=768)

    results = []
    for offset, row in enumerate(selected):
        row_idx = args.start_row + offset
        metadata = parse_metadata(row["metadata"])
        gt_pixel = [float(v) for v in metadata["goal_pixel"][0]]
        gt_image_index = int(metadata["goal_pixel_img_idx"][0])
        images = [Image.open(path).convert("RGB") for path in row["image_list"]]
        images, prompt = process_conversation(images, row["question"])
        prediction = model.chat_with_recon(
            tokenizer,
            new_token_ids,
            image_transform,
            dino_transform,
            images=images,
            prompt=prompt,
            max_length=args.max_length,
        )
        parsed = parse_prediction(prediction)
        result = {
            "row": row_idx,
            "question": row["question"],
            "label": row["answer"],
            "prediction": prediction,
            "gt_image_index": gt_image_index,
            "gt_pixel": gt_pixel,
            "metadata": metadata,
            "parsed_prediction": parsed,
        }
        if parsed:
            pred_pixel = [parsed["x"], parsed["y"]]
            result.update(
                {
                    "pred_image_index": parsed["image_index"],
                    "pred_pixel": pred_pixel,
                    "image_index_ok": int(parsed["image_index"] == gt_image_index),
                    "pixel_l2": l2(pred_pixel, gt_pixel),
                }
            )
        else:
            result.update({"image_index_ok": 0, "pixel_l2": None})
        results.append(result)
        (out_dir / f"row_{row_idx:05d}.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(
            f"[{offset + 1}/{len(selected)}] row={row_idx} "
            f"pred={parsed} gt=({gt_image_index}, {gt_pixel}) "
            f"err={result['pixel_l2']}"
        )

    summary = score_rows(results, thresholds)
    summary.update(
        {
            "parquet": str(parquet_path),
            "model_path": args.model_path,
            "start_row": args.start_row,
            "end_row": end_row,
            "out_dir": str(out_dir),
        }
    )
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nsummary:")
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"{key}: {value:.6g}")
        else:
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
