#!/usr/bin/env python3
"""Run or export a converted InternData-N1 replica_d435i sample for G2VLM."""

import argparse
import ast
import json
import os
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image


MODEL_ALLOW_PATTERNS = [
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
]


def parse_metadata(value):
    if isinstance(value, dict):
        return value
    return ast.literal_eval(value)


def default_parquet_path():
    root = os.environ.get(
        "G2VLM_INTERNDATA_N1_REPLICA_ROOT",
        "/mnt/data/wangqq/G2VLM/data/g2vlm_interndata_n1/replica_d435i",
    )
    return str(Path(root) / "parquets" / "interndata_n1_replica_d435i.parquet")


def read_row(parquet_path, row_index):
    table = pq.read_table(parquet_path)
    if row_index < 0 or row_index >= table.num_rows:
        raise SystemExit(f"row index out of range: {row_index}, total rows: {table.num_rows}")
    return table.slice(row_index, 1).to_pylist()[0], table.num_rows


def validate_paths(paths, label):
    missing = [path for path in paths if not Path(path).exists()]
    if missing:
        preview = "\n".join(f"- {path}" for path in missing[:10])
        raise SystemExit(f"missing {label} files:\n{preview}")


def sample_payload(row, row_index, total_rows):
    metadata = parse_metadata(row["metadata"])
    payload = {
        "row": row_index,
        "total_rows": total_rows,
        "question": row["question"],
        "answer": row["answer"],
        "metadata": metadata,
        "image_list": list(row["image_list"]),
        "depth_list": list(row["depth_list"]),
    }
    return payload


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

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit(
            "huggingface_hub is required to download a model repo. "
            "Run: python -m pip install --force-reinstall 'huggingface_hub==0.29.1'"
        ) from exc

    data_root = Path(os.environ.get("DATA_ROOT", "/mnt/data/wangqq/G2VLM/data"))
    local_dir = data_root / "models" / model_path.replace("/", "__")
    local_dir.mkdir(parents=True, exist_ok=True)
    resolved = snapshot_download(
        repo_id=model_path,
        local_dir=str(local_dir),
        allow_patterns=MODEL_ALLOW_PATTERNS,
        resume_download=True,
    )
    return str(Path(resolved).resolve())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", default=default_parquet_path())
    parser.add_argument("--row", type=int, default=0)
    parser.add_argument("--model-path", default="InternRobotics/G2VLM-2B-MoT")
    parser.add_argument("--max-length", type=int, default=120)
    parser.add_argument("--dry-run", action="store_true", help="Only export prompt/label metadata; do not load model.")
    parser.add_argument(
        "--out",
        default=None,
        help="Optional JSON output path. Defaults to converted_root/inference_samples/row_XXXXX.json.",
    )
    args = parser.parse_args()

    row, total_rows = read_row(args.parquet, args.row)
    payload = sample_payload(row, args.row, total_rows)
    validate_paths(payload["image_list"], "image")
    validate_paths(payload["depth_list"], "depth")

    out_path = Path(args.out) if args.out else Path(args.parquet).parents[1] / "inference_samples" / f"row_{args.row:05d}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"row: {args.row}/{total_rows}")
    print(f"images: {len(payload['image_list'])}")
    print(f"goal_pixel: {payload['metadata'].get('goal_pixel')}")
    print(f"goal_pixel_img_idx: {payload['metadata'].get('goal_pixel_img_idx')}")
    print("question:", payload["question"])
    print("label:", payload["answer"])

    if args.dry_run:
        payload["prediction"] = None
        payload["dry_run"] = True
        out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"saved dry-run sample: {out_path}")
        return

    from g2vlm_utils import build_transform, load_model_and_tokenizer, process_conversation

    args.model_path = resolve_model_path(args.model_path)
    print(f"resolved model_path: {args.model_path}")
    model, tokenizer, new_token_ids, _, dino_transform = load_model_and_tokenizer(args)
    image_transform = build_transform(pixel=768)
    images = [Image.open(path).convert("RGB") for path in payload["image_list"]]
    images, prompt = process_conversation(images, payload["question"])
    prediction = model.chat_with_recon(
        tokenizer,
        new_token_ids,
        image_transform,
        dino_transform,
        images=images,
        prompt=prompt,
        max_length=args.max_length,
    )

    payload["prediction"] = prediction
    payload["dry_run"] = False
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print("prediction:", prediction)
    print(f"saved inference sample: {out_path}")


if __name__ == "__main__":
    main()
