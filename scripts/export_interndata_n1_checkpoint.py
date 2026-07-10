#!/usr/bin/env python3
"""Create an inference-ready G2VLM directory from an InternData-N1 training checkpoint."""

import argparse
import json
import os
import shutil
from pathlib import Path


WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth", ".ckpt")
REQUIRED_BASE_FILES = ("text_config.json", "vit_config.json", "dino_config.json")
BASE_ALLOW_PATTERNS = [
    "*.json",
    "*.txt",
    "*.model",
    "tokenizer*",
    "vocab*",
    "merges*",
    "special_tokens_map.json",
    "added_tokens.json",
    "preprocessor_config.json",
    "generation_config.json",
]


def newest_path(paths):
    paths = list(paths)
    if not paths:
        return None
    return max(paths, key=lambda path: path.stat().st_mtime)


def latest_step_dir(run_dir):
    step_dirs = [path for path in run_dir.iterdir() if path.is_dir() and path.name.isdigit()]
    if not step_dirs:
        return None
    return max(step_dirs, key=lambda path: int(path.name))


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


def resolve_checkpoint(checkpoint):
    if checkpoint is not None:
        path = Path(checkpoint).expanduser().resolve()
        if path.is_file() and path.name == "model.safetensors":
            return path.parent
        if (path / "model.safetensors").exists():
            return path
        step_dir = latest_step_dir(path)
        if step_dir is not None and (step_dir / "model.safetensors").exists():
            return step_dir
        raise SystemExit(f"Could not find model.safetensors in checkpoint path: {path}")

    checkpoints_root = Path("checkpoints")
    run_dir = newest_path(checkpoints_root.glob("g2vlm_interndata_n1_replica_*")) if checkpoints_root.exists() else None
    if run_dir is None:
        raise SystemExit("No g2vlm_interndata_n1_replica_* run found under ./checkpoints")
    step_dir = latest_step_dir(run_dir)
    if step_dir is None or not (step_dir / "model.safetensors").exists():
        raise SystemExit(f"No step directory with model.safetensors found under {run_dir}")
    return step_dir.resolve()


def download_base_snapshot(repo_id, cache_dir):
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit(
            "huggingface_hub is required to download base config/tokenizer files. "
            "Run: python -m pip install --force-reinstall 'huggingface_hub==0.29.1'"
        ) from exc

    cache_dir.mkdir(parents=True, exist_ok=True)
    local_dir = cache_dir / repo_id.replace("/", "__")
    return Path(
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(local_dir),
            allow_patterns=BASE_ALLOW_PATTERNS,
            ignore_patterns=["*.safetensors", "*.bin", "*.pt", "*.pth", "*.ckpt"],
            resume_download=True,
        )
    ).resolve()


def resolve_base_model(base_model_path, out_dir):
    if looks_like_repo_id(base_model_path):
        return download_base_snapshot(base_model_path, out_dir.parent / "_base_model_cache")
    return Path(base_model_path).expanduser().resolve()


def copy_base_files(base_model_path, out_dir):
    base_model_path = resolve_base_model(base_model_path, out_dir)
    if not base_model_path.exists():
        raise SystemExit(
            f"Base model path does not exist: {base_model_path}\n"
            "Pass --base-model-path InternRobotics/G2VLM-2B-MoT to download config/tokenizer files "
            "or pass a local HF-format model directory."
        )

    missing = [name for name in REQUIRED_BASE_FILES if not (base_model_path / name).exists()]
    if missing:
        raise SystemExit(f"Base model path is missing required files: {missing}")

    copied = []
    for src in base_model_path.iterdir():
        if not src.is_file():
            continue
        if src.suffix in WEIGHT_SUFFIXES:
            continue
        dst = out_dir / src.name
        shutil.copy2(src, dst)
        copied.append(src.name)
    return base_model_path, copied


def install_weight(src_weight, dst_weight, copy_weights):
    if dst_weight.exists() or dst_weight.is_symlink():
        dst_weight.unlink()
    if copy_weights:
        shutil.copy2(src_weight, dst_weight)
        return "copied"
    try:
        dst_weight.symlink_to(src_weight)
        return "symlinked"
    except OSError:
        shutil.copy2(src_weight, dst_weight)
        return "copied_after_symlink_failed"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Step dir, run dir, or model.safetensors. Defaults to latest g2vlm_interndata_n1_replica run.",
    )
    parser.add_argument(
        "--base-model-path",
        default=os.environ.get("G2VLM_BASE_MODEL_PATH", "InternRobotics/G2VLM-2B-MoT"),
        help="HF repo id or local HF-format base G2VLM directory containing configs and tokenizer files.",
    )
    parser.add_argument("--out", default=None, help="Output model directory.")
    parser.add_argument("--copy-weights", action="store_true", help="Copy model.safetensors instead of symlinking it.")
    args = parser.parse_args()

    step_dir = resolve_checkpoint(args.checkpoint)
    src_weight = step_dir / "model.safetensors"
    if args.out:
        out_dir = Path(args.out).expanduser().resolve()
    else:
        out_dir = step_dir.parent / f"hf_export_{step_dir.name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    base_model_path, copied = copy_base_files(args.base_model_path, out_dir)
    weight_mode = install_weight(src_weight, out_dir / "model.safetensors", args.copy_weights)

    export_info = {
        "checkpoint_step_dir": str(step_dir),
        "source_weight": str(src_weight),
        "base_model_path": str(base_model_path),
        "output_dir": str(out_dir),
        "weight_mode": weight_mode,
        "copied_base_files": copied,
    }
    (out_dir / "export_info.json").write_text(json.dumps(export_info, indent=2), encoding="utf-8")

    print("exported:", out_dir)
    print("checkpoint:", step_dir)
    print("base model:", base_model_path)
    print("weight:", weight_mode)
    print("next sanity check:")
    print(f"python scripts/infer_interndata_n1_replica_sample.py --row 0 --model-path {out_dir}")


if __name__ == "__main__":
    main()
