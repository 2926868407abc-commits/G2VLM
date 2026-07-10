#!/usr/bin/env python3
"""Resolve a local path or Hugging Face repo id to a local directory."""

import argparse
import os
import sys
from pathlib import Path


CONFIG_ALLOW_PATTERNS = [
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
FULL_ALLOW_PATTERNS = CONFIG_ALLOW_PATTERNS + ["*.safetensors", "*.bin"]
WEIGHT_PATTERNS = ["*.safetensors", "*.bin", "*.pt", "*.pth", "*.ckpt"]
LOCAL_PREFIXES = {"checkpoints", "data", "docs", "models", "scripts", "mnt", "home", "root", "tmp"}


def looks_like_repo_id(value):
    path = Path(value).expanduser()
    if path.exists() or path.is_absolute():
        return False
    if value.startswith((".", "~")):
        return False
    if "\\" in value or ":" in value:
        return False
    first_part = value.split("/", 1)[0]
    if first_part in LOCAL_PREFIXES:
        return False
    return "/" in value


def download_repo(repo_id, local_root, mode):
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit(
            "huggingface_hub is required. Run: "
            "python -m pip install --force-reinstall 'huggingface_hub==0.29.1'"
        ) from exc

    local_root.mkdir(parents=True, exist_ok=True)
    local_dir = local_root / repo_id.replace("/", "__")
    allow_patterns = CONFIG_ALLOW_PATTERNS if mode == "config" else FULL_ALLOW_PATTERNS
    ignore_patterns = WEIGHT_PATTERNS if mode == "config" else None
    print(f"[resolve] downloading {repo_id} -> {local_dir} ({mode})", file=sys.stderr)
    return Path(
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(local_dir),
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
            resume_download=True,
        )
    ).resolve()


def check_required(path, required):
    missing = [name for name in required if not (path / name).exists()]
    if missing:
        raise SystemExit(f"Resolved path is missing required files {missing}: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-or-path", required=True)
    parser.add_argument("--local-root", default=None)
    parser.add_argument("--mode", choices=["config", "full"], default="full")
    parser.add_argument("--required", nargs="*", default=[])
    args = parser.parse_args()

    if looks_like_repo_id(args.repo_or_path):
        data_root = Path(os.environ.get("DATA_ROOT", "/mnt/data/wangqq/G2VLM/data"))
        local_root = Path(args.local_root) if args.local_root else data_root / "models"
        resolved = download_repo(args.repo_or_path, local_root, args.mode)
    else:
        resolved = Path(args.repo_or_path).expanduser().resolve()
        if not resolved.exists():
            raise SystemExit(
                f"Local path does not exist: {resolved}\n"
                "Pass an existing local path or a HF repo id like InternRobotics/G2VLM-2B-MoT."
            )

    check_required(resolved, args.required)
    print(resolved)


if __name__ == "__main__":
    main()
