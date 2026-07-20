#!/usr/bin/env python3
"""Check local InternData-N1 download progress and remaining disk needs."""

import argparse
import getpass
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path


DEFAULT_REPO_ID = "InternRobotics/InternData-N1"


def human_size(num_bytes):
    value = float(num_bytes)
    for unit in ["B", "KiB", "MiB", "GiB", "TiB", "PiB"]:
        if value < 1024.0 or unit == "PiB":
            return f"{value:.2f} {unit}"
        value /= 1024.0


def group_name(path):
    parts = path.split("/")
    if len(parts) >= 4 and parts[0] == "vln_n1" and parts[1] == "traj_data":
        return parts[2]
    if len(parts) >= 2:
        return "/".join(parts[:2])
    return parts[0] if parts else "unknown"


def iter_remote_files(api, args):
    try:
        items = api.list_repo_tree(
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            revision=args.revision,
            path_in_repo=args.prefix.rstrip("/"),
            recursive=True,
            expand=True,
        )
        for item in items:
            if getattr(item, "type", None) == "file":
                yield item
    except Exception as exc:
        raise SystemExit(
            "Failed to query Hugging Face file list. Check network/token/endpoint.\n"
            f"endpoint={args.endpoint}\n"
            f"repo={args.repo_id}\n"
            f"error={type(exc).__name__}: {exc}"
        ) from exc


def main():
    repo_root = Path(__file__).resolve().parents[1]
    data_root = Path(os.environ.get("DATA_ROOT", repo_root / "data"))

    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--repo-type", default="dataset")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--prefix", default="vln_n1")
    parser.add_argument("--local-dir", default=str(data_root / "InternData-N1"))
    parser.add_argument("--endpoint", default=os.environ.get("HF_ENDPOINT", "https://hf-mirror.com"))
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    parser.add_argument("--top", type=int, default=30)
    parser.add_argument("--missing-out", default="")
    args = parser.parse_args()

    if not args.token:
        args.token = getpass.getpass("HF Token: ")

    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise SystemExit(
            "huggingface_hub is not installed. Activate env first, then run:\n"
            "python -m pip install --force-reinstall 'huggingface_hub==0.29.1'"
        ) from exc

    local_root = Path(args.local_dir)
    incomplete_count = sum(1 for _ in local_root.rglob("*.incomplete")) if local_root.exists() else 0
    lock_count = sum(1 for _ in local_root.rglob("*.lock")) if local_root.exists() else 0
    local_file_count = (
        sum(1 for p in (local_root / args.prefix).rglob("*") if p.is_file())
        if (local_root / args.prefix).exists()
        else 0
    )

    api = HfApi(endpoint=args.endpoint, token=args.token)
    stats = defaultdict(lambda: {"files": 0, "complete": 0, "total": 0, "downloaded": 0, "missing": 0})
    missing_files = []
    unknown_size = 0
    total_files = 0
    complete_files = 0
    remote_total = 0
    downloaded_complete = 0
    missing_or_partial = 0

    for item in iter_remote_files(api, args):
        path = item.path
        size = getattr(item, "size", None)
        if size is None:
            unknown_size += 1
            continue

        total_files += 1
        remote_total += size
        local_path = local_root / path
        have = local_path.stat().st_size if local_path.exists() else 0
        group = group_name(path)
        stats[group]["files"] += 1
        stats[group]["total"] += size

        if local_path.exists() and have >= size:
            complete_files += 1
            downloaded_complete += size
            stats[group]["complete"] += 1
            stats[group]["downloaded"] += size
        else:
            left = max(size - have, 0)
            missing_or_partial += left
            stats[group]["missing"] += left
            missing_files.append((left, have, size, path))

    disk = shutil.disk_usage(local_root if local_root.exists() else local_root.parent)

    print("InternData-N1 download check")
    print(f"repo: {args.repo_id}")
    print(f"prefix: {args.prefix}")
    print(f"endpoint: {args.endpoint}")
    print(f"local_dir: {local_root}")
    print()
    print(f"remote files: {total_files}")
    print(f"local files under prefix: {local_file_count}")
    print(f"complete files: {complete_files}/{total_files}")
    print(f"incomplete files: {incomplete_count}")
    print(f"lock files: {lock_count}")
    print(f"unknown-size remote files: {unknown_size}")
    print()
    print(f"remote total size: {human_size(remote_total)}")
    print(f"downloaded complete size: {human_size(downloaded_complete)}")
    print(f"missing/partial remaining: {human_size(missing_or_partial)}")
    print(f"disk available: {human_size(disk.free)}")
    print(f"recommended free space: {human_size(missing_or_partial + 300 * 1024**3)}")
    print()

    if disk.free >= missing_or_partial + 300 * 1024**3:
        print("disk verdict: enough with ~300 GiB buffer")
    elif disk.free >= missing_or_partial:
        print("disk verdict: barely enough; no safety buffer")
    else:
        print("disk verdict: NOT enough for the remaining files")

    print("\nPer-subset progress:")
    for name in sorted(stats):
        item = stats[name]
        print(
            f"{name}: files {item['complete']}/{item['files']}, "
            f"done {human_size(item['downloaded'])}, "
            f"remaining {human_size(item['missing'])}, "
            f"total {human_size(item['total'])}"
        )

    if missing_files:
        print(f"\nTop {min(args.top, len(missing_files))} missing/partial files:")
        for left, have, size, path in sorted(missing_files, reverse=True)[: args.top]:
            print(
                f"left {human_size(left)} | "
                f"have {human_size(have)} / total {human_size(size)} | {path}"
            )

    if args.missing_out:
        out_path = Path(args.missing_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            for left, have, size, path in sorted(missing_files, reverse=True):
                f.write(f"{path}\tleft={left}\thave={have}\ttotal={size}\n")
        print(f"\nmissing list saved: {out_path}")

    if complete_files < total_files or incomplete_count:
        sys.exit(2)


if __name__ == "__main__":
    main()
