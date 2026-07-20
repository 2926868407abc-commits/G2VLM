#!/usr/bin/env python3
"""Estimate remaining InternData-N1 disk usage from local progress only.

This is a fallback for servers that cannot query the Hugging Face tree API.
It is not exact. It prints both an average-file estimate and a conservative
full-repository estimate.
"""

import argparse
import shutil
from pathlib import Path


def human(num_bytes):
    value = float(num_bytes)
    for unit in ["B", "GiB", "TiB"]:
        if unit == "B" and value < 1024**3:
            return f"{value:.2f} B"
        if unit == "GiB" and value < 1024:
            return f"{value:.2f} GiB"
        if unit == "TiB":
            return f"{value:.2f} TiB"
        value /= 1024
    return f"{value:.2f} TiB"


def dir_size(path):
    total = 0
    if not path.exists():
        return 0
    for p in path.rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except OSError:
                pass
    return total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-dir", default="/data/wqq/G2VLM/data/InternData-N1")
    parser.add_argument("--prefix", default="vln_n1")
    parser.add_argument("--expected-files", type=int, default=3774)
    parser.add_argument(
        "--full-repo-tb",
        type=float,
        default=7.78,
        help="HF page total file size in decimal TB. Used as a conservative upper estimate.",
    )
    parser.add_argument("--buffer-gib", type=float, default=300)
    args = parser.parse_args()

    local_root = Path(args.local_dir)
    prefix_root = local_root / args.prefix
    local_files = [p for p in prefix_root.rglob("*") if p.is_file()] if prefix_root.exists() else []
    local_file_count = len(local_files)
    prefix_size = sum(p.stat().st_size for p in local_files)
    total_local_size = dir_size(local_root)
    incomplete_count = len(list(local_root.rglob("*.incomplete"))) if local_root.exists() else 0
    lock_count = len(list(local_root.rglob("*.lock"))) if local_root.exists() else 0
    disk_free = shutil.disk_usage(local_root if local_root.exists() else local_root.parent).free

    avg_remaining = None
    if local_file_count > 0:
        avg_file_size = prefix_size / local_file_count
        avg_total = avg_file_size * args.expected_files
        avg_remaining = max(avg_total - prefix_size, 0)

    full_repo_bytes = args.full_repo_tb * 10**12
    full_repo_remaining = max(full_repo_bytes - total_local_size, 0)
    buffer = args.buffer_gib * 1024**3

    print("InternData-N1 local disk estimate")
    print(f"local_dir: {local_root}")
    print(f"prefix: {args.prefix}")
    print(f"local prefix files: {local_file_count}/{args.expected_files}")
    print(f"incomplete files: {incomplete_count}")
    print(f"lock files: {lock_count}")
    print(f"local prefix size: {human(prefix_size)}")
    print(f"local total size including cache: {human(total_local_size)}")
    print(f"disk available: {human(disk_free)}")
    print()

    if avg_remaining is None:
        print("average-file estimate remaining: unavailable, no local files")
    else:
        print(f"average-file estimate remaining: {human(avg_remaining)}")
        if disk_free >= avg_remaining + buffer:
            print("average-file verdict: enough with buffer")
        elif disk_free >= avg_remaining:
            print("average-file verdict: barely enough")
        else:
            print("average-file verdict: NOT enough")

    print()
    print(f"full-repo conservative remaining: {human(full_repo_remaining)}")
    if disk_free >= full_repo_remaining + buffer:
        print("full-repo conservative verdict: enough with buffer")
    elif disk_free >= full_repo_remaining:
        print("full-repo conservative verdict: barely enough")
    else:
        print("full-repo conservative verdict: NOT enough")

    print()
    print("Use the conservative verdict if you intend to keep downloading the full dataset.")


if __name__ == "__main__":
    main()
