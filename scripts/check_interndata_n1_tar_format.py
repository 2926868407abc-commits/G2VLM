#!/usr/bin/env python3
"""Sample-check InternData-N1 tar files before conversion/training."""

import argparse
import json
import tarfile
import tempfile
from pathlib import Path

import pyarrow.parquet as pq


REQUIRED_PARQUET_COLUMNS = {
    "observation.camera_extrinsic",
    "observation.camera_intrinsic",
    "action",
}


def choose_members(tar):
    names = tar.getnames()
    normalized = [name.rstrip("/") for name in names]
    meta_episodes = [n for n in normalized if n.endswith("/meta/episodes.jsonl") or n == "meta/episodes.jsonl"]
    meta_tasks = [n for n in normalized if n.endswith("/meta/tasks.jsonl") or n == "meta/tasks.jsonl"]
    data_parquets = [n for n in normalized if "/data/episode_" in n and n.endswith(".parquet")]
    rgb_files = [n for n in normalized if "/observation.images.rgb/" in n and n.lower().endswith((".jpg", ".png"))]
    depth_files = [n for n in normalized if "/observation.images.depth/" in n and n.lower().endswith((".png", ".jpg"))]
    return {
        "meta_episodes": meta_episodes,
        "meta_tasks": meta_tasks,
        "data_parquets": data_parquets,
        "rgb_files": rgb_files,
        "depth_files": depth_files,
    }


def read_first_jsonl(tar, member_name):
    f = tar.extractfile(member_name)
    if f is None:
        return None
    for raw in f:
        line = raw.decode("utf-8", errors="replace").strip()
        if line:
            return json.loads(line)
    return None


def check_tar(path):
    result = {
        "tar": str(path),
        "ok": False,
        "errors": [],
        "counts": {},
        "parquet_columns": [],
        "episode_keys": [],
        "task_keys": [],
    }

    try:
        with tarfile.open(path, "r:gz") as tar:
            members = choose_members(tar)
            result["counts"] = {key: len(value) for key, value in members.items()}

            for key in ["meta_episodes", "meta_tasks", "data_parquets", "rgb_files", "depth_files"]:
                if not members[key]:
                    result["errors"].append(f"missing {key}")

            if members["meta_episodes"]:
                episode = read_first_jsonl(tar, members["meta_episodes"][0])
                result["episode_keys"] = sorted(episode.keys()) if isinstance(episode, dict) else []

            if members["meta_tasks"]:
                task = read_first_jsonl(tar, members["meta_tasks"][0])
                result["task_keys"] = sorted(task.keys()) if isinstance(task, dict) else []

            if members["data_parquets"]:
                with tempfile.TemporaryDirectory() as tmp:
                    member = tar.getmember(members["data_parquets"][0])
                    tar.extract(member, tmp)
                    parquet_path = Path(tmp) / member.name
                    table = pq.read_table(parquet_path)
                    columns = set(table.column_names)
                    result["parquet_columns"] = sorted(columns)
                    missing = sorted(REQUIRED_PARQUET_COLUMNS - columns)
                    if missing:
                        result["errors"].append(f"parquet missing columns: {missing}")
                    if table.num_rows == 0:
                        result["errors"].append("sample parquet has 0 rows")
    except Exception as exc:
        result["errors"].append(f"{type(exc).__name__}: {exc}")

    result["ok"] = not result["errors"]
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/wqq/G2VLM/data/InternData-N1/vln_n1/traj_data")
    parser.add_argument(
        "--subsets",
        nargs="+",
        default=["3dfront_d435i", "gibson_d435i", "hm3d_d435i"],
    )
    parser.add_argument("--samples-per-subset", type=int, default=3)
    args = parser.parse_args()

    root = Path(args.root)
    all_ok = True
    for subset in args.subsets:
        subset_dir = root / subset
        tar_files = sorted(subset_dir.glob("*.tar.gz"))
        print(f"\n{subset}: {len(tar_files)} tar files")
        if not tar_files:
            all_ok = False
            print("  ERROR: no tar files")
            continue

        sample_files = tar_files[: args.samples_per_subset]
        for path in sample_files:
            result = check_tar(path)
            status = "OK" if result["ok"] else "BAD"
            print(f"  [{status}] {path.name}")
            print(f"    counts: {result['counts']}")
            if result["episode_keys"]:
                print(f"    episode keys: {result['episode_keys'][:20]}")
            if result["task_keys"]:
                print(f"    task keys: {result['task_keys'][:20]}")
            if result["parquet_columns"]:
                print(f"    parquet cols: {result['parquet_columns'][:20]}")
            for error in result["errors"]:
                print(f"    ERROR: {error}")
            all_ok = all_ok and result["ok"]

    raise SystemExit(0 if all_ok else 2)


if __name__ == "__main__":
    main()
