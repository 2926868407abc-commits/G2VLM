#!/usr/bin/env python3
"""Inspect InternData-N1 replica_d435i preparation status for G2VLM training."""

import argparse
import ast
import json
import os
import shutil
from pathlib import Path


def human_size(num_bytes):
    value = float(num_bytes)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if value < 1024 or unit == "TB":
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TB"


def count_jsonl(path):
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def discover_extracted_scenes(extract_root):
    if not extract_root.exists():
        return []
    return sorted({path.parent.parent for path in extract_root.rglob("meta/episodes.jsonl")})


def read_parquet_summary(parquet_path, parquet_info_path):
    summary = {
        "exists": parquet_path.exists(),
        "rows": 0,
        "row_groups": 0,
        "goal_pixel_rows": 0,
        "goal_pixel_sources": {},
        "error": None,
    }
    if not parquet_path.exists():
        return summary

    try:
        import pyarrow.parquet as pq

        pf = pq.ParquetFile(parquet_path)
        summary["row_groups"] = pf.num_row_groups
        summary["rows"] = pf.metadata.num_rows

        metadata_col = None
        schema_names = set(pf.schema_arrow.names)
        if "metadata" in schema_names:
            metadata_col = pq.read_table(parquet_path, columns=["metadata"]).column("metadata").to_pylist()

        if metadata_col is not None:
            for value in metadata_col:
                if isinstance(value, dict):
                    metadata = value
                else:
                    try:
                        metadata = ast.literal_eval(value)
                    except Exception:
                        metadata = {}
                if "goal_pixel" in metadata:
                    summary["goal_pixel_rows"] += 1
                source = metadata.get("goal_pixel_source", "missing")
                summary["goal_pixel_sources"][source] = summary["goal_pixel_sources"].get(source, 0) + 1

        if parquet_info_path.exists():
            with parquet_info_path.open("r", encoding="utf-8") as f:
                summary["parquet_info"] = json.load(f)
    except Exception as exc:
        summary["error"] = f"{type(exc).__name__}: {exc}"
    return summary


def print_kv(label, value):
    print(f"{label:<28} {value}")


def latest_training_checkpoint(checkpoint_root):
    if not checkpoint_root.exists():
        return None, None
    run_dirs = sorted(
        [path for path in checkpoint_root.glob("g2vlm_interndata_n1_replica_*") if path.is_dir()],
        key=lambda path: path.stat().st_mtime,
    )
    if not run_dirs:
        return None, None
    run_dir = run_dirs[-1]
    step_dirs = [path for path in run_dir.iterdir() if path.is_dir() and path.name.isdigit()]
    if not step_dirs:
        return run_dir, None
    step_dir = max(step_dirs, key=lambda path: int(path.name))
    return run_dir, step_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=os.environ.get("DATA_ROOT", "/mnt/data/wangqq/G2VLM/data"))
    parser.add_argument("--tar-root", default=None)
    parser.add_argument("--extract-root", default=None)
    parser.add_argument("--converted-root", default=os.environ.get("G2VLM_INTERNDATA_N1_REPLICA_ROOT"))
    parser.add_argument("--checkpoint-root", default="checkpoints")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    tar_root = Path(args.tar_root) if args.tar_root else data_root / "InternData-N1/vln_n1/traj_data/replica_d435i"
    extract_root = (
        Path(args.extract_root)
        if args.extract_root
        else data_root / "InternData-N1-extracted/vln_n1/traj_data/replica_d435i"
    )
    converted_root = (
        Path(args.converted_root)
        if args.converted_root
        else data_root / "g2vlm_interndata_n1/replica_d435i"
    )
    parquet_path = converted_root / "parquets/interndata_n1_replica_d435i.parquet"
    parquet_info_path = converted_root / "parquet_info.json"

    print("== Paths ==")
    print_kv("data_root", data_root)
    print_kv("tar_root", tar_root)
    print_kv("extract_root", extract_root)
    print_kv("converted_root", converted_root)

    print("\n== Disk ==")
    if data_root.exists():
        usage = shutil.disk_usage(data_root)
        print_kv("total", human_size(usage.total))
        print_kv("used", human_size(usage.used))
        print_kv("free", human_size(usage.free))
    else:
        print_kv("data_root exists", "no")

    print("\n== Downloaded Tar Files ==")
    tar_files = sorted(tar_root.glob("*.tar.gz")) if tar_root.exists() else []
    print_kv("tar count", len(tar_files))
    print_kv("tar size", human_size(sum(path.stat().st_size for path in tar_files)))
    for path in tar_files[:20]:
        print(f"  - {path.name:<32} {human_size(path.stat().st_size)}")
    if len(tar_files) > 20:
        print(f"  ... {len(tar_files) - 20} more")

    print("\n== Extracted Scenes ==")
    scenes = discover_extracted_scenes(extract_root)
    print_kv("scene count", len(scenes))
    total_episodes = 0
    total_tasks = 0
    for scene in scenes[:20]:
        episodes = count_jsonl(scene / "meta/episodes.jsonl")
        tasks = count_jsonl(scene / "meta/tasks.jsonl")
        data_parquets = len(list((scene / "data").rglob("*.parquet"))) if (scene / "data").exists() else 0
        total_episodes += episodes
        total_tasks += tasks
        print(f"  - {scene.name:<32} episodes={episodes:<5} tasks={tasks:<5} data_parquets={data_parquets}")
    if len(scenes) > 20:
        print(f"  ... {len(scenes) - 20} more")
    if scenes:
        print_kv("episodes shown", total_episodes)
        print_kv("tasks shown", total_tasks)

    print("\n== Converted Parquet ==")
    summary = read_parquet_summary(parquet_path, parquet_info_path)
    print_kv("parquet exists", "yes" if summary["exists"] else "no")
    print_kv("parquet path", parquet_path)
    if summary["exists"]:
        print_kv("parquet size", human_size(parquet_path.stat().st_size))
        print_kv("rows", summary["rows"])
        print_kv("row groups", summary["row_groups"])
        print_kv("rows with goal_pixel", summary["goal_pixel_rows"])
        print_kv("goal sources", summary["goal_pixel_sources"])
        if summary["error"]:
            print_kv("parquet read error", summary["error"])
    print_kv("parquet_info exists", "yes" if parquet_info_path.exists() else "no")

    print("\n== Training Checkpoints ==")
    checkpoint_root = Path(args.checkpoint_root)
    run_dir, step_dir = latest_training_checkpoint(checkpoint_root)
    print_kv("checkpoint_root", checkpoint_root)
    print_kv("latest run", run_dir if run_dir is not None else "none")
    print_kv("latest step", step_dir if step_dir is not None else "none")
    if step_dir is not None:
        print_kv("model.safetensors", "yes" if (step_dir / "model.safetensors").exists() else "no")
        print_kv("export suggestion", f"python scripts/export_interndata_n1_checkpoint.py --checkpoint {step_dir}")

    print("\n== Suggested Next Step ==")
    if not tar_files:
        print("Download a small replica_d435i subset first, then rerun this doctor script.")
    elif not scenes:
        print("Extract/convert data: bash scripts/prepare_interndata_n1_replica.sh")
    elif not summary["exists"] or summary["rows"] == 0:
        print("Convert extracted data: bash scripts/prepare_interndata_n1_replica.sh")
    elif summary["goal_pixel_rows"] != summary["rows"]:
        print("Rerun the converter so every row has goal_pixel metadata.")
    elif step_dir is not None and (step_dir / "model.safetensors").exists():
        print("Export latest checkpoint, then run one sample inference sanity check.")
    elif summary["rows"] < 100:
        print("Run the single-GPU smoke test: bash scripts/smoke_train_interndata_n1.sh")
    else:
        print("Run smoke if not done yet; after it prints loss, start 4-GPU training.")


if __name__ == "__main__":
    main()
