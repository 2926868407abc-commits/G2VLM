#!/usr/bin/env python3
"""Print the files that should be synced to the training server for InternData-N1."""

import argparse
from pathlib import Path


SYNC_FILES = [
    "data/preprocessing/convert_interndata_n1_replica_to_g2vlm.py",
    "data/preprocessing/preview_interndata_n1_goal.py",
    "data/configs/joint_train_interndata_n1_replica.yaml",
    "data/dataset_info.py",
    "data/interleave_datasets/__init__.py",
    "data/interleave_datasets/recon_then_und_dataset.py",
    "data/interleave_datasets/draw_marker.py",
    "data/draw_marker.py",
    "scripts/joint_train_single_node_interndata_n1.sh",
    "scripts/doctor_interndata_n1_replica.py",
    "scripts/export_interndata_n1_checkpoint.py",
    "scripts/infer_interndata_n1_replica_sample.py",
    "scripts/smoke_train_interndata_n1.sh",
    "scripts/prepare_interndata_n1_replica.sh",
    "scripts/resolve_hf_repo.py",
    "train/joint_train_unified_model.py",
    "docs/interndata_n1_replica_training.md",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="Exit nonzero if any listed file is missing.")
    parser.add_argument("--abs", action="store_true", help="Print absolute paths.")
    args = parser.parse_args()

    root = Path.cwd()
    missing = []
    for rel in SYNC_FILES:
        path = root / rel
        if not path.exists():
            missing.append(rel)
        print(path.resolve() if args.abs else rel)

    if missing:
        print("\nMissing sync files:")
        for rel in missing:
            print(f"- {rel}")
        if args.check:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
