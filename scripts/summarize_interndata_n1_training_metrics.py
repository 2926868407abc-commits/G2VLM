#!/usr/bin/env python3
"""Summarize InternData-N1 training loss metrics from G2VLM log files."""

import argparse
import math
import re
from pathlib import Path


STEP_RE = re.compile(r"\(step=(\d+)\)")
LOSS_RE = re.compile(r"Train Loss ([A-Za-z0-9_./-]+):\s*([-+0-9.eE]+)")


def newest_log(default_root):
    root = Path(default_root)
    candidates = list(root.rglob("log.txt")) if root.exists() else []
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def parse_log(path):
    rows = []
    with Path(path).open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            step_match = STEP_RE.search(line)
            if not step_match:
                continue
            losses = {
                key: float(value)
                for key, value in LOSS_RE.findall(line)
                if math.isfinite(float(value))
            }
            if not losses:
                continue
            rows.append({"step": int(step_match.group(1)), "losses": losses})
    return rows


def metric_values(rows, metric):
    values = []
    for row in rows:
        if metric in row["losses"]:
            values.append((row["step"], row["losses"][metric]))
    return values


def mean(values):
    return sum(values) / len(values) if values else float("nan")


def summarize_metric(values, window):
    first_step, first_value = values[0]
    last_step, last_value = values[-1]
    best_step, best_value = min(values, key=lambda item: item[1])
    recent = [value for _, value in values[-window:]]
    previous = [value for _, value in values[-2 * window : -window]]
    recent_mean = mean(recent)
    previous_mean = mean(previous)
    if previous and abs(previous_mean) > 1e-12:
        recent_change = (recent_mean - previous_mean) / abs(previous_mean)
    else:
        recent_change = float("nan")
    total_change = (last_value - first_value) / abs(first_value) if abs(first_value) > 1e-12 else float("nan")
    return {
        "first": (first_step, first_value),
        "last": (last_step, last_value),
        "best": (best_step, best_value),
        "recent_mean": recent_mean,
        "previous_mean": previous_mean,
        "recent_change": recent_change,
        "total_change": total_change,
    }


def convergence_label(summary, tolerance):
    change = summary["recent_change"]
    if math.isnan(change):
        return "need-more-points"
    if abs(change) <= tolerance:
        return "flat/converging"
    if change < -tolerance:
        return "still-decreasing"
    return "getting-worse"


def fmt(value):
    if math.isnan(value):
        return "nan"
    return f"{value:.6g}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", default=None, help="Path to train log.txt. Defaults to newest ./checkpoints/**/log.txt.")
    parser.add_argument("--checkpoint-root", default="checkpoints")
    parser.add_argument("--window", type=int, default=20, help="Number of recent logged points per convergence window.")
    parser.add_argument("--tolerance", type=float, default=0.02, help="Relative window-change tolerance for convergence.")
    args = parser.parse_args()

    log_path = Path(args.log) if args.log else newest_log(args.checkpoint_root)
    if log_path is None:
        raise SystemExit(f"No log.txt found under {args.checkpoint_root}")

    rows = parse_log(log_path)
    if not rows:
        raise SystemExit(f"No training loss rows found in {log_path}")

    metrics = sorted({key for row in rows for key in row["losses"]})
    print(f"log: {log_path}")
    print(f"logged points: {len(rows)}")
    print(f"step range: {rows[0]['step']} -> {rows[-1]['step']}")
    print(f"window: {args.window}, tolerance: {args.tolerance:.3g}")
    print()

    for metric in metrics:
        values = metric_values(rows, metric)
        if len(values) < 2:
            continue
        summary = summarize_metric(values, min(args.window, len(values)))
        label = convergence_label(summary, args.tolerance)
        first_step, first_value = summary["first"]
        last_step, last_value = summary["last"]
        best_step, best_value = summary["best"]
        print(
            f"{metric}: {label} | "
            f"first step {first_step}={fmt(first_value)}, "
            f"last step {last_step}={fmt(last_value)}, "
            f"best step {best_step}={fmt(best_value)}, "
            f"recent_mean={fmt(summary['recent_mean'])}, "
            f"prev_mean={fmt(summary['previous_mean'])}, "
            f"recent_change={fmt(summary['recent_change'] * 100)}%, "
            f"total_change={fmt(summary['total_change'] * 100)}%"
        )


if __name__ == "__main__":
    main()
