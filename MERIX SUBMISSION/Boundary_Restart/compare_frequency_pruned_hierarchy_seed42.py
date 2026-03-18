#!/usr/bin/env python3

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


OUTER_PIECES = ["M06-1", "M06-2", "M06-3"]
SEED = 42
MAX_EPOCHS = 60
EARLY_STOP_PATIENCE = 10

HIERARCHY_TARGETS = {
    "L6": ("level6_boundary", False),
    "L5+": ("level5plus_split56_boundary", True),
    "L4+": ("level4plus_split56_boundary", True),
    "L3+": ("level3plus_split56_boundary", True),
    "L2+": ("level2plus_split56_boundary", True),
    "L1+": ("level1plus_split56_boundary", True),
}

BASELINE_SUMMARIES = {
    "L6": "outputs/local_runs/full_split56_hierarchy_seed42/M06_outer_level6_boundary_seed42/summary.json",
    "L5+": "outputs/local_runs/full_split56_hierarchy_seed42/M06_outer_level5plus_split56_boundary_seed42/summary.json",
    "L4+": "outputs/local_runs/full_split56_hierarchy_seed42/M06_outer_level4plus_split56_boundary_seed42/summary.json",
    "L3+": "outputs/local_runs/full_split56_hierarchy_seed42/M06_outer_level3plus_split56_boundary_seed42/summary.json",
    "L2+": "outputs/local_runs/cumulative_training_split56_merge2_seed42/M06_outer_level2plus_split56_boundary_merge2_seed42/summary.json",
    "L1+": "outputs/local_runs/cumulative_training_split56_merge2_seed42/M06_outer_level1plus_split56_boundary_merge2_seed42/summary.json",
}


def run_training(train_script: Path, config: str, target: str, output_dir: Path, *, cumulative: bool, min_train_frequency_target: float) -> None:
    summary_path = output_dir / "summary.json"
    if summary_path.exists():
        return
    cmd = [
        sys.executable,
        str(train_script),
        "--config",
        config,
        "--heldout_piece",
        *OUTER_PIECES,
        "--model",
        "tcn",
        "--device",
        "mps",
        "--seed",
        str(SEED),
        "--detector_target",
        target,
        "--selection_metric",
        "weighted_recall",
        "--precision_metric",
        "union_precision",
        "--min_precision",
        "0.85",
        "--epochs",
        str(MAX_EPOCHS),
        "--early_stop_patience",
        str(EARLY_STOP_PATIENCE),
        "--skip_stage_grading",
        "--min_train_frequency_target",
        str(min_train_frequency_target),
        "--output_dir",
        str(output_dir),
    ]
    if cumulative:
        cmd.extend(["--cumulative_merge_tolerance", "2"])
    subprocess.run(cmd, check=True)


def metrics_from_summary(path: Path) -> dict:
    summary = json.loads(path.read_text(encoding="utf-8"))
    metrics = summary["union_metrics"]
    return {
        "best_epoch": summary.get("best_epoch"),
        "threshold": metrics.get("threshold"),
        "union_precision": metrics["union_precision"],
        "frequency_weighted_precision": metrics.get("frequency_weighted_precision"),
        "consensus_precision": metrics.get("consensus_precision"),
        "union_recall": metrics["union_recall"],
        "weighted_recall": metrics["weighted_recall"],
        "consensus_recall": metrics["consensus_recall"],
    }


def main() -> None:
    project_root = Path(__file__).resolve().parent
    train_script = project_root / "train_piece_union_protocol.py"
    config_path = project_root / "configs/salience_grouped3_hi8_score_only_xml_curated.yaml"
    run_root = project_root / "outputs/local_runs/frequency_pruned_hierarchy_seed42_e60"
    report_dir = project_root / "reports/frequency_pruned_hierarchy_seed42_e60"
    run_root.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    thresholds = [0.0, 0.03, 0.05, 0.10]
    rows: list[dict] = []

    for label, baseline_rel in BASELINE_SUMMARIES.items():
        metrics = metrics_from_summary(project_root / baseline_rel)
        rows.append(
            {
                "hierarchy_level": label,
                "detector_target": HIERARCHY_TARGETS[label][0],
                "min_train_frequency_target": 0.0,
                "variant": "baseline",
                **metrics,
            }
        )

    for min_freq in thresholds[1:]:
        for label, (target, cumulative) in HIERARCHY_TARGETS.items():
            suffix = str(min_freq).replace(".", "p")
            out_dir = run_root / f"M06_outer_{target}_freqfloor_{suffix}_seed{SEED}"
            run_training(
                train_script,
                str(config_path),
                target,
                out_dir,
                cumulative=cumulative,
                min_train_frequency_target=min_freq,
            )
            metrics = metrics_from_summary(out_dir / "summary.json")
            rows.append(
                {
                    "hierarchy_level": label,
                    "detector_target": target,
                    "min_train_frequency_target": min_freq,
                    "variant": f"train_freq_floor_{min_freq:.2f}",
                    **metrics,
                }
            )

    detail_path = report_dir / "hierarchy_compare.csv"
    with detail_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    avg = pd.DataFrame(rows)
    avg_path = report_dir / "hierarchy_compare_sorted.csv"
    avg.sort_values(["hierarchy_level", "min_train_frequency_target"]).to_csv(avg_path, index=False)

    print(detail_path)
    print(avg_path)


if __name__ == "__main__":
    main()
