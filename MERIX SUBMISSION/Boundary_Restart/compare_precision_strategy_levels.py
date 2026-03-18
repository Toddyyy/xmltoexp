#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


DEFAULT_PIECES = ["M06-1", "M06-2", "M06-3", "M17-1", "M30-1"]
LEVEL_TARGETS = [
    "level1_boundary",
    "level2_boundary",
    "level3_boundary",
    "level4_boundary",
    "level5_boundary",
    "level6_boundary",
    "level56_boundary",
]


def run_variant(
    *,
    train_script: Path,
    config: str,
    piece: str,
    target: str,
    device: str,
    seed: int,
    output_dir: Path,
    variant: str,
    baseline_min_precision: float,
    guarded_min_precision: float,
    guarded_min_union_precision_floor: float,
) -> Path:
    summary_path = output_dir / "summary.json"
    if summary_path.exists():
        return summary_path

    cmd = [
        sys.executable,
        str(train_script),
        "--config",
        config,
        "--heldout_piece",
        piece,
        "--model",
        "tcn",
        "--device",
        device,
        "--seed",
        str(seed),
        "--detector_target",
        target,
        "--selection_metric",
        "weighted_recall",
        "--skip_stage_grading",
        "--output_dir",
        str(output_dir),
    ]

    if variant == "baseline":
        cmd.extend(
            [
                "--precision_metric",
                "union_precision",
                "--min_precision",
                str(baseline_min_precision),
            ]
        )
    elif variant == "consensus_guarded":
        cmd.extend(
            [
                "--precision_metric",
                "consensus_precision",
                "--min_precision",
                str(guarded_min_precision),
                "--min_union_precision_floor",
                str(guarded_min_union_precision_floor),
            ]
        )
    else:
        raise ValueError(f"Unknown variant: {variant}")

    subprocess.run(cmd, check=True)
    return summary_path


def load_summary(summary_path: Path) -> dict:
    return json.loads(summary_path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare baseline vs guarded precision strategies across levels.")
    parser.add_argument(
        "--config",
        default="MERIX SUBMISSION/Boundary_Restart/configs/salience_grouped3_hi8_score_only_xml_curated.yaml",
    )
    parser.add_argument("--device", default="mps")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pieces", nargs="*", default=DEFAULT_PIECES)
    parser.add_argument("--targets", nargs="*", default=LEVEL_TARGETS)
    parser.add_argument("--baseline_min_precision", type=float, default=0.85)
    parser.add_argument("--guarded_min_precision", type=float, default=0.05)
    parser.add_argument("--guarded_min_union_precision_floor", type=float, default=0.60)
    parser.add_argument(
        "--run_root",
        default="MERIX SUBMISSION/Boundary_Restart/outputs/local_runs/strategy_compare_alllevels",
    )
    parser.add_argument(
        "--report_dir",
        default="MERIX SUBMISSION/Boundary_Restart/reports/strategy_compare_alllevels",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    train_script = project_root / "train_piece_union_protocol.py"
    run_root = Path(args.run_root)
    report_dir = Path(args.report_dir)
    run_root.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for piece in args.pieces:
        for target in args.targets:
            for variant in ("baseline", "consensus_guarded"):
                out_dir = run_root / f"{piece}_{target}_{variant}_seed{args.seed}"
                summary_path = run_variant(
                    train_script=train_script,
                    config=args.config,
                    piece=piece,
                    target=target,
                    device=args.device,
                    seed=int(args.seed),
                    output_dir=out_dir,
                    variant=variant,
                    baseline_min_precision=float(args.baseline_min_precision),
                    guarded_min_precision=float(args.guarded_min_precision),
                    guarded_min_union_precision_floor=float(args.guarded_min_union_precision_floor),
                )
                summary = load_summary(summary_path)
                metrics = summary["union_metrics"]
                rows.append(
                    {
                        "piece_id": piece,
                        "detector_target": target,
                        "variant": variant,
                        "seed": int(args.seed),
                        "precision_metric": summary.get("precision_metric"),
                        "precision_floors": json.dumps(summary.get("precision_floors", {}), sort_keys=True),
                        "best_epoch": summary.get("best_epoch"),
                        "threshold": metrics.get("threshold"),
                        "union_precision": metrics.get("union_precision"),
                        "frequency_weighted_precision": metrics.get("frequency_weighted_precision"),
                        "consensus_precision": metrics.get("consensus_precision"),
                        "union_recall": metrics.get("union_recall"),
                        "weighted_recall": metrics.get("weighted_recall"),
                        "consensus_recall": metrics.get("consensus_recall"),
                    }
                )

    fieldnames = list(rows[0].keys())
    detailed_csv = report_dir / "piece_level_strategy_compare.csv"
    with detailed_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    avg_rows: list[dict] = []
    for target in args.targets:
        for variant in ("baseline", "consensus_guarded"):
            subset = [row for row in rows if row["detector_target"] == target and row["variant"] == variant]
            avg_rows.append(
                {
                    "detector_target": target,
                    "variant": variant,
                    "mean_union_precision": sum(float(r["union_precision"]) for r in subset) / len(subset),
                    "mean_frequency_weighted_precision": sum(float(r["frequency_weighted_precision"]) for r in subset)
                    / len(subset),
                    "mean_consensus_precision": sum(float(r["consensus_precision"]) for r in subset) / len(subset),
                    "mean_union_recall": sum(float(r["union_recall"]) for r in subset) / len(subset),
                    "mean_weighted_recall": sum(float(r["weighted_recall"]) for r in subset) / len(subset),
                    "mean_consensus_recall": sum(float(r["consensus_recall"]) for r in subset) / len(subset),
                }
            )

    avg_csv = report_dir / "level_mean_strategy_compare.csv"
    with avg_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(avg_rows[0].keys()))
        writer.writeheader()
        writer.writerows(avg_rows)

    print(detailed_csv)
    print(avg_csv)


if __name__ == "__main__":
    main()
