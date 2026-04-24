#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

import pandas as pd


DEFAULT_CONFIG = "configs/salience_grouped3_hi8_score_only_xml_curated.yaml"
DEFAULT_OUTER = ("M06-1", "M06-2", "M06-3")
TARGET_SPECS = {
    "L1+": {"target": "level1plus_boundary", "min_precision": 0.85},
    "L2+": {"target": "level2plus_boundary", "min_precision": 0.85},
    "L3+": {"target": "level3plus_boundary", "min_precision": 0.85},
    "L4+": {"target": "level4plus_boundary", "min_precision": 0.85},
    "L5+6": {"target": "level56_boundary", "min_precision": 0.80},
}
WEIGHTS_JSON = '{"level56":1.0,"level4":0.64,"level3":0.46,"level2":0.28,"level1":0.16}'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the clean-outer BiLSTM baseline with the same weighted top-down target protocol."
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--targets", nargs="+", choices=list(TARGET_SPECS.keys()), default=list(TARGET_SPECS.keys()))
    parser.add_argument("--outer_heldout_piece", nargs="+", default=list(DEFAULT_OUTER))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--python_exec", default=sys.executable)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--early_stop_patience", type=int, default=10)
    parser.add_argument("--inner_mode", choices=["leave_one", "groupkfold"], default="leave_one")
    parser.add_argument("--max_inner_folds", type=int, default=None)
    parser.add_argument("--selection_metric", default="weighted_recall")
    parser.add_argument("--precision_metric", default="union_precision")
    parser.add_argument("--min_train_frequency_target", type=float, default=0.05)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--reuse_existing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_dir = Path(__file__).resolve().parent
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (project_dir / config_path).resolve()

    output_dir = Path(
        args.output_dir
        or (project_dir / "reports" / f"paper_outer_baselines_weighted_topdown_all_seed{args.seed}_bilstm")
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, object]] = []
    commands: list[str] = []

    for level_label in args.targets:
        spec = TARGET_SPECS[level_label]
        target_mode = str(spec["target"])
        target_out = output_dir / target_mode
        command = [
            str(args.python_exec),
            str((project_dir / "run_nested_piece_cv.py").resolve()),
            "--config",
            str(config_path),
            "--outer_heldout_piece",
            *list(args.outer_heldout_piece),
            "--inner_mode",
            str(args.inner_mode),
            "--model",
            "bilstm",
            "--detector_target",
            target_mode,
            "--selection_metric",
            str(args.selection_metric),
            "--precision_metric",
            str(args.precision_metric),
            "--min_precision",
            str(float(spec["min_precision"])),
            "--loss_type",
            "bce",
            "--min_train_frequency_target",
            str(float(args.min_train_frequency_target)),
            "--cumulative_merge_tolerance",
            "2",
            "--cumulative_component_weights_json",
            WEIGHTS_JSON,
            "--device",
            str(args.device),
            "--batch_size",
            str(int(args.batch_size)),
            "--epochs",
            str(int(args.epochs)),
            "--early_stop_patience",
            str(int(args.early_stop_patience)),
            "--seed",
            str(int(args.seed)),
            "--skip_stage_grading",
            "--run_outer_fit",
            "--output_dir",
            str(target_out),
        ]
        if args.max_inner_folds is not None:
            command.extend(["--max_inner_folds", str(int(args.max_inner_folds))])
        if args.reuse_existing:
            command.append("--reuse_existing")

        commands.append(" ".join(shlex.quote(part) for part in command))
        subprocess.run(command, check=True)

        payload = json.loads((target_out / "summary.json").read_text(encoding="utf-8"))
        outer_summary = payload.get("outer_test_summary")
        if outer_summary is None:
            raise RuntimeError(f"Missing outer_test_summary in {target_out / 'summary.json'}")
        union_metrics = outer_summary["union_metrics"]
        summary_rows.append(
            {
                "model": "bilstm",
                "target_design": "weighted_topdown",
                "feature_family": "all",
                "level_label": level_label,
                "target_mode": target_mode,
                "fixed_threshold": float(union_metrics["threshold"]),
                "context_window_radius": 0,
                "period_k": None,
                "threshold": float(union_metrics["threshold"]),
                "union_precision": float(union_metrics["union_precision"]),
                "frequency_weighted_precision": float(union_metrics["frequency_weighted_precision"]),
                "consensus_precision": float(union_metrics["consensus_precision"]),
                "union_recall": float(union_metrics["union_recall"]),
                "union_f1": float(union_metrics["union_f1"]),
                "weighted_recall": float(union_metrics["weighted_recall"]),
                "consensus_recall": float(union_metrics["consensus_recall"]),
                "mean_offset": union_metrics["mean_offset"],
                "matches": int(union_metrics["matches"]),
                "pred_events": int(union_metrics["pred_events"]),
                "true_union_events": int(union_metrics["true_union_events"]),
                "true_consensus_events": int(union_metrics["true_consensus_events"]),
                "matched_weight": float(union_metrics["matched_weight"]),
                "total_weight": float(union_metrics["total_weight"]),
                "candidate_slug": str(payload["best_candidate"]["candidate_slug"]),
            }
        )
        print(
            f"bilstm {level_label} | p={union_metrics['union_precision']:.4f} | "
            f"wr={union_metrics['weighted_recall']:.4f} | cr={union_metrics['consensus_recall']:.4f} | "
            f"thr={union_metrics['threshold']:.3f}"
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_dir / "outer_summary_by_level.csv", index=False)
    mean_df = (
        summary_df.groupby(["model", "target_design", "feature_family"], sort=False)[
            ["union_precision", "weighted_recall", "union_recall", "consensus_recall", "union_f1"]
        ]
        .mean()
        .reset_index()
    )
    mean_df.to_csv(output_dir / "outer_summary_mean.csv", index=False)
    (output_dir / "commands.txt").write_text("\n".join(commands) + "\n", encoding="utf-8")
    (output_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "config_path": str(config_path),
                "outer_heldout_pieces": list(args.outer_heldout_piece),
                "seed": int(args.seed),
                "targets": list(args.targets),
                "device": str(args.device),
                "batch_size": int(args.batch_size),
                "epochs": int(args.epochs),
                "early_stop_patience": int(args.early_stop_patience),
                "inner_mode": str(args.inner_mode),
                "max_inner_folds": None if args.max_inner_folds is None else int(args.max_inner_folds),
                "reuse_existing": bool(args.reuse_existing),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(str(output_dir))


if __name__ == "__main__":
    main()
