#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def run_command(cmd: list[str], cwd: Path) -> None:
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def load_direct_metrics(summary_path: Path) -> dict[str, float]:
    data = json.loads(summary_path.read_text())
    metrics = data["union_metrics"]
    return {
        "precision": float(metrics["union_precision"]),
        "union_recall": float(metrics["union_recall"]),
        "weighted_recall": float(metrics["weighted_recall"]),
        "consensus_recall": float(metrics["consensus_recall"]),
        "pred_events": int(metrics["pred_events"]),
        "true_union_events": int(metrics["true_union_events"]),
        "threshold": float(metrics["threshold"]),
        "best_epoch": int(data["best_epoch"]),
        "epochs_run": int(data["epochs_run"]),
    }


def load_shared_metrics(summary_path: Path) -> dict[str, float]:
    data = json.loads(summary_path.read_text())
    metrics = data["head_metrics"]["L56"]
    return {
        "precision": float(metrics["union_precision"]),
        "union_recall": float(metrics["union_recall"]),
        "weighted_recall": float(metrics["weighted_recall"]),
        "consensus_recall": float(metrics["consensus_recall"]),
        "pred_events": int(metrics["pred_events"]),
        "true_union_events": int(metrics["true_union_events"]),
        "threshold": float(metrics["threshold"]),
        "best_epoch": int(data["best_epoch"]),
        "epochs_run": int(data["epochs_run"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--pieces", nargs="+", default=["M06-2", "M17-1", "M30-1"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--min_precision", type=float, default=0.85)
    parser.add_argument("--selection_metric", default="union_recall")
    parser.add_argument(
        "--output_root",
        default="MERIX SUBMISSION/Boundary_Restart/outputs/local_runs/l56_shared_seed_sweep",
    )
    parser.add_argument("--skip_existing", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    script_dir = Path(__file__).resolve().parent
    output_root = (repo_root / args.output_root).resolve()
    direct_root = output_root / "direct"
    shared_root = output_root / "shared"
    direct_root.mkdir(parents=True, exist_ok=True)
    shared_root.mkdir(parents=True, exist_ok=True)

    direct_script = script_dir / "train_piece_union_protocol.py"
    shared_script = script_dir / "train_piece_union_fourgroup_shared.py"
    config_path = str((repo_root / args.config).resolve())

    rows: list[dict[str, object]] = []
    for piece in args.pieces:
        for seed in args.seeds:
            direct_out = direct_root / piece / f"seed{seed}"
            shared_out = shared_root / piece / f"seed{seed}"
            direct_summary = direct_out / "summary.json"
            shared_summary = shared_out / "summary.json"

            if not (args.skip_existing and direct_summary.exists()):
                direct_out.mkdir(parents=True, exist_ok=True)
                run_command(
                    [
                        sys.executable,
                        str(direct_script),
                        "--config",
                        config_path,
                        "--heldout_piece",
                        piece,
                        "--model",
                        "tcn",
                        "--device",
                        args.device,
                        "--detector_target",
                        "level56_boundary",
                        "--selection_metric",
                        args.selection_metric,
                        "--min_precision",
                        str(args.min_precision),
                        "--skip_stage_grading",
                        "--seed",
                        str(seed),
                        "--output_dir",
                        str(direct_out),
                    ],
                    cwd=repo_root,
                )

            if not (args.skip_existing and shared_summary.exists()):
                shared_out.mkdir(parents=True, exist_ok=True)
                run_command(
                    [
                        sys.executable,
                        str(shared_script),
                        "--config",
                        config_path,
                        "--heldout_piece",
                        piece,
                        "--device",
                        args.device,
                        "--selection_metric",
                        args.selection_metric,
                        "--min_precision",
                        str(args.min_precision),
                        "--seed",
                        str(seed),
                        "--output_dir",
                        str(shared_out),
                    ],
                    cwd=repo_root,
                )

            direct_metrics = load_direct_metrics(direct_summary)
            shared_metrics = load_shared_metrics(shared_summary)
            rows.append(
                {
                    "piece_id": piece,
                    "seed": seed,
                    **{f"direct_{k}": v for k, v in direct_metrics.items()},
                    **{f"shared_{k}": v for k, v in shared_metrics.items()},
                }
            )

    df = pd.DataFrame(rows).sort_values(["piece_id", "seed"]).reset_index(drop=True)
    per_piece = (
        df.groupby("piece_id")
        .agg(["mean", "std"])
        .reset_index()
    )
    overall = df.drop(columns=["piece_id", "seed"]).agg(["mean", "std"]).reset_index()

    csv_path = output_root / "direct_vs_shared_l56_seed_sweep.csv"
    piece_csv = output_root / "direct_vs_shared_l56_piece_summary.csv"
    overall_csv = output_root / "direct_vs_shared_l56_overall_summary.csv"
    df.to_csv(csv_path, index=False)
    per_piece.to_csv(piece_csv, index=False)
    overall.to_csv(overall_csv, index=False)

    direct_better = (df["shared_weighted_recall"] > df["direct_weighted_recall"]).sum()
    print()
    print(f"Wrote {csv_path}")
    print(f"Wrote {piece_csv}")
    print(f"Wrote {overall_csv}")
    print(
        "Shared beats direct on weighted_recall in "
        f"{direct_better}/{len(df)} piece-seed runs"
    )


if __name__ == "__main__":
    main()
