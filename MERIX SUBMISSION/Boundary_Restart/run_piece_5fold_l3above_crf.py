#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from boundary_restart.config import load_config, resolve_path
from boundary_restart.metrics import greedy_match_pairs, greedy_soft_match_pairs
from boundary_restart.table_io import load_table


TARGET_SPECS = {
    "L1+": {"target": "level1plus_boundary", "min_precision": 0.85},
    "L2+": {"target": "level2plus_boundary", "min_precision": 0.85},
    "L3+": {"target": "level3plus_boundary", "min_precision": 0.85},
    "L4+": {"target": "level4plus_boundary", "min_precision": 0.85},
    "L5": {"target": "level56_boundary", "min_precision": 0.80},
}
WEIGHTS_JSON = '{"level56":1.0,"level4":0.64,"level3":0.46,"level2":0.28,"level1":0.16}'


def build_piece_list(config_path: Path) -> list[str]:
    cfg = load_config(str(config_path))
    table_path = resolve_path(cfg, cfg["data"]["beat_table_path"])
    df = load_table(table_path)
    return sorted(df["piece_id"].dropna().unique().tolist())


def chunked_folds(items: list[str], n_splits: int) -> list[list[str]]:
    folds: list[list[str]] = [[] for _ in range(n_splits)]
    for idx, item in enumerate(items):
        folds[idx % n_splits].append(item)
    return [fold for fold in folds if fold]


def run_command(command: list[str], reuse_existing: bool, summary_path: Path) -> None:
    if reuse_existing and summary_path.exists():
        return
    subprocess.run(command, check=True)


def summarize_level_runs(output_dir: Path, folds: list[list[str]], targets: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for fold_idx, heldout in enumerate(folds, start=1):
        for level_label in targets:
            target = TARGET_SPECS[level_label]["target"]
            summary_path = output_dir / target / f"fold_{fold_idx:02d}" / "summary.json"
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            metrics = payload["union_metrics"]
            rows.append(
                {
                    "fold": fold_idx,
                    "heldout_pieces": ",".join(heldout),
                    "level": level_label,
                    "target": target,
                    "threshold": float(metrics["threshold"]),
                    "union_precision": float(metrics["union_precision"]),
                    "union_recall": float(metrics["union_recall"]),
                    "weighted_recall": float(metrics["weighted_recall"]),
                    "consensus_recall": float(metrics["consensus_recall"]),
                    "union_f1": float(metrics["union_f1"]),
                    "pred_events": int(metrics["pred_events"]),
                    "true_union_events": int(metrics["true_union_events"]),
                    "matches": float(metrics["matches"]),
                    "matched_weight": float(metrics["matched_weight"]),
                    "total_weight": float(metrics["total_weight"]),
                    "true_consensus_events": int(metrics["true_consensus_events"]),
                }
            )
    return pd.DataFrame(rows)


def combined_metrics_for_fold(output_dir: Path, fold_idx: int, mode: str, targets: list[str]) -> dict[str, float]:
    frames = []
    for level_label in targets:
        spec = TARGET_SPECS[level_label]
        path = output_dir / spec["target"] / f"fold_{fold_idx:02d}" / "val_predictions.csv.gz"
        frame = pd.read_csv(path)
        frame["target"] = spec["target"]
        frames.append(frame)
    all_df = pd.concat(frames, ignore_index=True)

    totals = {
        "pred_events": 0,
        "true_union_events": 0,
        "matches": 0.0,
        "matched_weight": 0.0,
        "total_weight": 0.0,
    }
    for _, group in all_df.groupby("piece_id", sort=True):
        pred_events = np.array(
            sorted(group.loc[group["decoded_boundary"] > 0.5, "beat_idx"].astype(int).unique()),
            dtype=np.int32,
        )
        truth_by_beat = group.groupby("beat_idx")["frequency_target"].max().astype(float)
        true_events = np.array(sorted(truth_by_beat.index[truth_by_beat > 0].astype(int).tolist()), dtype=np.int32)
        weights = {int(idx): float(value) for idx, value in truth_by_beat.loc[truth_by_beat > 0].items()}
        total_weight = float(sum(weights.values()))

        if mode == "hard_exact":
            matches = greedy_match_pairs(pred_events, true_events, tolerance=0)
            match_credit = float(len(matches))
            matched_weight = float(sum(weights[int(true_events[true_idx])] for _, true_idx, _ in matches))
        elif mode == "soft_decay":
            matches = greedy_soft_match_pairs(pred_events, true_events, tolerance=0)
            match_credit = float(sum(credit for _, _, _, credit in matches))
            matched_weight = float(
                sum(weights[int(true_events[true_idx])] * credit for _, true_idx, _, credit in matches)
            )
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        totals["pred_events"] += int(pred_events.size)
        totals["true_union_events"] += int(true_events.size)
        totals["matches"] += match_credit
        totals["matched_weight"] += matched_weight
        totals["total_weight"] += total_weight

    union_precision = totals["matches"] / totals["pred_events"] if totals["pred_events"] else 0.0
    union_recall = totals["matches"] / totals["true_union_events"] if totals["true_union_events"] else 0.0
    weighted_recall = totals["matched_weight"] / totals["total_weight"] if totals["total_weight"] else 0.0
    union_f1 = (
        2.0 * union_precision * union_recall / (union_precision + union_recall)
        if union_precision + union_recall
        else 0.0
    )
    return {
        "union_precision": float(union_precision),
        "union_recall": float(union_recall),
        "weighted_recall": float(weighted_recall),
        "union_f1": float(union_f1),
        **totals,
    }


def summarize_combined(output_dir: Path, folds: list[list[str]], targets: list[str]) -> pd.DataFrame:
    rows = []
    for fold_idx, heldout in enumerate(folds, start=1):
        for mode in ("soft_decay", "hard_exact"):
            metrics = combined_metrics_for_fold(output_dir, fold_idx, mode=mode, targets=targets)
            rows.append(
                {
                    "fold": fold_idx,
                    "heldout_pieces": ",".join(heldout),
                    "mode": mode,
                    **metrics,
                }
            )
    return pd.DataFrame(rows)


def aggregate(frame: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    metrics = ["union_precision", "union_recall", "weighted_recall", "union_f1", "pred_events"]
    return (
        frame.groupby(group_cols, sort=False)
        .agg(**{f"{metric}_mean": (metric, "mean") for metric in metrics}, **{f"{metric}_std": (metric, "std") for metric in metrics})
        .reset_index()
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run piece-level 5-fold CV for CRF boundary detectors.")
    parser.add_argument("--config", default="configs/salience_grouped3_hi8_score_only_xml_curated.yaml")
    parser.add_argument("--output_dir", default="reports/paper_5fold_multistate_crf_seed42_l3above")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--targets", nargs="+", choices=list(TARGET_SPECS), default=list(TARGET_SPECS))
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--model", default="cnn_crf")
    parser.add_argument("--loss_type", default="crf_nll")
    parser.add_argument("--selection_metric", default="weighted_recall")
    parser.add_argument("--precision_metric", default="union_precision")
    parser.add_argument("--min_train_frequency_target", type=float, default=0.05)
    parser.add_argument("--cumulative_merge_tolerance", type=int, default=0)
    parser.add_argument("--event_tolerance", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--early_stop_patience", type=int, default=10)
    parser.add_argument("--python_exec", default=sys.executable)
    parser.add_argument("--reuse_existing", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_dir = Path(__file__).resolve().parent
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (project_dir / config_path).resolve()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = (project_dir / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    pieces = build_piece_list(config_path)
    folds = chunked_folds(pieces, int(args.n_splits))
    (output_dir / "folds.json").write_text(json.dumps(folds, indent=2), encoding="utf-8")

    commands = []
    protocol = project_dir / "train_piece_union_protocol.py"
    for fold_idx, heldout in enumerate(folds, start=1):
        train_pieces = [piece for piece in pieces if piece not in set(heldout)]
        for level_label in args.targets:
            spec = TARGET_SPECS[level_label]
            target = spec["target"]
            fold_dir = output_dir / target / f"fold_{fold_idx:02d}"
            command = [
                str(args.python_exec),
                str(protocol),
                "--config",
                str(config_path),
                "--heldout_piece",
                *heldout,
                "--train_pieces",
                *train_pieces,
                "--output_dir",
                str(fold_dir),
                "--model",
                str(args.model),
                "--detector_target",
                target,
                "--selection_metric",
                str(args.selection_metric),
                "--precision_metric",
                str(args.precision_metric),
                "--min_precision",
                str(float(spec["min_precision"])),
                "--loss_type",
                str(args.loss_type),
                "--min_train_frequency_target",
                str(float(args.min_train_frequency_target)),
                "--cumulative_merge_tolerance",
                str(int(args.cumulative_merge_tolerance)),
                "--cumulative_component_weights_json",
                WEIGHTS_JSON,
                "--event_decoder",
                "crf",
                "--event_tolerance",
                str(int(args.event_tolerance)),
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
            ]
            commands.append(" ".join(shlex.quote(part) for part in command))
            print(f"fold {fold_idx}/{len(folds)} {level_label} heldout={','.join(heldout)}")
            if not args.dry_run:
                run_command(command, reuse_existing=bool(args.reuse_existing), summary_path=fold_dir / "summary.json")

    (output_dir / "commands.txt").write_text("\n".join(commands) + "\n", encoding="utf-8")
    if args.dry_run:
        print(output_dir)
        return

    level_rows = summarize_level_runs(output_dir, folds, targets=list(args.targets))
    combined_rows = summarize_combined(output_dir, folds, targets=list(args.targets))
    level_stats = aggregate(level_rows, ["level"])
    combined_stats = aggregate(combined_rows, ["mode"])

    target_slug = "_".join(level_label.replace("+", "").lower() for level_label in args.targets)
    prefix = f"fivefold_{target_slug}"
    level_rows.to_csv(output_dir / f"{prefix}_by_level.csv", index=False)
    level_stats.to_csv(output_dir / f"{prefix}_by_level_stats.csv", index=False)
    combined_rows.to_csv(output_dir / f"{prefix}_combined.csv", index=False)
    combined_stats.to_csv(output_dir / f"{prefix}_combined_stats.csv", index=False)
    print(output_dir)
    print(combined_stats.to_string(index=False))


if __name__ == "__main__":
    main()
