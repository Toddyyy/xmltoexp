#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import shlex
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from boundary_restart.config import load_config, resolve_path
from boundary_restart.metrics import decode_events, greedy_match_pairs, greedy_soft_match_pairs
from boundary_restart.table_io import load_table


TARGET = "weighted_all6_boundary"
WEIGHTS_JSON = '{"level6":1.0,"level5":0.82,"level4":0.64,"level3":0.46,"level2":0.28,"level1":0.16}'


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


def empty_totals() -> dict[str, float]:
    return {
        "pred_events": 0.0,
        "true_union_events": 0.0,
        "true_consensus_events": 0.0,
        "matches": 0.0,
        "consensus_matches": 0.0,
        "matched_weight": 0.0,
        "total_weight": 0.0,
    }


def finalize(totals: dict[str, float]) -> dict[str, float]:
    precision = totals["matches"] / totals["pred_events"] if totals["pred_events"] else 0.0
    recall = totals["matches"] / totals["true_union_events"] if totals["true_union_events"] else 0.0
    weighted_recall = totals["matched_weight"] / totals["total_weight"] if totals["total_weight"] else 0.0
    consensus_recall = (
        totals["consensus_matches"] / totals["true_consensus_events"] if totals["true_consensus_events"] else 0.0
    )
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "union_precision": float(precision),
        "union_recall": float(recall),
        "weighted_recall": float(weighted_recall),
        "consensus_recall": float(consensus_recall),
        "union_f1": float(f1),
        "pred_events": float(totals["pred_events"]),
        "true_union_events": float(totals["true_union_events"]),
        "true_consensus_events": float(totals["true_consensus_events"]),
        "matches": float(totals["matches"]),
        "matched_weight": float(totals["matched_weight"]),
        "total_weight": float(totals["total_weight"]),
    }


def score_events_for_piece(scores: np.ndarray, threshold: float, min_distance: int, event_decoder: str) -> np.ndarray:
    return decode_events(
        np.asarray(scores, dtype=np.float32),
        threshold=float(threshold),
        min_distance=int(min_distance),
        prominence=0.0,
        event_decoder=str(event_decoder),
    )


def evaluate_prediction_frame(
    pred_df: pd.DataFrame,
    threshold: float,
    min_distance: int,
    event_decoder: str,
    mode: str,
) -> dict[str, float]:
    totals = empty_totals()
    for _, group in pred_df.groupby("piece_id", sort=True):
        beat_idx = group["beat_idx"].to_numpy(dtype=np.int32)
        if str(event_decoder) == "decoded" and "decoded_boundary" in group.columns:
            pred_events = group.loc[group["decoded_boundary"] > 0.5, "beat_idx"].astype(int).to_numpy(dtype=np.int32)
        else:
            pred_event_idx = score_events_for_piece(
                group["detector_score"].to_numpy(dtype=np.float32),
                threshold=float(threshold),
                min_distance=int(min_distance),
                event_decoder=str(event_decoder),
            )
            pred_events = beat_idx[pred_event_idx[(pred_event_idx >= 0) & (pred_event_idx < beat_idx.shape[0])]]

        truth_by_beat = group.groupby("beat_idx")["frequency_target"].max().astype(float)
        true_events = np.asarray(sorted(truth_by_beat.index[truth_by_beat > 0.0].astype(int).tolist()), dtype=np.int32)
        consensus_events = np.asarray(
            sorted(truth_by_beat.index[truth_by_beat >= 0.5].astype(int).tolist()), dtype=np.int32
        )
        weights = {int(idx): float(value) for idx, value in truth_by_beat.loc[truth_by_beat > 0.0].items()}

        if mode == "hard_exact":
            matches = greedy_match_pairs(pred_events, true_events, tolerance=0)
            consensus_matches = greedy_match_pairs(pred_events, consensus_events, tolerance=0)
            match_credit = float(len(matches))
            consensus_credit = float(len(consensus_matches))
            matched_weight = float(sum(weights[int(true_events[true_idx])] for _, true_idx, _ in matches))
        elif mode == "soft_decay":
            matches = greedy_soft_match_pairs(pred_events, true_events, tolerance=0)
            consensus_matches = greedy_soft_match_pairs(pred_events, consensus_events, tolerance=0)
            match_credit = float(sum(credit for _, _, _, credit in matches))
            consensus_credit = float(sum(credit for _, _, _, credit in consensus_matches))
            matched_weight = float(
                sum(weights[int(true_events[true_idx])] * credit for _, true_idx, _, credit in matches)
            )
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        totals["pred_events"] += float(pred_events.size)
        totals["true_union_events"] += float(true_events.size)
        totals["true_consensus_events"] += float(consensus_events.size)
        totals["matches"] += match_credit
        totals["consensus_matches"] += consensus_credit
        totals["matched_weight"] += matched_weight
        totals["total_weight"] += float(sum(weights.values()))
    return finalize(totals)


def summarize_runs(output_dir: Path, folds: list[list[str]], min_distance: int, event_decoder: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for fold_idx, heldout in enumerate(folds, start=1):
        fold_dir = output_dir / TARGET / f"fold_{fold_idx:02d}"
        summary = json.loads((fold_dir / "summary.json").read_text(encoding="utf-8"))
        threshold = float(summary["union_metrics"]["threshold"])
        decoder = str(summary.get("event_decoder", event_decoder))
        pred_df = pd.read_csv(fold_dir / "val_predictions.csv.gz")
        for mode in ("hard_exact", "soft_decay"):
            metrics = evaluate_prediction_frame(
                pred_df,
                threshold=threshold,
                min_distance=int(min_distance),
                event_decoder=decoder,
                mode=mode,
            )
            rows.append(
                {
                    "fold": fold_idx,
                    "heldout_pieces": ",".join(heldout),
                    "target": TARGET,
                    "mode": mode,
                    "threshold": threshold,
                    **metrics,
                }
            )
    by_fold = pd.DataFrame(rows)
    stats_rows = []
    metrics = ["union_precision", "union_recall", "weighted_recall", "consensus_recall", "union_f1", "pred_events"]
    for mode, group in by_fold.groupby("mode", sort=False):
        row = {"mode": mode, "n_folds": int(len(group))}
        for metric in metrics:
            values = group[metric].astype(float)
            std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = std
            row[f"{metric}_ci95"] = float(1.96 * std / math.sqrt(len(values))) if len(values) > 1 else 0.0
        stats_rows.append(row)
    return by_fold, pd.DataFrame(stats_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run piece-level 5-fold CV for a single weighted all-six-layer target.")
    parser.add_argument("--config", default="configs/salience_grouped3_hi8_score_only_xml_curated.yaml")
    parser.add_argument("--output_dir", default="reports/paper_5fold_weighted_all6_seed42_cnn")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--model", default="cnn")
    parser.add_argument("--loss_type", default="bce_freq_weighted")
    parser.add_argument("--crf_aux_regression_weight", type=float, default=0.0)
    parser.add_argument("--crf_aux_regression_loss", choices=["smooth_l1", "mse"], default="smooth_l1")
    parser.add_argument("--selection_metric", default="weighted_recall")
    parser.add_argument("--precision_metric", default="union_precision")
    parser.add_argument("--min_precision", type=float, default=0.85)
    parser.add_argument("--min_train_frequency_target", type=float, default=0.0)
    parser.add_argument("--event_decoder", default="peak")
    parser.add_argument("--event_tolerance", type=int, default=0)
    parser.add_argument("--eval_min_distance", type=int, default=1)
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
        fold_dir = output_dir / TARGET / f"fold_{fold_idx:02d}"
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
            TARGET,
            "--selection_metric",
            str(args.selection_metric),
            "--precision_metric",
            str(args.precision_metric),
            "--min_precision",
            str(float(args.min_precision)),
            "--loss_type",
            str(args.loss_type),
            "--crf_aux_regression_weight",
            str(float(args.crf_aux_regression_weight)),
            "--crf_aux_regression_loss",
            str(args.crf_aux_regression_loss),
            "--min_train_frequency_target",
            str(float(args.min_train_frequency_target)),
            "--cumulative_component_weights_json",
            WEIGHTS_JSON,
            "--event_decoder",
            str(args.event_decoder),
            "--event_tolerance",
            str(int(args.event_tolerance)),
            "--eval_min_distance",
            str(int(args.eval_min_distance)),
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
        print(f"fold {fold_idx}/{len(folds)} {TARGET} heldout={','.join(heldout)}")
        if not args.dry_run:
            run_command(command, reuse_existing=bool(args.reuse_existing), summary_path=fold_dir / "summary.json")

    (output_dir / "commands.txt").write_text("\n".join(commands) + "\n", encoding="utf-8")
    if args.dry_run:
        print(output_dir)
        return

    by_fold, stats = summarize_runs(
        output_dir,
        folds,
        min_distance=int(args.eval_min_distance),
        event_decoder=str(args.event_decoder),
    )
    by_fold.to_csv(output_dir / "fivefold_weighted_all6_by_fold.csv", index=False)
    stats.to_csv(output_dir / "fivefold_weighted_all6_stats.csv", index=False)
    print(output_dir)
    print(stats.to_string(index=False))


if __name__ == "__main__":
    main()
