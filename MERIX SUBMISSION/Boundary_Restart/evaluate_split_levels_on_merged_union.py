#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

from boundary_restart.config import load_config, resolve_path
from boundary_restart.metrics import greedy_match_pairs
from boundary_restart.table_io import load_table
from train_piece_union_protocol import PeakConfig, build_piece_union_frame


DEFAULT_PIECES = ["M06-1", "M06-2", "M06-3", "M17-1", "M30-1"]


def evaluate_pred_events_against_union(
    pred_events: np.ndarray,
    true_union_events: np.ndarray,
    freq_targets: np.ndarray,
    *,
    tolerance: int,
    consensus_threshold: float,
) -> dict:
    true_consensus_events = np.flatnonzero(freq_targets >= float(consensus_threshold)).astype(np.int32)
    union_matches = greedy_match_pairs(pred_events, true_union_events, tolerance=tolerance)
    consensus_matches = greedy_match_pairs(pred_events, true_consensus_events, tolerance=tolerance)

    total_pred = int(pred_events.size)
    total_true_union = int(true_union_events.size)
    total_true_consensus = int(true_consensus_events.size)
    total_match = len(union_matches)
    total_consensus_match = len(consensus_matches)
    matched_weight = float(sum(freq_targets[true_union_events[true_idx]] for _, true_idx, _ in union_matches))
    total_weight = float(freq_targets[true_union_events].sum())
    offsets = [offset for _, _, offset in union_matches]

    union_precision = float(total_match / total_pred) if total_pred > 0 else 0.0
    frequency_weighted_precision = float(matched_weight / total_pred) if total_pred > 0 else 0.0
    consensus_precision = float(total_consensus_match / total_pred) if total_pred > 0 else 0.0
    union_recall = float(total_match / total_true_union) if total_true_union > 0 else 0.0
    denom = union_precision + union_recall
    union_f1 = float(2.0 * union_precision * union_recall / denom) if denom > 0 else 0.0
    weighted_recall = float(matched_weight / total_weight) if total_weight > 0 else 0.0
    consensus_recall = float(total_consensus_match / total_true_consensus) if total_true_consensus > 0 else 0.0
    mean_offset = float(np.mean(np.abs(offsets))) if offsets else None

    return {
        "union_precision": union_precision,
        "frequency_weighted_precision": frequency_weighted_precision,
        "consensus_precision": consensus_precision,
        "union_recall": union_recall,
        "union_f1": union_f1,
        "weighted_recall": weighted_recall,
        "consensus_recall": consensus_recall,
        "mean_offset": mean_offset,
        "matches": int(total_match),
        "pred_events": int(total_pred),
        "true_union_events": int(total_true_union),
        "true_consensus_events": int(total_true_consensus),
        "matched_weight": matched_weight,
        "total_weight": total_weight,
    }


def load_pred_events(path: Path, piece_id: str) -> np.ndarray:
    if not path.exists():
        return np.zeros(0, dtype=np.int32)
    frame = pd.read_csv(path)
    if "piece_id" in frame.columns:
        frame = frame[frame["piece_id"] == piece_id].copy()
    if frame.empty:
        return np.zeros(0, dtype=np.int32)
    return np.asarray(sorted(set(frame["beat_idx"].astype(int).tolist())), dtype=np.int32)


def load_summary_meta(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate split level predictions against merged level56 union labels.")
    parser.add_argument(
        "--config",
        default="MERIX SUBMISSION/Boundary_Restart/configs/salience_grouped3_hi8_score_only_xml_curated.yaml",
    )
    parser.add_argument("--pieces", nargs="*", default=DEFAULT_PIECES)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--consensus_threshold", type=float, default=0.5)
    parser.add_argument("--tolerance", type=int, default=1)
    parser.add_argument(
        "--split_run_root",
        default="MERIX SUBMISSION/Boundary_Restart/outputs/local_runs/strategy_compare_l5l6_u70",
    )
    parser.add_argument(
        "--merged_run_root",
        default="MERIX SUBMISSION/Boundary_Restart/outputs/local_runs/strategy_compare_alllevels",
    )
    parser.add_argument(
        "--report_dir",
        default="MERIX SUBMISSION/Boundary_Restart/reports/split_levels_on_merged_union",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    table_path = resolve_path(cfg, cfg["data"]["beat_table_path"])
    data_cfg = cfg.get("data", {})
    peak_cfg = PeakConfig(
        distance=int(data_cfg.get("peak_distance", 6)),
        height=float(data_cfg.get("peak_height", 0.15)),
        prominence=float(data_cfg.get("peak_prominence", 0.05)),
    )
    beat_unit_fallback = float(data_cfg.get("beat_unit_fallback", 1.0))

    df = load_table(table_path)
    if "protocol_split" not in df.columns:
        df = df.copy()
        df["protocol_split"] = "eval"
    piece_df = build_piece_union_frame(
        df,
        feature_cols=[],
        target_mode="level56_boundary",
        peak_cfg=peak_cfg,
        beat_unit_fallback=beat_unit_fallback,
    )

    split_run_root = Path(args.split_run_root)
    merged_run_root = Path(args.merged_run_root)
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    target_specs = [
        ("level5_boundary", split_run_root),
        ("level6_boundary", split_run_root),
        ("level56_boundary", merged_run_root),
    ]

    rows: list[dict] = []
    for piece_id in args.pieces:
        piece_truth = piece_df[piece_df["piece_id"] == piece_id].sort_values("beat_idx").reset_index(drop=True)
        freq_targets = piece_truth["frequency_target"].to_numpy(dtype=np.float32)
        true_union_events = np.flatnonzero(freq_targets > 0.0).astype(np.int32)
        for detector_target, run_root in target_specs:
            for variant in ("baseline", "consensus_guarded"):
                run_dir = run_root / f"{piece_id}_{detector_target}_{variant}_seed{int(args.seed)}"
                pred_events = load_pred_events(run_dir / "predicted_events.csv.gz", piece_id)
                metrics = evaluate_pred_events_against_union(
                    pred_events,
                    true_union_events,
                    freq_targets,
                    tolerance=int(args.tolerance),
                    consensus_threshold=float(args.consensus_threshold),
                )
                meta = load_summary_meta(run_dir / "summary.json")
                row = {
                    "piece_id": piece_id,
                    "detector_target": detector_target,
                    "variant": variant,
                    "seed": int(args.seed),
                    "precision_metric": meta.get("precision_metric"),
                    "precision_floors": json.dumps(meta.get("precision_floors", {}), sort_keys=True),
                    "best_epoch": meta.get("best_epoch"),
                    "threshold": meta.get("union_metrics", {}).get("threshold"),
                }
                row.update(metrics)
                rows.append(row)

    detail_path = report_dir / "piece_level_split_eval_on_merged_union.csv"
    with detail_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    avg_rows: list[dict] = []
    for detector_target, _ in target_specs:
        for variant in ("baseline", "consensus_guarded"):
            subset = [row for row in rows if row["detector_target"] == detector_target and row["variant"] == variant]
            avg_rows.append(
                {
                    "detector_target": detector_target,
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

    avg_path = report_dir / "level_mean_split_eval_on_merged_union.csv"
    with avg_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(avg_rows[0].keys()))
        writer.writeheader()
        writer.writerows(avg_rows)

    print(detail_path)
    print(avg_path)


if __name__ == "__main__":
    main()
