#!/usr/bin/env python3

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from boundary_restart.config import load_config, resolve_path
from boundary_restart.cumulative_targets import (
    COMPONENT_RAW_LEVELS,
    build_piece_frequency_for_raw_levels,
    build_topdown_cumulative_frequency,
    cumulative_components_for_target,
    merge_event_frames_topdown,
)
from boundary_restart.features import PeakConfig
from boundary_restart.metrics import greedy_match_pairs
from boundary_restart.table_io import load_table


OUTER_PIECES = ["M06-1", "M06-2", "M06-3"]
SEED = 42
MATCH_TOLERANCE = 1
MERGE_TOLERANCE = 2
CONSENSUS_THRESHOLD = 0.5

CUMULATIVE_TARGETS = {
    "L4+": "level4plus_boundary",
    "L3+": "level3plus_boundary",
    "L2+": "level2plus_boundary",
    "L1+": "level1plus_boundary",
}


def evaluate_pred_events_against_truth(
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

    union_precision = float(total_match / total_pred) if total_pred > 0 else 0.0
    frequency_weighted_precision = float(matched_weight / total_pred) if total_pred > 0 else 0.0
    consensus_precision = float(total_consensus_match / total_pred) if total_pred > 0 else 0.0
    union_recall = float(total_match / total_true_union) if total_true_union > 0 else 0.0
    weighted_recall = float(matched_weight / total_weight) if total_weight > 0 else 0.0
    consensus_recall = float(total_consensus_match / total_true_consensus) if total_true_consensus > 0 else 0.0
    return {
        "union_precision": union_precision,
        "frequency_weighted_precision": frequency_weighted_precision,
        "consensus_precision": consensus_precision,
        "union_recall": union_recall,
        "weighted_recall": weighted_recall,
        "consensus_recall": consensus_recall,
        "matches": int(total_match),
        "pred_events": int(total_pred),
        "true_union_events": int(total_true_union),
        "true_consensus_events": int(total_true_consensus),
        "matched_weight": matched_weight,
        "total_weight": total_weight,
    }


def build_truth_by_level(df: pd.DataFrame, peak_cfg: PeakConfig, beat_unit_fallback: float) -> dict[str, pd.DataFrame]:
    base_piece = (
        df.sort_values(["piece_id", "beat_idx", "sample_id"])
        .groupby(["piece_id", "beat_idx"], sort=False)
        .agg({"protocol_split": "first"})
        .reset_index()
    )
    component_map = {
        component_name: build_piece_frequency_for_raw_levels(
            df,
            raw_levels=raw_levels,
            peak_cfg=peak_cfg,
            beat_unit_fallback=beat_unit_fallback,
        )
        for component_name, raw_levels in COMPONENT_RAW_LEVELS.items()
    }
    truth = {}
    for label, target in CUMULATIVE_TARGETS.items():
        components = cumulative_components_for_target(target)
        if components is None:
            raise ValueError(f"Missing cumulative components for {target}")
        truth[label] = build_topdown_cumulative_frequency(
            base_piece[["piece_id", "beat_idx"]],
            component_map=component_map,
            component_order=components,
            tolerance=MERGE_TOLERANCE,
        )
    return truth


def load_direct_component_predictions(pred_root: Path, piece_id: str) -> dict[str, pd.DataFrame]:
    mapping = {
        "level1": pred_root / f"M06_outer_level1_seed{SEED}" / "predicted_events.csv.gz",
        "level2": pred_root / f"M06_outer_level2_seed{SEED}" / "predicted_events.csv.gz",
        "level3": pred_root / f"M06_outer_level3_seed{SEED}" / "predicted_events.csv.gz",
        "level4": pred_root / f"M06_outer_level4_seed{SEED}" / "predicted_events.csv.gz",
        "level56": pred_root / f"M06_outer_level56_seed{SEED}" / "predicted_events.csv.gz",
    }
    out = {}
    for component_name, path in mapping.items():
        frame = pd.read_csv(path)
        frame = frame[frame["piece_id"] == piece_id][["beat_idx", "detector_score"]].copy()
        out[component_name] = frame
    return out


def run_cumulative_training(train_script: Path, config: str, target: str, output_dir: Path) -> None:
    summary_path = output_dir / "summary.json"
    if summary_path.exists() and (output_dir / "predicted_events.csv.gz").exists():
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
        "--skip_stage_grading",
        "--cumulative_merge_tolerance",
        str(MERGE_TOLERANCE),
        "--output_dir",
        str(output_dir),
    ]
    subprocess.run(cmd, check=True)


def main() -> None:
    project_root = Path(__file__).resolve().parent
    cfg = load_config(project_root / "configs/salience_grouped3_hi8_score_only_xml_curated.yaml")
    table_path = resolve_path(cfg, cfg["data"]["beat_table_path"])
    df = load_table(table_path)
    df = df[df["piece_id"].isin(OUTER_PIECES)].copy()
    df["protocol_split"] = "eval"

    peak_cfg = PeakConfig(
        distance=int(cfg.get("data", {}).get("peak_distance", 6)),
        height=float(cfg.get("data", {}).get("peak_height", 0.15)),
        prominence=float(cfg.get("data", {}).get("peak_prominence", 0.05)),
    )
    beat_unit_fallback = float(cfg.get("data", {}).get("beat_unit_fallback", 1.0))
    truth_by_level = build_truth_by_level(df, peak_cfg, beat_unit_fallback)

    train_script = project_root / "train_piece_union_protocol.py"
    run_root = project_root / "outputs/local_runs/cumulative_training_merge2_seed42"
    report_dir = project_root / "reports/cumulative_training_merge2_seed42"
    direct_pred_root = project_root / "outputs/local_runs/clean_outer_test"
    run_root.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []

    for label, target in CUMULATIVE_TARGETS.items():
        out_dir = run_root / f"M06_outer_{target}_merge2_seed{SEED}"
        run_cumulative_training(train_script, str(cfg["_config_path"]), target, out_dir)
        pred_frame = pd.read_csv(out_dir / "predicted_events.csv.gz")
        summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
        for piece_id in OUTER_PIECES:
            truth_piece = truth_by_level[label]
            truth_piece = truth_piece[truth_piece["piece_id"] == piece_id].sort_values("beat_idx").reset_index(drop=True)
            freq_targets = truth_piece["frequency_target"].to_numpy(dtype=np.float32)
            true_union_events = np.flatnonzero(freq_targets > 0.0).astype(np.int32)
            piece_pred = pred_frame[pred_frame["piece_id"] == piece_id].copy()
            pred_events = np.asarray(sorted(set(piece_pred["beat_idx"].astype(int).tolist())), dtype=np.int32)
            metrics = evaluate_pred_events_against_truth(
                pred_events,
                true_union_events,
                freq_targets,
                tolerance=MATCH_TOLERANCE,
                consensus_threshold=CONSENSUS_THRESHOLD,
            )
            rows.append(
                {
                    "piece_id": piece_id,
                    "cumulative_level": label,
                    "training_mode": "cumulative_trained_merge2",
                    "detector_target": target,
                    "best_epoch": summary.get("best_epoch"),
                    "threshold": summary.get("union_metrics", {}).get("threshold"),
                    **metrics,
                }
            )

    for label, target in CUMULATIVE_TARGETS.items():
        components = cumulative_components_for_target(target)
        if components is None:
            continue
        for piece_id in OUTER_PIECES:
            pred_map = load_direct_component_predictions(direct_pred_root, piece_id)
            merged_pred = merge_event_frames_topdown(pred_map, components, tolerance=MERGE_TOLERANCE)
            pred_events = np.asarray(sorted(set(merged_pred["beat_idx"].astype(int).tolist())), dtype=np.int32)
            truth_piece = truth_by_level[label]
            truth_piece = truth_piece[truth_piece["piece_id"] == piece_id].sort_values("beat_idx").reset_index(drop=True)
            freq_targets = truth_piece["frequency_target"].to_numpy(dtype=np.float32)
            true_union_events = np.flatnonzero(freq_targets > 0.0).astype(np.int32)
            metrics = evaluate_pred_events_against_truth(
                pred_events,
                true_union_events,
                freq_targets,
                tolerance=MATCH_TOLERANCE,
                consensus_threshold=CONSENSUS_THRESHOLD,
            )
            rows.append(
                {
                    "piece_id": piece_id,
                    "cumulative_level": label,
                    "training_mode": "direct_trained_then_topdown_merge2_eval",
                    "detector_target": "+".join(components),
                    "best_epoch": None,
                    "threshold": None,
                    **metrics,
                }
            )

    detail_path = report_dir / "piece_level_compare.csv"
    with detail_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    avg_rows = []
    for label in CUMULATIVE_TARGETS:
        for training_mode in ("direct_trained_then_topdown_merge2_eval", "cumulative_trained_merge2"):
            subset = [row for row in rows if row["cumulative_level"] == label and row["training_mode"] == training_mode]
            avg_rows.append(
                {
                    "cumulative_level": label,
                    "training_mode": training_mode,
                    "mean_union_precision": sum(float(r["union_precision"]) for r in subset) / len(subset),
                    "mean_frequency_weighted_precision": sum(float(r["frequency_weighted_precision"]) for r in subset)
                    / len(subset),
                    "mean_consensus_precision": sum(float(r["consensus_precision"]) for r in subset) / len(subset),
                    "mean_union_recall": sum(float(r["union_recall"]) for r in subset) / len(subset),
                    "mean_weighted_recall": sum(float(r["weighted_recall"]) for r in subset) / len(subset),
                    "mean_consensus_recall": sum(float(r["consensus_recall"]) for r in subset) / len(subset),
                }
            )

    avg_path = report_dir / "level_mean_compare.csv"
    with avg_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(avg_rows[0].keys()))
        writer.writeheader()
        writer.writerows(avg_rows)

    print(detail_path)
    print(avg_path)


if __name__ == "__main__":
    main()
