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
from boundary_restart.features import PeakConfig, boundary_probs_to_binary, load_boundary_npz, replace_level_suffix
from boundary_restart.metrics import greedy_match_pairs
from boundary_restart.table_io import load_table


OUTER_PIECES = ["M06-1", "M06-2", "M06-3"]
SEED = 42
TOLERANCE = 1
CONSENSUS_THRESHOLD = 0.5

CUMULATIVE_TARGETS = {
    "L4+": "level4plus_boundary",
    "L3+": "level3plus_boundary",
    "L2+": "level2plus_boundary",
    "L1+": "level1plus_boundary",
}

DIRECT_REFERENCE_CSV = "reports/clean_outer_test_cumulative_seed42/piece_level_cumulative_metrics.csv"

TRUTH_LEVELS = {
    "L4+": (4, 5, 6),
    "L3+": (3, 4, 5, 6),
    "L2+": (2, 3, 4, 5, 6),
    "L1+": (1, 2, 3, 4, 5, 6),
}


def build_piece_union_truth(
    df: pd.DataFrame,
    raw_levels: tuple[int, ...],
    peak_cfg: PeakConfig,
    beat_unit_fallback: float,
) -> pd.DataFrame:
    frame = df.copy()
    frame["protocol_split"] = "eval"
    detector_binary = np.zeros(len(frame), dtype=np.float32)
    beat_idx = frame["beat_idx"].to_numpy(dtype=np.int32)
    for source_path, positions in frame.groupby("source_path", sort=False).indices.items():
        pos = np.asarray(positions, dtype=np.int64)
        boundary_binary = None
        for raw_level in raw_levels:
            level_path = replace_level_suffix(Path(str(source_path)), level=raw_level)
            loaded = load_boundary_npz(level_path, beat_unit_fallback=beat_unit_fallback)
            current_binary = boundary_probs_to_binary(
                np.asarray(loaded["boundary_probs"], dtype=np.float32),
                peak_cfg,
            ).astype(np.float32)
            boundary_binary = current_binary if boundary_binary is None else np.maximum(boundary_binary, current_binary)
        sample_beat_idx = beat_idx[pos]
        detector_binary[pos] = boundary_binary[sample_beat_idx].astype(np.float32)
    frame["detector_binary"] = detector_binary.astype(np.float32)
    piece = (
        frame.sort_values(["piece_id", "beat_idx", "sample_id"])
        .groupby(["piece_id", "beat_idx"], sort=False)
        .agg({"detector_binary": "mean"})
        .rename(columns={"detector_binary": "frequency_target"})
        .reset_index()
    )
    piece["union_target"] = (piece["frequency_target"] > 0.0).astype(np.float32)
    return piece


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

    peak_cfg = PeakConfig(
        distance=int(cfg.get("data", {}).get("peak_distance", 6)),
        height=float(cfg.get("data", {}).get("peak_height", 0.15)),
        prominence=float(cfg.get("data", {}).get("peak_prominence", 0.05)),
    )
    beat_unit_fallback = float(cfg.get("data", {}).get("beat_unit_fallback", 1.0))

    truth_by_level = {
        label: build_piece_union_truth(df, raw_levels, peak_cfg, beat_unit_fallback)
        for label, raw_levels in TRUTH_LEVELS.items()
    }

    train_script = project_root / "train_piece_union_protocol.py"
    run_root = project_root / "outputs/local_runs/cumulative_training_seed42"
    report_dir = project_root / "reports/cumulative_training_seed42"
    run_root.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for label, target in CUMULATIVE_TARGETS.items():
        out_dir = run_root / f"M06_outer_{target}_seed{SEED}"
        run_cumulative_training(train_script, str(cfg["_config_path"]), target, out_dir)
        pred_frame = pd.read_csv(out_dir / "predicted_events.csv.gz")
        for piece_id in OUTER_PIECES:
            piece_truth = truth_by_level[label]
            piece_truth = piece_truth[piece_truth["piece_id"] == piece_id].sort_values("beat_idx").reset_index(drop=True)
            freq_targets = piece_truth["frequency_target"].to_numpy(dtype=np.float32)
            true_union_events = np.flatnonzero(freq_targets > 0.0).astype(np.int32)
            piece_pred = pred_frame[pred_frame["piece_id"] == piece_id].copy()
            pred_events = np.asarray(sorted(set(piece_pred["beat_idx"].astype(int).tolist())), dtype=np.int32)
            metrics = evaluate_pred_events_against_truth(
                pred_events,
                true_union_events,
                freq_targets,
                tolerance=TOLERANCE,
                consensus_threshold=CONSENSUS_THRESHOLD,
            )
            summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
            rows.append(
                {
                    "piece_id": piece_id,
                    "cumulative_level": label,
                    "training_mode": "cumulative_trained",
                    "detector_target": target,
                    "best_epoch": summary.get("best_epoch"),
                    "threshold": summary.get("union_metrics", {}).get("threshold"),
                    **metrics,
                }
            )

    direct_ref = pd.read_csv(project_root / DIRECT_REFERENCE_CSV)
    for row in direct_ref.itertuples(index=False):
        if row.cumulative_level not in CUMULATIVE_TARGETS:
            continue
        rows.append(
            {
                "piece_id": row.piece_id,
                "cumulative_level": row.cumulative_level,
                "training_mode": "direct_trained_then_cumulative_eval",
                "detector_target": row.source_levels,
                "best_epoch": None,
                "threshold": None,
                "union_precision": row.union_precision,
                "frequency_weighted_precision": row.frequency_weighted_precision,
                "consensus_precision": row.consensus_precision,
                "union_recall": row.union_recall,
                "weighted_recall": row.weighted_recall,
                "consensus_recall": row.consensus_recall,
                "matches": row.matches,
                "pred_events": row.pred_events,
                "true_union_events": row.true_union_events,
                "true_consensus_events": row.true_consensus_events,
                "matched_weight": row.matched_weight,
                "total_weight": row.total_weight,
            }
        )

    detail_path = report_dir / "piece_level_compare.csv"
    with detail_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    avg_rows = []
    for label in CUMULATIVE_TARGETS:
        for training_mode in ("direct_trained_then_cumulative_eval", "cumulative_trained"):
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
