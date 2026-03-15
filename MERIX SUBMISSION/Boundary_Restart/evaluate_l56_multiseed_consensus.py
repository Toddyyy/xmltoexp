#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from boundary_restart.config import load_config, threshold_grid
from boundary_restart.metrics import (
    extract_events,
    greedy_match_pairs,
    search_union_frequency_threshold,
)


KEY_COLS = ["sample_id", "piece_id", "beat_idx", "union_target", "frequency_target", "performer_count"]


def sequence_maps(frame: pd.DataFrame, score_col: str) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    ordered = frame.sort_values(["sample_id", "beat_idx"]).copy()
    sequence_scores: dict[str, np.ndarray] = {}
    sequence_union: dict[str, np.ndarray] = {}
    sequence_freq: dict[str, np.ndarray] = {}
    for sample_id, group in ordered.groupby("sample_id", sort=False):
        sequence_scores[str(sample_id)] = group[score_col].to_numpy(dtype=np.float32)
        sequence_union[str(sample_id)] = group["union_target"].to_numpy(dtype=np.float32)
        sequence_freq[str(sample_id)] = group["frequency_target"].to_numpy(dtype=np.float32)
    return sequence_scores, sequence_union, sequence_freq


def load_seed_predictions(pred_paths: list[Path]) -> tuple[pd.DataFrame, list[tuple[str, pd.DataFrame]]]:
    merged = None
    seed_frames: list[tuple[str, pd.DataFrame]] = []
    for pred_path in pred_paths:
        seed_name = pred_path.parent.name
        frame = pd.read_csv(pred_path).sort_values(["sample_id", "beat_idx"]).reset_index(drop=True)
        score_col = f"detector_score_{seed_name}"
        renamed = frame[KEY_COLS + ["detector_score"]].rename(columns={"detector_score": score_col})
        seed_frames.append((seed_name, frame))
        if merged is None:
            merged = renamed.copy()
            continue
        merged = merged.merge(
            renamed,
            on=KEY_COLS,
            how="inner",
            validate="one_to_one",
        )
    if merged is None:
        raise ValueError("No prediction files were provided")
    expected_rows = len(seed_frames[0][1])
    if len(merged) != expected_rows:
        raise ValueError("Seed predictions are not perfectly aligned")
    return merged, seed_frames


def decode_seed_events(
    frame: pd.DataFrame,
    thresholds: np.ndarray,
    tolerance: int,
    min_distance: int,
    min_precision: float,
    consensus_threshold: float,
    prominence: float,
) -> tuple[dict[str, np.ndarray], dict[str, float], dict[str, dict]]:
    seed_events: dict[str, np.ndarray] = {}
    seed_thresholds: dict[str, float] = {}
    seed_metrics: dict[str, dict] = {}
    score_cols = [col for col in frame.columns if col.startswith("detector_score_seed")]
    for score_col in score_cols:
        seed_name = score_col.replace("detector_score_", "")
        seed_df = frame[KEY_COLS + [score_col]].rename(columns={score_col: "detector_score"})
        seq_scores, seq_union, seq_freq = sequence_maps(seed_df, score_col="detector_score")
        metrics = search_union_frequency_threshold(
            sequence_scores=seq_scores,
            sequence_union_labels=seq_union,
            sequence_frequency_targets=seq_freq,
            thresholds=thresholds,
            tolerance=tolerance,
            min_distance=min_distance,
            min_precision=min_precision,
            consensus_threshold=consensus_threshold,
            prominence=prominence,
        )
        sample_id, scores = next(iter(seq_scores.items()))
        events = extract_events(
            scores=scores,
            threshold=float(metrics.threshold),
            min_distance=min_distance,
            prominence=prominence,
        )
        seed_events[seed_name] = events
        seed_thresholds[seed_name] = float(metrics.threshold)
        seed_metrics[seed_name] = {
            "threshold": float(metrics.threshold),
            "union_precision": float(metrics.union_precision),
            "union_recall": float(metrics.union_recall),
            "union_f1": float(metrics.union_f1),
            "weighted_recall": float(metrics.weighted_recall),
            "consensus_recall": float(metrics.consensus_recall),
            "pred_events": int(metrics.pred_events),
            "true_union_events": int(metrics.true_union_events),
        }
    return seed_events, seed_thresholds, seed_metrics


def build_consensus_event_frame(
    pred_df: pd.DataFrame,
    threshold: float,
    min_distance: int,
    prominence: float,
    tolerance: int,
    support_tolerance: int,
    seed_events: dict[str, np.ndarray],
) -> pd.DataFrame:
    rows = []
    ordered = pred_df.sort_values(["sample_id", "beat_idx"]).copy()
    seed_score_cols = [col for col in ordered.columns if col.startswith("detector_score_seed")]

    for sample_id, group in ordered.groupby("sample_id", sort=False):
        group = group.reset_index(drop=True)
        scores = group["mean_detector_score"].to_numpy(dtype=np.float32)
        pred_events = extract_events(
            scores=scores,
            threshold=float(threshold),
            min_distance=int(min_distance),
            prominence=float(prominence),
        )
        true_union_events = np.flatnonzero(group["union_target"].to_numpy(dtype=np.float32) > 0.5).astype(np.int32)
        match_pairs = greedy_match_pairs(pred_events, true_union_events, tolerance=int(tolerance))
        match_map = {pred_idx: (true_idx, offset) for pred_idx, true_idx, offset in match_pairs}

        for event_rank, pred_idx in enumerate(pred_events.tolist(), start=1):
            row = group.iloc[int(pred_idx)]
            beat_idx = int(row["beat_idx"])
            support_details = []
            support_beats = []
            for seed_name, seed_pred_events in seed_events.items():
                diffs = np.abs(seed_pred_events - beat_idx)
                if diffs.size == 0:
                    continue
                nearest_idx = int(np.argmin(diffs))
                if int(diffs[nearest_idx]) <= int(support_tolerance):
                    support_details.append(seed_name)
                    support_beats.append(int(seed_pred_events[nearest_idx]))
            true_match = match_map.get(int(event_rank - 1))
            event_row = {
                "sample_id": str(sample_id),
                "piece_id": str(row["piece_id"]),
                "event_rank": int(event_rank),
                "beat_idx": beat_idx,
                "mean_detector_score": float(row["mean_detector_score"]),
                "std_detector_score": float(row["std_detector_score"]),
                "min_detector_score": float(row["min_detector_score"]),
                "max_detector_score": float(row["max_detector_score"]),
                "threshold": float(threshold),
                "union_target_at_beat": float(row["union_target"]),
                "frequency_target_at_beat": float(row["frequency_target"]),
                "performer_count": int(row["performer_count"]),
                "matched_union": bool(true_match is not None),
                "match_offset": int(true_match[1]) if true_match is not None else None,
                "matched_true_beat_idx": int(true_union_events[true_match[0]]) if true_match is not None else None,
                "seed_support_count": int(len(support_details)),
                "seed_support_ratio": float(len(support_details) / max(len(seed_events), 1)),
                "supporting_seeds": ",".join(support_details),
                "supporting_seed_beats": ",".join(str(v) for v in support_beats),
            }
            for score_col in seed_score_cols:
                event_row[score_col] = float(row[score_col])
            rows.append(event_row)
    return pd.DataFrame(rows)


def evaluate_piece(
    config_path: Path,
    pred_paths: list[Path],
    out_dir: Path,
    min_precision: float,
    support_tolerance: int,
) -> None:
    cfg = load_config(str(config_path))
    eval_cfg = cfg.get("evaluation", {})
    thresholds = threshold_grid(cfg)
    tolerance = int(eval_cfg.get("event_tolerance", 1))
    min_distance = int(eval_cfg.get("min_distance", 6))
    prominence = float(eval_cfg.get("prominence", 0.0))
    consensus_threshold = float(eval_cfg.get("consensus_threshold", 0.5))

    merged_df, seed_frames = load_seed_predictions(pred_paths)
    score_cols = [col for col in merged_df.columns if col.startswith("detector_score_seed")]
    merged_df["mean_detector_score"] = merged_df[score_cols].mean(axis=1)
    merged_df["std_detector_score"] = merged_df[score_cols].std(axis=1, ddof=0)
    merged_df["min_detector_score"] = merged_df[score_cols].min(axis=1)
    merged_df["max_detector_score"] = merged_df[score_cols].max(axis=1)

    seq_scores, seq_union, seq_freq = sequence_maps(merged_df, score_col="mean_detector_score")
    ensemble_metrics = search_union_frequency_threshold(
        sequence_scores=seq_scores,
        sequence_union_labels=seq_union,
        sequence_frequency_targets=seq_freq,
        thresholds=thresholds,
        tolerance=tolerance,
        min_distance=min_distance,
        min_precision=min_precision,
        consensus_threshold=consensus_threshold,
        prominence=prominence,
    )

    seed_events, seed_thresholds, seed_metrics = decode_seed_events(
        frame=merged_df,
        thresholds=thresholds,
        tolerance=tolerance,
        min_distance=min_distance,
        min_precision=min_precision,
        consensus_threshold=consensus_threshold,
        prominence=prominence,
    )

    event_df = build_consensus_event_frame(
        pred_df=merged_df,
        threshold=float(ensemble_metrics.threshold),
        min_distance=min_distance,
        prominence=prominence,
        tolerance=tolerance,
        support_tolerance=support_tolerance,
        seed_events=seed_events,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    merged_df.to_csv(out_dir / "mean_val_predictions.csv.gz", index=False, compression="gzip")
    event_df.to_csv(out_dir / "final_events.csv", index=False)
    pd.DataFrame(
        [
            {"seed": seed_name, **seed_metrics[seed_name]}
            for seed_name, _ in sorted(seed_frames, key=lambda item: item[0])
        ]
    ).to_csv(out_dir / "seed_metrics.csv", index=False)

    summary = {
        "piece_id": str(merged_df["piece_id"].iloc[0]),
        "seed_count": int(len(seed_frames)),
        "seed_names": [seed_name for seed_name, _ in sorted(seed_frames, key=lambda item: item[0])],
        "prediction_paths": [str(path) for path in pred_paths],
        "min_precision": float(min_precision),
        "support_tolerance": int(support_tolerance),
        "consensus_threshold": float(consensus_threshold),
        "ensemble_metrics": {
            "threshold": float(ensemble_metrics.threshold),
            "union_precision": float(ensemble_metrics.union_precision),
            "union_recall": float(ensemble_metrics.union_recall),
            "union_f1": float(ensemble_metrics.union_f1),
            "weighted_recall": float(ensemble_metrics.weighted_recall),
            "consensus_recall": float(ensemble_metrics.consensus_recall),
            "mean_offset": float(ensemble_metrics.mean_offset or 0.0) if ensemble_metrics.mean_offset is not None else None,
            "matches": int(ensemble_metrics.matches),
            "pred_events": int(ensemble_metrics.pred_events),
            "true_union_events": int(ensemble_metrics.true_union_events),
            "true_consensus_events": int(ensemble_metrics.true_consensus_events),
            "matched_weight": float(ensemble_metrics.matched_weight),
            "total_weight": float(ensemble_metrics.total_weight),
        },
        "seed_thresholds": seed_thresholds,
        "seed_metrics": seed_metrics,
        "support_summary": {
            "mean_seed_support_count": float(event_df["seed_support_count"].mean()) if not event_df.empty else 0.0,
            "min_seed_support_count": int(event_df["seed_support_count"].min()) if not event_df.empty else 0,
            "max_seed_support_count": int(event_df["seed_support_count"].max()) if not event_df.empty else 0,
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(
        f"{summary['piece_id']} | seeds={summary['seed_count']} | "
        f"union_precision={ensemble_metrics.union_precision:.4f} | "
        f"weighted_recall={ensemble_metrics.weighted_recall:.4f} | "
        f"consensus_recall={ensemble_metrics.consensus_recall:.4f} | "
        f"pred_events={ensemble_metrics.pred_events}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--pred_glob", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--min_precision", type=float, default=0.85)
    parser.add_argument("--support_tolerance", type=int, default=1)
    args = parser.parse_args()

    pred_paths = sorted(Path().glob(args.pred_glob))
    if not pred_paths:
        raise FileNotFoundError(f"No prediction files matched: {args.pred_glob}")

    evaluate_piece(
        config_path=Path(args.config).resolve(),
        pred_paths=[path.resolve() for path in pred_paths],
        out_dir=Path(args.output_dir).resolve(),
        min_precision=float(args.min_precision),
        support_tolerance=int(args.support_tolerance),
    )


if __name__ == "__main__":
    main()
