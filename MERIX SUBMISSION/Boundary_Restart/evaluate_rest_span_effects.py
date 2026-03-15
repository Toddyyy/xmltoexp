#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from boundary_restart.config import load_config, threshold_grid
from boundary_restart.metrics import extract_events, greedy_match_pairs, search_union_frequency_threshold
from boundary_restart.rest_spans import (
    build_rest_span_arrays,
    greedy_match_pairs_rest_aware,
    snap_events_to_rest_spans,
)


def build_piece_rest_arrays(
    beat_table: pd.DataFrame,
    piece_id: str,
    min_len: int,
    source_col: str,
    source_threshold: float,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    piece = (
        beat_table[beat_table["piece_id"] == piece_id]
        .sort_values(["piece_id", "beat_idx", "sample_id"])
        .groupby(["piece_id", "beat_idx"], sort=False)
        .first()
        .reset_index()
        .sort_values("beat_idx")
        .reset_index(drop=True)
    )
    empty_mask = piece[source_col].to_numpy(dtype=np.float32) > float(source_threshold)
    span_id, span_start, span_end = build_rest_span_arrays(empty_mask, min_len=min_len)
    return piece, span_id, span_start, span_end


def evaluate_event_lists(
    pred_events: np.ndarray,
    true_union_events: np.ndarray,
    true_consensus_events: np.ndarray,
    freq_targets: np.ndarray,
    tolerance: int,
    span_id: np.ndarray | None = None,
) -> dict:
    if span_id is None:
        union_matches = greedy_match_pairs(pred_events, true_union_events, tolerance=tolerance)
        consensus_matches = greedy_match_pairs(pred_events, true_consensus_events, tolerance=tolerance)
    else:
        union_matches = greedy_match_pairs_rest_aware(pred_events, true_union_events, tolerance=tolerance, span_id=span_id)
        consensus_matches = greedy_match_pairs_rest_aware(
            pred_events,
            true_consensus_events,
            tolerance=tolerance,
            span_id=span_id,
        )

    total_pred = int(len(pred_events))
    total_true_union = int(len(true_union_events))
    total_true_consensus = int(len(true_consensus_events))
    total_match = int(len(union_matches))
    total_consensus_match = int(len(consensus_matches))
    matched_weight = float(sum(freq_targets[true_union_events[true_i]] for _, true_i, _ in union_matches))
    total_weight = float(freq_targets[true_union_events].sum())
    union_precision = float(total_match / total_pred) if total_pred > 0 else 0.0
    union_recall = float(total_match / total_true_union) if total_true_union > 0 else 0.0
    denom = union_precision + union_recall
    union_f1 = float(2.0 * union_precision * union_recall / denom) if denom > 0 else 0.0
    weighted_recall = float(matched_weight / total_weight) if total_weight > 0 else 0.0
    consensus_recall = float(total_consensus_match / total_true_consensus) if total_true_consensus > 0 else 0.0
    offsets = [offset for _, _, offset in union_matches]
    return {
        "union_precision": union_precision,
        "union_recall": union_recall,
        "union_f1": union_f1,
        "weighted_recall": weighted_recall,
        "consensus_recall": consensus_recall,
        "mean_offset": float(np.mean(np.abs(offsets))) if offsets else None,
        "matches": total_match,
        "pred_events": total_pred,
        "true_union_events": total_true_union,
        "true_consensus_events": total_true_consensus,
        "matched_weight": matched_weight,
        "total_weight": total_weight,
    }


def evaluate_rest_aware_threshold_search(
    scores: np.ndarray,
    union_targets: np.ndarray,
    freq_targets: np.ndarray,
    thresholds: np.ndarray,
    tolerance: int,
    min_distance: int,
    min_precision: float,
    consensus_threshold: float,
    prominence: float,
    span_id: np.ndarray,
) -> dict:
    true_union_events = np.flatnonzero(union_targets > 0.5).astype(np.int32)
    true_consensus_events = np.flatnonzero(freq_targets >= float(consensus_threshold)).astype(np.int32)
    best_meeting = None
    best_fallback = None
    for threshold in thresholds.tolist():
        pred_events = extract_events(scores, threshold=float(threshold), min_distance=min_distance, prominence=prominence)
        metrics = evaluate_event_lists(
            pred_events=pred_events,
            true_union_events=true_union_events,
            true_consensus_events=true_consensus_events,
            freq_targets=freq_targets,
            tolerance=tolerance,
            span_id=span_id,
        )
        metrics["threshold"] = float(threshold)
        current_weighted_key = (
            metrics["weighted_recall"],
            metrics["union_precision"],
            metrics["consensus_recall"],
            -(metrics["mean_offset"] or 1e9),
            -metrics["threshold"],
        )
        current_precision_key = (
            metrics["union_precision"],
            metrics["weighted_recall"],
            metrics["consensus_recall"],
            -(metrics["mean_offset"] or 1e9),
            -metrics["threshold"],
        )
        if metrics["union_precision"] >= float(min_precision):
            if best_meeting is None:
                best_meeting = metrics
            else:
                best_key = (
                    best_meeting["weighted_recall"],
                    best_meeting["union_precision"],
                    best_meeting["consensus_recall"],
                    -(best_meeting["mean_offset"] or 1e9),
                    -best_meeting["threshold"],
                )
                if current_weighted_key > best_key:
                    best_meeting = metrics
        if best_fallback is None:
            best_fallback = metrics
        else:
            best_key = (
                best_fallback["union_precision"],
                best_fallback["weighted_recall"],
                best_fallback["consensus_recall"],
                -(best_fallback["mean_offset"] or 1e9),
                -best_fallback["threshold"],
            )
            if current_precision_key > best_key:
                best_fallback = metrics
    return best_meeting or best_fallback


def build_snapped_event_frame(pred_df: pd.DataFrame, pred_events: np.ndarray, snapped_events: np.ndarray) -> pd.DataFrame:
    score_by_beat = pred_df.set_index("beat_idx")["detector_score"].to_dict()
    rows = []
    for rank, (orig, snapped) in enumerate(zip(pred_events.tolist(), snapped_events.tolist()), start=1):
        row = pred_df[pred_df["beat_idx"] == int(orig)].iloc[0]
        rows.append(
            {
                "event_rank": int(rank),
                "original_beat_idx": int(orig),
                "snapped_beat_idx": int(snapped),
                "detector_score": float(row["detector_score"]),
                "union_target_at_original": float(row["union_target"]),
                "frequency_target_at_original": float(row["frequency_target"]),
            }
        )
    return pd.DataFrame(rows)


def evaluate_piece(
    config_path: Path,
    beat_table_path: Path,
    pred_path: Path,
    summary_path: Path,
    output_dir: Path,
    rest_span_min_len: int,
    rest_span_source_col: str,
    rest_span_source_threshold: float,
    snap_mode: str,
) -> None:
    cfg = load_config(str(config_path))
    eval_cfg = cfg.get("evaluation", {})
    thresholds = threshold_grid(cfg)
    tolerance = int(eval_cfg.get("event_tolerance", 1))
    min_distance = int(eval_cfg.get("min_distance", 6))
    prominence = float(eval_cfg.get("prominence", 0.0))
    consensus_threshold = float(eval_cfg.get("consensus_threshold", 0.5))

    beat_table = pd.read_csv(beat_table_path)
    pred_df = pd.read_csv(pred_path).sort_values("beat_idx").reset_index(drop=True)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    piece_id = str(pred_df["piece_id"].iloc[0])
    baseline_threshold = float(summary["union_metrics"]["threshold"])
    min_precision = float(summary["precision_floor"])

    piece_frame, span_id, span_start, span_end = build_piece_rest_arrays(
        beat_table,
        piece_id=piece_id,
        min_len=rest_span_min_len,
        source_col=rest_span_source_col,
        source_threshold=rest_span_source_threshold,
    )
    merged = pred_df.merge(
        piece_frame[["beat_idx", "is_empty", "xml_rest_duration_norm"]],
        on="beat_idx",
        how="left",
        validate="one_to_one",
    )
    scores = merged["detector_score"].to_numpy(dtype=np.float32)
    union_targets = merged["union_target"].to_numpy(dtype=np.float32)
    freq_targets = merged["frequency_target"].to_numpy(dtype=np.float32)
    true_union_events = np.flatnonzero(union_targets > 0.5).astype(np.int32)
    true_consensus_events = np.flatnonzero(freq_targets >= float(consensus_threshold)).astype(np.int32)

    pred_events = extract_events(scores, threshold=baseline_threshold, min_distance=min_distance, prominence=prominence)

    rest_aware_fixed = evaluate_event_lists(
        pred_events=pred_events,
        true_union_events=true_union_events,
        true_consensus_events=true_consensus_events,
        freq_targets=freq_targets,
        tolerance=tolerance,
        span_id=span_id,
    )
    rest_aware_fixed["threshold"] = baseline_threshold

    rest_aware_best = evaluate_rest_aware_threshold_search(
        scores=scores,
        union_targets=union_targets,
        freq_targets=freq_targets,
        thresholds=thresholds,
        tolerance=tolerance,
        min_distance=min_distance,
        min_precision=min_precision,
        consensus_threshold=consensus_threshold,
        prominence=prominence,
        span_id=span_id,
    )

    snapped_events = snap_events_to_rest_spans(pred_events, span_start=span_start, span_end=span_end, mode=snap_mode)
    snapped_standard = evaluate_event_lists(
        pred_events=snapped_events,
        true_union_events=true_union_events,
        true_consensus_events=true_consensus_events,
        freq_targets=freq_targets,
        tolerance=tolerance,
        span_id=None,
    )
    snapped_standard["threshold"] = baseline_threshold
    snapped_rest_aware = evaluate_event_lists(
        pred_events=snapped_events,
        true_union_events=true_union_events,
        true_consensus_events=true_consensus_events,
        freq_targets=freq_targets,
        tolerance=tolerance,
        span_id=span_id,
    )
    snapped_rest_aware["threshold"] = baseline_threshold

    output_dir.mkdir(parents=True, exist_ok=True)
    merged.assign(rest_span_id=span_id, rest_span_start=span_start, rest_span_end=span_end).to_csv(
        output_dir / "val_predictions_with_rest_spans.csv.gz",
        index=False,
        compression="gzip",
    )
    build_snapped_event_frame(merged, pred_events, snapped_events).to_csv(output_dir / "snapped_events.csv", index=False)

    result = {
        "piece_id": piece_id,
        "rest_span_min_len": int(rest_span_min_len),
        "rest_span_source_col": str(rest_span_source_col),
        "rest_span_source_threshold": float(rest_span_source_threshold),
        "snap_mode": snap_mode,
        "rest_span_count": int(len([v for v in np.unique(span_id) if int(v) >= 0])),
        "baseline_summary_metrics": summary["union_metrics"],
        "rest_aware_fixed_threshold": rest_aware_fixed,
        "rest_aware_best_threshold": rest_aware_best,
        "snapped_standard_eval": snapped_standard,
        "snapped_rest_aware_eval": snapped_rest_aware,
    }
    (output_dir / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        f"{piece_id} | baseline_wr={summary['union_metrics']['weighted_recall']:.4f} | "
        f"rest_fixed_wr={rest_aware_fixed['weighted_recall']:.4f} | "
        f"rest_best_wr={rest_aware_best['weighted_recall']:.4f} | "
        f"snapped_rest_wr={snapped_rest_aware['weighted_recall']:.4f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--beat_table", required=True)
    parser.add_argument("--pred", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--rest_span_min_len", type=int, default=2)
    parser.add_argument("--rest_span_source_col", default="xml_rest_duration_norm")
    parser.add_argument("--rest_span_source_threshold", type=float, default=1e-8)
    parser.add_argument("--snap_mode", choices=["start", "center", "end"], default="center")
    args = parser.parse_args()

    evaluate_piece(
        config_path=Path(args.config).resolve(),
        beat_table_path=Path(args.beat_table).resolve(),
        pred_path=Path(args.pred).resolve(),
        summary_path=Path(args.summary).resolve(),
        output_dir=Path(args.output_dir).resolve(),
        rest_span_min_len=int(args.rest_span_min_len),
        rest_span_source_col=str(args.rest_span_source_col),
        rest_span_source_threshold=float(args.rest_span_source_threshold),
        snap_mode=str(args.snap_mode),
    )


if __name__ == "__main__":
    main()
