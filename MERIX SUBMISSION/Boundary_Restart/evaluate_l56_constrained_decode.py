#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from boundary_restart.config import load_config, resolve_path, threshold_grid
from boundary_restart.metrics import (
    extract_events,
    greedy_match_pairs,
    search_union_frequency_threshold,
)


def sequence_maps(frame: pd.DataFrame) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    ordered = frame.sort_values(["sample_id", "beat_idx"])
    scores = {}
    unions = {}
    freqs = {}
    for sample_id, group in ordered.groupby("sample_id", sort=False):
        scores[sample_id] = group["detector_score"].to_numpy(dtype=np.float32)
        unions[sample_id] = group["union_target"].to_numpy(dtype=np.float32)
        freqs[sample_id] = group["frequency_target"].to_numpy(dtype=np.float32)
    return scores, unions, freqs


def build_constrained_scores(
    l34_df: pd.DataFrame,
    l56_df: pd.DataFrame,
    l34_threshold: float,
    support_radius: int,
    min_support_score: float,
    support_mode: str,
) -> pd.DataFrame:
    rows = []
    for sample_id, l56_group in l56_df.sort_values(["sample_id", "beat_idx"]).groupby("sample_id", sort=False):
        l56_group = l56_group.reset_index(drop=True)
        l34_group = (
            l34_df[l34_df["sample_id"] == sample_id]
            .sort_values("beat_idx")
            .reset_index(drop=True)
        )
        l56_scores = l56_group["detector_score"].to_numpy(dtype=np.float32)
        l34_scores = l34_group["detector_score"].to_numpy(dtype=np.float32)

        l34_events = extract_events(
            l34_scores,
            threshold=float(l34_threshold),
            min_distance=6,
            prominence=0.0,
        )
        event_mask = np.zeros_like(l56_scores, dtype=np.float32)
        for idx in l34_events.tolist():
            lo = max(int(idx) - int(support_radius), 0)
            hi = min(int(idx) + int(support_radius) + 1, event_mask.shape[0])
            event_mask[lo:hi] = 1.0
        score_mask = (l34_scores >= float(min_support_score)).astype(np.float32)

        if support_mode == "window_only":
            support_mask = event_mask
        elif support_mode == "score_only":
            support_mask = score_mask
        elif support_mode == "window_or_score":
            support_mask = np.maximum(event_mask, score_mask)
        elif support_mode == "window_and_score":
            support_mask = event_mask * score_mask
        elif support_mode == "window_soft":
            scaled = np.clip(
                (l34_scores - float(min_support_score)) / max(1.0 - float(min_support_score), 1e-6),
                0.0,
                1.0,
            ).astype(np.float32)
            support_mask = event_mask * scaled
        else:
            raise ValueError(f"Unsupported support_mode: {support_mode}")

        constrained = l56_scores * support_mask

        group = l56_group.copy()
        group["detector_score"] = constrained.astype(np.float32)
        group["support_mask"] = support_mask.astype(np.float32)
        group["event_mask"] = event_mask.astype(np.float32)
        group["score_mask"] = score_mask.astype(np.float32)
        group["l34_detector_score"] = l34_scores.astype(np.float32)
        rows.append(group)
    return pd.concat(rows, axis=0, ignore_index=True)


def evaluate_piece(
    config_path: Path,
    l34_pred_path: Path,
    l56_pred_path: Path,
    out_dir: Path,
    support_radius_values: list[int],
    min_support_score_values: list[float],
    support_modes: list[str],
    min_precision: float | None,
) -> None:
    cfg = load_config(str(config_path))
    eval_cfg = cfg.get("evaluation", {})
    thresholds = threshold_grid(cfg)
    tolerance = int(eval_cfg.get("event_tolerance", 1))
    consensus_threshold = float(eval_cfg.get("consensus_threshold", 0.5))
    prominence = float(eval_cfg.get("prominence", 0.0))

    l34_df = pd.read_csv(l34_pred_path)
    l56_df = pd.read_csv(l56_pred_path)
    l34_summary = json.loads((l34_pred_path.parent / "summary.json").read_text())
    l34_threshold = float(l34_summary["union_metrics"]["threshold"])

    base_scores, base_unions, base_freqs = sequence_maps(l56_df)
    base_metrics = search_union_frequency_threshold(
        sequence_scores=base_scores,
        sequence_union_labels=base_unions,
        sequence_frequency_targets=base_freqs,
        thresholds=thresholds,
        tolerance=tolerance,
        min_distance=12,
        min_precision=float(min_precision if min_precision is not None else 0.85),
        consensus_threshold=consensus_threshold,
        prominence=prominence,
    )
    precision_floor = float(base_metrics.union_precision if min_precision is None else min_precision)

    records = []
    best = None
    best_df = None
    for mode in support_modes:
        for radius in support_radius_values:
            for min_support_score in min_support_score_values:
                if mode == "window_only" and min_support_score_values:
                    # Keep output schema uniform; value is ignored by this mode.
                    min_support_score = float(min_support_score)
                constrained_df = build_constrained_scores(
                    l34_df=l34_df,
                    l56_df=l56_df,
                    l34_threshold=l34_threshold,
                    support_radius=int(radius),
                    min_support_score=float(min_support_score),
                    support_mode=mode,
                )
                scores, unions, freqs = sequence_maps(constrained_df)
                metrics = search_union_frequency_threshold(
                    sequence_scores=scores,
                    sequence_union_labels=unions,
                    sequence_frequency_targets=freqs,
                    thresholds=thresholds,
                    tolerance=tolerance,
                    min_distance=12,
                    min_precision=precision_floor,
                    consensus_threshold=consensus_threshold,
                    prominence=prominence,
                )
                record = {
                    "support_mode": mode,
                    "support_radius": int(radius),
                    "min_support_score": float(min_support_score),
                    "threshold": float(metrics.threshold),
                    "union_precision": float(metrics.union_precision),
                    "union_recall": float(metrics.union_recall),
                    "weighted_recall": float(metrics.weighted_recall),
                    "consensus_recall": float(metrics.consensus_recall),
                    "pred_events": int(metrics.pred_events),
                    "true_union_events": int(metrics.true_union_events),
                }
                records.append(record)
                key = (
                    float(metrics.union_precision >= 0.85),
                    float(metrics.weighted_recall),
                    float(metrics.union_precision),
                    float(metrics.consensus_recall),
                )
                if best is None or key > best[0]:
                    best = (key, record)
                    best_df = constrained_df.copy()

    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).sort_values(
        ["weighted_recall", "union_precision", "consensus_recall"],
        ascending=[False, False, False],
    ).to_csv(out_dir / "grid_results.csv", index=False)
    if best_df is not None:
        best_df.to_csv(out_dir / "best_val_predictions.csv.gz", index=False, compression="gzip")
    summary = {
        "base": {
            "threshold": float(base_metrics.threshold),
            "union_precision": float(base_metrics.union_precision),
            "union_recall": float(base_metrics.union_recall),
            "weighted_recall": float(base_metrics.weighted_recall),
            "consensus_recall": float(base_metrics.consensus_recall),
            "pred_events": int(base_metrics.pred_events),
            "true_union_events": int(base_metrics.true_union_events),
        },
        "best": best[1] if best is not None else None,
        "l34_threshold": l34_threshold,
        "precision_floor": precision_floor,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--l34_pred", required=True)
    parser.add_argument("--l56_pred", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--support_radius", nargs="+", type=int, default=[2, 4, 6, 8, 10, 12])
    parser.add_argument("--min_support_score", nargs="+", type=float, default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    parser.add_argument(
        "--support_mode",
        nargs="+",
        default=["window_only", "window_and_score", "window_or_score", "window_soft"],
    )
    parser.add_argument(
        "--min_precision",
        type=float,
        default=None,
        help="If omitted, use the baseline L56 precision as the required floor.",
    )
    args = parser.parse_args()

    evaluate_piece(
        config_path=Path(args.config).resolve(),
        l34_pred_path=Path(args.l34_pred).resolve(),
        l56_pred_path=Path(args.l56_pred).resolve(),
        out_dir=Path(args.output_dir).resolve(),
        support_radius_values=args.support_radius,
        min_support_score_values=args.min_support_score,
        support_modes=args.support_mode,
        min_precision=args.min_precision,
    )


if __name__ == "__main__":
    main()
