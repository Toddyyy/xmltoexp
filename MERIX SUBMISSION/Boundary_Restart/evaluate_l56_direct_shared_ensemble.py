#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from boundary_restart.config import load_config, threshold_grid
from boundary_restart.metrics import search_union_frequency_threshold


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


def merge_predictions(direct_df: pd.DataFrame, shared_df: pd.DataFrame) -> pd.DataFrame:
    shared = shared_df.rename(
        columns={
            "L56_union": "union_target",
            "L56_frequency": "frequency_target",
            "L56_score": "shared_score",
        }
    )[
        ["sample_id", "piece_id", "beat_idx", "union_target", "frequency_target", "shared_score"]
    ].copy()
    direct = direct_df.rename(columns={"detector_score": "direct_score"})[
        ["sample_id", "piece_id", "beat_idx", "union_target", "frequency_target", "direct_score"]
    ].copy()
    merged = direct.merge(
        shared,
        on=["sample_id", "piece_id", "beat_idx", "union_target", "frequency_target"],
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(direct) or len(merged) != len(shared):
        raise ValueError("Direct/shared predictions are not perfectly aligned")
    return merged


def apply_strategy(frame: pd.DataFrame, strategy: str) -> pd.DataFrame:
    merged = frame.copy()
    d = merged["direct_score"].to_numpy(dtype=np.float32)
    s = merged["shared_score"].to_numpy(dtype=np.float32)

    if strategy == "direct":
        score = d
    elif strategy == "shared":
        score = s
    elif strategy == "mean":
        score = 0.5 * (d + s)
    elif strategy == "max":
        score = np.maximum(d, s)
    elif strategy == "min":
        score = np.minimum(d, s)
    elif strategy == "product":
        score = d * s
    elif strategy.startswith("weighted_direct_"):
        w = float(strategy.split("_")[-1].replace("p", "."))
        score = w * d + (1.0 - w) * s
    else:
        raise ValueError(f"Unsupported strategy: {strategy}")

    merged["detector_score"] = score.astype(np.float32)
    return merged


def metrics_dict(metrics) -> dict:
    return {
        "threshold": float(metrics.threshold),
        "union_precision": float(metrics.union_precision),
        "union_recall": float(metrics.union_recall),
        "union_f1": float(metrics.union_f1),
        "weighted_recall": float(metrics.weighted_recall),
        "consensus_recall": float(metrics.consensus_recall),
        "mean_offset": float(metrics.mean_offset or 0.0) if metrics.mean_offset is not None else None,
        "matches": int(metrics.matches),
        "pred_events": int(metrics.pred_events),
        "true_union_events": int(metrics.true_union_events),
        "true_consensus_events": int(metrics.true_consensus_events),
    }


def evaluate_piece(
    config_path: Path,
    direct_pred_path: Path,
    shared_pred_path: Path,
    out_dir: Path,
    min_precision: float | None,
) -> None:
    cfg = load_config(str(config_path))
    eval_cfg = cfg.get("evaluation", {})
    thresholds = threshold_grid(cfg)
    tolerance = int(eval_cfg.get("event_tolerance", 1))
    min_distance = int(eval_cfg.get("min_distance", 6))
    prominence = float(eval_cfg.get("prominence", 0.0))
    consensus_threshold = float(eval_cfg.get("consensus_threshold", 0.5))

    direct_df = pd.read_csv(direct_pred_path)
    shared_df = pd.read_csv(shared_pred_path)
    merged = merge_predictions(direct_df, shared_df)

    strategies = [
        "direct",
        "shared",
        "mean",
        "max",
        "min",
        "product",
        "weighted_direct_0p25",
        "weighted_direct_0p40",
        "weighted_direct_0p50",
        "weighted_direct_0p60",
        "weighted_direct_0p75",
    ]

    records = []
    best_any = None
    best_floor = None
    floor_value = None
    baseline_metrics = {}

    for strategy in strategies:
        pred_df = apply_strategy(merged, strategy)
        seq_scores, seq_union, seq_freq = sequence_maps(pred_df)
        metrics = search_union_frequency_threshold(
            sequence_scores=seq_scores,
            sequence_union_labels=seq_union,
            sequence_frequency_targets=seq_freq,
            thresholds=thresholds,
            tolerance=tolerance,
            min_distance=min_distance,
            min_precision=float(min_precision if min_precision is not None else 0.85),
            consensus_threshold=consensus_threshold,
            prominence=prominence,
        )
        record = {"strategy": strategy, **metrics_dict(metrics)}
        records.append(record)
        if strategy in {"direct", "shared"}:
            baseline_metrics[strategy] = record

        key_any = (
            float(metrics.weighted_recall),
            float(metrics.union_precision),
            float(metrics.consensus_recall),
        )
        if best_any is None or key_any > best_any[0]:
            best_any = (key_any, record)

    if baseline_metrics:
        floor_value = max(m["union_precision"] for m in baseline_metrics.values()) if min_precision is None else float(min_precision)
        for record in records:
            if record["union_precision"] < floor_value:
                continue
            key_floor = (
                float(record["weighted_recall"]),
                float(record["union_precision"]),
                float(record["consensus_recall"]),
            )
            if best_floor is None or key_floor > best_floor[0]:
                best_floor = (key_floor, record)

    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).sort_values(
        ["weighted_recall", "union_precision", "consensus_recall"],
        ascending=[False, False, False],
    ).to_csv(out_dir / "leaderboard.csv", index=False)
    summary = {
        "direct_pred_path": str(direct_pred_path),
        "shared_pred_path": str(shared_pred_path),
        "baseline": baseline_metrics,
        "precision_floor": floor_value,
        "best_any": best_any[1] if best_any else None,
        "best_meets_floor": best_floor[1] if best_floor else None,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--direct_pred", required=True)
    parser.add_argument("--shared_pred", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--min_precision",
        type=float,
        default=None,
        help="If omitted, require precision >= max(direct, shared) baseline precision.",
    )
    args = parser.parse_args()
    evaluate_piece(
        config_path=Path(args.config).resolve(),
        direct_pred_path=Path(args.direct_pred).resolve(),
        shared_pred_path=Path(args.shared_pred).resolve(),
        out_dir=Path(args.output_dir).resolve(),
        min_precision=args.min_precision,
    )


if __name__ == "__main__":
    main()
