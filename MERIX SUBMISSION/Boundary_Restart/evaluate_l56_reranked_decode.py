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


def local_mass_score(scores: np.ndarray, radius: int, alpha: float, beta: float) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float32)
    if radius <= 0:
        local_mean = scores
        local_max = scores
    else:
        kernel = np.ones(2 * radius + 1, dtype=np.float32)
        local_sum = np.convolve(scores, kernel, mode="same")
        local_mean = local_sum / float(kernel.size)
        padded = np.pad(scores, radius, mode="edge")
        local_max = np.empty_like(scores)
        for idx in range(scores.size):
            local_max[idx] = float(np.max(padded[idx : idx + 2 * radius + 1]))
    reranked = scores + float(alpha) * local_mean + float(beta) * local_max
    return reranked.astype(np.float32)


def apply_rerank(frame: pd.DataFrame, radius: int, alpha: float, beta: float) -> pd.DataFrame:
    rows = []
    for sample_id, group in frame.sort_values(["sample_id", "beat_idx"]).groupby("sample_id", sort=False):
        group = group.copy().reset_index(drop=True)
        reranked = local_mass_score(
            group["detector_score"].to_numpy(dtype=np.float32),
            radius=int(radius),
            alpha=float(alpha),
            beta=float(beta),
        )
        group["base_detector_score"] = group["detector_score"].astype(np.float32)
        group["detector_score"] = reranked
        group["rerank_radius"] = int(radius)
        group["rerank_alpha"] = float(alpha)
        group["rerank_beta"] = float(beta)
        rows.append(group)
    return pd.concat(rows, axis=0, ignore_index=True)


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
    pred_path: Path,
    out_dir: Path,
    min_precision: float | None,
    radii: list[int],
    alphas: list[float],
    betas: list[float],
) -> None:
    cfg = load_config(str(config_path))
    eval_cfg = cfg.get("evaluation", {})
    thresholds = threshold_grid(cfg)
    tolerance = int(eval_cfg.get("event_tolerance", 1))
    min_distance = int(eval_cfg.get("min_distance", 6))
    prominence = float(eval_cfg.get("prominence", 0.0))
    consensus_threshold = float(eval_cfg.get("consensus_threshold", 0.5))

    base_df = pd.read_csv(pred_path)
    base_scores, base_unions, base_freqs = sequence_maps(base_df)
    base_metrics = search_union_frequency_threshold(
        sequence_scores=base_scores,
        sequence_union_labels=base_unions,
        sequence_frequency_targets=base_freqs,
        thresholds=thresholds,
        tolerance=tolerance,
        min_distance=min_distance,
        min_precision=float(min_precision if min_precision is not None else 0.85),
        consensus_threshold=consensus_threshold,
        prominence=prominence,
    )
    precision_floor = float(base_metrics.union_precision if min_precision is None else min_precision)

    records = []
    best = None
    best_df = None
    for radius in radii:
        for alpha in alphas:
            for beta in betas:
                reranked_df = apply_rerank(base_df, radius=radius, alpha=alpha, beta=beta)
                scores, unions, freqs = sequence_maps(reranked_df)
                metrics = search_union_frequency_threshold(
                    sequence_scores=scores,
                    sequence_union_labels=unions,
                    sequence_frequency_targets=freqs,
                    thresholds=thresholds,
                    tolerance=tolerance,
                    min_distance=min_distance,
                    min_precision=precision_floor,
                    consensus_threshold=consensus_threshold,
                    prominence=prominence,
                )
                record = {
                    "radius": int(radius),
                    "alpha": float(alpha),
                    "beta": float(beta),
                    **metrics_dict(metrics),
                }
                records.append(record)
                if float(metrics.union_precision) >= precision_floor:
                    key = (
                        float(metrics.weighted_recall),
                        float(metrics.union_precision),
                        float(metrics.consensus_recall),
                    )
                    if best is None or key > best[0]:
                        best = (key, record)
                        best_df = reranked_df.copy()

    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).sort_values(
        ["weighted_recall", "union_precision", "consensus_recall"],
        ascending=[False, False, False],
    ).to_csv(out_dir / "leaderboard.csv", index=False)
    if best_df is not None:
        best_df.to_csv(out_dir / "best_val_predictions.csv.gz", index=False, compression="gzip")
    summary = {
        "pred_path": str(pred_path),
        "base": metrics_dict(base_metrics),
        "precision_floor": precision_floor,
        "best": best[1] if best is not None else None,
        "meets_floor_found": bool(best is not None),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--pred", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--min_precision", type=float, default=None)
    parser.add_argument("--radius", nargs="+", type=int, default=[1, 2, 3, 4])
    parser.add_argument("--alpha", nargs="+", type=float, default=[0.25, 0.5, 1.0, 2.0])
    parser.add_argument("--beta", nargs="+", type=float, default=[0.0, 0.25, 0.5, 1.0])
    args = parser.parse_args()

    evaluate_piece(
        config_path=Path(args.config).resolve(),
        pred_path=Path(args.pred).resolve(),
        out_dir=Path(args.output_dir).resolve(),
        min_precision=args.min_precision,
        radii=args.radius,
        alphas=args.alpha,
        betas=args.beta,
    )


if __name__ == "__main__":
    main()
