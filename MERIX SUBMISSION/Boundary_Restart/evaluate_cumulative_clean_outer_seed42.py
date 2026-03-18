#!/usr/bin/env python3

from __future__ import annotations

import csv
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

PRED_GROUPS = {
    "L56+": ["level56"],
    "L4+": ["level4", "level56"],
    "L3+": ["level3", "level4", "level56"],
    "L2+": ["level2", "level3", "level4", "level56"],
    "L1+": ["level1", "level2", "level3", "level4", "level56"],
}

TRUTH_LEVELS = {
    "L56+": (5, 6),
    "L4+": (4, 5, 6),
    "L3+": (3, 4, 5, 6),
    "L2+": (2, 3, 4, 5, 6),
    "L1+": (1, 2, 3, 4, 5, 6),
}


def load_predicted_events(root: Path, piece_id: str, level_slug: str) -> pd.DataFrame:
    path = root / f"M06_outer_{level_slug}_seed{SEED}" / "predicted_events.csv.gz"
    if not path.exists():
        return pd.DataFrame(columns=["beat_idx", "detector_score"])
    frame = pd.read_csv(path)
    frame = frame[frame["piece_id"] == piece_id].copy()
    if frame.empty:
        return pd.DataFrame(columns=["beat_idx", "detector_score"])
    keep = [col for col in ["beat_idx", "detector_score"] if col in frame.columns]
    return frame[keep].copy()


def collapse_cross_level_duplicates(events: pd.DataFrame, tolerance: int) -> np.ndarray:
    if events.empty:
        return np.zeros(0, dtype=np.int32)
    frame = events.sort_values(["beat_idx", "detector_score"], ascending=[True, False]).reset_index(drop=True)
    clusters: list[list[tuple[int, float]]] = []
    for row in frame.itertuples(index=False):
        beat_idx = int(row.beat_idx)
        score = float(getattr(row, "detector_score", 0.0))
        if not clusters or beat_idx - clusters[-1][-1][0] > int(tolerance):
            clusters.append([(beat_idx, score)])
        else:
            clusters[-1].append((beat_idx, score))
    kept = []
    for cluster in clusters:
        best = sorted(cluster, key=lambda item: (-item[1], item[0]))[0]
        kept.append(best[0])
    return np.asarray(sorted(set(kept)), dtype=np.int32)


def build_piece_union_truth(
    df: pd.DataFrame,
    raw_levels: tuple[int, ...],
    peak_cfg: PeakConfig,
    beat_unit_fallback: float,
) -> pd.DataFrame:
    frame = df.copy()
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


def collapse_true_events(
    freq_targets: np.ndarray,
    *,
    collapse_tolerance: int,
    consensus_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    positive_beats = np.flatnonzero(freq_targets > 0.0).astype(np.int32)
    if positive_beats.size == 0:
        empty_i = np.zeros(0, dtype=np.int32)
        empty_f = np.zeros(0, dtype=np.float32)
        return empty_i, empty_f, empty_i

    clusters: list[list[int]] = []
    for beat_idx in positive_beats.tolist():
        if not clusters or beat_idx - clusters[-1][-1] > int(collapse_tolerance):
            clusters.append([beat_idx])
        else:
            clusters[-1].append(beat_idx)

    union_events: list[int] = []
    union_weights: list[float] = []
    consensus_events: list[int] = []
    for cluster in clusters:
        freqs = np.asarray([float(freq_targets[beat]) for beat in cluster], dtype=np.float32)
        center = float(cluster[0] + cluster[-1]) / 2.0
        best_idx = int(
            min(
                range(len(cluster)),
                key=lambda idx: (-float(freqs[idx]), abs(cluster[idx] - center), cluster[idx]),
            )
        )
        rep_beat = int(cluster[best_idx])
        rep_weight = float(freqs.max())
        union_events.append(rep_beat)
        union_weights.append(rep_weight)
        if float(freqs.max()) >= float(consensus_threshold):
            consensus_events.append(rep_beat)

    return (
        np.asarray(union_events, dtype=np.int32),
        np.asarray(union_weights, dtype=np.float32),
        np.asarray(consensus_events, dtype=np.int32),
    )


def evaluate_pred_events_against_truth(
    pred_events: np.ndarray,
    true_union_events: np.ndarray,
    true_union_weights: np.ndarray,
    true_consensus_events: np.ndarray,
    *,
    tolerance: int,
) -> dict:
    union_matches = greedy_match_pairs(pred_events, true_union_events, tolerance=tolerance)
    consensus_matches = greedy_match_pairs(pred_events, true_consensus_events, tolerance=tolerance)

    total_pred = int(pred_events.size)
    total_true_union = int(true_union_events.size)
    total_true_consensus = int(true_consensus_events.size)
    total_match = len(union_matches)
    total_consensus_match = len(consensus_matches)
    matched_weight = float(sum(true_union_weights[true_idx] for _, true_idx, _ in union_matches))
    total_weight = float(true_union_weights.sum())
    offsets = [offset for _, _, offset in union_matches]

    union_precision = float(total_match / total_pred) if total_pred > 0 else 0.0
    frequency_weighted_precision = float(matched_weight / total_pred) if total_pred > 0 else 0.0
    consensus_precision = float(total_consensus_match / total_pred) if total_pred > 0 else 0.0
    union_recall = float(total_match / total_true_union) if total_true_union > 0 else 0.0
    union_f1_denom = union_precision + union_recall
    union_f1 = float(2.0 * union_precision * union_recall / union_f1_denom) if union_f1_denom > 0 else 0.0
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


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate cumulative clean-outer metrics with configurable cross-level collapse tolerance.")
    parser.add_argument("--collapse_tolerance", type=int, default=TOLERANCE)
    parser.add_argument("--collapse_truth", action="store_true")
    parser.add_argument("--report_dir", default="reports/clean_outer_test_cumulative_seed42")
    args = parser.parse_args()

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

    pred_root = project_root / "outputs/local_runs/clean_outer_test"
    report_dir = project_root / args.report_dir
    report_dir.mkdir(parents=True, exist_ok=True)

    truth_by_group = {
        label: build_piece_union_truth(df, raw_levels, peak_cfg, beat_unit_fallback)
        for label, raw_levels in TRUTH_LEVELS.items()
    }

    rows: list[dict] = []
    for piece_id in OUTER_PIECES:
        for label, level_slugs in PRED_GROUPS.items():
            pred_frames = [load_predicted_events(pred_root, piece_id, slug) for slug in level_slugs]
            pred_frame = pd.concat(pred_frames, ignore_index=True) if pred_frames else pd.DataFrame(columns=["beat_idx", "detector_score"])
            pred_events = collapse_cross_level_duplicates(pred_frame, tolerance=int(args.collapse_tolerance))

            truth_piece = truth_by_group[label]
            truth_piece = truth_piece[truth_piece["piece_id"] == piece_id].sort_values("beat_idx").reset_index(drop=True)
            freq_targets = truth_piece["frequency_target"].to_numpy(dtype=np.float32)
            if args.collapse_truth:
                true_union_events, true_union_weights, true_consensus_events = collapse_true_events(
                    freq_targets,
                    collapse_tolerance=int(args.collapse_tolerance),
                    consensus_threshold=CONSENSUS_THRESHOLD,
                )
            else:
                true_union_events = np.flatnonzero(freq_targets > 0.0).astype(np.int32)
                true_union_weights = freq_targets[true_union_events].astype(np.float32)
                true_consensus_events = np.flatnonzero(freq_targets >= float(CONSENSUS_THRESHOLD)).astype(np.int32)
            metrics = evaluate_pred_events_against_truth(
                pred_events,
                true_union_events,
                true_union_weights,
                true_consensus_events,
                tolerance=TOLERANCE,
            )
            rows.append(
                {
                    "piece_id": piece_id,
                    "cumulative_level": label,
                    "source_levels": "+".join(level_slugs),
                    "collapse_tolerance": int(args.collapse_tolerance),
                    "collapse_truth": int(args.collapse_truth),
                    **metrics,
                }
            )

    detail_path = report_dir / "piece_level_cumulative_metrics.csv"
    with detail_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    avg_rows = []
    for label in PRED_GROUPS:
        subset = [row for row in rows if row["cumulative_level"] == label]
        avg_rows.append(
                {
                    "cumulative_level": label,
                    "source_levels": "+".join(PRED_GROUPS[label]),
                    "collapse_tolerance": int(args.collapse_tolerance),
                    "collapse_truth": int(args.collapse_truth),
                    "mean_union_precision": sum(float(r["union_precision"]) for r in subset) / len(subset),
                "mean_frequency_weighted_precision": sum(float(r["frequency_weighted_precision"]) for r in subset)
                / len(subset),
                "mean_consensus_precision": sum(float(r["consensus_precision"]) for r in subset) / len(subset),
                "mean_union_recall": sum(float(r["union_recall"]) for r in subset) / len(subset),
                "mean_weighted_recall": sum(float(r["weighted_recall"]) for r in subset) / len(subset),
                "mean_consensus_recall": sum(float(r["consensus_recall"]) for r in subset) / len(subset),
            }
        )

    avg_path = report_dir / "cumulative_mean_metrics.csv"
    with avg_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(avg_rows[0].keys()))
        writer.writeheader()
        writer.writerows(avg_rows)

    print(detail_path)
    print(avg_path)


if __name__ == "__main__":
    main()
