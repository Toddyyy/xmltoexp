from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
LABEL_DIR = ROOT / "MERIX SUBMISSION" / "MIREX_Model" / "beat_data_mazurka_performer_levels"
OUT_DIR = ROOT / "MERIX SUBMISSION" / "MIREX_Model" / "mazurkabl_boundary_patterns"
LEVELS = [1, 2, 3, 4, 5, 6]
PERIODS = [3, 4, 6, 8, 12, 16, 24, 32]


def parse_label_path(path: Path) -> tuple[str, str, int]:
    m = re.match(r"(M\d+-\d+)_(.+)_L(\d+)\.npz$", path.name)
    if not m:
        raise ValueError(path.name)
    return m.group(1), m.group(2), int(m.group(3))


def circular_distance_to_grid(indices: np.ndarray, period: int) -> np.ndarray:
    mod = np.mod(indices, period)
    return np.minimum(mod, period - mod)


def nearest_distance(events: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    if len(events) == 0 or len(anchors) == 0:
        return np.array([], dtype=float)
    anchors = np.sort(anchors)
    pos = np.searchsorted(anchors, events)
    out = np.full(len(events), np.inf, dtype=float)
    valid = pos < len(anchors)
    out[valid] = np.minimum(out[valid], np.abs(events[valid] - anchors[pos[valid]]))
    valid = pos > 0
    out[valid] = np.minimum(out[valid], np.abs(events[valid] - anchors[pos[valid] - 1]))
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    records = []
    for path in sorted(LABEL_DIR.glob("*_L*.npz")):
        piece, performer, level = parse_label_path(path)
        arr = np.load(path, allow_pickle=True)["boundary_probs"].astype(np.float32)
        records.append((piece, performer, level, arr))

    by_piece_perf: dict[tuple[str, str], dict[int, np.ndarray]] = {}
    by_piece_level: dict[tuple[str, int], list[np.ndarray]] = {}
    for piece, performer, level, arr in records:
        by_piece_perf.setdefault((piece, performer), {})[level] = arr
        by_piece_level.setdefault((piece, level), []).append(arr)

    spacing_rows = []
    period_rows = []
    nesting_rows = []
    top_consensus_rows = []

    for (piece, performer), levels in sorted(by_piece_perf.items()):
        for level in LEVELS:
            if level not in levels:
                continue
            events = np.flatnonzero(levels[level] > 0.5)
            gaps = np.diff(events)
            if len(gaps):
                spacing_rows.append(
                    {
                        "piece": piece,
                        "performer": performer,
                        "level": level,
                        "event_count": len(events),
                        "gap_median": float(np.median(gaps)),
                        "gap_mean": float(np.mean(gaps)),
                        "gap_q25": float(np.quantile(gaps, 0.25)),
                        "gap_q75": float(np.quantile(gaps, 0.75)),
                        "gap_mode": int(pd.Series(gaps).mode().iloc[0]),
                        "gap_multiple_of_4_rate": float(np.mean(gaps % 4 == 0)),
                        "gap_multiple_of_8_rate": float(np.mean(gaps % 8 == 0)),
                        "gap_multiple_of_12_rate": float(np.mean(gaps % 12 == 0)),
                        "gap_multiple_of_16_rate": float(np.mean(gaps % 16 == 0)),
                    }
                )
            for period in PERIODS:
                if len(events):
                    dist = circular_distance_to_grid(events, period)
                    period_rows.append(
                        {
                            "piece": piece,
                            "performer": performer,
                            "level": level,
                            "period": period,
                            "event_count": len(events),
                            "on_grid_rate": float(np.mean(dist == 0)),
                            "within1_grid_rate": float(np.mean(dist <= 1)),
                            "mean_grid_distance": float(np.mean(dist)),
                        }
                    )
        for lo, hi in zip(LEVELS[:-1], LEVELS[1:]):
            if lo not in levels or hi not in levels:
                continue
            lo_events = np.flatnonzero(levels[lo] > 0.5)
            hi_events = np.flatnonzero(levels[hi] > 0.5)
            dist = nearest_distance(hi_events, lo_events)
            nesting_rows.append(
                {
                    "piece": piece,
                    "performer": performer,
                    "lower_level": lo,
                    "higher_level": hi,
                    "higher_event_count": len(hi_events),
                    "exact_nested_rate": float(np.mean(dist == 0)) if len(dist) else np.nan,
                    "within1_nested_rate": float(np.mean(dist <= 1)) if len(dist) else np.nan,
                    "mean_distance_to_lower": float(np.mean(dist)) if len(dist) else np.nan,
                }
            )

    for (piece, level), arrays in sorted(by_piece_level.items()):
        masks = np.stack(arrays, axis=0)
        consensus = masks.mean(axis=0)
        nonzero = np.flatnonzero(consensus > 0)
        if len(nonzero) == 0:
            continue
        order = nonzero[np.argsort(-consensus[nonzero])]
        top_k = min(12, len(order))
        for rank, beat in enumerate(order[:top_k], start=1):
            prev_gap = int(beat - order[rank - 2]) if rank > 1 else np.nan
            top_consensus_rows.append(
                {
                    "piece": piece,
                    "level": level,
                    "rank": rank,
                    "beat": int(beat),
                    "consensus": float(consensus[beat]),
                    "gap_from_previous_ranked_beat": prev_gap,
                }
            )

    spacing_df = pd.DataFrame(spacing_rows)
    period_df = pd.DataFrame(period_rows)
    nesting_df = pd.DataFrame(nesting_rows)
    top_df = pd.DataFrame(top_consensus_rows)
    spacing_df.to_csv(OUT_DIR / "spacing_by_performer.csv", index=False)
    period_df.to_csv(OUT_DIR / "period_grid_alignment_by_performer.csv", index=False)
    nesting_df.to_csv(OUT_DIR / "level_nesting_by_performer.csv", index=False)
    top_df.to_csv(OUT_DIR / "top_consensus_beats_by_piece_level.csv", index=False)

    spacing_summary = (
        spacing_df.groupby("level")
        .agg(
            mean_event_count=("event_count", "mean"),
            median_gap=("gap_median", "median"),
            mean_gap=("gap_mean", "mean"),
            median_gap_mode=("gap_mode", "median"),
            multiple4=("gap_multiple_of_4_rate", "mean"),
            multiple8=("gap_multiple_of_8_rate", "mean"),
            multiple12=("gap_multiple_of_12_rate", "mean"),
            multiple16=("gap_multiple_of_16_rate", "mean"),
        )
        .reset_index()
    )
    spacing_summary.to_csv(OUT_DIR / "spacing_summary_by_level.csv", index=False)

    period_summary = (
        period_df.groupby(["level", "period"])
        .agg(
            on_grid_rate=("on_grid_rate", "mean"),
            within1_grid_rate=("within1_grid_rate", "mean"),
            mean_grid_distance=("mean_grid_distance", "mean"),
        )
        .reset_index()
    )
    period_summary.to_csv(OUT_DIR / "period_grid_alignment_summary.csv", index=False)

    nesting_summary = (
        nesting_df.groupby(["lower_level", "higher_level"])
        .agg(
            exact_nested_rate=("exact_nested_rate", "mean"),
            within1_nested_rate=("within1_nested_rate", "mean"),
            mean_distance_to_lower=("mean_distance_to_lower", "mean"),
        )
        .reset_index()
    )
    nesting_summary.to_csv(OUT_DIR / "level_nesting_summary.csv", index=False)

    best_period = (
        period_summary.sort_values(["level", "within1_grid_rate", "on_grid_rate"], ascending=[True, False, False])
        .groupby("level")
        .head(3)
    )
    best_period.to_csv(OUT_DIR / "best_periods_by_level.csv", index=False)

    print("Spacing summary:")
    print(spacing_summary.round(4).to_string(index=False))
    print("\nBest periods by level:")
    print(best_period.round(4).to_string(index=False))
    print("\nLevel nesting summary:")
    print(nesting_summary.round(4).to_string(index=False))
    print(f"\nWrote {OUT_DIR}")


if __name__ == "__main__":
    main()
