from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks


ROOT = Path(__file__).resolve().parents[2]
PLOT_DIR = ROOT / "MERIX SUBMISSION" / "MIREX_Model" / "mazurka_project_l1_l6_tempo_human_phrase_plots"
OUT_DIR = PLOT_DIR / "human_phrase_target_offset_analysis"
WINDOW = 6


def local_maxima(scores: np.ndarray) -> np.ndarray:
    peaks, _ = find_peaks(scores)
    candidates = set(peaks.tolist())
    if scores.size:
        candidates.add(int(np.nanargmax(scores)))
        candidates.add(0)
        candidates.add(len(scores) - 1)
    return np.array(sorted(candidates), dtype=int)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for target_csv in sorted(PLOT_DIR.glob("*_beat_target_values.csv")):
        piece_id = target_csv.name.replace("_beat_target_values.csv", "")
        df = pd.read_csv(target_csv)
        scores = df["target_weighted_l2plus"].to_numpy(dtype=float)
        peaks = local_maxima(scores)
        human = df.loc[df["human_phrase_label"].fillna("").astype(str).str.len() > 0].copy()
        for _, row in human.iterrows():
            h = int(row["beat"]) - 1
            in_window = peaks[np.abs(peaks - h) <= WINDOW]
            if in_window.size:
                best = int(in_window[np.argmax(scores[in_window])])
                nearest = int(in_window[np.argmin(np.abs(in_window - h))])
            else:
                best = int(peaks[np.argmax(scores[peaks])])
                nearest = int(peaks[np.argmin(np.abs(peaks - h))])
            rows.append(
                {
                    "piece_id": piece_id,
                    "human_beat": h + 1,
                    "human_label": row["human_phrase_label"],
                    "human_target": float(scores[h]),
                    "best_peak_beat_within_window": best + 1,
                    "best_peak_target_within_window": float(scores[best]),
                    "offset_human_minus_best_peak": int(h - best),
                    "nearest_peak_beat": nearest + 1,
                    "nearest_peak_target": float(scores[nearest]),
                    "offset_human_minus_nearest_peak": int(h - nearest),
                    "window": WINDOW,
                }
            )

    detail = pd.DataFrame(rows)
    detail.to_csv(OUT_DIR / "human_phrase_target_peak_offsets.csv", index=False)

    summary = (
        detail.groupby("offset_human_minus_best_peak")
        .size()
        .rename("count")
        .reset_index()
        .sort_values("offset_human_minus_best_peak")
    )
    summary["fraction"] = summary["count"] / max(int(summary["count"].sum()), 1)
    summary.to_csv(OUT_DIR / "offset_distribution_best_peak.csv", index=False)

    by_piece = (
        detail.assign(is_plus1=detail["offset_human_minus_best_peak"] == 1)
        .groupby("piece_id")
        .agg(
            human_count=("human_beat", "size"),
            plus1_count=("is_plus1", "sum"),
            plus1_fraction=("is_plus1", "mean"),
            median_offset=("offset_human_minus_best_peak", "median"),
            mean_offset=("offset_human_minus_best_peak", "mean"),
        )
        .reset_index()
    )
    by_piece.to_csv(OUT_DIR / "offset_summary_by_piece.csv", index=False)

    print("Offset distribution: human_beat - local_best_target_peak_beat")
    print(summary.to_string(index=False))
    print("\nBy piece:")
    print(by_piece.to_string(index=False))
    print(f"\nWrote {OUT_DIR}")


if __name__ == "__main__":
    main()
