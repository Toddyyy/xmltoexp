from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import music21 as m21
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from compare_vienna_atepp_k331_tempo_curves import (  # noqa: E402
    ATEPP_DIR,
    OUT_DIR,
    load_atepp_curves,
    load_vienna_curves,
    stack_mean,
)


VIENNA_SCORE = ROOT / "datasets" / "Vienna4x4" / "vienna4x22_rematched-master" / "musicxml" / "Mozart_K331_1st-mov.musicxml"
ATEPP_SCORE = ATEPP_DIR / "musicxml_cleaned.musicxml"


def score_onset_tokens(path: Path, grid_per_quarter: int = 2) -> list[tuple[int, ...]]:
    score = m21.converter.parse(str(path))
    length = int(np.ceil(float(score.highestTime) * grid_per_quarter))
    by_grid: dict[int, set[int]] = {}
    for part in score.parts:
        for note in part.recurse().notes:
            grid_idx = int(round(float(note.getOffsetInHierarchy(score)) * grid_per_quarter))
            pitches = note.pitches if note.isChord else [note.pitch]
            by_grid.setdefault(grid_idx, set()).update(int(p.midi) for p in pitches)
    return [tuple(sorted(by_grid.get(i, set()))) for i in range(length)]


def pitch_jaccard_window(query: list[tuple[int, ...]], target: list[tuple[int, ...]]) -> pd.DataFrame:
    query_len = len(query)
    rows = []
    for start in range(0, len(target) - query_len + 1):
        inter = 0
        union = 0
        exact = 0
        compared = 0
        for i, query_token in enumerate(query):
            query_set = set(query_token)
            target_set = set(target[start + i])
            if query_set or target_set:
                compared += 1
                exact += int(query_set == target_set)
                inter += len(query_set & target_set)
                union += len(query_set | target_set)
        rows.append(
            {
                "atepp_start_beat": start,
                "atepp_end_beat_exclusive": start + query_len,
                "pitch_jaccard": inter / union if union else 0.0,
                "exact_onset_rate": exact / compared if compared else 0.0,
                "nonempty_compared_positions": compared,
            }
        )
    return pd.DataFrame(rows).sort_values(["pitch_jaccard", "exact_onset_rate"], ascending=False)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    vienna_tokens = score_onset_tokens(VIENNA_SCORE)
    atepp_tokens = score_onset_tokens(ATEPP_SCORE)
    alignment = pitch_jaccard_window(vienna_tokens, atepp_tokens)
    best = alignment.iloc[0].to_dict()
    start = int(best["atepp_start_beat"])
    length = len(vienna_tokens)
    end = start + length

    vienna = load_vienna_curves()
    atepp = load_atepp_curves()
    v_arr, v_mean = stack_mean(vienna, length=length)
    a_full_arr, _ = stack_mean(atepp)
    a_aligned_arr = np.vstack([curve[start:end] for curve in atepp.values() if len(curve) >= end])
    a_aligned_mean = np.nanmean(a_aligned_arr, axis=0)

    stats = {
        "alignment_method": "musicxml_onset_pitch_sliding_jaccard",
        "vienna_score_beats": length,
        "atepp_score_beats": len(atepp_tokens),
        "atepp_aligned_start_beat": start,
        "atepp_aligned_end_beat_exclusive": end,
        "pitch_jaccard": float(best["pitch_jaccard"]),
        "exact_onset_rate": float(best["exact_onset_rate"]),
        "vienna_curves": len(vienna),
        "atepp_curves": int(a_aligned_arr.shape[0]),
        "mean_abs_diff_bpm": float(np.nanmean(np.abs(v_mean - a_aligned_mean))),
        "rmse_bpm": float(np.sqrt(np.nanmean((v_mean - a_aligned_mean) ** 2))),
        "corr": float(np.corrcoef(v_mean, a_aligned_mean)[0, 1]),
        "vienna_mean_bpm": float(np.nanmean(v_mean)),
        "atepp_aligned_mean_bpm": float(np.nanmean(a_aligned_mean)),
    }

    alignment.head(20).to_csv(OUT_DIR / "k331_score_alignment_top_windows.csv", index=False)
    pd.DataFrame([stats]).to_csv(OUT_DIR / "k331_score_aligned_tempo_stats.csv", index=False)
    pd.DataFrame(
        {
            "aligned_beat_idx": np.arange(length),
            "vienna_mean_bpm": v_mean,
            "atepp_score_aligned_mean_bpm": a_aligned_mean,
        }
    ).to_csv(OUT_DIR / "k331_score_aligned_mean_tempo.csv", index=False)

    fig, ax = plt.subplots(1, 1, figsize=(13, 4.8))
    x = np.arange(length)
    for curve in v_arr:
        ax.plot(x, curve, color="#4C78A8", alpha=0.16, linewidth=0.7)
    for curve in a_aligned_arr:
        ax.plot(x, curve, color="#F58518", alpha=0.22, linewidth=0.8)
    ax.plot(x, v_mean, color="#1f4e8c", linewidth=2.4, label=f"Vienna4x4 score-aligned mean (n={len(vienna)})")
    ax.plot(x, a_aligned_mean, color="#c45a00", linewidth=2.4, label=f"ATEPP score-aligned mean (n={a_aligned_arr.shape[0]})")
    ax.set_title("Mozart K331 tempo curves on the same score-aligned fragment")
    ax.set_xlabel("Aligned score beat index")
    ax.set_ylabel("Tempo (quarter-note BPM)")
    ax.grid(alpha=0.22)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig_path = OUT_DIR / "k331_vienna_vs_atepp_score_aligned_tempo.png"
    fig.savefig(fig_path, dpi=180)
    plt.close(fig)

    print(fig_path)
    print(pd.Series(stats).to_string())


if __name__ == "__main__":
    main()
