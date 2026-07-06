from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
MIREX = ROOT / "MERIX SUBMISSION" / "MIREX_Model"
BUILD_SCRIPT = MIREX / "build_mazurka_beat_npz_performer_levels.py"
BEAT_TIME = ROOT / "MazurkaBL-master" / "beat_time" / "M17-4beat_time.csv"
OUT_DIR = MIREX / "m17_4_tempo_l1_l6_performer_choices"
STR_VEC = [3, 2, 2, 2, 2, 2]


def load_builder():
    spec = importlib.util.spec_from_file_location("mazurka_builder", BUILD_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {BUILD_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["mazurka_builder_for_m17_choices"] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    builder = load_builder()
    df, performer_cols = builder.load_beat_time(BEAT_TIME)
    tempo_curves = builder.compute_tempo_curves(df, performer_cols, smooth_window=3)
    n_beats = len(df)

    matrices = {level: np.zeros((len(performer_cols), n_beats), dtype=np.uint8) for level in range(1, 7)}
    event_counts = {level: [] for level in range(1, 7)}

    for row, performer in enumerate(performer_cols):
        curve = tempo_curves[performer]
        _, level_sets = builder.group_analysis_hierarchy(curve, STR_VEC, enforce_nested=False)
        for level in range(1, 7):
            idx = np.asarray(level_sets[level], dtype=int)
            idx = idx[(idx >= 0) & (idx < n_beats)]
            matrices[level][row, idx] = 1
            event_counts[level].append(len(idx))

    fig, axes = plt.subplots(6, 1, figsize=(18, 12), sharex=True, constrained_layout=True)
    for level, ax in zip(range(1, 7), axes):
        mat = matrices[level]
        ax.imshow(
            mat,
            aspect="auto",
            interpolation="nearest",
            cmap="Greys",
            origin="lower",
            extent=[1, n_beats, 1, len(performer_cols)],
            vmin=0,
            vmax=1,
        )
        counts = np.asarray(event_counts[level], dtype=float)
        consensus = mat.mean(axis=0)
        strong = np.flatnonzero(consensus >= 0.10) + 1
        for x in strong:
            ax.axvline(x, color="#d62728", alpha=0.16, linewidth=0.7)
        ax.set_ylabel(f"L{level}\nperformer")
        ax.set_title(
            f"L{level}: mean events/performer={counts.mean():.1f}, "
            f"median={np.median(counts):.0f}, consensus>=10% beats={len(strong)}",
            loc="left",
            fontsize=10,
        )
        ax.set_yticks([1, len(performer_cols)])
        ax.grid(axis="x", color="#cccccc", alpha=0.12, linewidth=0.4)

    axes[-1].set_xlabel("score beat")
    fig.suptitle(
        "M17-4 tempo-curve L1-L6 raw local-minima boundary choices by performer "
        "(black = selected by performer; red guide = >=10% performer consensus)",
        fontsize=14,
    )
    png = OUT_DIR / "M17-4_tempo_L1-L6_performer_choice_raster_raw_non_nested.png"
    pdf = OUT_DIR / "M17-4_tempo_L1-L6_performer_choice_raster_raw_non_nested.pdf"
    fig.savefig(png, dpi=180)
    fig.savefig(pdf)
    plt.close(fig)

    summary_rows = []
    for level in range(1, 7):
        mat = matrices[level]
        counts = np.asarray(event_counts[level], dtype=float)
        consensus = mat.mean(axis=0)
        summary_rows.append(
            {
                "level": level,
                "performers": len(performer_cols),
                "beats": n_beats,
                "mean_events_per_performer": counts.mean(),
                "median_events_per_performer": np.median(counts),
                "min_events_per_performer": counts.min(),
                "max_events_per_performer": counts.max(),
                "consensus_ge_0p10_beats": int((consensus >= 0.10).sum()),
                "consensus_ge_0p25_beats": int((consensus >= 0.25).sum()),
                "consensus_ge_0p50_beats": int((consensus >= 0.50).sum()),
                "max_consensus": float(consensus.max()),
            }
        )
        np.savetxt(
            OUT_DIR / f"M17-4_L{level}_performer_by_beat_raw_non_nested.csv",
            mat,
            fmt="%d",
            delimiter=",",
        )

    import pandas as pd

    pd.DataFrame(summary_rows).to_csv(OUT_DIR / "M17-4_level_summary_raw_non_nested.csv", index=False)
    print(png)
    print(pdf)
    print(OUT_DIR / "M17-4_level_summary_raw_non_nested.csv")


if __name__ == "__main__":
    main()
