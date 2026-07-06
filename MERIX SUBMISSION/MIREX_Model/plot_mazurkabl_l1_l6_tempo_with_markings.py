from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
BUILD_SCRIPT = ROOT / "MERIX SUBMISSION" / "MIREX_Model" / "build_mazurka_beat_npz_performer_levels.py"
BEAT_TIME_DIR = ROOT / "MazurkaBL-master" / "beat_time"
MARKINGS_DIR = ROOT / "MazurkaBL-master" / "markings"
OUT_DIR = ROOT / "MERIX SUBMISSION" / "MIREX_Model" / "mazurkabl_l1_l6_tempo_markings_plots"

PIECES = ["M17-4", "M24-2", "M30-2", "M63-3", "M68-3"]
STR_VEC = [3, 2, 2, 2, 2, 2]

spec = importlib.util.spec_from_file_location("mazurka_level_builder", BUILD_SCRIPT)
builder = importlib.util.module_from_spec(spec)
sys.modules["mazurka_level_builder"] = builder
assert spec.loader is not None
spec.loader.exec_module(builder)


def read_mean_tempo(piece_id: str) -> tuple[pd.DataFrame, np.ndarray, dict[str, np.ndarray]]:
    path = BEAT_TIME_DIR / f"{piece_id}beat_time.csv"
    df, performer_cols = builder.load_beat_time(path)
    curves = builder.compute_tempo_curves(df, performer_cols, smooth_window=3, clip_max=600)
    stacked = np.vstack([curves[col] for col in performer_cols])
    mean_tempo = np.nanmean(stacked, axis=0)
    return df, mean_tempo, curves


def read_markings(piece_id: str) -> tuple[np.ndarray, list[str]]:
    path = MARKINGS_DIR / f"{piece_id}markings.csv"
    if not path.exists():
        return np.array([], dtype=int), []
    raw = pd.read_csv(path, header=None)
    if raw.shape[0] < 2:
        return np.array([], dtype=int), []
    labels = [str(x) for x in raw.iloc[0].dropna().tolist()]
    positions = pd.to_numeric(raw.iloc[1], errors="coerce").dropna().to_numpy(dtype=int) - 1
    positions = positions[positions >= 0]
    return positions, labels


def consensus_level_sets(curves: dict[str, np.ndarray], n_beats: int) -> dict[int, np.ndarray]:
    counts = {level: np.zeros(n_beats, dtype=np.float32) for level in range(1, 7)}
    for curve in curves.values():
        _, level_sets = builder.group_analysis_hierarchy(curve, STR_VEC, enforce_nested=True)
        for level in range(1, 7):
            idx = level_sets[level]
            idx = idx[(idx >= 0) & (idx < n_beats)]
            counts[level][idx] += 1.0
    denom = max(len(curves), 1)
    return {level: counts[level] / denom for level in range(1, 7)}


def draw_piece(piece_id: str) -> dict:
    df, mean_tempo, curves = read_mean_tempo(piece_id)
    n_beats = len(mean_tempo)
    beats = np.arange(n_beats)
    level_probs = consensus_level_sets(curves, n_beats)
    marking_pos, marking_labels = read_markings(piece_id)
    marking_pos = marking_pos[marking_pos < n_beats]

    fig, (ax_tempo, ax_marks) = plt.subplots(
        2,
        1,
        figsize=(16, 7.6),
        sharex=True,
        gridspec_kw={"height_ratios": [3.2, 2.2]},
    )

    ax_tempo.plot(beats + 1, mean_tempo, color="#222222", linewidth=1.6, label="mean tempo curve")
    ax_tempo.set_title(f"{piece_id}: tempo curve, computed L1-L6 boundaries, score markings")
    ax_tempo.set_ylabel("BPM")
    ax_tempo.grid(True, axis="y", alpha=0.25)
    ax_tempo.legend(loc="upper right")

    colors = {
        1: "#4c78a8",
        2: "#f58518",
        3: "#54a24b",
        4: "#b279a2",
        5: "#e45756",
        6: "#72b7b2",
    }
    rows = []
    for level in range(1, 7):
        probs = level_probs[level]
        idx = np.flatnonzero(probs > 0)
        rows.append(
            {
                "piece_id": piece_id,
                "level": level,
                "boundary_count_any_performer": int(len(idx)),
                "boundary_count_ge_0p10": int(np.count_nonzero(probs >= 0.10)),
                "boundary_count_ge_0p50": int(np.count_nonzero(probs >= 0.50)),
            }
        )
        if idx.size:
            sizes = 8 + 42 * probs[idx]
            ax_marks.scatter(
                idx + 1,
                np.full_like(idx, 7 - level, dtype=float),
                s=sizes,
                color=colors[level],
                alpha=0.80,
                edgecolors="none",
                label=f"L{level}",
            )

    if marking_pos.size:
        ax_marks.scatter(
            marking_pos + 1,
            np.full_like(marking_pos, 0, dtype=float),
            s=34,
            marker="|",
            linewidths=1.8,
            color="#111111",
            label="score markings",
        )
        for x, label in zip(marking_pos[:40], marking_labels[:40]):
            ax_marks.text(x + 1, -0.35, label.replace("+", " "), rotation=70, fontsize=6, ha="right", va="top")

    ax_marks.set_yticks([6, 5, 4, 3, 2, 1, 0])
    ax_marks.set_yticklabels(["L1", "L2", "L3", "L4", "L5", "L6", "markings"])
    ax_marks.set_ylim(-1.1, 6.8)
    ax_marks.set_xlabel("score beat")
    ax_marks.grid(True, axis="x", alpha=0.18)
    ax_marks.legend(loc="upper right", ncol=4, fontsize=8)

    fig.tight_layout()
    png_path = OUT_DIR / f"{piece_id}_tempo_L1-L6_score_markings.png"
    pdf_path = OUT_DIR / f"{piece_id}_tempo_L1-L6_score_markings.pdf"
    fig.savefig(png_path, dpi=180)
    fig.savefig(pdf_path)
    plt.close(fig)

    for row in rows:
        row["num_beats"] = n_beats
        row["num_performers"] = len(curves)
        row["score_marking_count"] = int(len(marking_pos))
    return {
        "piece_id": piece_id,
        "num_beats": n_beats,
        "num_performers": len(curves),
        "score_marking_count": int(len(marking_pos)),
        "png": str(png_path),
        "pdf": str(pdf_path),
        "level_rows": rows,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summaries = [draw_piece(piece_id) for piece_id in PIECES]
    pd.DataFrame([{
        "piece_id": s["piece_id"],
        "num_beats": s["num_beats"],
        "num_performers": s["num_performers"],
        "score_marking_count": s["score_marking_count"],
        "png": s["png"],
        "pdf": s["pdf"],
    } for s in summaries]).to_csv(OUT_DIR / "plot_summary.csv", index=False)
    pd.DataFrame([row for s in summaries for row in s["level_rows"]]).to_csv(
        OUT_DIR / "level_boundary_count_summary.csv", index=False
    )
    (OUT_DIR / "metadata.json").write_text(
        json.dumps(
            {
                "pieces": PIECES,
                "tempo_curve": "mean of smoothed per-performer tempo curves from MazurkaBL beat_time",
                "computed_boundaries": "L1-L6 from group_analysis_hierarchy with STR_VEC=[1,2,4,8,16,32]; marker size is performer consensus frequency",
                "manual_reference": "MazurkaBL score expressive markings from markings/*.csv; these are not structural boundary ground truth",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
