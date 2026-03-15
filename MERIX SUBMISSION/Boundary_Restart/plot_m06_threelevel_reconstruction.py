from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
MIREX_MODEL_DIR = REPO_ROOT / "MERIX SUBMISSION" / "MIREX_Model"
if str(MIREX_MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MIREX_MODEL_DIR))

from build_mazurka_beat_npz_performer_levels import (  # noqa: E402
    compute_tempo_curves,
    load_beat_time,
)


DEFAULT_PIECE_ID = "M06-2"
THRESHOLD = 0.3
LEVELS = ("low", "mid", "high")
TARGET_COLS = {
    "low": "target_peak_G1",
    "mid": "target_peak_G2",
    "high": "target_peak_G3",
}
LEVEL_COLORS = {
    "low": "#4C78A8",
    "mid": "#F58518",
    "high": "#E45756",
}


def load_piece_mean_tempo(beat_time_dir: Path, piece_id: str) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    beat_time_path = beat_time_dir / f"{piece_id}beat_time.csv"
    df_bt, performer_cols = load_beat_time(beat_time_path)
    tempo_arrays = compute_tempo_curves(
        df_bt,
        performer_cols,
        smooth_window=3,
        bpm_range=(0, 5000),
        clip_max=600,
    )
    mean_tempo = np.nanmean(np.vstack([tempo_arrays[name] for name in tempo_arrays]), axis=0)
    return mean_tempo, tempo_arrays


def build_level_basis(num_beats: int, boundaries: np.ndarray, strengths: np.ndarray | None = None) -> np.ndarray:
    b = np.asarray(boundaries, dtype=int)
    b = b[(b >= 0) & (b < num_beats)]
    b = np.unique(b)
    if b.size == 0:
        b = np.array([0, num_beats - 1], dtype=int)
    if b[0] != 0:
        b = np.insert(b, 0, 0)
    if b[-1] != num_beats - 1:
        b = np.append(b, num_beats - 1)
    b.sort()

    if strengths is None or len(strengths) == 0:
        s = np.ones(len(b), dtype=float)
    else:
        s = np.zeros(len(b), dtype=float)
        orig_b = np.asarray(boundaries, dtype=int)
        orig_s = np.asarray(strengths, dtype=float)
        for i, bi in enumerate(b):
            if orig_b.size == 0:
                s[i] = 1.0
            else:
                j = int(np.argmin(np.abs(orig_b - bi)))
                s[i] = orig_s[j]

    basis = np.zeros(num_beats, dtype=float)
    for i in range(len(b) - 1):
        start = int(b[i])
        end = int(b[i + 1])
        if end <= start:
            continue
        amp = 0.5 * (s[i] + s[i + 1])
        t = np.arange(start, end + 1)
        u = (t - start) / (end - start)
        basis[t] = amp * (-4.0 * u * (1.0 - u))
    return basis


def build_design_matrix(num_beats: int, level_sets: dict[str, np.ndarray], strengths_by_level: dict[str, np.ndarray] | None = None) -> np.ndarray:
    cols = [np.ones(num_beats, dtype=float)]
    for level in LEVELS:
        strengths = None if strengths_by_level is None else strengths_by_level.get(level)
        cols.append(build_level_basis(num_beats, level_sets[level], strengths=strengths))
    return np.stack(cols, axis=1)


def apply_params(mean_tempo: np.ndarray, level_sets: dict[str, np.ndarray], beta: np.ndarray, strengths_by_level: dict[str, np.ndarray] | None = None) -> tuple[np.ndarray, dict[str, float]]:
    X = build_design_matrix(len(mean_tempo), level_sets, strengths_by_level)
    y_hat = X @ beta
    rmse = float(np.sqrt(np.mean((y_hat - mean_tempo) ** 2)))
    corr = float(np.corrcoef(mean_tempo, y_hat)[0, 1])
    return y_hat, {"rmse": rmse, "corr": corr}


def aggregate_piece_frequencies(df: pd.DataFrame, piece_id: str) -> pd.DataFrame:
    piece_df = df[df["piece_id"] == piece_id]
    agg = (
        piece_df.groupby("beat_idx")
        .agg({TARGET_COLS[level]: "mean" for level in LEVELS})
        .rename(columns={TARGET_COLS[level]: level for level in LEVELS})
        .sort_index()
    )
    return agg.reset_index()


def fit_beta(mean_tempo_by_piece: dict[str, np.ndarray], freq_by_piece: dict[str, pd.DataFrame], heldout_piece: str) -> np.ndarray:
    X_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []
    for piece_id, y in mean_tempo_by_piece.items():
        if piece_id == heldout_piece:
            continue
        agg = freq_by_piece[piece_id]
        if len(agg) != len(y):
            n = min(len(agg), len(y))
            agg = agg.iloc[:n].copy()
            y = y[:n]
        level_sets = {
            level: agg.loc[agg[level] >= THRESHOLD, "beat_idx"].to_numpy(dtype=int)
            for level in LEVELS
        }
        X_list.append(build_design_matrix(len(y), level_sets))
        y_list.append(y)
    X_all = np.concatenate(X_list, axis=0)
    y_all = np.concatenate(y_list, axis=0)
    beta, *_ = np.linalg.lstsq(X_all, y_all, rcond=None)
    return beta


def load_predicted_events(base_dir: Path) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    level_sets: dict[str, np.ndarray] = {}
    strengths: dict[str, np.ndarray] = {}
    for level in LEVELS:
        pred_path = base_dir / f"tcn_{level}_boundary_union_recall_cpu" / "predicted_events.csv.gz"
        pred_df = pd.read_csv(pred_path)
        level_sets[level] = pred_df["beat_idx"].to_numpy(dtype=int)
        strengths[level] = pred_df["detector_score"].to_numpy(dtype=float)
    return level_sets, strengths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot three-level boundary reconstruction vs formal tempo curve.")
    parser.add_argument("--piece_id", default=DEFAULT_PIECE_ID, help="Mazurka piece id, e.g. M06-2")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    piece_id = args.piece_id
    beat_time_dir = REPO_ROOT / "MazurkaBL-master" / "beat_time"
    table_path = REPO_ROOT / "MERIX SUBMISSION" / "Boundary_Restart" / "outputs" / "salience_grouped3_hi8_xml" / "beat_table_salience_grouped3_hi8_xml.csv.gz"
    pred_base_dir = REPO_ROOT / "MERIX SUBMISSION" / "Boundary_Restart" / "outputs" / "local_runs" / piece_id
    output_dir = pred_base_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(table_path, usecols=["piece_id", "beat_idx", *TARGET_COLS.values()])
    pieces = sorted(df["piece_id"].unique())
    mean_tempo_by_piece = {}
    tempo_arrays_by_piece = {}
    freq_by_piece = {}
    for train_piece_id in pieces:
        mean_tempo, tempo_arrays = load_piece_mean_tempo(beat_time_dir, train_piece_id)
        mean_tempo_by_piece[train_piece_id] = mean_tempo
        tempo_arrays_by_piece[train_piece_id] = tempo_arrays
        freq_by_piece[train_piece_id] = aggregate_piece_frequencies(df, train_piece_id)

    beta = fit_beta(mean_tempo_by_piece, freq_by_piece, heldout_piece=piece_id)

    mean_tempo = mean_tempo_by_piece[piece_id]
    tempo_arrays = tempo_arrays_by_piece[piece_id]
    true_freq = freq_by_piece[piece_id]

    true_level_sets = {
        level: true_freq.loc[true_freq[level] >= THRESHOLD, "beat_idx"].to_numpy(dtype=int)
        for level in LEVELS
    }
    true_recon, true_metrics = apply_params(mean_tempo, true_level_sets, beta)

    pred_level_sets, pred_strengths = load_predicted_events(pred_base_dir)
    pred_recon, pred_metrics = apply_params(mean_tempo, pred_level_sets, beta, strengths_by_level=pred_strengths)

    x = np.arange(len(mean_tempo))
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(4, 1, height_ratios=[2.2, 1, 1, 1], hspace=0.18)

    ax0 = fig.add_subplot(gs[0])
    for nm, curve in tempo_arrays.items():
        ax0.plot(x, curve, color="0.8", linewidth=0.7, alpha=0.25)
    ax0.plot(x, mean_tempo, color="black", linewidth=2.0, label="Formal tempo curve (mean)")
    ax0.plot(
        x,
        true_recon,
        color="#1f77b4",
        linewidth=1.8,
        linestyle="--",
        label=f"Reconstruction from true low/mid/high union (rmse={true_metrics['rmse']:.2f}, corr={true_metrics['corr']:.3f})",
    )
    ax0.plot(
        x,
        pred_recon,
        color="#d62728",
        linewidth=1.8,
        label=f"Reconstruction from TCN direct low/mid/high (rmse={pred_metrics['rmse']:.2f}, corr={pred_metrics['corr']:.3f})",
    )
    ax0.set_ylabel("Tempo (BPM)")
    ax0.set_title(f"{piece_id}: three-level boundary reconstruction vs formal tempo curve")
    ax0.grid(alpha=0.25)
    ax0.legend(frameon=False, fontsize=9, loc="upper right")

    for row_idx, level in enumerate(LEVELS, start=1):
        ax = fig.add_subplot(gs[row_idx], sharex=ax0)
        ax.plot(
            true_freq["beat_idx"],
            true_freq[level],
            color=LEVEL_COLORS[level],
            linewidth=1.3,
            label=f"True {level} union frequency",
        )
        pred_beats = pred_level_sets[level]
        pred_scores = pred_strengths[level]
        ax.scatter(
            pred_beats,
            pred_scores,
            color=LEVEL_COLORS[level],
            edgecolors="black",
            linewidths=0.3,
            s=28,
            alpha=0.9,
            label=f"Predicted {level} event score",
            zorder=3,
        )
        ax.axhline(THRESHOLD, color="0.6", linestyle="--", linewidth=0.8)
        ax.set_ylim(-0.02, 1.05)
        ax.set_ylabel(level)
        ax.grid(alpha=0.2)
        ax.legend(frameon=False, fontsize=8, loc="upper right")

    fig.axes[-1].set_xlabel("Beat index")
    fig.tight_layout()

    stem = f"{piece_id}_threelevel_reconstruction_vs_tempo"
    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"
    fig.savefig(png_path, dpi=180, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    metrics = {
        "piece_id": piece_id,
        "threshold": THRESHOLD,
        "beta": beta.tolist(),
        "true_reconstruction": true_metrics,
        "pred_reconstruction": pred_metrics,
        "pred_event_counts": {level: int(len(pred_level_sets[level])) for level in LEVELS},
        "true_union_counts_over_threshold": {
            level: int(np.sum(true_freq[level].to_numpy() >= THRESHOLD)) for level in LEVELS
        },
    }
    json_path = output_dir / f"{stem}.json"
    json_path.write_text(json.dumps(metrics, indent=2))

    print(f"Saved figure to {png_path}")
    print(f"Saved figure to {pdf_path}")
    print(f"Saved metrics to {json_path}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
