from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
MIREX = ROOT / "MERIX SUBMISSION" / "MIREX_Model"
VELOCITY = ROOT / "MERIX SUBMISSION" / "Velocity"
BEAT_DYN_DIR = ROOT / "MazurkaBL-master" / "beat_dyn"
MARKINGS_DYN_DIR = ROOT / "MazurkaBL-master" / "markings_dyn"
OUT_DIR = MIREX / "mazurkabl_dynamics_example_plots"
STR_VEC = [3, 2, 2, 2, 2, 2]
PIECES = ["M17-4", "M24-2"]


def load_velocity_module():
    path = VELOCITY / "build_mazurka_velocity_npz_performer_levels.py"
    spec = importlib.util.spec_from_file_location("velocity_builder", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_markings(raw_id: str):
    path = MARKINGS_DYN_DIR / f"{raw_id}markings.csv"
    if not path.exists():
        return []
    rows = list(csv.reader(path.open()))
    if len(rows) < 2:
        return []
    labels = rows[2] if len(rows) >= 3 else rows[0]
    out = []
    for i, cell in enumerate(rows[1]):
        try:
            beat = int(float(cell)) - 1
        except ValueError:
            continue
        label = labels[i].strip() if i < len(labels) else ""
        out.append((beat, label))
    return out


def plot_piece(piece: str, vel):
    raw_id = piece.replace("M0", "M", 1)
    dyn_path = BEAT_DYN_DIR / f"{raw_id}beat_dynNORM.csv"
    df, performer_cols = vel.load_beat_dyn(dyn_path)
    curves = vel.compute_dyn_curves(df, performer_cols, smooth_window=3)
    mat = np.vstack([curves[k] for k in performer_cols])
    mean = np.nanmean(mat, axis=0)
    q25 = np.nanpercentile(mat, 25, axis=0)
    q75 = np.nanpercentile(mat, 75, axis=0)
    n = mat.shape[1]
    beats = np.arange(n)

    counts = {level: np.zeros(n, dtype=float) for level in range(1, 7)}
    for curve in curves.values():
        _, level_sets = vel.group_analysis_hierarchy(curve, STR_VEC, enforce_nested=True)
        for level in range(1, 7):
            idx = np.asarray(level_sets[level], dtype=int)
            idx = idx[(idx >= 0) & (idx < n)]
            counts[level][idx] += 1.0
    consensus = {level: counts[level] / len(curves) for level in range(1, 7)}
    markings = parse_markings(raw_id)

    fig, axes = plt.subplots(
        7,
        1,
        figsize=(16, 10),
        sharex=True,
        gridspec_kw={"height_ratios": [3.2, 0.65, 0.65, 0.65, 0.65, 0.65, 0.65]},
    )

    ax = axes[0]
    ax.fill_between(beats, q25, q75, color="#cfd8dc", alpha=0.85, label="performer IQR")
    ax.plot(beats, mean, color="#263238", linewidth=1.4, label="mean dynamics")
    ax.set_ylabel("dyn")
    ax.set_ylim(-0.03, 1.03)
    ax.set_title(f"{piece} dynamics curve, score dynamics, and L1-L6 dynamic boundaries")
    ax.legend(loc="upper right", frameon=False)

    for beat, label in markings:
        if 0 <= beat < n:
            ax.axvline(beat, color="#d84315", alpha=0.35, linewidth=0.9)
            if label:
                ax.text(
                    beat,
                    1.02,
                    label,
                    rotation=90,
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    color="#bf360c",
                )

    colors = ["#1565c0", "#00897b", "#7b1fa2", "#ef6c00", "#c2185b", "#455a64"]
    for level in range(1, 7):
        axl = axes[level]
        y = consensus[level]
        axl.plot(beats, y, color=colors[level - 1], linewidth=0.9)
        hit = np.where(y >= 0.05)[0]
        axl.scatter(hit, y[hit], s=8 + y[hit] * 60, color=colors[level - 1], alpha=0.85)
        for beat, _ in markings:
            if 0 <= beat < n:
                axl.axvline(beat, color="#d84315", alpha=0.18, linewidth=0.7)
        axl.set_ylim(-0.02, max(0.2, float(y.max()) + 0.05))
        axl.set_ylabel(f"L{level}")
        axl.grid(axis="y", alpha=0.2)

    axes[-1].set_xlabel("beat index")
    fig.tight_layout()
    out_path = OUT_DIR / f"{piece}_dynamics_l1_l6_markings.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(out_path)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    vel = load_velocity_module()
    for piece in PIECES:
        plot_piece(piece, vel)


if __name__ == "__main__":
    main()
