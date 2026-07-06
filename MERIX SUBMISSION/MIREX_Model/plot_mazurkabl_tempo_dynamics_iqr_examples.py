from __future__ import annotations

import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
MIREX = ROOT / "MERIX SUBMISSION" / "MIREX_Model"
BEAT_TIME_DIR = ROOT / "MazurkaBL-master" / "beat_time"
BEAT_DYN_DIR = ROOT / "MazurkaBL-master" / "beat_dyn"
DATASETS_DIR = ROOT / "datasets"
OUT_DIR = MIREX / "mazurkabl_tempo_dynamics_iqr_plots"
PIECES = ["M17-4", "M24-2"]


def performer_cols(df: pd.DataFrame) -> list[str]:
    meta = {"Unnamed: 0", "measure_number", "beat_number"}
    return [c for c in df.columns if c not in meta]


def raw_id(piece: str) -> str:
    return piece.replace("M0", "M", 1)


def timestamps_to_bpm(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    curves = []
    for col in cols:
        t = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        dt = np.diff(t)
        bpm = np.full(len(t), np.nan, dtype=float)
        valid = np.isfinite(dt) & (dt > 0)
        bpm[1:][valid] = 60.0 / dt[valid]
        if len(bpm) > 1:
            bpm[0] = bpm[1]
        s = pd.Series(bpm)
        s = s.interpolate("linear", limit_direction="both")
        s = s.rolling(window=3, center=True, min_periods=1).mean()
        curves.append(s.to_numpy())
    return np.vstack(curves)


def dynamics_matrix(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    curves = []
    for col in cols:
        vals = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        s = pd.Series(vals)
        s = s.where((s >= 0.0) & (s <= 1.0))
        s = s.interpolate("linear", limit_direction="both")
        s = s.rolling(window=3, center=True, min_periods=1).mean()
        curves.append(s.clip(0.0, 1.0).to_numpy())
    return np.vstack(curves)


def is_phrase_label(value: object) -> bool:
    if pd.isna(value):
        return False
    text = str(value).strip()
    if not text:
        return False
    low = text.lower()
    skip = {
        "nan",
        ".",
        "expressions/notes",
        "measure",
        "beat",
        "non-event",
        "minrhy",
    }
    if low in skip:
        return False
    if re.fullmatch(r"\d+(\.\d+)?", text):
        return False
    phrase_words = ("phrase", "section", "verse", "intro", "cod", "coda")
    if any(w in low for w in phrase_words):
        return True
    if re.fullmatch(r"[A-H](?:['’])?", text):
        return True
    return False


def parse_human_phrase_labels(piece: str):
    opus, num = piece[1:].split("-")
    path = DATASETS_DIR / f"mazurka{int(opus)}-{int(num)}.xls"
    if not path.exists():
        return []
    raw = pd.read_excel(path, sheet_name="Summary", header=None)
    phrase_col = 3
    data = raw.iloc[2:].reset_index(drop=True)
    if phrase_col >= data.shape[1]:
        return []
    out = []
    for row_idx, value in data.iloc[:, phrase_col].items():
        if is_phrase_label(value):
            out.append((int(row_idx), str(value).strip()))
    return out


def plot_piece(piece: str):
    piece_raw = raw_id(piece)
    time_df = pd.read_csv(BEAT_TIME_DIR / f"{piece_raw}beat_time.csv")
    dyn_df = pd.read_csv(BEAT_DYN_DIR / f"{piece_raw}beat_dynNORM.csv")
    common_cols = sorted(set(performer_cols(time_df)) & set(performer_cols(dyn_df)))
    tempo = timestamps_to_bpm(time_df, common_cols)
    dyn = dynamics_matrix(dyn_df, common_cols)
    n = min(tempo.shape[1], dyn.shape[1])
    tempo = tempo[:, :n]
    dyn = dyn[:, :n]
    beats = np.arange(n) + 1

    tempo_mean = np.nanmean(tempo, axis=0)
    tempo_q25 = np.nanpercentile(tempo, 25, axis=0)
    tempo_q75 = np.nanpercentile(tempo, 75, axis=0)
    dyn_mean = np.nanmean(dyn, axis=0)
    dyn_q25 = np.nanpercentile(dyn, 25, axis=0)
    dyn_q75 = np.nanpercentile(dyn, 75, axis=0)
    phrase_labels = parse_human_phrase_labels(piece)

    fig, axes = plt.subplots(2, 1, figsize=(16, 7.5), sharex=True)
    fig.suptitle(f"{piece}: tempo and dynamics mean/IQR with XLS human phrase labels")

    ax = axes[0]
    ax.fill_between(beats, tempo_q25, tempo_q75, color="#bbdefb", alpha=0.8, label="tempo IQR")
    ax.plot(beats, tempo_mean, color="#0d47a1", linewidth=1.35, label="mean tempo")
    ax.set_ylabel("tempo (BPM)")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(loc="upper right", frameon=False)

    ax = axes[1]
    ax.fill_between(beats, dyn_q25, dyn_q75, color="#cfd8dc", alpha=0.85, label="dynamics IQR")
    ax.plot(beats, dyn_mean, color="#263238", linewidth=1.35, label="mean dynamics")
    ax.set_ylabel("dynamics")
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("beat index")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(loc="upper right", frameon=False)

    for ax in axes:
        for beat_idx, _ in phrase_labels:
            x = beat_idx + 1
            if 1 <= x <= n:
                ax.axvline(x, color="#d62728", alpha=0.34, linewidth=1.0)

    top = axes[0]
    ymin, ymax = top.get_ylim()
    for beat_idx, label in phrase_labels:
        x = beat_idx + 1
        if 1 <= x <= n and label:
            top.text(
                x,
                ymax,
                label,
                rotation=90,
                ha="center",
                va="bottom",
                fontsize=7,
                color="#bf360c",
            )
    top.set_ylim(ymin, ymax)

    fig.tight_layout()
    out = OUT_DIR / f"{piece}_tempo_dynamics_iqr_xls_phrases.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print(out)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for piece in PIECES:
        plot_piece(piece)


if __name__ == "__main__":
    main()
