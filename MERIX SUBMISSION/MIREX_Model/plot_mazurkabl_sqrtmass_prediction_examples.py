from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
MIREX = ROOT / "MERIX SUBMISSION" / "MIREX_Model"
BEAT_TIME_DIR = ROOT / "MazurkaBL-master" / "beat_time"
SQRT_SCRIPT = MIREX / "run_mazurkabl_l2plus_sqrtmass_branchwise_cnn.py"
OUT_DIR = MIREX / "mazurkabl_sqrtmass_prediction_example_plots"

SETTING = "handcrafted_plus_branchwise"
EXAMPLE_PIECES = ["M17-4", "M24-2", "M30-2", "M68-3"]


def load_sqrt_runner():
    os.environ.setdefault("MAZURKA_EVENT_MIN", "0.01")
    os.environ.setdefault("MAZURKA_DENSITY_MODE", "fixed_2bars")
    os.environ.setdefault("MAZURKA_DENSITY_BEATS", "6")
    os.environ.setdefault("MAZURKA_DENSITY_MIN_DISTANCE", "1")
    spec = importlib.util.spec_from_file_location("mazurka_sqrtmass_for_plots", SQRT_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {SQRT_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["mazurka_sqrtmass_for_plots"] = module
    spec.loader.exec_module(module)
    return module


sqrt_runner = load_sqrt_runner()
bw = sqrt_runner.bw


def raw_id(piece: str) -> str:
    return piece.replace("M0", "M", 1)


def performer_cols(df: pd.DataFrame) -> list[str]:
    meta = {"Unnamed: 0", "measure_number", "beat_number"}
    return [c for c in df.columns if c not in meta]


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
        s = pd.Series(bpm).interpolate("linear", limit_direction="both")
        s = s.rolling(window=3, center=True, min_periods=1).mean()
        curves.append(s.to_numpy())
    return np.vstack(curves)


def load_tempo_iqr(piece: str, n_beats: int):
    path = BEAT_TIME_DIR / f"{raw_id(piece)}beat_time.csv"
    df = pd.read_csv(path)
    curves = timestamps_to_bpm(df, performer_cols(df))[:, :n_beats]
    return {
        "mean": np.nanmean(curves, axis=0),
        "q25": np.nanpercentile(curves, 25, axis=0),
        "q75": np.nanpercentile(curves, 75, axis=0),
        "num_performers": curves.shape[0],
    }


def density_predictions(labels, scores, val_pieces, train_pieces):
    out = {}
    for piece in val_pieces:
        expected = bw.runner.base.expected_count_from_train_density(labels, train_pieces, len(labels[piece]))
        pred = bw.runner.base.extract_top_density(scores[piece], expected)
        out[piece] = pred.astype(int)
    return out


def apply_latest_density_settings():
    base = bw.runner.base
    base.EVENT_MIN = float(sqrt_runner.EVENT_MIN_OVERRIDE)

    def fixed_two_bar_density(_labels, _train_pieces, num_beats):
        return max(1, int(round(float(num_beats) / max(float(sqrt_runner.DENSITY_BEATS), 1e-9))))

    original_extract = base.extract_top_density

    def top_density(scores, expected_count, min_distance=None):
        return original_extract(
            scores,
            expected_count,
            min_distance=max(int(sqrt_runner.DENSITY_MIN_DISTANCE), 1),
        )

    if sqrt_runner.DENSITY_MODE == "fixed_2bars":
        base.expected_count_from_train_density = fixed_two_bar_density
        base.extract_top_density = top_density
    elif sqrt_runner.DENSITY_MODE != "train_density":
        raise ValueError(f"Unsupported density mode: {sqrt_runner.DENSITY_MODE}")


def plot_piece(piece: str, labels, scores, pred_events, out_dir: Path):
    n = len(labels[piece])
    tempo = load_tempo_iqr(piece, n)
    beats = np.arange(n) + 1
    target = labels[piece]
    true_events = np.flatnonzero(target >= bw.runner.base.EVENT_MIN)

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(16, 8.5),
        sharex=True,
        gridspec_kw={"height_ratios": [2.0, 1.0, 0.9]},
    )
    fig.suptitle(
        f"{piece}: tempo curve, true target events, predicted events "
        f"({SETTING}, target >= {bw.runner.base.EVENT_MIN:g})"
    )

    ax = axes[0]
    ax.fill_between(beats, tempo["q25"], tempo["q75"], color="#bbdefb", alpha=0.75, label="tempo IQR")
    ax.plot(beats, tempo["mean"], color="#0d47a1", linewidth=1.35, label="mean tempo")
    ax.set_ylabel("BPM")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper right", frameon=False)

    ax = axes[1]
    ax.plot(beats, target, color="#4e342e", linewidth=1.25, label="target value")
    ax.axhline(bw.runner.base.EVENT_MIN, color="#8d6e63", linestyle="--", linewidth=1.0, label="true threshold")
    ax.set_ylabel("target")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper right", frameon=False)

    ax = axes[2]
    ax.plot(beats, scores[piece], color="#455a64", linewidth=1.15, label="model score")
    ax.vlines(true_events + 1, 0.0, 0.48, color="#d32f2f", alpha=0.55, linewidth=1.0, label="true")
    ax.vlines(pred_events[piece] + 1, 0.52, 1.0, color="#2e7d32", alpha=0.65, linewidth=1.0, label="pred")
    ax.set_ylim(-0.02, 1.02)
    ax.set_ylabel("score / events")
    ax.set_xlabel("beat index")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(loc="upper right", ncol=3, frameon=False)

    for ax in axes:
        ax.set_xlim(1, n)

    fig.tight_layout()
    png = out_dir / f"{piece}_tempo_true_pred.png"
    pdf = out_dir / f"{piece}_tempo_true_pred.pdf"
    fig.savefig(png, dpi=180)
    fig.savefig(pdf)
    plt.close(fig)
    return {
        "piece": piece,
        "num_beats": n,
        "num_performers": tempo["num_performers"],
        "true_events": int(len(true_events)),
        "pred_events": int(len(pred_events[piece])),
        "png": str(png),
        "pdf": str(pdf),
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = bw.runner.load_config()
    pieces, labels, _, _ = sqrt_runner.load_sqrtmass_l2plus_labels()
    pieces = sorted(pieces)
    base_features, _ = bw.runner.load_piece_features(pieces, cfg)
    rich_features = bw.runner.load_rich_features(pieces)
    folds = bw.runner.base.make_folds(pieces, n_folds=5, seed=42)
    device = bw.runner.resolve_device()
    apply_latest_density_settings()

    all_scores = {}
    all_preds = {}
    for fold_idx, val_pieces in enumerate(folds, start=1):
        wanted = [p for p in val_pieces if p in EXAMPLE_PIECES]
        if not wanted:
            continue
        train_pieces = [p for p in pieces if p not in set(val_pieces)]
        features, base_dim = bw.build_setting_features(
            SETTING,
            base_features,
            rich_features,
            pieces,
            seed=200000 + fold_idx,
        )
        model, mean, std = bw.train_one(
            cfg,
            features,
            labels,
            train_pieces,
            base_dim=base_dim,
            seed=9900 + fold_idx,
            device=device,
        )
        val_scores = bw.predict(model, features, val_pieces, mean, std, device)
        pred_events = density_predictions(labels, val_scores, val_pieces, train_pieces)
        for piece in wanted:
            all_scores[piece] = val_scores[piece]
            all_preds[piece] = pred_events[piece]

    rows = []
    for piece in EXAMPLE_PIECES:
        if piece not in all_scores:
            continue
        rows.append(plot_piece(piece, labels, all_scores, all_preds, OUT_DIR))
    pd.DataFrame(rows).to_csv(OUT_DIR / "plot_summary.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"Wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
