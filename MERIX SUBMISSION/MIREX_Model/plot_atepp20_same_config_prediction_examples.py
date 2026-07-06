from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
MIREX = ROOT / "MERIX SUBMISSION" / "MIREX_Model"
BASE_SCRIPT = MIREX / "run_atepp20_l2plus_weighted_target_experiment.py"
OUT_DIR = MIREX / "atepp20_same_config_prediction_example_plots"

EVENT_MIN = 0.01
DENSITY_BEATS = 6.0
MIN_DISTANCE = 1
EXAMPLE_SHORT_NAMES = [
    "wolfgang_amadeus_mozart_piano_sonata_no_12_in_f_k_332_1_allegro",
    "robert_schumann_arabeske_op_18",
    "ludwig_van_beethoven_piano_sonata_no_8_in_c_minor_op_13_pathe_tique_ii_adagio_cantabile",
    "franz_schubert_piano_sonata_no_13_in_a_d_664_2_andante",
]


def load_base():
    spec = importlib.util.spec_from_file_location("atepp20_same_config_base", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["atepp20_same_config_base"] = module
    spec.loader.exec_module(module)
    return module


base = load_base()
quick = base.base


def extract_top_density(scores: np.ndarray, expected_count: int) -> np.ndarray:
    return quick.extract_top_density(scores, expected_count, min_distance=MIN_DISTANCE).astype(np.int32)


def expected_count(num_beats: int) -> int:
    return max(1, int(round(float(num_beats) / DENSITY_BEATS)))


def match_true_indices(pred: np.ndarray, true: np.ndarray, tolerance: int = 1) -> list[int]:
    used = set()
    matched = []
    for p in pred.tolist():
        best = None
        best_dist = tolerance + 1
        for j, t in enumerate(true.tolist()):
            if j in used:
                continue
            dist = abs(int(p) - int(t))
            if dist <= tolerance and dist < best_dist:
                best = j
                best_dist = dist
        if best is not None:
            used.add(best)
            matched.append(int(true[best]))
    return matched


def metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float | int]:
    true = np.flatnonzero(target >= EVENT_MIN).astype(np.int32)
    m = quick.metrics_from_events(pred, true, tolerance=1)
    matched = match_true_indices(pred, true, tolerance=1)
    wr_num = float(target[matched].sum()) if matched else 0.0
    wr_den = float(target[true].sum()) if len(true) else 0.0
    up = m.matches / m.pred_events if m.pred_events else 0.0
    rec = m.matches / m.true_events if m.true_events else 0.0
    f1 = 2 * up * rec / (up + rec) if up + rec else 0.0
    return {
        "pred": int(m.pred_events),
        "true": int(m.true_events),
        "match": int(m.matches),
        "UP": float(up),
        "recall": float(rec),
        "WR": float(wr_num / wr_den) if wr_den else 0.0,
        "F1": float(f1),
    }


def run_same_config_predictions():
    old_event_min = base.EVENT_MIN
    base.EVENT_MIN = EVENT_MIN
    pieces, labels, _components = base.load_l2plus_weighted_labels()
    features, feature_cols = base.load_piece_features(pieces)
    pieces = sorted(pieces)
    labels = {p: labels[p] for p in pieces}
    features = {p: features[p] for p in pieces}
    folds = quick.make_folds(pieces, n_folds=5, seed=42)

    rows = []
    all_scores = {}
    all_preds = {}
    totals = {"pred": 0, "true": 0, "match": 0, "wr_num": 0.0, "wr_den": 0.0}
    for fold_idx, val_pieces in enumerate(folds, start=1):
        train_pieces = [p for p in pieces if p not in set(val_pieces)]
        model, mean, std = quick.train_one(features, labels, train_pieces, seed=8200 + fold_idx)
        scores = quick.predict(model, features, val_pieces, mean, std)
        for piece in val_pieces:
            pred = extract_top_density(scores[piece], expected_count(len(labels[piece])))
            true = np.flatnonzero(labels[piece] >= EVENT_MIN).astype(np.int32)
            matched = match_true_indices(pred, true, tolerance=1)
            m = metrics(pred, labels[piece])
            totals["pred"] += int(m["pred"])
            totals["true"] += int(m["true"])
            totals["match"] += int(m["match"])
            totals["wr_num"] += float(labels[piece][matched].sum()) if matched else 0.0
            totals["wr_den"] += float(labels[piece][true].sum()) if len(true) else 0.0
            row = {"fold": fold_idx, "piece": piece, "num_beats": len(labels[piece]), **m}
            rows.append(row)
            all_scores[piece] = scores[piece]
            all_preds[piece] = pred

    up = totals["match"] / totals["pred"] if totals["pred"] else 0.0
    rec = totals["match"] / totals["true"] if totals["true"] else 0.0
    aggregate = {
        "setting": "atepp20_baseline_cnn_same_config",
        "event_min": EVENT_MIN,
        "density_beats": DENSITY_BEATS,
        "min_distance": MIN_DISTANCE,
        "pred": int(totals["pred"]),
        "true": int(totals["true"]),
        "match": int(totals["match"]),
        "UP": up,
        "recall": rec,
        "WR": totals["wr_num"] / totals["wr_den"] if totals["wr_den"] else 0.0,
        "F1": 2 * up * rec / (up + rec) if up + rec else 0.0,
    }
    base.EVENT_MIN = old_event_min
    return pieces, labels, all_scores, all_preds, pd.DataFrame(rows), aggregate, feature_cols


def display_name(piece: str) -> str:
    return piece.replace("_", " ")[:88]


def plot_piece(piece: str, labels, scores, preds):
    target = labels[piece]
    score = scores[piece]
    pred = preds[piece]
    true = np.flatnonzero(target >= EVENT_MIN)
    beats = np.arange(len(target)) + 1

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(16, 8.5),
        sharex=True,
        gridspec_kw={"height_ratios": [1.35, 1.0, 0.9]},
    )
    fig.suptitle(f"ATEPP20 {display_name(piece)}: target, model score, true/pred events")

    ax = axes[0]
    ax.plot(beats, target, color="#4e342e", linewidth=1.1, label="target value")
    ax.axhline(EVENT_MIN, color="#8d6e63", linestyle="--", linewidth=1.0, label="true threshold")
    ax.vlines(true + 1, 0, max(float(target.max()), EVENT_MIN), color="#d32f2f", alpha=0.25, linewidth=0.8)
    ax.set_ylabel("target")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper right", frameon=False)

    ax = axes[1]
    ax.plot(beats, score, color="#455a64", linewidth=1.15, label="model score")
    ax.set_ylabel("score")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper right", frameon=False)

    ax = axes[2]
    ax.vlines(true + 1, 0.0, 0.48, color="#d32f2f", alpha=0.55, linewidth=1.0, label="true")
    ax.vlines(pred + 1, 0.52, 1.0, color="#2e7d32", alpha=0.65, linewidth=1.0, label="pred")
    ax.set_ylim(-0.02, 1.02)
    ax.set_ylabel("events")
    ax.set_xlabel("beat index")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(loc="upper right", ncol=2, frameon=False)

    for ax in axes:
        ax.set_xlim(1, len(target))

    fig.tight_layout()
    stem = piece[:80]
    png = OUT_DIR / f"{stem}_target_score_true_pred.png"
    pdf = OUT_DIR / f"{stem}_target_score_true_pred.pdf"
    fig.savefig(png, dpi=180)
    fig.savefig(pdf)
    plt.close(fig)
    return png, pdf


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pieces, labels, scores, preds, piece_df, aggregate, feature_cols = run_same_config_predictions()
    piece_df.to_csv(OUT_DIR / "piece_summary.csv", index=False)
    pd.DataFrame([aggregate]).to_csv(OUT_DIR / "aggregate_totals.csv", index=False)

    selected = [p for p in EXAMPLE_SHORT_NAMES if p in scores]
    if len(selected) < 4:
        selected.extend([p for p in pieces if p not in selected][: 4 - len(selected)])
    plot_rows = []
    for piece in selected[:4]:
        png, pdf = plot_piece(piece, labels, scores, preds)
        row = piece_df[piece_df["piece"] == piece].iloc[0].to_dict()
        row.update({"png": str(png), "pdf": str(pdf)})
        plot_rows.append(row)
    pd.DataFrame(plot_rows).to_csv(OUT_DIR / "plot_summary.csv", index=False)
    (OUT_DIR / "metadata.json").write_text(
        json.dumps(
            {
                "event_min": EVENT_MIN,
                "density_beats": DENSITY_BEATS,
                "min_distance": MIN_DISTANCE,
                "model": "baseline_cnn",
                "target_rule": "ATEPP L2+ max(weight_L * performer-consensus_L), original weights",
                "feature_columns": feature_cols,
                "note": "ATEPP20 exported label/features do not include performed beat timestamps, so plots show target and model score rather than BPM tempo curve.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("Aggregate:")
    print(pd.DataFrame([aggregate]).round(4).to_string(index=False))
    print("\nPlots:")
    print(pd.DataFrame(plot_rows)[["piece", "num_beats", "true", "pred", "match", "UP", "WR", "F1", "png"]].round(4).to_string(index=False))
    print(f"\nWrote {OUT_DIR}")


if __name__ == "__main__":
    main()
