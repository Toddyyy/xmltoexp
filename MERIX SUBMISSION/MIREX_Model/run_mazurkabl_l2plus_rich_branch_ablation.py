from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
MIREX = ROOT / "MERIX SUBMISSION" / "MIREX_Model"
RICH_RUN_SCRIPT = MIREX / "run_mazurkabl_l2plus_rich_midibert_mlp_cnn.py"
OUT_DIR = MIREX / "mazurkabl_l2plus_rich_branch_ablation"
HIDDEN_DIM = 768

BRANCHES = {
    "onset_mean": (0, HIDDEN_DIM),
    "sustain_mean": (HIDDEN_DIM, HIDDEN_DIM * 2),
    "all_mean": (HIDDEN_DIM * 2, HIDDEN_DIM * 3),
    "highest_note": (HIDDEN_DIM * 3, HIDDEN_DIM * 4),
    "lowest_note": (HIDDEN_DIM * 4, HIDDEN_DIM * 5),
    "duration_weighted": (HIDDEN_DIM * 5, HIDDEN_DIM * 6),
}


def load_rich_runner():
    spec = importlib.util.spec_from_file_location("mazurka_rich_runner", RICH_RUN_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {RICH_RUN_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["mazurka_rich_runner"] = module
    spec.loader.exec_module(module)
    return module


runner = load_rich_runner()


def slice_branch_features(rich_features: dict[str, np.ndarray], branch: str) -> dict[str, np.ndarray]:
    start, end = BRANCHES[branch]
    return {piece: values[:, start:end].astype(np.float32) for piece, values in rich_features.items()}


def run_branch(branch: str, cfg, pieces, labels, rich_features, folds, device):
    features = slice_branch_features(rich_features, branch)
    rows = []
    th_total = runner.Result()
    den_total = runner.Result()
    for fold_idx, val_pieces in enumerate(folds, start=1):
        train_pieces = [p for p in pieces if p not in set(val_pieces)]
        model, mean, std = runner.train_one(
            cfg,
            "mlp_cnn",
            HIDDEN_DIM,
            0,
            HIDDEN_DIM,
            features,
            labels,
            train_pieces,
            seed=9700 + fold_idx,
            device=device,
        )
        train_scores = runner.predict(model, features, train_pieces, mean, std, device)
        val_scores = runner.predict(model, features, val_pieces, mean, std, device)
        threshold, train_metric = runner.base.choose_threshold(
            train_scores,
            {p: labels[p] for p in train_pieces},
            tolerance=1,
        )
        th, den = runner.evaluate_fold(labels, val_scores, val_pieces, threshold, train_pieces)
        for attr in ("pred", "true", "match", "wr_num", "wr_den"):
            setattr(th_total, attr, getattr(th_total, attr) + getattr(th, attr))
            setattr(den_total, attr, getattr(den_total, attr) + getattr(den, attr))
        row = {
            "branch": branch,
            "fold": fold_idx,
            "threshold": float(threshold),
            "train_f1_tol1": float(train_metric.f1),
            **runner.pack("threshold", th),
            **runner.pack("density", den),
        }
        rows.append(row)
        print(
            f"{branch} fold {fold_idx}: "
            f"threshold UP/WR/F1={row['threshold_UP']:.3f}/{row['threshold_WR']:.3f}/{row['threshold_f1']:.3f}; "
            f"density UP/WR/F1={row['density_UP']:.3f}/{row['density_WR']:.3f}/{row['density_f1']:.3f}"
        )
    return pd.DataFrame(rows), {"branch": branch, **runner.pack("threshold", th_total), **runner.pack("density", den_total)}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = runner.load_config()
    pieces, labels, _ = runner.base.load_l2plus_weighted_labels()
    rich_features = runner.load_rich_features(pieces)
    missing = sorted(set(pieces) - set(rich_features))
    if missing:
        raise RuntimeError(f"Missing rich features: {missing}")
    bad = [
        (piece, len(labels[piece]), len(rich_features[piece]))
        for piece in pieces
        if len(labels[piece]) != len(rich_features[piece])
    ]
    if bad:
        raise RuntimeError(f"Label/rich length mismatch: {bad}")
    pieces = sorted(pieces)
    folds = runner.base.make_folds(pieces, n_folds=5, seed=42)
    device = runner.resolve_device()
    print(f"device={device}; branches={list(BRANCHES)}; branch_dim={HIDDEN_DIM}; beat_mlp_dim={runner.BEAT_MLP_DIM}")

    fold_frames = []
    aggregates = []
    for branch in BRANCHES:
        fold_df, aggregate = run_branch(branch, cfg, pieces, labels, rich_features, folds, device)
        fold_frames.append(fold_df)
        aggregates.append(aggregate)

    pd.concat(fold_frames, ignore_index=True).to_csv(OUT_DIR / "fold_summary.csv", index=False)
    agg = pd.DataFrame(aggregates)
    agg.to_csv(OUT_DIR / "aggregate_totals.csv", index=False)
    (OUT_DIR / "metadata.json").write_text(
        json.dumps(
            {
                "rich_dir": str(runner.RICH_DIR),
                "source_script": str(RICH_RUN_SCRIPT),
                "branch_dim": HIDDEN_DIM,
                "beat_mlp_dim": runner.BEAT_MLP_DIM,
                "branches": BRANCHES,
                "pieces": pieces,
                "folds": folds,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("\nAggregate:")
    print(agg.round(4).to_string(index=False))
    print(f"\nWrote {OUT_DIR}")


if __name__ == "__main__":
    main()
