from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
MIREX = ROOT / "MERIX SUBMISSION" / "MIREX_Model"
COMBINED_SCRIPT = MIREX / "run_combined_mazurkabl_asap40_l2plus_sqrtmass.py"
OUT_DIR = MIREX / "combined_mazurkabl_asap40_l2plus_sqrtmass_baseline_branchwise"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


cmb = load_module("combined_per_piece_source", COMBINED_SCRIPT)
bw = cmb.bw
runner = cmb.runner
base = cmb.base


def pack_piece(prefix: str, pred: np.ndarray, true: np.ndarray, labels: np.ndarray) -> dict:
    metric = base.metrics_from_events(pred, true, tolerance=1)
    matched_true = runner.match_true_indices(pred, true, tolerance=1)
    wr_num = float(labels[matched_true].sum()) if matched_true else 0.0
    wr_den = float(labels[true].sum()) if len(true) else 0.0
    up = metric.matches / metric.pred_events if metric.pred_events else 0.0
    recall = metric.matches / metric.true_events if metric.true_events else 0.0
    wr = wr_num / wr_den if wr_den else 0.0
    f1 = 2 * up * recall / (up + recall) if up + recall else 0.0
    return {
        f"{prefix}_pred_events": int(metric.pred_events),
        f"{prefix}_true_events": int(metric.true_events),
        f"{prefix}_matches_tol1": int(metric.matches),
        f"{prefix}_UP": float(up),
        f"{prefix}_recall": float(recall),
        f"{prefix}_WR": float(wr),
        f"{prefix}_f1": float(f1),
    }


def evaluate_per_piece(setting: str, cfg, pieces, labels, base_features, rich_features, folds, device):
    rows = []
    for fold_idx, val_pieces in enumerate(folds, start=1):
        train_pieces = [p for p in pieces if p not in set(val_pieces)]
        if setting == "baseline_cnn":
            features, model_kind, base_dim, rich_dim = runner.build_setting_features(
                setting,
                base_features,
                rich_features,
                pieces,
                seed=100000 + fold_idx,
            )
            model, mean, std = runner.train_one(
                cfg,
                model_kind,
                features[pieces[0]].shape[1],
                base_dim,
                rich_dim,
                features,
                labels,
                train_pieces,
                seed=9400 + fold_idx,
                device=device,
            )
            train_scores = runner.predict(model, features, train_pieces, mean, std, device)
            val_scores = runner.predict(model, features, val_pieces, mean, std, device)
        elif setting == "handcrafted_plus_branchwise":
            features, base_dim = bw.build_setting_features(
                setting,
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
            train_scores = bw.predict(model, features, train_pieces, mean, std, device)
            val_scores = bw.predict(model, features, val_pieces, mean, std, device)
        else:
            raise ValueError(setting)

        threshold, train_metric = base.choose_threshold(
            train_scores,
            {p: labels[p] for p in train_pieces},
            tolerance=1,
        )
        for piece in val_pieces:
            true = np.flatnonzero(labels[piece] >= cmb.EVENT_MIN).astype(np.int32)
            pred_th = base.extract_events(val_scores[piece], threshold=threshold)
            expected = base.expected_count_from_train_density(labels, train_pieces, len(labels[piece]))
            pred_den = base.extract_top_density(val_scores[piece], expected)
            dataset, raw_piece = piece.split("__", 1)
            row = {
                "setting": setting,
                "fold": int(fold_idx),
                "dataset": dataset,
                "piece": raw_piece,
                "piece_key": piece,
                "num_beats": int(len(labels[piece])),
                "threshold": float(threshold),
                "train_f1_tol1": float(train_metric.f1),
                "density_expected_events": int(expected),
            }
            row.update(pack_piece("threshold", pred_th, true, labels[piece]))
            row.update(pack_piece("density", pred_den, true, labels[piece]))
            rows.append(row)
        print(f"{setting} fold {fold_idx}: wrote {len(val_pieces)} piece rows", flush=True)
    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = runner.load_config()

    mz_pieces_raw, mz_labels_raw, _, _ = cmb.mz.load_sqrtmass_l2plus_labels()
    mz_base_raw, mz_cols = runner.load_piece_features(mz_pieces_raw, cfg)
    mz_rich_raw = runner.load_rich_features(mz_pieces_raw)

    asap_pieces_raw, asap_labels_raw, _, _ = cmb.asap.load_sqrtmass_labels()
    asap_base_raw, asap_cols = cmb.asap.load_piece_features(asap_pieces_raw, cfg)
    asap_rich_raw = cmb.asap.load_rich_features(asap_pieces_raw)

    common_cols = [col for col in mz_cols if col in set(asap_cols)]
    mz_base_common = cmb.subset_features(mz_base_raw, mz_cols, common_cols)
    asap_base_common = cmb.subset_features(asap_base_raw, asap_cols, common_cols)

    labels = {**cmb.prefix_dict("mazurka", mz_labels_raw), **cmb.prefix_dict("asap40", asap_labels_raw)}
    base_features = {
        **cmb.prefix_dict("mazurka", mz_base_common),
        **cmb.prefix_dict("asap40", asap_base_common),
    }
    rich_features = {
        **cmb.prefix_dict("mazurka", mz_rich_raw),
        **cmb.prefix_dict("asap40", asap_rich_raw),
    }
    pieces = sorted(labels)
    folds = base.make_folds(pieces, n_folds=5, seed=42)
    device = runner.resolve_device()

    original_event_min = base.EVENT_MIN
    original_min_distance = base.MIN_DISTANCE
    original_expected_count = base.expected_count_from_train_density
    original_extract_top_density = base.extract_top_density
    base.EVENT_MIN = float(cmb.EVENT_MIN)
    base.MIN_DISTANCE = int(cmb.DENSITY_MIN_DISTANCE)
    base.expected_count_from_train_density = cmb.fixed_density

    def top_density_min_distance(scores, expected_count, min_distance=None):
        return original_extract_top_density(scores, expected_count, min_distance=max(int(cmb.DENSITY_MIN_DISTANCE), 1))

    base.extract_top_density = top_density_min_distance
    try:
        frames = []
        for setting in ["baseline_cnn", "handcrafted_plus_branchwise"]:
            frames.append(evaluate_per_piece(setting, cfg, pieces, labels, base_features, rich_features, folds, device))
        out = pd.concat(frames, ignore_index=True)
    finally:
        base.EVENT_MIN = original_event_min
        base.MIN_DISTANCE = original_min_distance
        base.expected_count_from_train_density = original_expected_count
        base.extract_top_density = original_extract_top_density

    out.to_csv(OUT_DIR / "per_piece_validation_summary.csv", index=False)
    numeric = [
        "threshold_pred_events",
        "threshold_true_events",
        "threshold_matches_tol1",
        "threshold_UP",
        "threshold_recall",
        "threshold_WR",
        "threshold_f1",
        "density_pred_events",
        "density_true_events",
        "density_matches_tol1",
        "density_UP",
        "density_recall",
        "density_WR",
        "density_f1",
    ]
    dataset_summary = out.groupby(["setting", "dataset"], as_index=False)[numeric[:3] + numeric[7:10]].sum()
    dataset_summary.to_csv(OUT_DIR / "per_piece_validation_dataset_totals.csv", index=False)
    print(f"wrote {OUT_DIR / 'per_piece_validation_summary.csv'}")


if __name__ == "__main__":
    main()
