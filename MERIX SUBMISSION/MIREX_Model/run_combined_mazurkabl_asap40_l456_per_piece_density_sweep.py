from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
MIREX = ROOT / "MERIX SUBMISSION" / "MIREX_Model"
SOURCE_SCRIPT = MIREX / "run_combined_mazurkabl_asap40_l23_l456_targets.py"
OUT_DIR = MIREX / "combined_mazurkabl_asap40_l456_per_piece_density_sweep"

EVENT_MIN = 0.01
TOLERANCE = 1
DENSITY_BEATS_LIST = [6.0, 12.0, 24.0]
L456_LEVELS = [4, 5, 6]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


src = load_module("combined_l456_piece_source", SOURCE_SCRIPT)
cmb = src.cmb
bw = src.bw
runner = src.runner
base = src.base


def pack_piece(prefix: str, pred: np.ndarray, true: np.ndarray, labels: np.ndarray) -> dict:
    metric = base.metrics_from_events(pred, true, tolerance=TOLERANCE)
    matched_true = runner.match_true_indices(pred, true, tolerance=TOLERANCE)
    wr_num = float(labels[matched_true].sum()) if matched_true else 0.0
    wr_den = float(labels[true].sum()) if len(true) else 0.0
    up = metric.matches / metric.pred_events if metric.pred_events else 0.0
    recall = metric.matches / metric.true_events if metric.true_events else 0.0
    wr = wr_num / wr_den if wr_den else 0.0
    return {
        f"{prefix}_pred": int(metric.pred_events),
        f"{prefix}_true": int(metric.true_events),
        f"{prefix}_match": int(metric.matches),
        f"{prefix}_UP": float(up),
        f"{prefix}_recall": float(recall),
        f"{prefix}_WR": float(wr),
    }


def density_count(num_beats: int, density_beats: float) -> int:
    return max(1, int(round(float(num_beats) / max(float(density_beats), 1e-9))))


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
            val_scores = bw.predict(model, features, val_pieces, mean, std, device)
        else:
            raise ValueError(setting)

        for piece in val_pieces:
            true = np.flatnonzero(labels[piece] >= EVENT_MIN).astype(np.int32)
            dataset, raw_piece = piece.split("__", 1)
            row_base = {
                "setting": setting,
                "fold": int(fold_idx),
                "dataset": dataset,
                "piece": raw_piece,
                "piece_key": piece,
                "num_beats": int(len(labels[piece])),
            }
            for density_beats in DENSITY_BEATS_LIST:
                expected = density_count(len(labels[piece]), density_beats)
                pred = base.extract_top_density(val_scores[piece], expected, min_distance=1)
                row = {
                    **row_base,
                    "density_beats": float(density_beats),
                    "density_expected_events": int(expected),
                }
                row.update(pack_piece("density", pred, true, labels[piece]))
                rows.append(row)
        print(f"{setting} fold {fold_idx}: wrote {len(val_pieces) * len(DENSITY_BEATS_LIST)} rows", flush=True)
    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = runner.load_config()

    mz_pieces_raw, _mz_labels_raw, mz_components_raw, _mz_argmax = cmb.mz.load_sqrtmass_l2plus_labels()
    mz_base_raw, mz_cols = runner.load_piece_features(mz_pieces_raw, cfg)
    mz_rich_raw = runner.load_rich_features(mz_pieces_raw)

    asap_pieces_raw, _asap_labels_raw, asap_components_raw, _asap_argmax = cmb.asap.load_sqrtmass_labels()
    asap_base_raw, asap_cols = cmb.asap.load_piece_features(asap_pieces_raw, cfg)
    asap_rich_raw = cmb.asap.load_rich_features(asap_pieces_raw)

    common_cols = [col for col in mz_cols if col in set(asap_cols)]
    mz_base_common = cmb.subset_features(mz_base_raw, mz_cols, common_cols)
    asap_base_common = cmb.subset_features(asap_base_raw, asap_cols, common_cols)

    mz_group = src.build_group_labels(mz_components_raw, L456_LEVELS)
    asap_group = src.build_group_labels(asap_components_raw, L456_LEVELS)
    labels = {**cmb.prefix_dict("mazurka", mz_group), **cmb.prefix_dict("asap40", asap_group)}
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
    base.EVENT_MIN = float(EVENT_MIN)
    base.MIN_DISTANCE = 1
    try:
        frames = []
        for setting in ["baseline_cnn", "handcrafted_plus_branchwise"]:
            frames.append(evaluate_per_piece(setting, cfg, pieces, labels, base_features, rich_features, folds, device))
            pd.concat(frames, ignore_index=True).to_csv(OUT_DIR / "per_piece_validation_summary.csv", index=False)
    finally:
        base.EVENT_MIN = original_event_min
        base.MIN_DISTANCE = original_min_distance

    out = pd.concat(frames, ignore_index=True)
    out.to_csv(OUT_DIR / "per_piece_validation_summary.csv", index=False)
    compact = out[
        [
            "density_beats",
            "setting",
            "fold",
            "dataset",
            "piece",
            "num_beats",
            "density_pred",
            "density_true",
            "density_match",
            "density_UP",
            "density_WR",
        ]
    ].sort_values(["density_beats", "setting", "density_UP", "density_WR"], ascending=[True, True, False, False])
    compact.to_csv(OUT_DIR / "per_piece_validation_compact.csv", index=False)
    print(f"wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
