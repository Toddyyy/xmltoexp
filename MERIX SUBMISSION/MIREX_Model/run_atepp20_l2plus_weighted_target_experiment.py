from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
BASE_SCRIPT = ROOT / "MERIX SUBMISSION" / "MIREX_Model" / "run_mazurkabl_l2plus_weighted_target_experiment.py"
BEAT_TABLE = ROOT / "MERIX SUBMISSION" / "MIREX_Model_meter_auto" / "outputs" / "atepp_top20_mixed_by_segment_nonan_beat_table.csv.gz"
LABEL_DIR = ROOT / "MERIX SUBMISSION" / "MIREX_Model_meter_auto" / "beat_data_atepp_top20_mixed_by_segment_performer_levels"
OUT_DIR = ROOT / "MERIX SUBMISSION" / "MIREX_Model" / "atepp20_l2plus_weighted_target_experiment"

LEVEL_WEIGHTS = {2: 0.28, 3: 0.46, 4: 0.64, 5: 0.82, 6: 1.00}
EVENT_MIN = 0.05

spec = importlib.util.spec_from_file_location("mazurkabl_quick_base", BASE_SCRIPT)
base = importlib.util.module_from_spec(spec)
sys.modules["mazurkabl_quick_base"] = base
assert spec.loader is not None
spec.loader.exec_module(base)


def piece_id_from_npz(path: Path) -> str:
    match = re.match(r"(.+)_\d+_L[1-6]\.npz$", path.name)
    if not match:
        raise ValueError(f"Cannot parse ATEPP piece id from {path.name}")
    return match.group(1)


def load_piece_features(pieces: list[str]) -> tuple[dict[str, np.ndarray], list[str]]:
    df = pd.read_csv(BEAT_TABLE)
    df = df[df["piece_id"].isin(pieces)].copy()
    if df.empty:
        raise RuntimeError(f"No selected pieces found in {BEAT_TABLE}")
    feature_cols = [
        col
        for col in df.columns
        if base.is_feature_column(col)
        and col not in {"protocol_split", "local_time_signature", "meter_group", "segment_id"}
        and pd.api.types.is_numeric_dtype(df[col])
    ]
    out: dict[str, np.ndarray] = {}
    for piece in pieces:
        piece_df = df[df["piece_id"] == piece].sort_values(["beat_idx", "sample_id"])
        one = piece_df.groupby("beat_idx", as_index=False).first().sort_values("beat_idx")
        expected = np.arange(len(one), dtype=int)
        got = one["beat_idx"].to_numpy(dtype=int)
        if not np.array_equal(got, expected):
            raise RuntimeError(f"{piece}: non-contiguous beat_idx in beat table")
        out[piece] = one[feature_cols].to_numpy(dtype=np.float32)
    return out, feature_cols


def load_level_frequency(piece: str, level: int) -> np.ndarray:
    files = sorted(LABEL_DIR.glob(f"{piece}_*_L{level}.npz"))
    if not files:
        raise FileNotFoundError(f"No ATEPP label files for {piece} L{level}")
    arrays = [np.load(path, allow_pickle=True)["boundary_probs"].astype(np.float32) for path in files]
    lengths = {len(a) for a in arrays}
    if len(lengths) != 1:
        raise RuntimeError(f"{piece} L{level}: inconsistent label lengths {sorted(lengths)}")
    return np.mean(np.stack(arrays, axis=0), axis=0).astype(np.float32)


def load_l2plus_weighted_labels() -> tuple[list[str], dict[str, np.ndarray], dict[str, dict[int, np.ndarray]]]:
    pieces = sorted({piece_id_from_npz(path) for path in LABEL_DIR.glob("*_L2.npz")})
    labels: dict[str, np.ndarray] = {}
    components: dict[str, dict[int, np.ndarray]] = {}
    for piece in pieces:
        weighted = []
        components[piece] = {}
        for level, weight in LEVEL_WEIGHTS.items():
            freq = load_level_frequency(piece, level)
            components[piece][level] = freq
            weighted.append(float(weight) * freq)
        labels[piece] = np.maximum.reduce(weighted).astype(np.float32)
    return pieces, labels, components


def weighted_metrics_from_events(pred: np.ndarray, target: np.ndarray, tolerance: int) -> dict:
    true = np.flatnonzero(target >= EVENT_MIN).astype(np.int32)
    matches, offsets = base.match_events(pred, true, tolerance=tolerance)
    used = set()
    matched_true = []
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
            matched_true.append(int(true[best]))
    matched_weight = float(np.sum(target[matched_true])) if matched_true else 0.0
    total_weight = float(np.sum(target[true])) if true.size else 0.0
    precision = matches / len(pred) if len(pred) else 0.0
    recall = matches / len(true) if len(true) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "pred_events": int(len(pred)),
        "true_events": int(len(true)),
        "matches": int(matches),
        "weighted_recall": float(matched_weight / total_weight) if total_weight > 0 else 0.0,
        "matched_weight": matched_weight,
        "total_weight": total_weight,
        "mean_offset": float(np.mean(offsets)) if offsets else None,
    }


def aggregate_weighted(items: list[dict]) -> dict:
    pred = sum(int(x["pred_events"]) for x in items)
    true = sum(int(x["true_events"]) for x in items)
    matches = sum(int(x["matches"]) for x in items)
    matched_weight = sum(float(x["matched_weight"]) for x in items)
    total_weight = sum(float(x["total_weight"]) for x in items)
    precision = matches / pred if pred else 0.0
    recall = matches / true if true else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "pred_events": int(pred),
        "true_events": int(true),
        "matches": int(matches),
        "weighted_recall": float(matched_weight / total_weight) if total_weight > 0 else 0.0,
        "matched_weight": float(matched_weight),
        "total_weight": float(total_weight),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pieces, labels, components = load_l2plus_weighted_labels()
    features, feature_cols = load_piece_features(pieces)
    missing = sorted(set(pieces) - set(features))
    if missing:
        raise RuntimeError(f"Missing feature rows for pieces: {missing}")
    length_mismatch = [
        (piece, len(labels[piece]), len(features[piece]))
        for piece in pieces
        if len(labels[piece]) != len(features[piece])
    ]
    if length_mismatch:
        raise RuntimeError(f"Label/feature length mismatch: {length_mismatch[:10]}")

    label_stats = []
    for piece in pieces:
        label_stats.append(
            {
                "piece": piece,
                "num_beats": len(labels[piece]),
                "target_sum": float(labels[piece].sum()),
                "event_count_target_ge_0p05": int(np.count_nonzero(labels[piece] >= EVENT_MIN)),
                **{f"L{level}_support": int(np.count_nonzero(components[piece][level] > 0)) for level in LEVEL_WEIGHTS},
            }
        )
    pd.DataFrame(label_stats).to_csv(OUT_DIR / "label_stats.csv", index=False)

    folds = base.make_folds(pieces, n_folds=5, seed=42)
    rows = []
    piece_rows = []
    for fold_idx, val_pieces in enumerate(folds, start=1):
        train_pieces = [p for p in pieces if p not in set(val_pieces)]
        model, mean, std = base.train_one(features, labels, train_pieces, seed=8200 + fold_idx)
        train_scores = base.predict(model, features, train_pieces, mean, std)
        val_scores = base.predict(model, features, val_pieces, mean, std)
        threshold, train_metric = base.choose_threshold(train_scores, {p: labels[p] for p in train_pieces}, tolerance=1)

        th_items_1 = []
        th_items_0 = []
        den_items_1 = []
        den_items_0 = []
        for piece in val_pieces:
            true = np.flatnonzero(labels[piece] >= EVENT_MIN).astype(np.int32)
            threshold_pred = base.extract_events(val_scores[piece], threshold=threshold)
            expected = base.expected_count_from_train_density(labels, train_pieces, len(labels[piece]))
            density_pred = base.extract_top_density(val_scores[piece], expected)
            th1 = weighted_metrics_from_events(threshold_pred, labels[piece], tolerance=1)
            th0 = weighted_metrics_from_events(threshold_pred, labels[piece], tolerance=0)
            den1 = weighted_metrics_from_events(density_pred, labels[piece], tolerance=1)
            den0 = weighted_metrics_from_events(density_pred, labels[piece], tolerance=0)
            th_items_1.append(th1)
            th_items_0.append(th0)
            den_items_1.append(den1)
            den_items_0.append(den0)
            piece_rows.append(
                {
                    "fold": fold_idx,
                    "piece": piece,
                    "num_beats": len(labels[piece]),
                    "true_events": len(true),
                    "threshold_pred_events": len(threshold_pred),
                    "threshold_precision_tol1": th1["precision"],
                    "threshold_recall_tol1": th1["recall"],
                    "threshold_f1_tol1": th1["f1"],
                    "threshold_weighted_recall_tol1": th1["weighted_recall"],
                    "density_expected_events": expected,
                    "density_pred_events": len(density_pred),
                    "density_precision_tol1": den1["precision"],
                    "density_recall_tol1": den1["recall"],
                    "density_f1_tol1": den1["f1"],
                    "density_weighted_recall_tol1": den1["weighted_recall"],
                }
            )
            pd.DataFrame(
                {
                    "beat_idx": np.arange(len(val_scores[piece])),
                    "score": val_scores[piece],
                    "target_l2plus_weighted": labels[piece],
                    "target_event_ge_0p05": (labels[piece] >= EVENT_MIN).astype(np.int8),
                    "threshold_pred": np.isin(np.arange(len(val_scores[piece])), threshold_pred).astype(np.int8),
                    "density_pred": np.isin(np.arange(len(val_scores[piece])), density_pred).astype(np.int8),
                }
            ).to_csv(OUT_DIR / f"fold{fold_idx}_{piece}_val_predictions.csv", index=False)

        th_tol1 = aggregate_weighted(th_items_1)
        th_tol0 = aggregate_weighted(th_items_0)
        den_tol1 = aggregate_weighted(den_items_1)
        den_tol0 = aggregate_weighted(den_items_0)
        rows.append(
            {
                "fold": fold_idx,
                "train_pieces": " ".join(train_pieces),
                "val_pieces": " ".join(val_pieces),
                "threshold_from_train_tol1": threshold,
                "train_f1_tol1": train_metric.f1,
                "threshold_precision_tol1": th_tol1["precision"],
                "threshold_recall_tol1": th_tol1["recall"],
                "threshold_f1_tol1": th_tol1["f1"],
                "threshold_weighted_recall_tol1": th_tol1["weighted_recall"],
                "threshold_precision_tol0": th_tol0["precision"],
                "threshold_recall_tol0": th_tol0["recall"],
                "threshold_f1_tol0": th_tol0["f1"],
                "threshold_weighted_recall_tol0": th_tol0["weighted_recall"],
                "density_precision_tol1": den_tol1["precision"],
                "density_recall_tol1": den_tol1["recall"],
                "density_f1_tol1": den_tol1["f1"],
                "density_weighted_recall_tol1": den_tol1["weighted_recall"],
                "density_precision_tol0": den_tol0["precision"],
                "density_recall_tol0": den_tol0["recall"],
                "density_f1_tol0": den_tol0["f1"],
                "density_weighted_recall_tol0": den_tol0["weighted_recall"],
                "density_pred_events": den_tol1["pred_events"],
                "threshold_pred_events": th_tol1["pred_events"],
                "true_events": den_tol1["true_events"],
            }
        )
        print(
            f"fold {fold_idx}: threshold tol1 P/R/F1/WR="
            f"{th_tol1['precision']:.3f}/{th_tol1['recall']:.3f}/{th_tol1['f1']:.3f}/{th_tol1['weighted_recall']:.3f}; "
            f"density tol1 P/R/F1/WR="
            f"{den_tol1['precision']:.3f}/{den_tol1['recall']:.3f}/{den_tol1['f1']:.3f}/{den_tol1['weighted_recall']:.3f}"
        )

    summary = pd.DataFrame(rows)
    piece_summary = pd.DataFrame(piece_rows)
    summary.to_csv(OUT_DIR / "fold_summary.csv", index=False)
    piece_summary.to_csv(OUT_DIR / "piece_summary.csv", index=False)
    metric_cols = [
        "threshold_precision_tol1",
        "threshold_recall_tol1",
        "threshold_f1_tol1",
        "threshold_weighted_recall_tol1",
        "threshold_precision_tol0",
        "threshold_recall_tol0",
        "threshold_f1_tol0",
        "threshold_weighted_recall_tol0",
        "density_precision_tol1",
        "density_recall_tol1",
        "density_f1_tol1",
        "density_weighted_recall_tol1",
        "density_precision_tol0",
        "density_recall_tol0",
        "density_f1_tol0",
        "density_weighted_recall_tol0",
        "density_pred_events",
        "threshold_pred_events",
        "true_events",
    ]
    mean = summary[metric_cols].mean()
    mean.to_frame("mean").to_csv(OUT_DIR / "fold_mean.csv")
    metadata = {
        "beat_table": str(BEAT_TABLE),
        "label_dir": str(LABEL_DIR),
        "feature_columns": feature_cols,
        "pieces": pieces,
        "folds": folds,
        "level_weights": LEVEL_WEIGHTS,
        "target_rule": "max(weight_L * mean_performer_boundary_L_frequency) for L2-L6",
        "event_min": EVENT_MIN,
        "min_distance": base.MIN_DISTANCE,
    }
    (OUT_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("\nMean:")
    print(mean.round(4).to_string())
    print(f"\nWrote {OUT_DIR}")


if __name__ == "__main__":
    main()
