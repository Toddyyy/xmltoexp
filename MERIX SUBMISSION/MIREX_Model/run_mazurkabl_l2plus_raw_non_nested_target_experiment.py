from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
BASE_SCRIPT = ROOT / "MERIX SUBMISSION" / "MIREX_Model" / "run_mazurkabl_l2plus_weighted_target_experiment.py"
BUILD_SCRIPT = ROOT / "MERIX SUBMISSION" / "MIREX_Model" / "build_mazurka_beat_npz_performer_levels.py"
BEAT_TIME_DIR = ROOT / "MazurkaBL-master" / "beat_time"
OUT_DIR = ROOT / "MERIX SUBMISSION" / "MIREX_Model" / "mazurkabl_l2plus_raw_non_nested_target_experiment"

STR_VEC = [3, 2, 2, 2, 2, 2]
LEVEL_WEIGHTS = {2: 0.28, 3: 0.46, 4: 0.64, 5: 0.82, 6: 1.00}
EVENT_MIN = 0.05

spec = importlib.util.spec_from_file_location("mazurkabl_base_raw_non_nested", BASE_SCRIPT)
base = importlib.util.module_from_spec(spec)
sys.modules["mazurkabl_base_raw_non_nested"] = base
assert spec.loader is not None
spec.loader.exec_module(base)

builder_spec = importlib.util.spec_from_file_location("mazurka_level_builder_raw_non_nested_train", BUILD_SCRIPT)
builder = importlib.util.module_from_spec(builder_spec)
sys.modules["mazurka_level_builder_raw_non_nested_train"] = builder
assert builder_spec.loader is not None
builder_spec.loader.exec_module(builder)


def normalize_piece_from_file(path: Path) -> str:
    return path.name.replace("beat_time.csv", "")


def load_raw_non_nested_labels(pieces: list[str]) -> tuple[dict[str, np.ndarray], dict[str, dict[int, np.ndarray]], pd.DataFrame]:
    labels: dict[str, np.ndarray] = {}
    components: dict[str, dict[int, np.ndarray]] = {}
    stats = []
    for piece in pieces:
        path = BEAT_TIME_DIR / f"{piece}beat_time.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        df, performer_cols = builder.load_beat_time(path)
        curves = builder.compute_tempo_curves(df, performer_cols, smooth_window=3, clip_max=600)
        n_beats = len(df)
        counts = {level: np.zeros(n_beats, dtype=np.float32) for level in range(1, 7)}
        for curve in curves.values():
            raw, _level_sets = builder.group_analysis_hierarchy(curve, STR_VEC, enforce_nested=False)
            for level in range(1, 7):
                current = raw[level - 1].astype(np.float32)
                if len(current) != n_beats:
                    raise RuntimeError(f"{piece} L{level}: raw length {len(current)} != {n_beats}")
                counts[level] += current
        components[piece] = {level: counts[level] / max(len(curves), 1) for level in range(1, 7)}
        weighted = [LEVEL_WEIGHTS[level] * components[piece][level] for level in LEVEL_WEIGHTS]
        labels[piece] = np.maximum.reduce(weighted).astype(np.float32)
        row = {
            "piece": piece,
            "num_beats": n_beats,
            "num_performers": len(curves),
            "target_sum": float(labels[piece].sum()),
            "target_ge_0p05": int(np.count_nonzero(labels[piece] >= EVENT_MIN)),
        }
        for level in range(1, 7):
            row[f"L{level}_mean_events_per_performer"] = float(counts[level].sum() / max(len(curves), 1))
            row[f"L{level}_consensus_ge_0p05"] = int(np.count_nonzero(components[piece][level] >= EVENT_MIN))
        stats.append(row)
    return labels, components, pd.DataFrame(stats)


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
        "mean_offset": float(np.mean(offsets)) if offsets else np.nan,
    }


def aggregate(items: list[dict]) -> dict:
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
        "matched_weight": matched_weight,
        "total_weight": total_weight,
    }


def choose_threshold(scores: dict[str, np.ndarray], labels: dict[str, np.ndarray], tolerance: int) -> tuple[float, dict]:
    best_th = 0.05
    best = None
    for th in np.linspace(0.05, 0.95, 37):
        items = []
        for piece, score in scores.items():
            pred = base.extract_events(score, threshold=float(th))
            items.append(weighted_metrics_from_events(pred, labels[piece], tolerance=tolerance))
        metrics = aggregate(items)
        key = (metrics["f1"], metrics["precision"], metrics["weighted_recall"], metrics["recall"])
        if best is None or key > (best["f1"], best["precision"], best["weighted_recall"], best["recall"]):
            best = metrics
            best_th = float(th)
    assert best is not None
    return best_th, best


def expected_count(labels: dict[str, np.ndarray], train_pieces: list[str], heldout_len: int) -> int:
    train_events = sum(int(np.count_nonzero(labels[p] >= EVENT_MIN)) for p in train_pieces)
    train_beats = sum(int(len(labels[p])) for p in train_pieces)
    return max(1, int(round(heldout_len * train_events / max(train_beats, 1))))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    old_pieces, _old_labels, _old_components = base.load_l2plus_weighted_labels()
    pieces = sorted(old_pieces)
    features, feature_cols = base.load_piece_features(pieces)
    labels, components, label_stats = load_raw_non_nested_labels(pieces)
    label_stats.to_csv(OUT_DIR / "label_stats.csv", index=False)

    folds = base.make_folds(pieces, n_folds=5, seed=42)
    rows = []
    piece_rows = []
    for fold_idx, val_pieces in enumerate(folds, start=1):
        train_pieces = [p for p in pieces if p not in set(val_pieces)]
        model, mean, std = base.train_one(features, labels, train_pieces, seed=13000 + fold_idx)
        train_scores = base.predict(model, features, train_pieces, mean, std)
        val_scores = base.predict(model, features, val_pieces, mean, std)
        threshold, train_metric = choose_threshold(train_scores, {p: labels[p] for p in train_pieces}, tolerance=1)

        th1_items = []
        th0_items = []
        den1_items = []
        den0_items = []
        for piece in val_pieces:
            target = labels[piece]
            threshold_pred = base.extract_events(val_scores[piece], threshold=threshold)
            density_pred = base.extract_top_density(val_scores[piece], expected_count(labels, train_pieces, len(target)))
            th1 = weighted_metrics_from_events(threshold_pred, target, tolerance=1)
            th0 = weighted_metrics_from_events(threshold_pred, target, tolerance=0)
            den1 = weighted_metrics_from_events(density_pred, target, tolerance=1)
            den0 = weighted_metrics_from_events(density_pred, target, tolerance=0)
            th1_items.append(th1)
            th0_items.append(th0)
            den1_items.append(den1)
            den0_items.append(den0)
            piece_rows.append(
                {
                    "fold": fold_idx,
                    "piece": piece,
                    "num_beats": len(target),
                    "true_events": th1["true_events"],
                    "threshold_pred_events": th1["pred_events"],
                    "threshold_f1_tol1": th1["f1"],
                    "threshold_weighted_recall_tol1": th1["weighted_recall"],
                    "density_pred_events": den1["pred_events"],
                    "density_f1_tol1": den1["f1"],
                    "density_weighted_recall_tol1": den1["weighted_recall"],
                }
            )

        th1 = aggregate(th1_items)
        th0 = aggregate(th0_items)
        den1 = aggregate(den1_items)
        den0 = aggregate(den0_items)
        rows.append(
            {
                "fold": fold_idx,
                "train_pieces": " ".join(train_pieces),
                "val_pieces": " ".join(val_pieces),
                "threshold_from_train_tol1": threshold,
                "train_f1_tol1": train_metric["f1"],
                "threshold_precision_tol1": th1["precision"],
                "threshold_recall_tol1": th1["recall"],
                "threshold_f1_tol1": th1["f1"],
                "threshold_weighted_recall_tol1": th1["weighted_recall"],
                "threshold_matches_tol1": th1["matches"],
                "threshold_precision_tol0": th0["precision"],
                "threshold_recall_tol0": th0["recall"],
                "threshold_f1_tol0": th0["f1"],
                "threshold_weighted_recall_tol0": th0["weighted_recall"],
                "density_precision_tol1": den1["precision"],
                "density_recall_tol1": den1["recall"],
                "density_f1_tol1": den1["f1"],
                "density_weighted_recall_tol1": den1["weighted_recall"],
                "density_matches_tol1": den1["matches"],
                "density_precision_tol0": den0["precision"],
                "density_recall_tol0": den0["recall"],
                "density_f1_tol0": den0["f1"],
                "density_weighted_recall_tol0": den0["weighted_recall"],
                "threshold_pred_events": th1["pred_events"],
                "density_pred_events": den1["pred_events"],
                "true_events": th1["true_events"],
                "threshold_matched_weight_tol1": th1["matched_weight"],
                "density_matched_weight_tol1": den1["matched_weight"],
                "total_weight_tol1": th1["total_weight"],
            }
        )
        print(
            f"fold {fold_idx}: threshold P/R/F1/WR="
            f"{th1['precision']:.3f}/{th1['recall']:.3f}/{th1['f1']:.3f}/{th1['weighted_recall']:.3f}; "
            f"density P/R/F1/WR={den1['precision']:.3f}/{den1['recall']:.3f}/{den1['f1']:.3f}/{den1['weighted_recall']:.3f}",
            flush=True,
        )

    summary = pd.DataFrame(rows)
    summary.to_csv(OUT_DIR / "fold_summary.csv", index=False)
    pd.DataFrame(piece_rows).to_csv(OUT_DIR / "piece_summary.csv", index=False)
    metric_cols = [c for c in summary.columns if c not in {"train_pieces", "val_pieces"}]
    summary[metric_cols].mean(numeric_only=True).to_frame("mean").to_csv(OUT_DIR / "fold_mean.csv")
    totals = {
        "threshold_pred_events": int(summary["threshold_pred_events"].sum()),
        "density_pred_events": int(summary["density_pred_events"].sum()),
        "true_events": int(summary["true_events"].sum()),
        "threshold_matches_tol1": int(summary["threshold_matches_tol1"].sum()),
        "density_matches_tol1": int(summary["density_matches_tol1"].sum()),
        "threshold_precision_tol1": float(summary["threshold_matches_tol1"].sum() / max(summary["threshold_pred_events"].sum(), 1)),
        "density_precision_tol1": float(summary["density_matches_tol1"].sum() / max(summary["density_pred_events"].sum(), 1)),
        "threshold_recall_tol1": float(summary["threshold_matches_tol1"].sum() / max(summary["true_events"].sum(), 1)),
        "density_recall_tol1": float(summary["density_matches_tol1"].sum() / max(summary["true_events"].sum(), 1)),
        "threshold_weighted_recall_tol1": float(summary["threshold_matched_weight_tol1"].sum() / max(summary["total_weight_tol1"].sum(), 1e-12)),
        "density_weighted_recall_tol1": float(summary["density_matched_weight_tol1"].sum() / max(summary["total_weight_tol1"].sum(), 1e-12)),
    }
    for prefix in ("threshold", "density"):
        p = totals[f"{prefix}_precision_tol1"]
        r = totals[f"{prefix}_recall_tol1"]
        totals[f"{prefix}_f1_tol1"] = float(2 * p * r / (p + r)) if p + r else 0.0
    pd.Series(totals).to_frame("total").to_csv(OUT_DIR / "aggregate_totals.csv")

    metadata = {
        "beat_time_dir": str(BEAT_TIME_DIR),
        "beat_table": str(base.BEAT_TABLE),
        "feature_columns": feature_cols,
        "pieces": pieces,
        "folds": folds,
        "str_vec": STR_VEC,
        "enforce_nested": False,
        "level_weights": LEVEL_WEIGHTS,
        "target_rule": "max(weight_L * mean_performer_raw_non_nested_boundary_L_frequency) for L2-L6",
        "event_min": EVENT_MIN,
        "min_distance": base.MIN_DISTANCE,
    }
    (OUT_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("\nAggregate totals:")
    print(pd.Series(totals).round(4).to_string())
    print(f"\nWrote {OUT_DIR}")


if __name__ == "__main__":
    main()
