from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
BASE_SCRIPT = ROOT / "MERIX SUBMISSION" / "MIREX_Model" / "run_mazurkabl_l2plus_weighted_target_experiment.py"
OUT_DIR = ROOT / "MERIX SUBMISSION" / "MIREX_Model" / "mazurkabl_l1_l5_cascade_experiment"
EVENT_MIN = 0.05

spec = importlib.util.spec_from_file_location("mazurkabl_quick_base_cascade", BASE_SCRIPT)
base = importlib.util.module_from_spec(spec)
sys.modules["mazurkabl_quick_base_cascade"] = base
assert spec.loader is not None
spec.loader.exec_module(base)

LEVEL_NAMES = ["L1", "L2", "L3", "L4", "L5plus6"]


def load_level_labels() -> tuple[list[str], dict[str, dict[str, np.ndarray]]]:
    pieces, _weighted, components = base.load_l2plus_weighted_labels()
    labels: dict[str, dict[str, np.ndarray]] = {}
    for piece in pieces:
        labels[piece] = {
            "L1": base.load_level_frequency(piece, 1),
            "L2": components[piece][2],
            "L3": components[piece][3],
            "L4": components[piece][4],
            "L5plus6": np.maximum(components[piece][5], components[piece][6]).astype(np.float32),
        }
    return pieces, labels


def labels_for_level(labels: dict[str, dict[str, np.ndarray]], level_name: str, pieces: list[str]) -> dict[str, np.ndarray]:
    return {piece: labels[piece][level_name] for piece in pieces}


def append_score_columns(
    features: dict[str, np.ndarray],
    score_columns: dict[str, list[np.ndarray]],
    pieces: list[str],
) -> dict[str, np.ndarray]:
    out = {}
    for piece in pieces:
        if score_columns[piece]:
            cols = [np.asarray(col, dtype=np.float32).reshape(-1, 1) for col in score_columns[piece]]
            out[piece] = np.concatenate([features[piece], *cols], axis=1).astype(np.float32)
        else:
            out[piece] = features[piece].astype(np.float32)
    return out


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


def evaluate_density(
    scores: dict[str, np.ndarray],
    train_labels: dict[str, np.ndarray],
    val_labels: dict[str, np.ndarray],
    train_pieces: list[str],
    val_pieces: list[str],
    tolerance: int,
) -> dict:
    items = []
    for piece in val_pieces:
        pred = base.extract_top_density(scores[piece], expected_count(train_labels, train_pieces, len(val_labels[piece])))
        items.append(weighted_metrics_from_events(pred, val_labels[piece], tolerance=tolerance))
    return aggregate(items)


def evaluate_threshold(scores: dict[str, np.ndarray], labels: dict[str, np.ndarray], val_pieces: list[str], threshold: float, tolerance: int) -> dict:
    items = []
    for piece in val_pieces:
        pred = base.extract_events(scores[piece], threshold=threshold)
        items.append(weighted_metrics_from_events(pred, labels[piece], tolerance=tolerance))
    return aggregate(items)


def run_mode(
    mode: str,
    base_features: dict[str, np.ndarray],
    labels: dict[str, dict[str, np.ndarray]],
    pieces: list[str],
    feature_cols: list[str],
) -> tuple[list[dict], list[dict]]:
    folds = base.make_folds(pieces, n_folds=5, seed=42)
    rows = []
    piece_rows = []
    for fold_idx, val_pieces in enumerate(folds, start=1):
        train_pieces = [p for p in pieces if p not in set(val_pieces)]
        train_score_cols = {p: [] for p in train_pieces}
        val_score_cols = {p: [] for p in val_pieces}
        for level_idx, level_name in enumerate(LEVEL_NAMES, start=1):
            train_features = append_score_columns(base_features, train_score_cols, train_pieces)
            val_features = append_score_columns(base_features, val_score_cols, val_pieces)
            current_labels_train = labels_for_level(labels, level_name, train_pieces)
            current_labels_val = labels_for_level(labels, level_name, val_pieces)
            model, mean, std = base.train_one(train_features, current_labels_train, train_pieces, seed=11000 + fold_idx * 100 + level_idx)
            train_scores = base.predict(model, train_features, train_pieces, mean, std)
            val_scores = base.predict(model, val_features, val_pieces, mean, std)
            threshold, train_metric = choose_threshold(train_scores, current_labels_train, tolerance=1)
            th1 = evaluate_threshold(val_scores, current_labels_val, val_pieces, threshold, tolerance=1)
            th0 = evaluate_threshold(val_scores, current_labels_val, val_pieces, threshold, tolerance=0)
            den1 = evaluate_density(val_scores, current_labels_train, current_labels_val, train_pieces, val_pieces, tolerance=1)
            den0 = evaluate_density(val_scores, current_labels_train, current_labels_val, train_pieces, val_pieces, tolerance=0)
            rows.append(
                {
                    "mode": mode,
                    "fold": fold_idx,
                    "level": level_name,
                    "input_dim": int(train_features[train_pieces[0]].shape[1]),
                    "threshold_from_train_tol1": threshold,
                    "train_f1_tol1": train_metric["f1"],
                    "threshold_precision_tol1": th1["precision"],
                    "threshold_recall_tol1": th1["recall"],
                    "threshold_f1_tol1": th1["f1"],
                    "threshold_weighted_recall_tol1": th1["weighted_recall"],
                    "threshold_precision_tol0": th0["precision"],
                    "threshold_recall_tol0": th0["recall"],
                    "threshold_f1_tol0": th0["f1"],
                    "threshold_weighted_recall_tol0": th0["weighted_recall"],
                    "threshold_pred_events": th1["pred_events"],
                    "threshold_true_events": th1["true_events"],
                    "threshold_matches_tol1": th1["matches"],
                    "density_precision_tol1": den1["precision"],
                    "density_recall_tol1": den1["recall"],
                    "density_f1_tol1": den1["f1"],
                    "density_weighted_recall_tol1": den1["weighted_recall"],
                    "density_precision_tol0": den0["precision"],
                    "density_recall_tol0": den0["recall"],
                    "density_f1_tol0": den0["f1"],
                    "density_weighted_recall_tol0": den0["weighted_recall"],
                    "density_pred_events": den1["pred_events"],
                    "density_true_events": den1["true_events"],
                    "density_matches_tol1": den1["matches"],
                }
            )
            for piece in val_pieces:
                target = current_labels_val[piece]
                threshold_pred = base.extract_events(val_scores[piece], threshold=threshold)
                density_pred = base.extract_top_density(
                    val_scores[piece],
                    expected_count(current_labels_train, train_pieces, len(target)),
                )
                piece_rows.append(
                    {
                        "mode": mode,
                        "fold": fold_idx,
                        "level": level_name,
                        "piece": piece,
                        "num_beats": len(target),
                        "true_events": int(np.count_nonzero(target >= EVENT_MIN)),
                        "threshold_pred_events": int(len(threshold_pred)),
                        "density_pred_events": int(len(density_pred)),
                    }
                )
            if mode == "cascade":
                for piece in train_pieces:
                    train_score_cols[piece].append(train_scores[piece])
                for piece in val_pieces:
                    val_score_cols[piece].append(val_scores[piece])
        print(f"{mode} fold {fold_idx} done", flush=True)
    return rows, piece_rows


def aggregate_totals(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (mode, level), group in summary.groupby(["mode", "level"], sort=False):
        for prefix in ["threshold", "density"]:
            pred = int(group[f"{prefix}_pred_events"].sum())
            true = int(group[f"{prefix}_true_events"].sum())
            matches = int(group[f"{prefix}_matches_tol1"].sum())
            precision = matches / pred if pred else 0.0
            recall = matches / true if true else 0.0
            rows.append(
                {
                    "mode": mode,
                    "level": level,
                    "decode": prefix,
                    "pred_events": pred,
                    "true_events": true,
                    "matches_tol1": matches,
                    "precision_tol1": precision,
                    "recall_tol1": recall,
                    "f1_tol1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
                    "mean_wr_tol1": float(group[f"{prefix}_weighted_recall_tol1"].mean()),
                    "mean_f1_tol0": float(group[f"{prefix}_f1_tol0"].mean()),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pieces, labels = load_level_labels()
    features, feature_cols = base.load_piece_features(pieces)
    label_rows = []
    for piece in pieces:
        row = {"piece": piece, "num_beats": len(features[piece])}
        for level in LEVEL_NAMES:
            row[f"{level}_events_ge_0p05"] = int(np.count_nonzero(labels[piece][level] >= EVENT_MIN))
            row[f"{level}_target_sum"] = float(labels[piece][level].sum())
        label_rows.append(row)
    pd.DataFrame(label_rows).to_csv(OUT_DIR / "label_stats.csv", index=False)

    all_rows = []
    all_piece_rows = []
    for mode in ["plain", "cascade"]:
        rows, piece_rows = run_mode(mode, features, labels, pieces, feature_cols)
        all_rows.extend(rows)
        all_piece_rows.extend(piece_rows)

    summary = pd.DataFrame(all_rows)
    piece_summary = pd.DataFrame(all_piece_rows)
    summary.to_csv(OUT_DIR / "fold_level_summary.csv", index=False)
    piece_summary.to_csv(OUT_DIR / "piece_level_summary.csv", index=False)
    mean = summary.groupby(["mode", "level"], sort=False).mean(numeric_only=True).reset_index()
    mean.to_csv(OUT_DIR / "fold_level_mean.csv", index=False)
    totals = aggregate_totals(summary)
    totals.to_csv(OUT_DIR / "aggregate_totals.csv", index=False)
    metadata = {
        "label_source": str(base.LABEL_DIR),
        "beat_table": str(base.BEAT_TABLE),
        "feature_columns": feature_cols,
        "pieces": pieces,
        "levels": LEVEL_NAMES,
        "L5_definition": "max(consensus_L5, consensus_L6)",
        "event_min": EVENT_MIN,
        "cascade_rule": "Lk model input appends previous predicted score columns L1..L(k-1); train and validation both use model predictions, not gold previous labels",
    }
    (OUT_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("\nAggregate totals:")
    print(totals.round(4).to_string(index=False))
    print(f"\nWrote {OUT_DIR}")


if __name__ == "__main__":
    main()
