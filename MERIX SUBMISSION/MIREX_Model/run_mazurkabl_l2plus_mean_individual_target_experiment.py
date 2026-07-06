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
OUT_DIR = ROOT / "MERIX SUBMISSION" / "MIREX_Model" / "mazurkabl_l2plus_mean_individual_target_experiment"
LEVEL_WEIGHTS = {2: 0.28, 3: 0.46, 4: 0.64, 5: 0.82, 6: 1.00}
EVENT_MIN = 0.05

spec = importlib.util.spec_from_file_location("mazurkabl_base_for_mean_individual", BASE_SCRIPT)
base = importlib.util.module_from_spec(spec)
sys.modules["mazurkabl_base_for_mean_individual"] = base
assert spec.loader is not None
spec.loader.exec_module(base)


def parse_npz_name(path: Path) -> tuple[str, str, int]:
    match = re.match(r"(M\d+-\d+)_(.+)_L([1-6])\.npz$", path.name)
    if not match:
        raise ValueError(f"Cannot parse label filename: {path.name}")
    return match.group(1), match.group(2), int(match.group(3))


def load_mean_individual_labels() -> tuple[list[str], dict[str, np.ndarray], dict[str, np.ndarray], dict[str, int]]:
    grouped: dict[tuple[str, str], dict[int, Path]] = {}
    for path in sorted(base.LABEL_DIR.glob("*_L[2-6].npz")):
        piece, performer, level = parse_npz_name(path)
        grouped.setdefault((piece, performer), {})[level] = path
    per_piece_targets: dict[str, list[np.ndarray]] = {}
    performer_count: dict[str, int] = {}
    for (piece, performer), paths in sorted(grouped.items()):
        if any(level not in paths for level in LEVEL_WEIGHTS):
            continue
        weighted = []
        for level, weight in LEVEL_WEIGHTS.items():
            arr = np.load(paths[level], allow_pickle=True)["boundary_probs"].astype(np.float32)
            weighted.append(float(weight) * arr)
        target = np.maximum.reduce(weighted).astype(np.float32)
        per_piece_targets.setdefault(piece, []).append(target)
    labels = {}
    for piece, arrays in per_piece_targets.items():
        lengths = {len(a) for a in arrays}
        if len(lengths) != 1:
            raise RuntimeError(f"{piece}: inconsistent performer target lengths {sorted(lengths)}")
        labels[piece] = np.mean(np.stack(arrays, axis=0), axis=0).astype(np.float32)
        performer_count[piece] = len(arrays)
    pieces = sorted(labels)
    # Existing target for direct comparison: max_l weight_l * consensus_l(b).
    _, consensus_max_labels, _ = base.load_l2plus_weighted_labels()
    return pieces, labels, consensus_max_labels, performer_count


def weighted_metrics(pred: np.ndarray, target: np.ndarray, tolerance: int) -> dict:
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


def aggregate(items: list[dict]) -> dict:
    pred = sum(x["pred_events"] for x in items)
    true = sum(x["true_events"] for x in items)
    matches = sum(x["matches"] for x in items)
    matched_weight = sum(x["matched_weight"] for x in items)
    total_weight = sum(x["total_weight"] for x in items)
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


def evaluate_scores(scores: dict[str, np.ndarray], labels: dict[str, np.ndarray], threshold: float, pieces: list[str]) -> tuple[dict, dict]:
    threshold_items = []
    density_items = []
    for piece in pieces:
        pred_th = base.extract_events(scores[piece], threshold=threshold)
        expected = max(1, int(round(len(labels[piece]) * sum(np.count_nonzero(labels[p] >= EVENT_MIN) for p in labels) / max(sum(len(labels[p]) for p in labels), 1))))
        pred_den = base.extract_top_density(scores[piece], expected)
        threshold_items.append(weighted_metrics(pred_th, labels[piece], tolerance=1))
        density_items.append(weighted_metrics(pred_den, labels[piece], tolerance=1))
    return aggregate(threshold_items), aggregate(density_items)


def choose_threshold(scores: dict[str, np.ndarray], labels: dict[str, np.ndarray], pieces: list[str]) -> tuple[float, dict]:
    best_th = 0.05
    best = None
    for th in np.linspace(0.05, 0.95, 37):
        items = [weighted_metrics(base.extract_events(scores[p], float(th)), labels[p], tolerance=1) for p in pieces]
        m = aggregate(items)
        key = (m["f1"], m["precision"], m["weighted_recall"], m["recall"])
        if best is None or key > (best["f1"], best["precision"], best["weighted_recall"], best["recall"]):
            best = m
            best_th = float(th)
    assert best is not None
    return best_th, best


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pieces, train_labels, consensus_max_labels, performer_count = load_mean_individual_labels()
    features, feature_cols = base.load_piece_features(pieces)
    length_mismatch = [(p, len(train_labels[p]), len(features[p])) for p in pieces if len(train_labels[p]) != len(features[p])]
    if length_mismatch:
        raise RuntimeError(f"Length mismatch: {length_mismatch[:5]}")
    folds = base.make_folds(pieces, n_folds=5, seed=42)
    rows = []
    for fold_idx, val_pieces in enumerate(folds, start=1):
        train_pieces = [p for p in pieces if p not in set(val_pieces)]
        model, mean, std = base.train_one(features, train_labels, train_pieces, seed=10200 + fold_idx)
        train_scores = base.predict(model, features, train_pieces, mean, std)
        val_scores = base.predict(model, features, val_pieces, mean, std)
        threshold, _ = choose_threshold(train_scores, train_labels, train_pieces)
        mean_target_th, mean_target_den = evaluate_scores(val_scores, train_labels, threshold, val_pieces)
        consensus_th, consensus_den = evaluate_scores(val_scores, consensus_max_labels, threshold, val_pieces)
        rows.append(
            {
                "fold": fold_idx,
                "threshold": threshold,
                "mean_target_threshold_precision": mean_target_th["precision"],
                "mean_target_threshold_recall": mean_target_th["recall"],
                "mean_target_threshold_f1": mean_target_th["f1"],
                "mean_target_threshold_wr": mean_target_th["weighted_recall"],
                "mean_target_density_precision": mean_target_den["precision"],
                "mean_target_density_recall": mean_target_den["recall"],
                "mean_target_density_f1": mean_target_den["f1"],
                "mean_target_density_wr": mean_target_den["weighted_recall"],
                "consensus_eval_threshold_precision": consensus_th["precision"],
                "consensus_eval_threshold_recall": consensus_th["recall"],
                "consensus_eval_threshold_f1": consensus_th["f1"],
                "consensus_eval_threshold_wr": consensus_th["weighted_recall"],
                "consensus_eval_density_precision": consensus_den["precision"],
                "consensus_eval_density_recall": consensus_den["recall"],
                "consensus_eval_density_f1": consensus_den["f1"],
                "consensus_eval_density_wr": consensus_den["weighted_recall"],
                "consensus_eval_density_pred_events": consensus_den["pred_events"],
                "consensus_eval_true_events": consensus_den["true_events"],
            }
        )
        print(
            f"fold {fold_idx}: mean-target density F1/WR={mean_target_den['f1']:.3f}/{mean_target_den['weighted_recall']:.3f}; "
            f"consensus-eval density F1/WR={consensus_den['f1']:.3f}/{consensus_den['weighted_recall']:.3f}"
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT_DIR / "fold_summary.csv", index=False)
    mean = summary.drop(columns=["fold"]).mean(numeric_only=True)
    mean.to_frame("mean").to_csv(OUT_DIR / "fold_mean.csv")
    pd.DataFrame(
        [
            {
                "piece": p,
                "performer_count": performer_count[p],
                "mean_individual_events_ge_0p05": int(np.count_nonzero(train_labels[p] >= EVENT_MIN)),
                "consensus_max_events_ge_0p05": int(np.count_nonzero(consensus_max_labels[p] >= EVENT_MIN)),
                "mean_individual_sum": float(np.sum(train_labels[p])),
                "consensus_max_sum": float(np.sum(consensus_max_labels[p])),
            }
            for p in pieces
        ]
    ).to_csv(OUT_DIR / "label_stats.csv", index=False)
    metadata = {
        "target_rule": "mean over performers of individual max_l(weight_l * boundary_l)",
        "comparison_target_rule": "original max_l(weight_l * consensus_l)",
        "note": "Equivalent to per-performer duplicated score-feature BCE training when all performers have identical score features.",
        "feature_columns": feature_cols,
        "pieces": pieces,
        "folds": folds,
        "event_min": EVENT_MIN,
    }
    (OUT_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("\nMean:")
    print(mean.round(4).to_string())
    print(f"\nWrote {OUT_DIR}")


if __name__ == "__main__":
    main()
