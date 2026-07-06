from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[2]
BASE_SCRIPT = ROOT / "MERIX SUBMISSION" / "MIREX_Model" / "run_mazurkabl_l2plus_weighted_target_experiment.py"
OUT_DIR = ROOT / "MERIX SUBMISSION" / "MIREX_Model" / "mazurkabl_l2plus_soft_spread_target_experiment"
SIGMA = 1.0
RADIUS = 2


spec = importlib.util.spec_from_file_location("mazurkabl_base", BASE_SCRIPT)
base = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules["mazurkabl_base"] = base
spec.loader.exec_module(base)


@dataclass
class WeightedMetrics:
    precision: float
    recall: float
    f1: float
    pred_events: int
    true_events: int
    matches: int
    weighted_recall: float
    matched_weight: float
    total_weight: float
    mean_offset: float | None


def gaussian_spread_target(target: np.ndarray, sigma: float = SIGMA, radius: int = RADIUS) -> np.ndarray:
    target = np.asarray(target, dtype=np.float32)
    out = target.copy()
    centers = np.flatnonzero(target >= base.EVENT_MIN)
    for center in centers.tolist():
        center_value = float(target[center])
        for distance in range(1, int(radius) + 1):
            value = center_value * float(np.exp(-(distance**2) / (2.0 * float(sigma) ** 2)))
            left = center - distance
            right = center + distance
            if left >= 0:
                out[left] = max(float(out[left]), value)
            if right < len(out):
                out[right] = max(float(out[right]), value)
    return out.astype(np.float32)


def weighted_metrics_from_events(pred: np.ndarray, hard_target: np.ndarray, tolerance: int) -> WeightedMetrics:
    hard_target = np.asarray(hard_target, dtype=np.float32)
    true = np.flatnonzero(hard_target >= base.EVENT_MIN).astype(np.int32)
    matches, offsets = base.match_events(pred, true, tolerance=tolerance)

    # Rebuild matched true indices for weighted recall.
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
    matched_weight = float(np.sum(hard_target[matched_true])) if matched_true else 0.0
    total_weight = float(np.sum(hard_target[true])) if true.size else 0.0
    precision = matches / len(pred) if len(pred) else 0.0
    recall = matches / len(true) if len(true) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    weighted_recall = matched_weight / total_weight if total_weight > 0 else 0.0
    return WeightedMetrics(
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        pred_events=int(len(pred)),
        true_events=int(len(true)),
        matches=int(matches),
        weighted_recall=float(weighted_recall),
        matched_weight=matched_weight,
        total_weight=total_weight,
        mean_offset=float(np.mean(offsets)) if offsets else None,
    )


def aggregate_weighted(items: list[WeightedMetrics]) -> WeightedMetrics:
    pred = sum(x.pred_events for x in items)
    true = sum(x.true_events for x in items)
    matches = sum(x.matches for x in items)
    matched_weight = sum(x.matched_weight for x in items)
    total_weight = sum(x.total_weight for x in items)
    precision = matches / pred if pred else 0.0
    recall = matches / true if true else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    weighted_recall = matched_weight / total_weight if total_weight > 0 else 0.0
    return WeightedMetrics(
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        pred_events=int(pred),
        true_events=int(true),
        matches=int(matches),
        weighted_recall=float(weighted_recall),
        matched_weight=float(matched_weight),
        total_weight=float(total_weight),
        mean_offset=None,
    )


def train_one_soft(features: dict[str, np.ndarray], train_labels: dict[str, np.ndarray], train_pieces: list[str], seed: int):
    base.set_seed(seed)
    train_feats = [features[p] for p in train_pieces]
    train_labs = [train_labels[p] for p in train_pieces]
    stacked = np.concatenate(train_feats, axis=0)
    mean = stacked.mean(axis=0).astype(np.float32)
    std = stacked.std(axis=0).astype(np.float32)
    std[std < 1e-6] = 1.0

    x, y, mask = base.pad_batch(train_feats, train_labs, mean, std)
    model = base.TinyMLP(input_dim=x.shape[-1])
    pos = float(y.sum().item())
    neg = float(mask.sum().item() - pos)
    pos_weight = torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best_state = None
    best_loss = float("inf")
    stale = 0
    for _epoch in range(1, 81):
        model.train()
        opt.zero_grad()
        logits = model(x)
        loss = (loss_fn(logits, y) * mask).sum() / mask.sum().clamp(min=1.0)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        value = float(loss.item())
        if value < best_loss - 1e-5:
            best_loss = value
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= 10:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, mean, std


def evaluate_threshold_weighted(
    scores_by_piece: dict[str, np.ndarray],
    hard_labels: dict[str, np.ndarray],
    threshold: float,
    tolerance: int,
) -> WeightedMetrics:
    items = []
    for piece, scores in scores_by_piece.items():
        pred = base.extract_events(scores, threshold=threshold)
        items.append(weighted_metrics_from_events(pred, hard_labels[piece], tolerance=tolerance))
    return aggregate_weighted(items)


def choose_threshold_weighted(
    scores_by_piece: dict[str, np.ndarray],
    hard_labels: dict[str, np.ndarray],
    tolerance: int,
) -> tuple[float, WeightedMetrics]:
    best_th = 0.05
    best = None
    for th in np.linspace(0.05, 0.95, 37):
        metrics = evaluate_threshold_weighted(scores_by_piece, hard_labels, float(th), tolerance=tolerance)
        key = (metrics.f1, metrics.precision, metrics.weighted_recall, metrics.recall)
        if best is None or key > (best.f1, best.precision, best.weighted_recall, best.recall):
            best = metrics
            best_th = float(th)
    assert best is not None
    return best_th, best


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pieces, hard_labels, components = base.load_l2plus_weighted_labels()
    soft_labels = {piece: gaussian_spread_target(hard_labels[piece]) for piece in pieces}
    features, feature_cols = base.load_piece_features(pieces)

    label_stats = []
    for piece in pieces:
        label_stats.append(
            {
                "piece": piece,
                "num_beats": len(hard_labels[piece]),
                "hard_sum": float(hard_labels[piece].sum()),
                "soft_sum": float(soft_labels[piece].sum()),
                "hard_events_ge_0p05": int(np.count_nonzero(hard_labels[piece] >= base.EVENT_MIN)),
                "soft_events_ge_0p05": int(np.count_nonzero(soft_labels[piece] >= base.EVENT_MIN)),
                **{f"L{level}_support": int(np.count_nonzero(components[piece][level] > 0)) for level in base.LEVEL_WEIGHTS},
            }
        )
    pd.DataFrame(label_stats).to_csv(OUT_DIR / "label_stats.csv", index=False)

    folds = base.make_folds(pieces, n_folds=5, seed=42)
    rows = []
    piece_rows = []
    for fold_idx, val_pieces in enumerate(folds, start=1):
        train_pieces = [p for p in pieces if p not in set(val_pieces)]
        model, mean, std = train_one_soft(features, soft_labels, train_pieces, seed=7200 + fold_idx)
        train_scores = base.predict(model, features, train_pieces, mean, std)
        val_scores = base.predict(model, features, val_pieces, mean, std)
        threshold, train_metric = choose_threshold_weighted(
            train_scores,
            {p: hard_labels[p] for p in train_pieces},
            tolerance=1,
        )
        th_tol1 = evaluate_threshold_weighted(val_scores, {p: hard_labels[p] for p in val_pieces}, threshold, tolerance=1)
        th_tol0 = evaluate_threshold_weighted(val_scores, {p: hard_labels[p] for p in val_pieces}, threshold, tolerance=0)

        density_tol1_items = []
        density_tol0_items = []
        for piece in val_pieces:
            expected = base.expected_count_from_train_density(hard_labels, train_pieces, len(hard_labels[piece]))
            pred = base.extract_top_density(val_scores[piece], expected)
            m1 = weighted_metrics_from_events(pred, hard_labels[piece], tolerance=1)
            m0 = weighted_metrics_from_events(pred, hard_labels[piece], tolerance=0)
            density_tol1_items.append(m1)
            density_tol0_items.append(m0)
            piece_rows.append(
                {
                    "fold": fold_idx,
                    "piece": piece,
                    "num_beats": len(hard_labels[piece]),
                    "true_events": m1.true_events,
                    "density_expected_events": expected,
                    "density_pred_events": len(pred),
                    "density_precision_tol1": m1.precision,
                    "density_recall_tol1": m1.recall,
                    "density_f1_tol1": m1.f1,
                    "density_weighted_recall_tol1": m1.weighted_recall,
                    "density_precision_tol0": m0.precision,
                    "density_recall_tol0": m0.recall,
                    "density_f1_tol0": m0.f1,
                    "density_weighted_recall_tol0": m0.weighted_recall,
                }
            )
            pd.DataFrame(
                {
                    "beat_idx": np.arange(len(val_scores[piece])),
                    "score": val_scores[piece],
                    "hard_target_l2plus_weighted_max": hard_labels[piece],
                    "soft_train_target_gaussian": soft_labels[piece],
                    "hard_target_event_ge_0p05": (hard_labels[piece] >= base.EVENT_MIN).astype(np.int8),
                    "density_pred": np.isin(np.arange(len(val_scores[piece])), pred).astype(np.int8),
                }
            ).to_csv(OUT_DIR / f"fold{fold_idx}_{piece}_val_predictions.csv", index=False)

        den_tol1 = aggregate_weighted(density_tol1_items)
        den_tol0 = aggregate_weighted(density_tol0_items)
        rows.append(
            {
                "fold": fold_idx,
                "train_pieces": " ".join(train_pieces),
                "val_pieces": " ".join(val_pieces),
                "threshold_from_train_tol1": threshold,
                "train_f1_tol1": train_metric.f1,
                "threshold_precision_tol1": th_tol1.precision,
                "threshold_recall_tol1": th_tol1.recall,
                "threshold_f1_tol1": th_tol1.f1,
                "threshold_weighted_recall_tol1": th_tol1.weighted_recall,
                "threshold_precision_tol0": th_tol0.precision,
                "threshold_recall_tol0": th_tol0.recall,
                "threshold_f1_tol0": th_tol0.f1,
                "threshold_weighted_recall_tol0": th_tol0.weighted_recall,
                "density_precision_tol1": den_tol1.precision,
                "density_recall_tol1": den_tol1.recall,
                "density_f1_tol1": den_tol1.f1,
                "density_weighted_recall_tol1": den_tol1.weighted_recall,
                "density_precision_tol0": den_tol0.precision,
                "density_recall_tol0": den_tol0.recall,
                "density_f1_tol0": den_tol0.f1,
                "density_weighted_recall_tol0": den_tol0.weighted_recall,
                "density_pred_events": den_tol1.pred_events,
                "threshold_pred_events": th_tol1.pred_events,
                "true_events": den_tol1.true_events,
            }
        )
        print(
            f"fold {fold_idx}: threshold tol1 P/R/F1/WR="
            f"{th_tol1.precision:.3f}/{th_tol1.recall:.3f}/{th_tol1.f1:.3f}/{th_tol1.weighted_recall:.3f}; "
            f"density tol1 P/R/F1/WR="
            f"{den_tol1.precision:.3f}/{den_tol1.recall:.3f}/{den_tol1.f1:.3f}/{den_tol1.weighted_recall:.3f}"
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
        "feature_columns": feature_cols,
        "pieces": pieces,
        "folds": folds,
        "label_source": str(base.LABEL_DIR),
        "level_weights": base.LEVEL_WEIGHTS,
        "hard_target_rule": "max(weight_L * mean_performer_boundary_L_frequency) for L2-L6",
        "train_target_rule": "gaussian spread of hard target by max over centers",
        "sigma": SIGMA,
        "radius": RADIUS,
        "event_min": base.EVENT_MIN,
        "min_distance": base.MIN_DISTANCE,
    }
    (OUT_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("\nMean:")
    print(mean.round(4).to_string())
    print(f"\nWrote {OUT_DIR}")


if __name__ == "__main__":
    main()
