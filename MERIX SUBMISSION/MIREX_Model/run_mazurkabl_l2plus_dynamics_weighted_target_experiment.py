from __future__ import annotations

import importlib.util
import json
import re
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
MIREX = ROOT / "MERIX SUBMISSION" / "MIREX_Model"
VELOCITY = ROOT / "MERIX SUBMISSION" / "Velocity"
BASE_SCRIPT = MIREX / "run_mazurkabl_l2plus_weighted_target_experiment.py"
VELOCITY_SCRIPT = VELOCITY / "build_mazurka_velocity_npz_performer_levels.py"
BEAT_DYN_DIR = ROOT / "MazurkaBL-master" / "beat_dyn"
OUT_DIR = MIREX / "mazurkabl_l2plus_dynamics_weighted_target_experiment"

LEVEL_WEIGHTS = {2: 0.28, 3: 0.46, 4: 0.64, 5: 0.82, 6: 1.00}
STR_VEC = [3, 2, 2, 2, 2, 2]
EVENT_MIN = 0.05


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


base = load_module(BASE_SCRIPT, "tempo_weighted_base")
velocity = load_module(VELOCITY_SCRIPT, "velocity_builder_for_dyn_target")


def normalize_raw_id(raw_id: str) -> str:
    match = re.match(r"^M(\d+)-(\d+)$", raw_id)
    if not match:
        raise ValueError(f"Bad raw id: {raw_id}")
    return f"M{int(match.group(1)):02d}-{int(match.group(2))}"


def load_dynamics_l2plus_weighted_labels() -> tuple[list[str], dict[str, np.ndarray], dict[str, dict[int, np.ndarray]]]:
    labels: dict[str, np.ndarray] = {}
    components: dict[str, dict[int, np.ndarray]] = {}

    for path in sorted(BEAT_DYN_DIR.glob("*beat_dynNORM.csv")):
        raw_id = path.name.replace("beat_dynNORM.csv", "")
        piece = normalize_raw_id(raw_id)
        df, performer_cols = velocity.load_beat_dyn(path)
        curves = velocity.compute_dyn_curves(df, performer_cols, smooth_window=3)
        n_beats = len(df)
        counts = {level: np.zeros(n_beats, dtype=np.float32) for level in range(1, 7)}

        for curve in curves.values():
            _, level_sets = velocity.group_analysis_hierarchy(curve, STR_VEC, enforce_nested=True)
            for level in range(1, 7):
                idx = np.asarray(level_sets[level], dtype=int)
                idx = idx[(idx >= 0) & (idx < n_beats)]
                counts[level][idx] += 1.0

        denom = max(len(curves), 1)
        components[piece] = {level: counts[level] / denom for level in range(1, 7)}
        weighted = [float(weight) * components[piece][level] for level, weight in LEVEL_WEIGHTS.items()]
        labels[piece] = np.maximum.reduce(weighted).astype(np.float32)

    pieces = sorted(labels)
    return pieces, labels, components


def match_true_indices(pred: np.ndarray, true: np.ndarray, tolerance: int) -> list[int]:
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
    return matched_true


def weighted_recall(pred: np.ndarray, label: np.ndarray, tolerance: int) -> float:
    true = np.flatnonzero(label >= EVENT_MIN).astype(np.int32)
    denom = float(label[true].sum()) if len(true) else 0.0
    if denom <= 0:
        return 0.0
    matched_true = match_true_indices(pred, true, tolerance)
    return float(label[matched_true].sum()) / denom if matched_true else 0.0


def evaluate_threshold_with_wr(scores_by_piece: dict[str, np.ndarray], labels_by_piece: dict[str, np.ndarray], threshold: float, tolerance: int):
    items = []
    wr_num = 0.0
    wr_den = 0.0
    for piece, scores in scores_by_piece.items():
        pred = base.extract_events(scores, threshold=threshold)
        true = np.flatnonzero(labels_by_piece[piece] >= EVENT_MIN).astype(np.int32)
        items.append(base.metrics_from_events(pred, true, tolerance))
        matched = match_true_indices(pred, true, tolerance)
        wr_num += float(labels_by_piece[piece][matched].sum()) if matched else 0.0
        wr_den += float(labels_by_piece[piece][true].sum()) if len(true) else 0.0
    metric = base.aggregate_metrics(items)
    return metric, (wr_num / wr_den if wr_den > 0 else 0.0)


def density_metrics_with_wr(scores: np.ndarray, label: np.ndarray, expected: int, tolerance: int):
    true = np.flatnonzero(label >= EVENT_MIN).astype(np.int32)
    pred = base.extract_top_density(scores, expected)
    metric = base.metrics_from_events(pred, true, tolerance)
    wr = weighted_recall(pred, label, tolerance)
    return pred, metric, wr


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pieces, labels, components = load_dynamics_l2plus_weighted_labels()
    features, feature_cols = base.load_piece_features(pieces)
    skipped = []
    truncated = []
    kept = []
    for piece in pieces:
        feat_len = len(features.get(piece, [])) if piece in features else 0
        lab_len = len(labels[piece])
        if feat_len <= 0:
            skipped.append({"piece": piece, "reason": "no_features", "feature_len": feat_len, "label_len": lab_len})
            continue
        if feat_len != lab_len:
            n = min(feat_len, lab_len)
            truncated.append({"piece": piece, "feature_len": feat_len, "label_len": lab_len, "used_len": n})
            features[piece] = features[piece][:n]
            labels[piece] = labels[piece][:n]
            for level in components[piece]:
                components[piece][level] = components[piece][level][:n]
        kept.append(piece)
    pieces = kept
    labels = {p: labels[p] for p in pieces}
    components = {p: components[p] for p in pieces}

    label_stats = []
    for piece in pieces:
        label_stats.append(
            {
                "piece": piece,
                "num_beats": len(labels[piece]),
                "target_sum": float(labels[piece].sum()),
                "event_count_target_ge_0p05": int(np.count_nonzero(labels[piece] >= EVENT_MIN)),
                **{
                    f"L{level}_support": int(np.count_nonzero(components[piece][level] > 0))
                    for level in LEVEL_WEIGHTS
                },
            }
        )
    pd.DataFrame(label_stats).to_csv(OUT_DIR / "label_stats.csv", index=False)

    folds = base.make_folds(pieces, n_folds=5, seed=42)
    rows = []
    piece_rows = []
    totals = {
        "threshold_pred": 0,
        "threshold_true": 0,
        "threshold_match": 0,
        "threshold_wr_num": 0.0,
        "threshold_wr_den": 0.0,
        "density_pred": 0,
        "density_true": 0,
        "density_match": 0,
        "density_wr_num": 0.0,
        "density_wr_den": 0.0,
    }

    for fold_idx, val_pieces in enumerate(folds, start=1):
        train_pieces = [p for p in pieces if p not in set(val_pieces)]
        model, mean, std = base.train_one(features, labels, train_pieces, seed=7300 + fold_idx)
        train_scores = base.predict(model, features, train_pieces, mean, std)
        val_scores = base.predict(model, features, val_pieces, mean, std)
        threshold, train_metric = base.choose_threshold(train_scores, {p: labels[p] for p in train_pieces}, tolerance=1)
        th_tol1, th_wr1 = evaluate_threshold_with_wr(val_scores, {p: labels[p] for p in val_pieces}, threshold, tolerance=1)
        th_tol0, th_wr0 = evaluate_threshold_with_wr(val_scores, {p: labels[p] for p in val_pieces}, threshold, tolerance=0)

        density_tol1_items = []
        density_tol0_items = []
        density_wr1_nums = []
        density_wr1_dens = []
        density_wr0_nums = []
        density_wr0_dens = []
        for piece in val_pieces:
            true = np.flatnonzero(labels[piece] >= EVENT_MIN).astype(np.int32)
            expected = base.expected_count_from_train_density(labels, train_pieces, len(labels[piece]))
            pred, m1, _ = density_metrics_with_wr(val_scores[piece], labels[piece], expected, tolerance=1)
            _, m0, _ = density_metrics_with_wr(val_scores[piece], labels[piece], expected, tolerance=0)
            matched1 = match_true_indices(pred, true, tolerance=1)
            matched0 = match_true_indices(pred, true, tolerance=0)
            den = float(labels[piece][true].sum()) if len(true) else 0.0
            num1 = float(labels[piece][matched1].sum()) if matched1 else 0.0
            num0 = float(labels[piece][matched0].sum()) if matched0 else 0.0
            density_wr1_nums.append(num1)
            density_wr1_dens.append(den)
            density_wr0_nums.append(num0)
            density_wr0_dens.append(den)
            density_tol1_items.append(m1)
            density_tol0_items.append(m0)
            totals["density_wr_num"] += num1
            totals["density_wr_den"] += den
            piece_rows.append(
                {
                    "fold": fold_idx,
                    "piece": piece,
                    "num_beats": len(labels[piece]),
                    "true_events": len(true),
                    "density_expected_events": expected,
                    "density_pred_events": len(pred),
                    "density_precision_tol1_UP": m1.precision,
                    "density_recall_tol1": m1.recall,
                    "density_weighted_recall_tol1_WR": num1 / den if den > 0 else 0.0,
                    "density_f1_tol1": m1.f1,
                    "density_precision_tol0_UP": m0.precision,
                    "density_recall_tol0": m0.recall,
                    "density_weighted_recall_tol0_WR": num0 / den if den > 0 else 0.0,
                    "density_f1_tol0": m0.f1,
                }
            )
            pd.DataFrame(
                {
                    "beat_idx": np.arange(len(val_scores[piece])),
                    "score": val_scores[piece],
                    "target_l2plus_dynamics_weighted": labels[piece],
                    "target_event_ge_0p05": (labels[piece] >= EVENT_MIN).astype(np.int8),
                    "density_pred": np.isin(np.arange(len(val_scores[piece])), pred).astype(np.int8),
                }
            ).to_csv(OUT_DIR / f"fold{fold_idx}_{piece}_val_predictions.csv", index=False)

        den_tol1 = base.aggregate_metrics(density_tol1_items)
        den_tol0 = base.aggregate_metrics(density_tol0_items)
        den_wr1 = sum(density_wr1_nums) / sum(density_wr1_dens) if sum(density_wr1_dens) > 0 else 0.0
        den_wr0 = sum(density_wr0_nums) / sum(density_wr0_dens) if sum(density_wr0_dens) > 0 else 0.0

        totals["threshold_pred"] += th_tol1.pred_events
        totals["threshold_true"] += th_tol1.true_events
        totals["threshold_match"] += th_tol1.matches
        for piece in val_pieces:
            true = np.flatnonzero(labels[piece] >= EVENT_MIN).astype(np.int32)
            pred = base.extract_events(val_scores[piece], threshold=threshold)
            matched = match_true_indices(pred, true, tolerance=1)
            totals["threshold_wr_num"] += float(labels[piece][matched].sum()) if matched else 0.0
            totals["threshold_wr_den"] += float(labels[piece][true].sum()) if len(true) else 0.0
        totals["density_pred"] += den_tol1.pred_events
        totals["density_true"] += den_tol1.true_events
        totals["density_match"] += den_tol1.matches

        rows.append(
            {
                "fold": fold_idx,
                "train_pieces": " ".join(train_pieces),
                "val_pieces": " ".join(val_pieces),
                "threshold_from_train_tol1": threshold,
                "train_f1_tol1": train_metric.f1,
                "threshold_precision_tol1_UP": th_tol1.precision,
                "threshold_recall_tol1": th_tol1.recall,
                "threshold_weighted_recall_tol1_WR": th_wr1,
                "threshold_f1_tol1": th_tol1.f1,
                "threshold_precision_tol0_UP": th_tol0.precision,
                "threshold_recall_tol0": th_tol0.recall,
                "threshold_weighted_recall_tol0_WR": th_wr0,
                "threshold_f1_tol0": th_tol0.f1,
                "density_precision_tol1_UP": den_tol1.precision,
                "density_recall_tol1": den_tol1.recall,
                "density_weighted_recall_tol1_WR": den_wr1,
                "density_f1_tol1": den_tol1.f1,
                "density_precision_tol0_UP": den_tol0.precision,
                "density_recall_tol0": den_tol0.recall,
                "density_weighted_recall_tol0_WR": den_wr0,
                "density_f1_tol0": den_tol0.f1,
                "threshold_pred_events": th_tol1.pred_events,
                "density_pred_events": den_tol1.pred_events,
                "true_events": den_tol1.true_events,
            }
        )
        print(
            f"fold {fold_idx}: threshold UP/WR/F1="
            f"{th_tol1.precision:.3f}/{th_wr1:.3f}/{th_tol1.f1:.3f}; "
            f"density UP/WR/F1={den_tol1.precision:.3f}/{den_wr1:.3f}/{den_tol1.f1:.3f}"
        )

    summary = pd.DataFrame(rows)
    piece_summary = pd.DataFrame(piece_rows)
    summary.to_csv(OUT_DIR / "fold_summary.csv", index=False)
    piece_summary.to_csv(OUT_DIR / "piece_summary.csv", index=False)
    mean_cols = [
        "threshold_precision_tol1_UP",
        "threshold_recall_tol1",
        "threshold_weighted_recall_tol1_WR",
        "threshold_f1_tol1",
        "density_precision_tol1_UP",
        "density_recall_tol1",
        "density_weighted_recall_tol1_WR",
        "density_f1_tol1",
    ]
    mean = summary[mean_cols].mean()
    mean.to_frame("mean").to_csv(OUT_DIR / "fold_mean.csv")

    total_threshold_up = totals["threshold_match"] / totals["threshold_pred"] if totals["threshold_pred"] else 0.0
    total_threshold_recall = totals["threshold_match"] / totals["threshold_true"] if totals["threshold_true"] else 0.0
    total_threshold_f1 = (
        2 * total_threshold_up * total_threshold_recall / (total_threshold_up + total_threshold_recall)
        if total_threshold_up + total_threshold_recall
        else 0.0
    )
    total_density_up = totals["density_match"] / totals["density_pred"] if totals["density_pred"] else 0.0
    total_density_recall = totals["density_match"] / totals["density_true"] if totals["density_true"] else 0.0
    total_density_f1 = (
        2 * total_density_up * total_density_recall / (total_density_up + total_density_recall)
        if total_density_up + total_density_recall
        else 0.0
    )
    aggregate = {
        "threshold_pred_events": totals["threshold_pred"],
        "threshold_true_events": totals["threshold_true"],
        "threshold_matches_tol1": totals["threshold_match"],
        "threshold_precision_tol1_UP": total_threshold_up,
        "threshold_recall_tol1": total_threshold_recall,
        "threshold_weighted_recall_tol1_WR": totals["threshold_wr_num"] / totals["threshold_wr_den"],
        "threshold_f1_tol1": total_threshold_f1,
        "density_pred_events": totals["density_pred"],
        "density_true_events": totals["density_true"],
        "density_matches_tol1": totals["density_match"],
        "density_precision_tol1_UP": total_density_up,
        "density_recall_tol1": total_density_recall,
        "density_weighted_recall_tol1_WR": totals["density_wr_num"] / totals["density_wr_den"],
        "density_f1_tol1": total_density_f1,
    }
    pd.DataFrame([aggregate]).to_csv(OUT_DIR / "aggregate_totals.csv", index=False)
    metadata = {
        "feature_columns": feature_cols,
        "pieces": pieces,
        "folds": folds,
        "label_source": str(BEAT_DYN_DIR),
        "level_weights": LEVEL_WEIGHTS,
        "str_vec": STR_VEC,
        "target_rule": "max(weight_L * mean_performer_dynamics_boundary_L_frequency) for L2-L6",
        "hierarchy_enforce_nested": True,
        "event_min": EVENT_MIN,
        "base_training_script": str(BASE_SCRIPT),
        "skipped": skipped,
        "truncated": truncated,
    }
    (OUT_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("\nFold mean:")
    print(mean.round(4).to_string())
    print("\nAggregate totals:")
    print(pd.Series(aggregate).round(4).to_string())
    print(f"\nWrote {OUT_DIR}")


if __name__ == "__main__":
    main()
