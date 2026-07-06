from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[2]
BASE_SCRIPT = ROOT / "MERIX SUBMISSION" / "MIREX_Model" / "run_mazurkabl_l2plus_weighted_target_experiment.py"
OUT_DIR = ROOT / "MERIX SUBMISSION" / "MIREX_Model" / "mazurkabl_l2plus_performer_sample_experiment"
LEVEL_WEIGHTS = {2: 0.28, 3: 0.46, 4: 0.64, 5: 0.82, 6: 1.00}
EVENT_MIN = 0.05


spec = importlib.util.spec_from_file_location("mazurkabl_piece_base", BASE_SCRIPT)
base = importlib.util.module_from_spec(spec)
sys.modules["mazurkabl_piece_base"] = base
assert spec.loader is not None
spec.loader.exec_module(base)


def parse_npz_name(path: Path) -> tuple[str, str, int]:
    match = re.match(r"(M\d+-\d+)_(.+)_L([1-6])\.npz$", path.name)
    if not match:
        raise ValueError(f"Cannot parse label filename: {path.name}")
    return match.group(1), match.group(2), int(match.group(3))


def load_performer_labels() -> tuple[list[str], dict[str, str], dict[str, np.ndarray], dict[str, dict[int, np.ndarray]]]:
    grouped: dict[tuple[str, str], dict[int, Path]] = {}
    for path in sorted(base.LABEL_DIR.glob("*_L2.npz")):
        piece, performer, _ = parse_npz_name(path)
        grouped.setdefault((piece, performer), {})
    for path in sorted(base.LABEL_DIR.glob("*_L[2-6].npz")):
        piece, performer, level = parse_npz_name(path)
        if level in LEVEL_WEIGHTS:
            grouped.setdefault((piece, performer), {})[level] = path

    sample_to_piece: dict[str, str] = {}
    labels: dict[str, np.ndarray] = {}
    components: dict[str, dict[int, np.ndarray]] = {}
    for (piece, performer), paths in sorted(grouped.items()):
        if any(level not in paths for level in LEVEL_WEIGHTS):
            continue
        sample_id = f"{piece}_{performer}"
        weighted = []
        components[sample_id] = {}
        for level, weight in LEVEL_WEIGHTS.items():
            arr = np.load(paths[level], allow_pickle=True)["boundary_probs"].astype(np.float32)
            components[sample_id][level] = arr
            weighted.append(float(weight) * arr)
        lengths = {len(x) for x in weighted}
        if len(lengths) != 1:
            raise RuntimeError(f"{sample_id}: inconsistent level lengths {sorted(lengths)}")
        labels[sample_id] = np.maximum.reduce(weighted).astype(np.float32)
        sample_to_piece[sample_id] = piece
    samples = sorted(labels)
    return samples, sample_to_piece, labels, components


def make_sample_feature_map(piece_features: dict[str, np.ndarray], sample_to_piece: dict[str, str]) -> dict[str, np.ndarray]:
    return {sample_id: piece_features[piece] for sample_id, piece in sample_to_piece.items()}


def train_one_samplewise(
    features: dict[str, np.ndarray],
    labels: dict[str, np.ndarray],
    train_samples: list[str],
    seed: int,
) -> tuple[nn.Module, np.ndarray, np.ndarray]:
    base.set_seed(seed)
    train_feats = [features[s] for s in train_samples]
    train_labs = [labels[s] for s in train_samples]
    stacked = np.concatenate(train_feats, axis=0)
    mean = stacked.mean(axis=0).astype(np.float32)
    std = stacked.std(axis=0).astype(np.float32)
    std[std < 1e-6] = 1.0
    x_all = np.concatenate(train_feats, axis=0).astype(np.float32)
    y_all = np.concatenate(train_labs, axis=0).astype(np.float32)
    x_all = ((x_all - mean) / std).astype(np.float32)
    model = base.TinyMLP(input_dim=x_all.shape[-1])
    pos = float(np.sum(y_all))
    neg = float(len(y_all) - pos)
    pos_weight = torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best_state = None
    best_loss = float("inf")
    stale = 0
    rng = np.random.default_rng(seed)
    batch_size = 32768
    x_tensor = torch.from_numpy(x_all)
    y_tensor = torch.from_numpy(y_all)
    for _epoch in range(1, 16):
        model.train()
        order = rng.permutation(len(y_all))
        total_loss = 0.0
        total_weight = 0.0
        for start in range(0, len(order), batch_size):
            idx = torch.from_numpy(order[start : start + batch_size])
            x = x_tensor[idx]
            y = y_tensor[idx]
            opt.zero_grad()
            logits = model(x)
            loss_matrix = loss_fn(logits, y)
            loss = loss_matrix.mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += float(loss_matrix.sum().item())
            total_weight += float(len(idx))
        value = total_loss / max(total_weight, 1.0)
        if value < best_loss - 1e-4:
            best_loss = value
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= 4:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, mean, std


def match_weighted(pred: np.ndarray, target: np.ndarray, tolerance: int) -> dict:
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


def choose_threshold_samplewise(scores: dict[str, np.ndarray], labels: dict[str, np.ndarray], samples: list[str]) -> tuple[float, dict]:
    best_th = 0.05
    best = None
    for th in np.linspace(0.05, 0.95, 37):
        metrics = aggregate([match_weighted(base.extract_events(scores[s], float(th)), labels[s], tolerance=1) for s in samples])
        key = (metrics["f1"], metrics["precision"], metrics["weighted_recall"], metrics["recall"])
        if best is None or key > (best["f1"], best["precision"], best["weighted_recall"], best["recall"]):
            best = metrics
            best_th = float(th)
    assert best is not None
    return best_th, best


def expected_count_from_sample_density(labels: dict[str, np.ndarray], train_samples: list[str], heldout_len: int) -> int:
    events = sum(int(np.count_nonzero(labels[s] >= EVENT_MIN)) for s in train_samples)
    beats = sum(int(len(labels[s])) for s in train_samples)
    return max(1, int(round(heldout_len * events / max(beats, 1))))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pieces, consensus_labels, consensus_components = base.load_l2plus_weighted_labels()
    piece_features, feature_cols = base.load_piece_features(pieces)
    samples, sample_to_piece, performer_labels, performer_components = load_performer_labels()
    features = make_sample_feature_map(piece_features, sample_to_piece)
    samples = [s for s in samples if sample_to_piece[s] in piece_features and len(features[s]) == len(performer_labels[s])]

    folds = base.make_folds(pieces, n_folds=5, seed=42)
    fold_rows = []
    piece_eval_rows = []
    performer_eval_rows = []
    for fold_idx, val_pieces in enumerate(folds, start=1):
        val_piece_set = set(val_pieces)
        train_pieces = [p for p in pieces if p not in val_piece_set]
        train_samples = [s for s in samples if sample_to_piece[s] in set(train_pieces)]
        val_samples = [s for s in samples if sample_to_piece[s] in val_piece_set]
        model, mean, std = train_one_samplewise(features, performer_labels, train_samples, seed=9200 + fold_idx)

        train_scores = base.predict(model, features, train_samples, mean, std)
        val_scores_sample = base.predict(model, features, val_samples, mean, std)
        threshold, train_metric = choose_threshold_samplewise(train_scores, performer_labels, train_samples)

        performer_th_items_1 = []
        performer_th_items_0 = []
        performer_den_items_1 = []
        performer_den_items_0 = []
        for sample in val_samples:
            expected = expected_count_from_sample_density(performer_labels, train_samples, len(performer_labels[sample]))
            th_pred = base.extract_events(val_scores_sample[sample], threshold)
            den_pred = base.extract_top_density(val_scores_sample[sample], expected)
            th1 = match_weighted(th_pred, performer_labels[sample], tolerance=1)
            th0 = match_weighted(th_pred, performer_labels[sample], tolerance=0)
            den1 = match_weighted(den_pred, performer_labels[sample], tolerance=1)
            den0 = match_weighted(den_pred, performer_labels[sample], tolerance=0)
            performer_th_items_1.append(th1)
            performer_th_items_0.append(th0)
            performer_den_items_1.append(den1)
            performer_den_items_0.append(den0)
            performer_eval_rows.append(
                {
                    "fold": fold_idx,
                    "sample_id": sample,
                    "piece": sample_to_piece[sample],
                    "threshold_pred_events": th1["pred_events"],
                    "density_pred_events": den1["pred_events"],
                    "true_events": den1["true_events"],
                    "threshold_f1_tol1": th1["f1"],
                    "threshold_weighted_recall_tol1": th1["weighted_recall"],
                    "density_f1_tol1": den1["f1"],
                    "density_weighted_recall_tol1": den1["weighted_recall"],
                }
            )

        # Directly comparable piece-consensus evaluation: one prediction per held-out piece.
        val_piece_scores = base.predict(model, piece_features, val_pieces, mean, std)
        train_piece_scores = base.predict(model, piece_features, train_pieces, mean, std)
        piece_threshold, _ = base.choose_threshold(train_piece_scores, {p: consensus_labels[p] for p in train_pieces}, tolerance=1)
        piece_th_items_1 = []
        piece_th_items_0 = []
        piece_den_items_1 = []
        piece_den_items_0 = []
        for piece in val_pieces:
            expected = base.expected_count_from_train_density(consensus_labels, train_pieces, len(consensus_labels[piece]))
            th_pred = base.extract_events(val_piece_scores[piece], piece_threshold)
            den_pred = base.extract_top_density(val_piece_scores[piece], expected)
            th1 = match_weighted(th_pred, consensus_labels[piece], tolerance=1)
            th0 = match_weighted(th_pred, consensus_labels[piece], tolerance=0)
            den1 = match_weighted(den_pred, consensus_labels[piece], tolerance=1)
            den0 = match_weighted(den_pred, consensus_labels[piece], tolerance=0)
            piece_th_items_1.append(th1)
            piece_th_items_0.append(th0)
            piece_den_items_1.append(den1)
            piece_den_items_0.append(den0)
            piece_eval_rows.append(
                {
                    "fold": fold_idx,
                    "piece": piece,
                    "threshold_pred_events": th1["pred_events"],
                    "density_pred_events": den1["pred_events"],
                    "true_events": den1["true_events"],
                    "threshold_f1_tol1": th1["f1"],
                    "threshold_weighted_recall_tol1": th1["weighted_recall"],
                    "density_f1_tol1": den1["f1"],
                    "density_weighted_recall_tol1": den1["weighted_recall"],
                }
            )
            pd.DataFrame(
                {
                    "beat_idx": np.arange(len(val_piece_scores[piece])),
                    "piece_score": val_piece_scores[piece],
                    "consensus_target": consensus_labels[piece],
                    "threshold_pred": np.isin(np.arange(len(val_piece_scores[piece])), th_pred).astype(np.int8),
                    "density_pred": np.isin(np.arange(len(val_piece_scores[piece])), den_pred).astype(np.int8),
                }
            ).to_csv(OUT_DIR / f"fold{fold_idx}_{piece}_piece_consensus_predictions.csv", index=False)

        perf_th1 = aggregate(performer_th_items_1)
        perf_th0 = aggregate(performer_th_items_0)
        perf_den1 = aggregate(performer_den_items_1)
        perf_den0 = aggregate(performer_den_items_0)
        piece_th1 = aggregate(piece_th_items_1)
        piece_th0 = aggregate(piece_th_items_0)
        piece_den1 = aggregate(piece_den_items_1)
        piece_den0 = aggregate(piece_den_items_0)
        row = {
            "fold": fold_idx,
            "train_piece_count": len(train_pieces),
            "val_piece_count": len(val_pieces),
            "train_sample_count": len(train_samples),
            "val_sample_count": len(val_samples),
            "performer_threshold": threshold,
            "piece_consensus_threshold": piece_threshold,
            "performer_threshold_precision_tol1": perf_th1["precision"],
            "performer_threshold_recall_tol1": perf_th1["recall"],
            "performer_threshold_f1_tol1": perf_th1["f1"],
            "performer_threshold_weighted_recall_tol1": perf_th1["weighted_recall"],
            "performer_threshold_f1_tol0": perf_th0["f1"],
            "performer_threshold_weighted_recall_tol0": perf_th0["weighted_recall"],
            "performer_density_precision_tol1": perf_den1["precision"],
            "performer_density_recall_tol1": perf_den1["recall"],
            "performer_density_f1_tol1": perf_den1["f1"],
            "performer_density_weighted_recall_tol1": perf_den1["weighted_recall"],
            "performer_density_f1_tol0": perf_den0["f1"],
            "performer_density_weighted_recall_tol0": perf_den0["weighted_recall"],
            "piece_threshold_precision_tol1": piece_th1["precision"],
            "piece_threshold_recall_tol1": piece_th1["recall"],
            "piece_threshold_f1_tol1": piece_th1["f1"],
            "piece_threshold_weighted_recall_tol1": piece_th1["weighted_recall"],
            "piece_threshold_f1_tol0": piece_th0["f1"],
            "piece_threshold_weighted_recall_tol0": piece_th0["weighted_recall"],
            "piece_density_precision_tol1": piece_den1["precision"],
            "piece_density_recall_tol1": piece_den1["recall"],
            "piece_density_f1_tol1": piece_den1["f1"],
            "piece_density_weighted_recall_tol1": piece_den1["weighted_recall"],
            "piece_density_f1_tol0": piece_den0["f1"],
            "piece_density_weighted_recall_tol0": piece_den0["weighted_recall"],
            "piece_density_pred_events": piece_den1["pred_events"],
            "piece_true_events": piece_den1["true_events"],
            "performer_density_pred_events": perf_den1["pred_events"],
            "performer_true_events": perf_den1["true_events"],
        }
        fold_rows.append(row)
        print(
            f"fold {fold_idx}: performer density F1/WR={perf_den1['f1']:.3f}/{perf_den1['weighted_recall']:.3f}; "
            f"piece consensus density F1/WR={piece_den1['f1']:.3f}/{piece_den1['weighted_recall']:.3f}"
        )

    fold_summary = pd.DataFrame(fold_rows)
    fold_summary.to_csv(OUT_DIR / "fold_summary.csv", index=False)
    pd.DataFrame(piece_eval_rows).to_csv(OUT_DIR / "piece_consensus_eval.csv", index=False)
    pd.DataFrame(performer_eval_rows).to_csv(OUT_DIR / "performer_eval.csv", index=False)
    mean = fold_summary.drop(columns=["fold"]).select_dtypes(include=[np.number]).mean()
    mean.to_frame("mean").to_csv(OUT_DIR / "fold_mean.csv")
    metadata = {
        "feature_columns": feature_cols,
        "pieces": pieces,
        "samples": samples,
        "folds": folds,
        "level_weights": LEVEL_WEIGHTS,
        "event_min": EVENT_MIN,
        "sample_definition": "one performer recording is one training sequence; piece-level folds",
        "piece_eval_definition": "same trained model evaluated once per piece against consensus target for baseline comparability",
    }
    (OUT_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("\nMean:")
    print(mean.round(4).to_string())
    print(f"\nWrote {OUT_DIR}")


if __name__ == "__main__":
    main()
