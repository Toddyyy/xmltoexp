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
BEAT_TABLE = ROOT / "MERIX SUBMISSION" / "MIREX_Model_meter_auto" / "outputs" / "atepp_top20_mixed_by_segment_nonan_beat_table.csv.gz"
LABEL_DIR = ROOT / "MERIX SUBMISSION" / "MIREX_Model_meter_auto" / "beat_data_atepp_top20_mixed_by_segment_performer_levels"
OUT_DIR = ROOT / "MERIX SUBMISSION" / "MIREX_Model" / "atepp20_l2plus_performer_sample_experiment"

LEVEL_WEIGHTS = {2: 0.28, 3: 0.46, 4: 0.64, 5: 0.82, 6: 1.00}
EVENT_MIN = 0.05

spec = importlib.util.spec_from_file_location("mazurkabl_quick_base_perf", BASE_SCRIPT)
base = importlib.util.module_from_spec(spec)
sys.modules["mazurkabl_quick_base_perf"] = base
assert spec.loader is not None
spec.loader.exec_module(base)


class PointLinear(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(-1)


def performer_prefix(sample_id: str) -> str:
    text = str(sample_id)
    return text.removesuffix("_G3SAL")


def load_performer_label(prefix: str) -> np.ndarray:
    weighted = []
    for level, weight in LEVEL_WEIGHTS.items():
        path = LABEL_DIR / f"{prefix}_L{level}.npz"
        if not path.exists():
            raise FileNotFoundError(path)
        arr = np.load(path, allow_pickle=True)["boundary_probs"].astype(np.float32)
        weighted.append(float(weight) * arr)
    return np.maximum.reduce(weighted).astype(np.float32)


def load_samples() -> tuple[list[dict], list[str]]:
    df = pd.read_csv(BEAT_TABLE)
    feature_cols = [
        col
        for col in df.columns
        if base.is_feature_column(col)
        and col not in {"protocol_split", "local_time_signature", "meter_group", "segment_id"}
        and pd.api.types.is_numeric_dtype(df[col])
    ]
    samples = []
    for sample_id, group in df.sort_values(["sample_id", "beat_idx"]).groupby("sample_id", sort=False):
        group = group.groupby("beat_idx", as_index=False).first().sort_values("beat_idx")
        prefix = performer_prefix(sample_id)
        label = load_performer_label(prefix)
        if len(label) != len(group):
            raise RuntimeError(f"{sample_id}: label length {len(label)} != feature length {len(group)}")
        samples.append(
            {
                "sample_id": str(sample_id),
                "piece_id": str(group["piece_id"].iloc[0]),
                "performer_id": str(group["performer_id"].iloc[0]),
                "features": group[feature_cols].to_numpy(dtype=np.float32),
                "label": label,
            }
        )
    return samples, feature_cols


def train_one_sample_level(samples: list[dict], train_pieces: list[str], seed: int):
    base.set_seed(seed)
    train = [s for s in samples if s["piece_id"] in set(train_pieces)]
    stacked = np.concatenate([s["features"] for s in train], axis=0)
    mean = stacked.mean(axis=0).astype(np.float32)
    std = stacked.std(axis=0).astype(np.float32)
    std[std < 1e-6] = 1.0

    model = PointLinear(input_dim=train[0]["features"].shape[1])
    train_x = np.concatenate([s["features"] for s in train], axis=0)
    train_labels = np.concatenate([s["label"] for s in train], axis=0).astype(np.float32)
    train_x = ((train_x - mean) / std).astype(np.float32)
    pos = float(train_labels.sum())
    neg = float(train_labels.shape[0] - pos)
    pos_weight = torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)

    rng = np.random.default_rng(seed)
    best_state = None
    best_loss = float("inf")
    stale = 0
    batch_size = 8192
    x_all = torch.from_numpy(train_x)
    y_all = torch.from_numpy(train_labels)
    for _epoch in range(1, 9):
        order = rng.permutation(len(train_labels))
        model.train()
        losses = []
        for start in range(0, len(order), batch_size):
            idx = torch.from_numpy(order[start : start + batch_size].astype(np.int64))
            x = x_all[idx]
            y = y_all[idx]
            opt.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.item()))
        epoch_loss = float(np.mean(losses)) if losses else float("inf")
        if epoch_loss < best_loss - 1e-5:
            best_loss = epoch_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= 3:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, mean, std


def predict_samples(model, samples: list[dict], pieces: list[str], mean: np.ndarray, std: np.ndarray) -> dict[str, np.ndarray]:
    selected = {s["sample_id"]: s["features"] for s in samples if s["piece_id"] in set(pieces)}
    return base.predict(model, selected, list(selected.keys()), mean, std)


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


def choose_threshold_sample_level(scores: dict[str, np.ndarray], labels: dict[str, np.ndarray], tolerance: int) -> tuple[float, dict]:
    best_th = 0.05
    best = None
    for th in np.linspace(0.05, 0.95, 37):
        items = []
        for sample_id, score in scores.items():
            pred = base.extract_events(score, threshold=float(th))
            items.append(weighted_metrics_from_events(pred, labels[sample_id], tolerance=tolerance))
        metrics = aggregate(items)
        key = (metrics["f1"], metrics["precision"], metrics["weighted_recall"], metrics["recall"])
        if best is None or key > (best["f1"], best["precision"], best["weighted_recall"], best["recall"]):
            best = metrics
            best_th = float(th)
    assert best is not None
    return best_th, best


def expected_count_from_train_density_sample_level(labels: dict[str, np.ndarray], train_ids: list[str], heldout_len: int) -> int:
    train_events = sum(int(np.count_nonzero(labels[sid] >= EVENT_MIN)) for sid in train_ids)
    train_beats = sum(int(len(labels[sid])) for sid in train_ids)
    return max(1, int(round(heldout_len * train_events / max(train_beats, 1))))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    samples, feature_cols = load_samples()
    pieces = sorted({s["piece_id"] for s in samples})
    labels = {s["sample_id"]: s["label"] for s in samples}
    sample_by_id = {s["sample_id"]: s for s in samples}
    folds = base.make_folds(pieces, n_folds=5, seed=42)

    label_stats = []
    for s in samples:
        label_stats.append(
            {
                "sample_id": s["sample_id"],
                "piece_id": s["piece_id"],
                "performer_id": s["performer_id"],
                "num_beats": len(s["label"]),
                "target_sum": float(np.sum(s["label"])),
                "event_count_target_ge_0p05": int(np.count_nonzero(s["label"] >= EVENT_MIN)),
            }
        )
    pd.DataFrame(label_stats).to_csv(OUT_DIR / "label_stats_by_performer.csv", index=False)

    rows = []
    piece_rows = []
    for fold_idx, val_pieces in enumerate(folds, start=1):
        train_pieces = [p for p in pieces if p not in set(val_pieces)]
        model, mean, std = train_one_sample_level(samples, train_pieces, seed=9200 + fold_idx)
        train_scores = predict_samples(model, samples, train_pieces, mean, std)
        val_scores = predict_samples(model, samples, val_pieces, mean, std)
        train_labels = {sid: labels[sid] for sid in train_scores}
        threshold, train_metric = choose_threshold_sample_level(train_scores, train_labels, tolerance=1)

        th1_items = []
        th0_items = []
        den1_items = []
        den0_items = []
        train_ids = list(train_scores.keys())
        for sid, score in val_scores.items():
            target = labels[sid]
            threshold_pred = base.extract_events(score, threshold=threshold)
            expected = expected_count_from_train_density_sample_level(labels, train_ids, len(target))
            density_pred = base.extract_top_density(score, expected)
            th1 = weighted_metrics_from_events(threshold_pred, target, tolerance=1)
            th0 = weighted_metrics_from_events(threshold_pred, target, tolerance=0)
            den1 = weighted_metrics_from_events(density_pred, target, tolerance=1)
            den0 = weighted_metrics_from_events(density_pred, target, tolerance=0)
            th1_items.append(th1)
            th0_items.append(th0)
            den1_items.append(den1)
            den0_items.append(den0)
            meta = sample_by_id[sid]
            piece_rows.append(
                {
                    "fold": fold_idx,
                    "sample_id": sid,
                    "piece_id": meta["piece_id"],
                    "performer_id": meta["performer_id"],
                    "true_events": den1["true_events"],
                    "threshold_pred_events": th1["pred_events"],
                    "threshold_precision_tol1": th1["precision"],
                    "threshold_recall_tol1": th1["recall"],
                    "threshold_f1_tol1": th1["f1"],
                    "threshold_weighted_recall_tol1": th1["weighted_recall"],
                    "density_pred_events": den1["pred_events"],
                    "density_precision_tol1": den1["precision"],
                    "density_recall_tol1": den1["recall"],
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
                "train_samples": len(train_scores),
                "val_samples": len(val_scores),
                "threshold_from_train_tol1": threshold,
                "train_f1_tol1": train_metric["f1"],
                "threshold_precision_tol1": th1["precision"],
                "threshold_recall_tol1": th1["recall"],
                "threshold_f1_tol1": th1["f1"],
                "threshold_weighted_recall_tol1": th1["weighted_recall"],
                "threshold_matches_tol1": th1["matches"],
                "threshold_matched_weight_tol1": th1["matched_weight"],
                "threshold_total_weight_tol1": th1["total_weight"],
                "threshold_precision_tol0": th0["precision"],
                "threshold_recall_tol0": th0["recall"],
                "threshold_f1_tol0": th0["f1"],
                "threshold_weighted_recall_tol0": th0["weighted_recall"],
                "threshold_matches_tol0": th0["matches"],
                "threshold_matched_weight_tol0": th0["matched_weight"],
                "threshold_total_weight_tol0": th0["total_weight"],
                "density_precision_tol1": den1["precision"],
                "density_recall_tol1": den1["recall"],
                "density_f1_tol1": den1["f1"],
                "density_weighted_recall_tol1": den1["weighted_recall"],
                "density_matches_tol1": den1["matches"],
                "density_matched_weight_tol1": den1["matched_weight"],
                "density_total_weight_tol1": den1["total_weight"],
                "density_precision_tol0": den0["precision"],
                "density_recall_tol0": den0["recall"],
                "density_f1_tol0": den0["f1"],
                "density_weighted_recall_tol0": den0["weighted_recall"],
                "density_matches_tol0": den0["matches"],
                "density_matched_weight_tol0": den0["matched_weight"],
                "density_total_weight_tol0": den0["total_weight"],
                "density_pred_events": den1["pred_events"],
                "threshold_pred_events": th1["pred_events"],
                "true_events": den1["true_events"],
            }
        )
        print(
            f"fold {fold_idx}: samples train/val={len(train_scores)}/{len(val_scores)} "
            f"threshold P/R/F1/WR={th1['precision']:.3f}/{th1['recall']:.3f}/{th1['f1']:.3f}/{th1['weighted_recall']:.3f}; "
            f"density P/R/F1/WR={den1['precision']:.3f}/{den1['recall']:.3f}/{den1['f1']:.3f}/{den1['weighted_recall']:.3f}"
        )

    summary = pd.DataFrame(rows)
    per_sample = pd.DataFrame(piece_rows)
    summary.to_csv(OUT_DIR / "fold_summary.csv", index=False)
    per_sample.to_csv(OUT_DIR / "sample_summary.csv", index=False)
    metric_cols = [
        "threshold_precision_tol1",
        "threshold_recall_tol1",
        "threshold_f1_tol1",
        "threshold_weighted_recall_tol1",
        "threshold_matches_tol1",
        "threshold_precision_tol0",
        "threshold_recall_tol0",
        "threshold_f1_tol0",
        "threshold_weighted_recall_tol0",
        "threshold_matches_tol0",
        "density_precision_tol1",
        "density_recall_tol1",
        "density_f1_tol1",
        "density_weighted_recall_tol1",
        "density_matches_tol1",
        "density_precision_tol0",
        "density_recall_tol0",
        "density_f1_tol0",
        "density_weighted_recall_tol0",
        "density_matches_tol0",
        "density_pred_events",
        "threshold_pred_events",
        "true_events",
        "train_samples",
        "val_samples",
    ]
    mean = summary[metric_cols].mean()
    mean.to_frame("mean").to_csv(OUT_DIR / "fold_mean.csv")
    totals = {
        "threshold_pred_events": int(summary["threshold_pred_events"].sum()),
        "density_pred_events": int(summary["density_pred_events"].sum()),
        "true_events": int(summary["true_events"].sum()),
        "threshold_matches_tol1": int(summary["threshold_matches_tol1"].sum()),
        "density_matches_tol1": int(summary["density_matches_tol1"].sum()),
        "threshold_matches_tol0": int(summary["threshold_matches_tol0"].sum()),
        "density_matches_tol0": int(summary["density_matches_tol0"].sum()),
        "threshold_precision_tol1": float(summary["threshold_matches_tol1"].sum() / max(summary["threshold_pred_events"].sum(), 1)),
        "density_precision_tol1": float(summary["density_matches_tol1"].sum() / max(summary["density_pred_events"].sum(), 1)),
        "threshold_recall_tol1": float(summary["threshold_matches_tol1"].sum() / max(summary["true_events"].sum(), 1)),
        "density_recall_tol1": float(summary["density_matches_tol1"].sum() / max(summary["true_events"].sum(), 1)),
        "threshold_weighted_recall_tol1": float(summary["threshold_matched_weight_tol1"].sum() / max(summary["threshold_total_weight_tol1"].sum(), 1e-12)),
        "density_weighted_recall_tol1": float(summary["density_matched_weight_tol1"].sum() / max(summary["density_total_weight_tol1"].sum(), 1e-12)),
    }
    for prefix in ("threshold", "density"):
        precision = totals[f"{prefix}_precision_tol1"]
        recall = totals[f"{prefix}_recall_tol1"]
        totals[f"{prefix}_f1_tol1"] = float(2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    pd.Series(totals).to_frame("total").to_csv(OUT_DIR / "aggregate_totals.csv")
    metadata = {
        "beat_table": str(BEAT_TABLE),
        "label_dir": str(LABEL_DIR),
        "feature_columns": feature_cols,
        "pieces": pieces,
        "folds": folds,
        "sample_count": len(samples),
        "target_rule": "per-performer max(weight_L * boundary_L) for L2-L6; no performer consensus averaging",
        "event_min": EVENT_MIN,
    }
    (OUT_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("\nMean:")
    print(mean.round(4).to_string())
    print("\nAggregate totals:")
    print(pd.Series(totals).round(4).to_string())
    print(f"\nWrote {OUT_DIR}")


if __name__ == "__main__":
    main()
