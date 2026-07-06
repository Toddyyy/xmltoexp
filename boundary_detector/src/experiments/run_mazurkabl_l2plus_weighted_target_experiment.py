from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.signal import find_peaks
from torch import nn


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
BEAT_TABLE = DATA / "features" / "mazurkabl_handcrafted_beat_table.csv.gz"
LABEL_DIR = DATA / "labels" / "mazurkabl_performer_levels"
OUT_DIR = RESULTS / "mazurkabl_l2plus_weighted_target_experiment"
LEVEL_WEIGHTS = {2: 0.28, 3: 0.46, 4: 0.64, 5: 0.82, 6: 1.00}
EVENT_MIN = 0.05
MIN_DISTANCE = 6

META_PREFIXES = ("boundary_", "target_", "salience_", "stage_")
META_COLUMNS = {
    "source_path",
    "sample_id",
    "piece_id",
    "performer_id",
    "level",
    "split",
    "beat_idx",
    "num_beats",
    "boundary_prob",
    "boundary_peak",
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def is_feature_column(col: str) -> bool:
    if col in META_COLUMNS:
        return False
    return not any(col.startswith(prefix) for prefix in META_PREFIXES)


def piece_id_from_npz(path: Path) -> str:
    match = re.match(r"(M\d+-\d+)_", path.name)
    if not match:
        raise ValueError(f"Cannot parse piece id from {path.name}")
    return match.group(1)


def load_piece_features(pieces: list[str]) -> tuple[dict[str, np.ndarray], list[str]]:
    df = pd.read_csv(BEAT_TABLE)
    df = df[df["piece_id"].isin(pieces)].copy()
    feature_cols = [col for col in df.columns if is_feature_column(col)]
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
        raise FileNotFoundError(f"No label files for {piece} L{level}")
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


class TinyMLP(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 96, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def pad_batch(features: list[np.ndarray], labels: list[np.ndarray], mean: np.ndarray, std: np.ndarray):
    batch = len(features)
    max_len = max(len(x) for x in features)
    feat_dim = features[0].shape[1]
    x = np.zeros((batch, max_len, feat_dim), dtype=np.float32)
    y = np.zeros((batch, max_len), dtype=np.float32)
    m = np.zeros((batch, max_len), dtype=np.float32)
    for i, (feat, lab) in enumerate(zip(features, labels)):
        n = len(feat)
        x[i, :n] = (feat - mean) / std
        y[i, :n] = lab
        m[i, :n] = 1.0
    return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(m)


def extract_events(scores: np.ndarray, threshold: float, min_distance: int = MIN_DISTANCE) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float32)
    peaks, _ = find_peaks(scores, height=threshold, distance=max(int(min_distance), 1))
    candidates = list(peaks.astype(int))
    if scores.size == 1 and scores[0] >= threshold:
        candidates.append(0)
    elif scores.size > 1:
        if scores[0] >= threshold and scores[0] >= scores[1]:
            candidates.append(0)
        if scores[-1] >= threshold and scores[-1] >= scores[-2]:
            candidates.append(scores.size - 1)
    candidates = sorted(set(candidates), key=lambda idx: (-float(scores[idx]), idx))
    kept = []
    for idx in candidates:
        if all(abs(idx - prev) >= max(int(min_distance), 1) for prev in kept):
            kept.append(idx)
    return np.asarray(sorted(kept), dtype=np.int32)


def extract_top_density(scores: np.ndarray, expected_count: int, min_distance: int = MIN_DISTANCE) -> np.ndarray:
    if expected_count <= 0:
        return np.zeros(0, dtype=np.int32)
    scores = np.asarray(scores, dtype=np.float32)
    peaks, _ = find_peaks(scores, distance=max(int(min_distance), 1))
    candidates = list(peaks.astype(int))
    if scores.size:
        candidates.extend([0, scores.size - 1])
    candidates = sorted(set(candidates), key=lambda idx: (-float(scores[idx]), idx))
    kept = []
    for idx in candidates:
        if all(abs(idx - prev) >= max(int(min_distance), 1) for prev in kept):
            kept.append(idx)
            if len(kept) >= expected_count:
                break
    return np.asarray(sorted(kept), dtype=np.int32)


def match_events(pred: np.ndarray, true: np.ndarray, tolerance: int) -> tuple[int, list[int]]:
    used = set()
    offsets = []
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
            offsets.append(int(p) - int(true[best]))
    return len(offsets), offsets


@dataclass
class Metrics:
    precision: float
    recall: float
    f1: float
    pred_events: int
    true_events: int
    matches: int
    mean_offset: float | None


def metrics_from_events(pred: np.ndarray, true: np.ndarray, tolerance: int) -> Metrics:
    matches, offsets = match_events(pred, true, tolerance)
    precision = matches / len(pred) if len(pred) else 0.0
    recall = matches / len(true) if len(true) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return Metrics(
        precision=precision,
        recall=recall,
        f1=f1,
        pred_events=int(len(pred)),
        true_events=int(len(true)),
        matches=int(matches),
        mean_offset=float(np.mean(offsets)) if offsets else None,
    )


def aggregate_metrics(items: list[Metrics]) -> Metrics:
    pred = sum(m.pred_events for m in items)
    true = sum(m.true_events for m in items)
    matches = sum(m.matches for m in items)
    precision = matches / pred if pred else 0.0
    recall = matches / true if true else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return Metrics(precision, recall, f1, pred, true, matches, None)


def evaluate_threshold(
    scores_by_piece: dict[str, np.ndarray],
    labels_by_piece: dict[str, np.ndarray],
    threshold: float,
    tolerance: int,
) -> Metrics:
    items = []
    for piece, scores in scores_by_piece.items():
        pred = extract_events(scores, threshold=threshold)
        true = np.flatnonzero(labels_by_piece[piece] >= EVENT_MIN).astype(np.int32)
        items.append(metrics_from_events(pred, true, tolerance))
    return aggregate_metrics(items)


def choose_threshold(scores_by_piece: dict[str, np.ndarray], labels_by_piece: dict[str, np.ndarray], tolerance: int) -> tuple[float, Metrics]:
    best_th = 0.05
    best = None
    for th in np.linspace(0.05, 0.95, 37):
        metrics = evaluate_threshold(scores_by_piece, labels_by_piece, float(th), tolerance)
        key = (metrics.f1, metrics.precision, metrics.recall)
        if best is None or key > (best.f1, best.precision, best.recall):
            best = metrics
            best_th = float(th)
    assert best is not None
    return best_th, best


@torch.no_grad()
def predict(model: nn.Module, features: dict[str, np.ndarray], pieces: list[str], mean: np.ndarray, std: np.ndarray) -> dict[str, np.ndarray]:
    model.eval()
    out = {}
    for piece in pieces:
        x = torch.from_numpy(((features[piece] - mean) / std)[None, :, :].astype(np.float32))
        out[piece] = torch.sigmoid(model(x)).squeeze(0).cpu().numpy().astype(np.float32)
    return out


def train_one(features: dict[str, np.ndarray], labels: dict[str, np.ndarray], train_pieces: list[str], seed: int) -> tuple[TinyMLP, np.ndarray, np.ndarray]:
    set_seed(seed)
    train_feats = [features[p] for p in train_pieces]
    train_labs = [labels[p] for p in train_pieces]
    stacked = np.concatenate(train_feats, axis=0)
    mean = stacked.mean(axis=0).astype(np.float32)
    std = stacked.std(axis=0).astype(np.float32)
    std[std < 1e-6] = 1.0

    x, y, mask = pad_batch(train_feats, train_labs, mean, std)
    model = TinyMLP(input_dim=x.shape[-1])
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


def make_folds(pieces: list[str], n_folds: int = 5, seed: int = 42) -> list[list[str]]:
    rng = random.Random(seed)
    shuffled = list(pieces)
    rng.shuffle(shuffled)
    return [sorted(shuffled[i::n_folds]) for i in range(n_folds)]


def expected_count_from_train_density(labels: dict[str, np.ndarray], train_pieces: list[str], heldout_len: int) -> int:
    train_events = sum(int(np.count_nonzero(labels[p] >= EVENT_MIN)) for p in train_pieces)
    train_beats = sum(int(len(labels[p])) for p in train_pieces)
    return max(1, int(round(heldout_len * train_events / max(train_beats, 1))))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pieces, labels, components = load_l2plus_weighted_labels()
    features, feature_cols = load_piece_features(pieces)

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

    folds = make_folds(pieces, n_folds=5, seed=42)
    rows = []
    piece_rows = []
    for fold_idx, val_pieces in enumerate(folds, start=1):
        train_pieces = [p for p in pieces if p not in set(val_pieces)]
        model, mean, std = train_one(features, labels, train_pieces, seed=6200 + fold_idx)
        train_scores = predict(model, features, train_pieces, mean, std)
        val_scores = predict(model, features, val_pieces, mean, std)
        threshold, train_metric = choose_threshold(train_scores, {p: labels[p] for p in train_pieces}, tolerance=1)
        th_tol1 = evaluate_threshold(val_scores, {p: labels[p] for p in val_pieces}, threshold, tolerance=1)
        th_tol0 = evaluate_threshold(val_scores, {p: labels[p] for p in val_pieces}, threshold, tolerance=0)

        density_tol1_items = []
        density_tol0_items = []
        for piece in val_pieces:
            true = np.flatnonzero(labels[piece] >= EVENT_MIN).astype(np.int32)
            expected = expected_count_from_train_density(labels, train_pieces, len(labels[piece]))
            pred = extract_top_density(val_scores[piece], expected)
            m1 = metrics_from_events(pred, true, tolerance=1)
            m0 = metrics_from_events(pred, true, tolerance=0)
            density_tol1_items.append(m1)
            density_tol0_items.append(m0)
            piece_rows.append(
                {
                    "fold": fold_idx,
                    "piece": piece,
                    "num_beats": len(labels[piece]),
                    "true_events": len(true),
                    "density_expected_events": expected,
                    "density_pred_events": len(pred),
                    "density_precision_tol1": m1.precision,
                    "density_recall_tol1": m1.recall,
                    "density_f1_tol1": m1.f1,
                    "density_precision_tol0": m0.precision,
                    "density_recall_tol0": m0.recall,
                    "density_f1_tol0": m0.f1,
                }
            )
            pd.DataFrame(
                {
                    "beat_idx": np.arange(len(val_scores[piece])),
                    "score": val_scores[piece],
                    "target_l2plus_weighted": labels[piece],
                    "target_event_ge_0p05": (labels[piece] >= EVENT_MIN).astype(np.int8),
                    "density_pred": np.isin(np.arange(len(val_scores[piece])), pred).astype(np.int8),
                }
            ).to_csv(OUT_DIR / f"fold{fold_idx}_{piece}_val_predictions.csv", index=False)

        den_tol1 = aggregate_metrics(density_tol1_items)
        den_tol0 = aggregate_metrics(density_tol0_items)
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
                "threshold_precision_tol0": th_tol0.precision,
                "threshold_recall_tol0": th_tol0.recall,
                "threshold_f1_tol0": th_tol0.f1,
                "density_precision_tol1": den_tol1.precision,
                "density_recall_tol1": den_tol1.recall,
                "density_f1_tol1": den_tol1.f1,
                "density_precision_tol0": den_tol0.precision,
                "density_recall_tol0": den_tol0.recall,
                "density_f1_tol0": den_tol0.f1,
                "density_pred_events": den_tol1.pred_events,
                "true_events": den_tol1.true_events,
            }
        )
        print(
            f"fold {fold_idx}: threshold tol1 P/R/F1="
            f"{th_tol1.precision:.3f}/{th_tol1.recall:.3f}/{th_tol1.f1:.3f} "
            f"tol0 F1={th_tol0.f1:.3f}; density tol1 P/R/F1="
            f"{den_tol1.precision:.3f}/{den_tol1.recall:.3f}/{den_tol1.f1:.3f} "
            f"tol0 F1={den_tol0.f1:.3f}"
        )

    summary = pd.DataFrame(rows)
    piece_summary = pd.DataFrame(piece_rows)
    summary.to_csv(OUT_DIR / "fold_summary.csv", index=False)
    piece_summary.to_csv(OUT_DIR / "piece_summary.csv", index=False)
    mean = summary[
        [
            "threshold_precision_tol1",
            "threshold_recall_tol1",
            "threshold_f1_tol1",
            "threshold_precision_tol0",
            "threshold_recall_tol0",
            "threshold_f1_tol0",
            "density_precision_tol1",
            "density_recall_tol1",
            "density_f1_tol1",
            "density_precision_tol0",
            "density_recall_tol0",
            "density_f1_tol0",
        ]
    ].mean()
    mean.to_frame("mean").to_csv(OUT_DIR / "fold_mean.csv")
    metadata = {
        "feature_columns": feature_cols,
        "pieces": pieces,
        "folds": folds,
        "label_source": str(LABEL_DIR),
        "level_weights": LEVEL_WEIGHTS,
        "target_rule": "max(weight_L * mean_performer_boundary_L_frequency) for L2-L6",
        "event_min": EVENT_MIN,
        "min_distance": MIN_DISTANCE,
    }
    (OUT_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("\nMean:")
    print(mean.round(4).to_string())
    print(f"\nWrote {OUT_DIR}")


if __name__ == "__main__":
    main()
