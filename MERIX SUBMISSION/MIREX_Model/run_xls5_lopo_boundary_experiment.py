from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.signal import find_peaks
from torch import nn


ROOT = Path(__file__).resolve().parents[2]
BEAT_TABLE = ROOT / "MERIX SUBMISSION" / "MIREX_Model_meter_auto" / "outputs" / "beat_table_salience_auto_meter_hi8_xml.csv.gz"
LABEL_DIR = ROOT / "MERIX SUBMISSION" / "MIREX_Model" / "xls_mazurka_boundary_compare"
OUT_DIR = ROOT / "MERIX SUBMISSION" / "MIREX_Model" / "xls5_lopo_boundary_experiment"
PIECES = ["M17-4", "M24-2", "M30-2", "M63-3", "M68-3"]
LEVELS = [1, 2, 3, 4, 5, 6]

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


def load_piece_features() -> tuple[dict[str, np.ndarray], list[str]]:
    df = pd.read_csv(BEAT_TABLE)
    df = df[df["piece_id"].isin(PIECES)].copy()
    feature_cols = [col for col in df.columns if is_feature_column(col)]
    out: dict[str, np.ndarray] = {}
    for piece in PIECES:
        piece_df = df[df["piece_id"] == piece].sort_values(["beat_idx", "sample_id"])
        one = piece_df.groupby("beat_idx", as_index=False).first().sort_values("beat_idx")
        expected = np.arange(len(one), dtype=int)
        got = one["beat_idx"].to_numpy(dtype=int)
        if not np.array_equal(got, expected):
            raise RuntimeError(f"{piece}: non-contiguous beat_idx in beat table")
        out[piece] = one[feature_cols].to_numpy(dtype=np.float32)
    return out, feature_cols


def load_xls_labels() -> dict[int, dict[str, np.ndarray]]:
    labels: dict[int, dict[str, np.ndarray]] = {level: {} for level in LEVELS}
    for piece in PIECES:
        df = pd.read_csv(LABEL_DIR / f"{piece}_frequency_compare.csv")
        for level in LEVELS:
            labels[level][piece] = df[f"xls_L{level}"].to_numpy(dtype=np.float32)
    return labels


class TinyCNN(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 64, dropout: float = 0.2):
        super().__init__()
        self.in_proj = nn.Conv1d(input_dim, hidden, kernel_size=1)
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
                    nn.BatchNorm1d(hidden),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
                    nn.BatchNorm1d(hidden),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
                for _ in range(3)
            ]
        )
        self.out = nn.Conv1d(hidden, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: B, T, F
        y = self.in_proj(x.transpose(1, 2))
        for block in self.blocks:
            y = y + block(y)
        return self.out(y).squeeze(1)


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


def extract_events(scores: np.ndarray, threshold: float, min_distance: int = 6) -> np.ndarray:
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
    if not candidates:
        return np.zeros(0, dtype=np.int32)
    candidates = sorted(set(candidates), key=lambda idx: (-float(scores[idx]), idx))
    kept = []
    for idx in candidates:
        if all(abs(idx - prev) >= max(int(min_distance), 1) for prev in kept):
            kept.append(idx)
    return np.asarray(sorted(kept), dtype=np.int32)


def match_events(pred: np.ndarray, true: np.ndarray, tolerance: int) -> int:
    used = set()
    matched = 0
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
            matched += 1
    return matched


@dataclass
class Metrics:
    threshold: float
    precision: float
    recall: float
    f1: float
    pred_events: int
    true_events: int
    matches: int


def evaluate(scores_by_piece: dict[str, np.ndarray], labels_by_piece: dict[str, np.ndarray], threshold: float, tolerance: int) -> Metrics:
    pred_total = true_total = match_total = 0
    for piece, scores in scores_by_piece.items():
        pred = extract_events(scores, threshold=threshold, min_distance=6)
        true = np.flatnonzero(labels_by_piece[piece] > 0).astype(np.int32)
        pred_total += int(len(pred))
        true_total += int(len(true))
        match_total += match_events(pred, true, tolerance=tolerance)
    precision = match_total / pred_total if pred_total else 0.0
    recall = match_total / true_total if true_total else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return Metrics(float(threshold), precision, recall, f1, pred_total, true_total, match_total)


def choose_threshold(scores_by_piece: dict[str, np.ndarray], labels_by_piece: dict[str, np.ndarray], tolerance: int) -> Metrics:
    best = None
    for th in np.linspace(0.05, 0.95, 37):
        metrics = evaluate(scores_by_piece, labels_by_piece, float(th), tolerance=tolerance)
        key = (metrics.f1, metrics.precision, metrics.recall)
        if best is None or key > (best.f1, best.precision, best.recall):
            best = metrics
    assert best is not None
    return best


@torch.no_grad()
def predict(model: nn.Module, features: dict[str, np.ndarray], pieces: list[str], mean: np.ndarray, std: np.ndarray) -> dict[str, np.ndarray]:
    model.eval()
    out = {}
    for piece in pieces:
        x = torch.from_numpy(((features[piece] - mean) / std)[None, :, :].astype(np.float32))
        out[piece] = torch.sigmoid(model(x)).squeeze(0).cpu().numpy().astype(np.float32)
    return out


def train_one(features: dict[str, np.ndarray], labels: dict[str, np.ndarray], train_pieces: list[str], seed: int) -> tuple[TinyCNN, np.ndarray, np.ndarray]:
    set_seed(seed)
    train_feats = [features[p] for p in train_pieces]
    train_labs = [labels[p] for p in train_pieces]
    stacked = np.concatenate(train_feats, axis=0)
    mean = stacked.mean(axis=0).astype(np.float32)
    std = stacked.std(axis=0).astype(np.float32)
    std[std < 1e-6] = 1.0

    x, y, mask = pad_batch(train_feats, train_labs, mean, std)
    model = TinyCNN(input_dim=x.shape[-1])
    pos = float(y.sum().item())
    neg = float(mask.sum().item() - pos)
    pos_weight = torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best_state = None
    best_loss = float("inf")
    patience = 20
    stale = 0
    for _epoch in range(1, 121):
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
            if stale >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, mean, std


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    features, feature_cols = load_piece_features()
    labels_by_level = load_xls_labels()
    rows = []

    for level in LEVELS:
        labels = labels_by_level[level]
        for heldout in PIECES:
            train_pieces = [p for p in PIECES if p != heldout]
            model, mean, std = train_one(features, labels, train_pieces, seed=4200 + level * 10 + PIECES.index(heldout))
            train_scores = predict(model, features, train_pieces, mean, std)
            val_scores = predict(model, features, [heldout], mean, std)
            train_metric_tol1 = choose_threshold(train_scores, {p: labels[p] for p in train_pieces}, tolerance=1)
            val_tol1 = evaluate(val_scores, {heldout: labels[heldout]}, train_metric_tol1.threshold, tolerance=1)
            val_tol0 = evaluate(val_scores, {heldout: labels[heldout]}, train_metric_tol1.threshold, tolerance=0)

            row = {
                "level": level,
                "heldout_piece": heldout,
                "train_pieces": " ".join(train_pieces),
                "threshold_from_train_tol1": train_metric_tol1.threshold,
                "train_f1_tol1": train_metric_tol1.f1,
                "val_precision_tol1": val_tol1.precision,
                "val_recall_tol1": val_tol1.recall,
                "val_f1_tol1": val_tol1.f1,
                "val_pred_events_tol1": val_tol1.pred_events,
                "val_true_events": val_tol1.true_events,
                "val_matches_tol1": val_tol1.matches,
                "val_precision_tol0": val_tol0.precision,
                "val_recall_tol0": val_tol0.recall,
                "val_f1_tol0": val_tol0.f1,
                "val_matches_tol0": val_tol0.matches,
            }
            rows.append(row)
            pd.DataFrame({"beat_idx": np.arange(len(val_scores[heldout])), "score": val_scores[heldout], "label_freq": labels[heldout]}).to_csv(
                OUT_DIR / f"L{level}_{heldout}_val_predictions.csv", index=False
            )
            print(
                f"L{level} heldout {heldout}: "
                f"tol1 P/R/F1={val_tol1.precision:.3f}/{val_tol1.recall:.3f}/{val_tol1.f1:.3f} "
                f"tol0 F1={val_tol0.f1:.3f} th={train_metric_tol1.threshold:.2f}"
            )

    summary = pd.DataFrame(rows)
    summary.to_csv(OUT_DIR / "lopo_summary.csv", index=False)
    by_level = summary.groupby("level")[
        ["val_precision_tol1", "val_recall_tol1", "val_f1_tol1", "val_precision_tol0", "val_recall_tol0", "val_f1_tol0"]
    ].mean()
    by_level.to_csv(OUT_DIR / "lopo_summary_by_level.csv")
    metadata = {"feature_columns": feature_cols, "pieces": PIECES, "levels": LEVELS, "label_source": str(LABEL_DIR)}
    (OUT_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("\nMean by level:")
    print(by_level.round(4).to_string())
    print(f"\nWrote {OUT_DIR}")


if __name__ == "__main__":
    main()
