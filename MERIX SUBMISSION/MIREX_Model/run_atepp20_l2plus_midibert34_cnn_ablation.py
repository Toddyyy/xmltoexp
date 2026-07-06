from __future__ import annotations

import importlib.util
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torch import nn


ROOT = Path(__file__).resolve().parents[2]
MIREX = ROOT / "MERIX SUBMISSION" / "MIREX_Model"
BR = ROOT / "MERIX SUBMISSION" / "Boundary_Restart"
BASE_SCRIPT = MIREX / "run_atepp20_l2plus_weighted_target_experiment.py"
CONFIG_PATH = BR / "configs" / "mazurkabl_l2plus_weighted_auto_meter.yaml"
EMB_DIR = MIREX / "atepp20_midibert_beat_embeddings_regenerated"
OUT_DIR = MIREX / "atepp20_l2plus_midibert34_cnn_ablation_regenerated"
EMBED_DIM = 128


def load_base():
    spec = importlib.util.spec_from_file_location("atepp20_l2plus_base_cnn", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["atepp20_l2plus_base_cnn"] = module
    spec.loader.exec_module(module)
    return module


base = load_base()
quick = base.base
sys.path.insert(0, str(BR.resolve()))
from boundary_restart.models import build_sequence_model  # noqa: E402


@dataclass
class CnnResult:
    pred: int
    true: int
    match: int
    wr_num: float
    wr_den: float


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_piece_features(pieces: list[str], cfg: dict) -> tuple[dict[str, np.ndarray], list[str]]:
    return base.load_piece_features(pieces)


def load_midibert_embeddings(pieces: list[str]) -> dict[str, np.ndarray]:
    out = {}
    for piece in pieces:
        path = EMB_DIR / f"{piece}_midibert_beat_embeddings.npz"
        if not path.exists():
            continue
        with np.load(path) as data:
            out[piece] = np.asarray(data["beat_embeddings"], dtype=np.float32)
    return out


def fit_embedding_pca(embeddings: dict[str, np.ndarray], train_pieces: list[str]) -> tuple[StandardScaler, PCA]:
    stacked = np.concatenate([embeddings[p] for p in train_pieces], axis=0)
    scaler = StandardScaler()
    z = scaler.fit_transform(stacked)
    n_components = min(EMBED_DIM, z.shape[0] - 1, z.shape[1])
    pca = PCA(n_components=n_components, random_state=42)
    pca.fit(z)
    return scaler, pca


def augment_features(
    features: dict[str, np.ndarray],
    embeddings: dict[str, np.ndarray],
    pieces: list[str],
    scaler: StandardScaler,
    pca: PCA,
    *,
    embedding_only: bool = False,
) -> dict[str, np.ndarray]:
    out = {}
    for piece in pieces:
        feat = features[piece]
        emb = embeddings[piece]
        n = min(len(feat), len(emb))
        emb_p = pca.transform(scaler.transform(emb[:n])).astype(np.float32)
        if embedding_only:
            out[piece] = emb_p
        else:
            out[piece] = np.concatenate([feat[:n].astype(np.float32), emb_p], axis=1)
    return out


def pad_batch(features: list[np.ndarray], labels: list[np.ndarray], mean: np.ndarray, std: np.ndarray):
    batch = len(features)
    max_len = max(len(x) for x in features)
    feat_dim = features[0].shape[1]
    x = np.zeros((batch, max_len, feat_dim), dtype=np.float32)
    y = np.zeros((batch, max_len), dtype=np.float32)
    mask = np.zeros((batch, max_len), dtype=bool)
    lengths = np.zeros(batch, dtype=np.int64)
    for i, (feat, lab) in enumerate(zip(features, labels)):
        n = len(feat)
        x[i, :n] = (feat - mean) / std
        y[i, :n] = lab
        mask[i, :n] = True
        lengths[i] = n
    return (
        torch.from_numpy(x),
        torch.from_numpy(y),
        torch.from_numpy(mask),
        torch.from_numpy(lengths),
    )


def train_one(
    cfg: dict,
    features: dict[str, np.ndarray],
    labels: dict[str, np.ndarray],
    train_pieces: list[str],
    seed: int,
    device: torch.device,
):
    set_seed(seed)
    train_feats = [features[p] for p in train_pieces]
    train_labs = [labels[p] for p in train_pieces]
    stacked = np.concatenate(train_feats, axis=0)
    mean = stacked.mean(axis=0).astype(np.float32)
    std = stacked.std(axis=0).astype(np.float32)
    std[std < 1e-6] = 1.0
    x, y, mask, lengths = pad_batch(train_feats, train_labs, mean, std)
    x = x.to(device)
    y = y.to(device)
    mask = mask.to(device)
    lengths = lengths.to(device)

    model = build_sequence_model("cnn", input_dim=x.shape[-1], cfg=cfg).to(device)
    pos = float(y[mask].sum().item())
    neg = float(mask.sum().item() - pos)
    pos_weight = torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32, device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")
    seq_cfg = cfg.get("sequence", {})
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(seq_cfg.get("lr", 1e-3)),
        weight_decay=float(seq_cfg.get("weight_decay", 1e-4)),
    )
    epochs = int(seq_cfg.get("epochs", 18))
    patience = int(seq_cfg.get("early_stop_patience", 5))
    grad_clip = float(seq_cfg.get("grad_clip", 1.0))
    best_loss = float("inf")
    best_state = None
    stale = 0
    for _epoch in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        logits = model(x, lengths=lengths)
        loss = (loss_fn(logits, y) * mask.float()).sum() / mask.float().sum().clamp(min=1.0)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        value = float(loss.item())
        if value < best_loss - 1e-5:
            best_loss = value
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, mean, std


@torch.no_grad()
def predict(
    model,
    features: dict[str, np.ndarray],
    pieces: list[str],
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
) -> dict[str, np.ndarray]:
    model.eval()
    out = {}
    for piece in pieces:
        x = ((features[piece] - mean) / std)[None].astype(np.float32)
        lengths = torch.tensor([x.shape[1]], dtype=torch.int64, device=device)
        logits = model(torch.from_numpy(x).to(device), lengths=lengths)
        out[piece] = torch.sigmoid(logits).squeeze(0).detach().cpu().numpy().astype(np.float32)
    return out


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


def evaluate_fold(
    labels: dict[str, np.ndarray],
    scores: dict[str, np.ndarray],
    val_pieces: list[str],
    threshold: float,
    train_labels: dict[str, np.ndarray],
    train_pieces: list[str],
) -> tuple[CnnResult, CnnResult]:
    th = CnnResult(0, 0, 0, 0.0, 0.0)
    den = CnnResult(0, 0, 0, 0.0, 0.0)
    for piece in val_pieces:
        true = np.flatnonzero(labels[piece] >= base.EVENT_MIN).astype(np.int32)

        pred_th = quick.extract_events(scores[piece], threshold=threshold)
        m_th = quick.metrics_from_events(pred_th, true, tolerance=1)
        matched_th = match_true_indices(pred_th, true, tolerance=1)
        th.pred += m_th.pred_events
        th.true += m_th.true_events
        th.match += m_th.matches
        th.wr_num += float(labels[piece][matched_th].sum()) if matched_th else 0.0
        th.wr_den += float(labels[piece][true].sum()) if len(true) else 0.0

        expected = quick.expected_count_from_train_density(train_labels, train_pieces, len(labels[piece]))
        pred_den = quick.extract_top_density(scores[piece], expected)
        m_den = quick.metrics_from_events(pred_den, true, tolerance=1)
        matched_den = match_true_indices(pred_den, true, tolerance=1)
        den.pred += m_den.pred_events
        den.true += m_den.true_events
        den.match += m_den.matches
        den.wr_num += float(labels[piece][matched_den].sum()) if matched_den else 0.0
        den.wr_den += float(labels[piece][true].sum()) if len(true) else 0.0
    return th, den


def pack(prefix: str, totals: CnnResult) -> dict[str, float | int]:
    up = totals.match / totals.pred if totals.pred else 0.0
    recall = totals.match / totals.true if totals.true else 0.0
    wr = totals.wr_num / totals.wr_den if totals.wr_den else 0.0
    f1 = 2 * up * recall / (up + recall) if up + recall else 0.0
    return {
        f"{prefix}_pred_events": int(totals.pred),
        f"{prefix}_true_events": int(totals.true),
        f"{prefix}_matches_tol1": int(totals.match),
        f"{prefix}_UP": up,
        f"{prefix}_recall": recall,
        f"{prefix}_WR": wr,
        f"{prefix}_f1": f1,
    }


def augment_random_features(features: dict[str, np.ndarray], pieces: list[str], seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    out = {}
    for piece in pieces:
        feat = features[piece]
        random_emb = rng.standard_normal((len(feat), EMBED_DIM)).astype(np.float32)
        out[piece] = np.concatenate([feat.astype(np.float32), random_emb], axis=1)
    return out


def run_setting(setting: str, cfg, pieces, labels, features, embeddings, folds, device) -> tuple[pd.DataFrame, dict]:
    rows = []
    th_total = CnnResult(0, 0, 0, 0.0, 0.0)
    den_total = CnnResult(0, 0, 0, 0.0, 0.0)
    for fold_idx, val_pieces in enumerate(folds, start=1):
        train_pieces = [p for p in pieces if p not in set(val_pieces)]
        if setting == "midibert34_pca":
            scaler, pca = fit_embedding_pca(embeddings, train_pieces)
            fold_features = augment_features(features, embeddings, pieces, scaler, pca)
        elif setting == "midibert34_only":
            scaler, pca = fit_embedding_pca(embeddings, train_pieces)
            fold_features = augment_features(features, embeddings, pieces, scaler, pca, embedding_only=True)
        elif setting == "random128":
            fold_features = augment_random_features(features, pieces, seed=123000 + fold_idx)
        else:
            fold_features = features
        model, mean, std = train_one(cfg, fold_features, labels, train_pieces, seed=9300 + fold_idx, device=device)
        train_scores = predict(model, fold_features, train_pieces, mean, std, device)
        val_scores = predict(model, fold_features, val_pieces, mean, std, device)
        threshold, train_metric = quick.choose_threshold(
            train_scores,
            {p: labels[p] for p in train_pieces},
            tolerance=1,
        )
        th, den = evaluate_fold(labels, val_scores, val_pieces, threshold, labels, train_pieces)
        for attr in ("pred", "true", "match", "wr_num", "wr_den"):
            setattr(th_total, attr, getattr(th_total, attr) + getattr(th, attr))
            setattr(den_total, attr, getattr(den_total, attr) + getattr(den, attr))
        row = {
            "setting": setting,
            "fold": fold_idx,
            "threshold": threshold,
            "train_f1_tol1": train_metric.f1,
            **pack("threshold", th),
            **pack("density", den),
        }
        rows.append(row)
        print(
            f"{setting} fold {fold_idx}: "
            f"threshold UP/WR/F1={row['threshold_UP']:.3f}/{row['threshold_WR']:.3f}/{row['threshold_f1']:.3f}; "
            f"density UP/WR/F1={row['density_UP']:.3f}/{row['density_WR']:.3f}/{row['density_f1']:.3f}"
        )
    return pd.DataFrame(rows), {"setting": setting, **pack("threshold", th_total), **pack("density", den_total)}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    pieces, labels, _ = base.load_l2plus_weighted_labels()
    features, feature_cols = load_piece_features(pieces, cfg)
    embeddings = load_midibert_embeddings(pieces)

    missing_embeddings = sorted(set(pieces) - set(embeddings))
    if missing_embeddings:
        raise RuntimeError(f"Missing MidiBERT embeddings for pieces: {missing_embeddings}")
    length_mismatch = []
    for piece in pieces:
        if len(features[piece]) != len(labels[piece]) or len(labels[piece]) != len(embeddings[piece]):
            length_mismatch.append(
                {
                    "piece": piece,
                    "feature_len": len(features[piece]),
                    "label_len": len(labels[piece]),
                    "emb_len": len(embeddings[piece]),
                }
            )
    if length_mismatch:
        raise RuntimeError(f"Feature/label/embedding length mismatch: {length_mismatch}")
    pieces = sorted(pieces)
    features = {p: features[p] for p in pieces}
    labels = {p: labels[p] for p in pieces}
    embeddings = {p: embeddings[p] for p in pieces}
    folds = quick.make_folds(pieces, n_folds=5, seed=42)
    device = resolve_device()
    print(f"device={device}; feature_dim={features[pieces[0]].shape[1]}; emb_dim={embeddings[pieces[0]].shape[1]}")

    rows = []
    aggregates = []
    for setting in ["baseline_cnn", "random128", "midibert34_only", "midibert34_pca"]:
        fold_rows, aggregate = run_setting(setting, cfg, pieces, labels, features, embeddings, folds, device)
        rows.append(fold_rows)
        aggregates.append(aggregate)
    pd.concat(rows, ignore_index=True).to_csv(OUT_DIR / "fold_summary.csv", index=False)
    agg = pd.DataFrame(aggregates)
    agg.to_csv(OUT_DIR / "aggregate_totals.csv", index=False)
    (OUT_DIR / "metadata.json").write_text(
        json.dumps(
            {
                "config_path": str(CONFIG_PATH),
                "base_script": str(BASE_SCRIPT),
                "embedding_dir": str(EMB_DIR),
                "regenerated_note_feats": str(MIREX / "atepp20_regenerated_note_feats"),
                "embedding_dim_before_pca": 768,
                "embedding_dim_after_pca": EMBED_DIM,
                "base_feature_dim": int(features[pieces[0]].shape[1]),
                "augmented_feature_dim": int(features[pieces[0]].shape[1] + EMBED_DIM),
                "pieces": pieces,
                "folds": folds,
                "length_mismatch_policy": "strict_raise",
                "feature_columns": feature_cols,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("\nAggregate:")
    print(agg.round(4).to_string(index=False))
    print(f"\nWrote {OUT_DIR}")


if __name__ == "__main__":
    main()
