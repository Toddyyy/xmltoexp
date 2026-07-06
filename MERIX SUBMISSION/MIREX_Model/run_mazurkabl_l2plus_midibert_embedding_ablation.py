from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
MIREX = ROOT / "MERIX SUBMISSION" / "MIREX_Model"
BASE_SCRIPT = MIREX / "run_mazurkabl_l2plus_weighted_target_experiment.py"
EMB_DIR = MIREX / "mazurkabl_midibert_beat_embeddings"
OUT_DIR = MIREX / "mazurkabl_l2plus_midibert_embedding_ablation"
EMBED_DIM = 128


def load_base():
    spec = importlib.util.spec_from_file_location("mazurka_l2plus_base_midibert", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["mazurka_l2plus_base_midibert"] = module
    spec.loader.exec_module(module)
    return module


base = load_base()


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
) -> dict[str, np.ndarray]:
    out = {}
    for piece in pieces:
        feat = features[piece]
        emb = embeddings[piece]
        n = min(len(feat), len(emb))
        emb_z = scaler.transform(emb[:n])
        emb_p = pca.transform(emb_z).astype(np.float32)
        out[piece] = np.concatenate([feat[:n].astype(np.float32), emb_p], axis=1)
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


def weighted_recall_for_pred(pred: np.ndarray, label: np.ndarray, tolerance: int = 1) -> float:
    true = np.flatnonzero(label >= base.EVENT_MIN).astype(np.int32)
    denom = float(label[true].sum()) if len(true) else 0.0
    if denom <= 0:
        return 0.0
    matched = match_true_indices(pred, true, tolerance)
    return float(label[matched].sum()) / denom if matched else 0.0


def run_setting(setting: str, pieces, labels, features, embeddings, folds) -> tuple[pd.DataFrame, dict]:
    rows = []
    totals = {k: 0.0 for k in [
        "threshold_pred", "threshold_true", "threshold_match", "threshold_wr_num", "threshold_wr_den",
        "density_pred", "density_true", "density_match", "density_wr_num", "density_wr_den",
    ]}
    for fold_idx, val_pieces in enumerate(folds, start=1):
        train_pieces = [p for p in pieces if p not in set(val_pieces)]
        if setting == "midibert_pca":
            scaler, pca = fit_embedding_pca(embeddings, train_pieces)
            fold_features = augment_features(features, embeddings, pieces, scaler, pca)
        else:
            fold_features = features
        model, mean, std = base.train_one(fold_features, labels, train_pieces, seed=9100 + fold_idx)
        train_scores = base.predict(model, fold_features, train_pieces, mean, std)
        val_scores = base.predict(model, fold_features, val_pieces, mean, std)
        threshold, train_metric = base.choose_threshold(train_scores, {p: labels[p] for p in train_pieces}, tolerance=1)
        threshold_items = []
        density_items = []
        threshold_wr_num = threshold_wr_den = 0.0
        density_wr_num = density_wr_den = 0.0
        for piece in val_pieces:
            true = np.flatnonzero(labels[piece] >= base.EVENT_MIN).astype(np.int32)
            pred_th = base.extract_events(val_scores[piece], threshold=threshold)
            m_th = base.metrics_from_events(pred_th, true, tolerance=1)
            threshold_items.append(m_th)
            matched_th = match_true_indices(pred_th, true, tolerance=1)
            threshold_wr_num += float(labels[piece][matched_th].sum()) if matched_th else 0.0
            threshold_wr_den += float(labels[piece][true].sum()) if len(true) else 0.0

            expected = base.expected_count_from_train_density(labels, train_pieces, len(labels[piece]))
            pred_den = base.extract_top_density(val_scores[piece], expected)
            m_den = base.metrics_from_events(pred_den, true, tolerance=1)
            density_items.append(m_den)
            matched_den = match_true_indices(pred_den, true, tolerance=1)
            density_wr_num += float(labels[piece][matched_den].sum()) if matched_den else 0.0
            density_wr_den += float(labels[piece][true].sum()) if len(true) else 0.0

        th = base.aggregate_metrics(threshold_items)
        den = base.aggregate_metrics(density_items)
        th_wr = threshold_wr_num / threshold_wr_den if threshold_wr_den else 0.0
        den_wr = density_wr_num / density_wr_den if density_wr_den else 0.0

        totals["threshold_pred"] += th.pred_events
        totals["threshold_true"] += th.true_events
        totals["threshold_match"] += th.matches
        totals["threshold_wr_num"] += threshold_wr_num
        totals["threshold_wr_den"] += threshold_wr_den
        totals["density_pred"] += den.pred_events
        totals["density_true"] += den.true_events
        totals["density_match"] += den.matches
        totals["density_wr_num"] += density_wr_num
        totals["density_wr_den"] += density_wr_den

        rows.append({
            "setting": setting,
            "fold": fold_idx,
            "threshold": threshold,
            "train_f1_tol1": train_metric.f1,
            "threshold_UP": th.precision,
            "threshold_recall": th.recall,
            "threshold_WR": th_wr,
            "threshold_f1": th.f1,
            "threshold_pred_events": th.pred_events,
            "density_UP": den.precision,
            "density_recall": den.recall,
            "density_WR": den_wr,
            "density_f1": den.f1,
            "density_pred_events": den.pred_events,
            "true_events": den.true_events,
        })
        print(
            f"{setting} fold {fold_idx}: "
            f"threshold UP/WR/F1={th.precision:.3f}/{th_wr:.3f}/{th.f1:.3f}; "
            f"density UP/WR/F1={den.precision:.3f}/{den_wr:.3f}/{den.f1:.3f}"
        )

    def pack(prefix: str):
        up = totals[f"{prefix}_match"] / totals[f"{prefix}_pred"] if totals[f"{prefix}_pred"] else 0.0
        recall = totals[f"{prefix}_match"] / totals[f"{prefix}_true"] if totals[f"{prefix}_true"] else 0.0
        f1 = 2 * up * recall / (up + recall) if up + recall else 0.0
        wr = totals[f"{prefix}_wr_num"] / totals[f"{prefix}_wr_den"] if totals[f"{prefix}_wr_den"] else 0.0
        return {
            f"{prefix}_pred_events": int(totals[f"{prefix}_pred"]),
            f"{prefix}_true_events": int(totals[f"{prefix}_true"]),
            f"{prefix}_matches_tol1": int(totals[f"{prefix}_match"]),
            f"{prefix}_UP": up,
            f"{prefix}_recall": recall,
            f"{prefix}_WR": wr,
            f"{prefix}_f1": f1,
        }

    return pd.DataFrame(rows), {"setting": setting, **pack("threshold"), **pack("density")}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pieces, labels, _ = base.load_l2plus_weighted_labels()
    features, feature_cols = base.load_piece_features(pieces)
    embeddings = load_midibert_embeddings(pieces)
    keep = []
    truncated = []
    for piece in pieces:
        if piece not in embeddings:
            continue
        n = min(len(features[piece]), len(labels[piece]), len(embeddings[piece]))
        if n <= 0:
            continue
        if len(features[piece]) != n or len(labels[piece]) != n or len(embeddings[piece]) != n:
            truncated.append({"piece": piece, "feature_len": len(features[piece]), "label_len": len(labels[piece]), "emb_len": len(embeddings[piece]), "used": n})
            features[piece] = features[piece][:n]
            labels[piece] = labels[piece][:n]
            embeddings[piece] = embeddings[piece][:n]
        keep.append(piece)
    pieces = sorted(keep)
    features = {p: features[p] for p in pieces}
    labels = {p: labels[p] for p in pieces}
    embeddings = {p: embeddings[p] for p in pieces}
    folds = base.make_folds(pieces, n_folds=5, seed=42)

    rows = []
    aggregates = []
    for setting in ["baseline", "midibert_pca"]:
        fold_rows, aggregate = run_setting(setting, pieces, labels, features, embeddings, folds)
        rows.append(fold_rows)
        aggregates.append(aggregate)
    pd.concat(rows, ignore_index=True).to_csv(OUT_DIR / "fold_summary.csv", index=False)
    agg = pd.DataFrame(aggregates)
    agg.to_csv(OUT_DIR / "aggregate_totals.csv", index=False)
    (OUT_DIR / "metadata.json").write_text(json.dumps({
        "embedding_dir": str(EMB_DIR),
        "embedding_dim_before_pca": 768,
        "embedding_dim_after_pca": EMBED_DIM,
        "pieces": pieces,
        "folds": folds,
        "truncated": truncated,
        "feature_columns": feature_cols,
    }, indent=2), encoding="utf-8")
    print("\nAggregate:")
    print(agg.round(4).to_string(index=False))
    print(f"\nWrote {OUT_DIR}")


if __name__ == "__main__":
    main()
