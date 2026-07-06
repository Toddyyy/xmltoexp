from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
MIREX = ROOT / "MERIX SUBMISSION" / "MIREX_Model"
BASE_SCRIPT = MIREX / "run_mazurkabl_l2plus_weighted_target_experiment.py"
OUT_DIR = MIREX / "mazurkabl_l2plus_context_embedding_ablation"

WINDOW_RADIUS = 4
EMBED_DIM = 64


def load_base():
    spec = importlib.util.spec_from_file_location("mazurka_l2plus_base", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["mazurka_l2plus_base"] = module
    spec.loader.exec_module(module)
    return module


base = load_base()


def context_matrix(x: np.ndarray, radius: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    n, d = x.shape
    pads = np.pad(x, ((radius, radius), (0, 0)), mode="edge")
    windows = [pads[offset : offset + n] for offset in range(0, 2 * radius + 1)]
    return np.concatenate(windows, axis=1).astype(np.float32)


def fit_context_embedding(features: dict[str, np.ndarray], train_pieces: list[str]) -> tuple[StandardScaler, TruncatedSVD]:
    ctx = np.concatenate([context_matrix(features[p], WINDOW_RADIUS) for p in train_pieces], axis=0)
    scaler = StandardScaler()
    ctx_z = scaler.fit_transform(ctx)
    n_components = min(EMBED_DIM, ctx_z.shape[1] - 1, ctx_z.shape[0] - 1)
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    svd.fit(ctx_z)
    return scaler, svd


def augment_features(
    features: dict[str, np.ndarray],
    pieces: list[str],
    scaler: StandardScaler,
    svd: TruncatedSVD,
) -> dict[str, np.ndarray]:
    out = {}
    for piece in pieces:
        ctx = context_matrix(features[piece], WINDOW_RADIUS)
        emb = svd.transform(scaler.transform(ctx)).astype(np.float32)
        # Normalize per piece to keep the appended embedding numerically tame.
        emb_mean = emb.mean(axis=0, keepdims=True)
        emb_std = emb.std(axis=0, keepdims=True)
        emb_std[emb_std < 1e-6] = 1.0
        emb = (emb - emb_mean) / emb_std
        out[piece] = np.concatenate([features[piece].astype(np.float32), emb.astype(np.float32)], axis=1)
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


def evaluate_threshold_with_wr(scores_by_piece, labels_by_piece, threshold: float, tolerance: int):
    items = []
    wr_num = 0.0
    wr_den = 0.0
    for piece, scores in scores_by_piece.items():
        pred = base.extract_events(scores, threshold=threshold)
        true = np.flatnonzero(labels_by_piece[piece] >= base.EVENT_MIN).astype(np.int32)
        items.append(base.metrics_from_events(pred, true, tolerance))
        matched = match_true_indices(pred, true, tolerance)
        wr_num += float(labels_by_piece[piece][matched].sum()) if matched else 0.0
        wr_den += float(labels_by_piece[piece][true].sum()) if len(true) else 0.0
    metric = base.aggregate_metrics(items)
    return metric, (wr_num / wr_den if wr_den > 0 else 0.0)


def run_setting(setting: str, pieces, labels, features, folds) -> tuple[pd.DataFrame, dict]:
    rows = []
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
        if setting == "context_pca":
            scaler, svd = fit_context_embedding(features, train_pieces)
            fold_features = augment_features(features, pieces, scaler, svd)
        else:
            fold_features = features
        model, mean, std = base.train_one(fold_features, labels, train_pieces, seed=8100 + fold_idx)
        train_scores = base.predict(model, fold_features, train_pieces, mean, std)
        val_scores = base.predict(model, fold_features, val_pieces, mean, std)
        threshold, train_metric = base.choose_threshold(train_scores, {p: labels[p] for p in train_pieces}, tolerance=1)
        th_metric, th_wr = evaluate_threshold_with_wr(val_scores, {p: labels[p] for p in val_pieces}, threshold, 1)

        density_items = []
        density_wr_num = 0.0
        density_wr_den = 0.0
        for piece in val_pieces:
            true = np.flatnonzero(labels[piece] >= base.EVENT_MIN).astype(np.int32)
            expected = base.expected_count_from_train_density(labels, train_pieces, len(labels[piece]))
            pred = base.extract_top_density(val_scores[piece], expected)
            m = base.metrics_from_events(pred, true, tolerance=1)
            density_items.append(m)
            matched = match_true_indices(pred, true, tolerance=1)
            density_wr_num += float(labels[piece][matched].sum()) if matched else 0.0
            density_wr_den += float(labels[piece][true].sum()) if len(true) else 0.0
        den_metric = base.aggregate_metrics(density_items)
        den_wr = density_wr_num / density_wr_den if density_wr_den > 0 else 0.0

        for piece in val_pieces:
            true = np.flatnonzero(labels[piece] >= base.EVENT_MIN).astype(np.int32)
            pred = base.extract_events(val_scores[piece], threshold=threshold)
            matched = match_true_indices(pred, true, tolerance=1)
            totals["threshold_wr_num"] += float(labels[piece][matched].sum()) if matched else 0.0
            totals["threshold_wr_den"] += float(labels[piece][true].sum()) if len(true) else 0.0
        totals["threshold_pred"] += th_metric.pred_events
        totals["threshold_true"] += th_metric.true_events
        totals["threshold_match"] += th_metric.matches
        totals["density_pred"] += den_metric.pred_events
        totals["density_true"] += den_metric.true_events
        totals["density_match"] += den_metric.matches
        totals["density_wr_num"] += density_wr_num
        totals["density_wr_den"] += density_wr_den

        rows.append(
            {
                "setting": setting,
                "fold": fold_idx,
                "threshold": threshold,
                "train_f1_tol1": train_metric.f1,
                "threshold_UP": th_metric.precision,
                "threshold_recall": th_metric.recall,
                "threshold_WR": th_wr,
                "threshold_f1": th_metric.f1,
                "threshold_pred_events": th_metric.pred_events,
                "density_UP": den_metric.precision,
                "density_recall": den_metric.recall,
                "density_WR": den_wr,
                "density_f1": den_metric.f1,
                "density_pred_events": den_metric.pred_events,
                "true_events": den_metric.true_events,
            }
        )
        print(
            f"{setting} fold {fold_idx}: "
            f"threshold UP/WR/F1={th_metric.precision:.3f}/{th_wr:.3f}/{th_metric.f1:.3f}; "
            f"density UP/WR/F1={den_metric.precision:.3f}/{den_wr:.3f}/{den_metric.f1:.3f}"
        )

    def pack(prefix: str) -> dict:
        up = totals[f"{prefix}_match"] / totals[f"{prefix}_pred"] if totals[f"{prefix}_pred"] else 0.0
        rec = totals[f"{prefix}_match"] / totals[f"{prefix}_true"] if totals[f"{prefix}_true"] else 0.0
        f1 = 2 * up * rec / (up + rec) if up + rec else 0.0
        wr = totals[f"{prefix}_wr_num"] / totals[f"{prefix}_wr_den"] if totals[f"{prefix}_wr_den"] else 0.0
        return {
            f"{prefix}_pred_events": totals[f"{prefix}_pred"],
            f"{prefix}_true_events": totals[f"{prefix}_true"],
            f"{prefix}_matches_tol1": totals[f"{prefix}_match"],
            f"{prefix}_UP": up,
            f"{prefix}_recall": rec,
            f"{prefix}_WR": wr,
            f"{prefix}_f1": f1,
        }

    aggregate = {"setting": setting, **pack("threshold"), **pack("density")}
    return pd.DataFrame(rows), aggregate


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pieces, labels, _ = base.load_l2plus_weighted_labels()
    features, feature_cols = base.load_piece_features(pieces)
    folds = base.make_folds(pieces, n_folds=5, seed=42)

    all_rows = []
    aggregates = []
    for setting in ["baseline", "context_pca"]:
        rows, aggregate = run_setting(setting, pieces, labels, features, folds)
        all_rows.append(rows)
        aggregates.append(aggregate)

    fold_df = pd.concat(all_rows, ignore_index=True)
    agg_df = pd.DataFrame(aggregates)
    fold_df.to_csv(OUT_DIR / "fold_summary.csv", index=False)
    agg_df.to_csv(OUT_DIR / "aggregate_totals.csv", index=False)
    (OUT_DIR / "metadata.json").write_text(
        json.dumps(
            {
                "base_script": str(BASE_SCRIPT),
                "idea": "cheap validation of adding contextual score beat embeddings before trying MidiBERT",
                "context_window_radius": WINDOW_RADIUS,
                "embedding_dim": EMBED_DIM,
                "feature_columns": feature_cols,
                "pieces": pieces,
                "folds": folds,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("\nAggregate:")
    print(agg_df.round(4).to_string(index=False))
    print(f"\nWrote {OUT_DIR}")


if __name__ == "__main__":
    main()
