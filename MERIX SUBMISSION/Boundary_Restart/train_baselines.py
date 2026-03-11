#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from boundary_restart.config import load_config, resolve_path, threshold_grid
from boundary_restart.metrics import search_best_threshold
from boundary_restart.table_io import feature_columns, load_table


def build_logreg(cfg: dict):
    model_cfg = cfg.get("baseline", {}).get("logistic", {})
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=float(model_cfg.get("C", 1.0)),
                    max_iter=int(model_cfg.get("max_iter", 3000)),
                    class_weight="balanced",
                    solver="liblinear",
                    random_state=42,
                ),
            ),
        ]
    )


def build_xgboost(cfg: dict, pos_weight: float):
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise RuntimeError("xgboost is not installed") from exc

    model_cfg = cfg.get("baseline", {}).get("xgboost", {})
    return XGBClassifier(
        n_estimators=int(model_cfg.get("n_estimators", 400)),
        learning_rate=float(model_cfg.get("learning_rate", 0.05)),
        max_depth=int(model_cfg.get("max_depth", 4)),
        subsample=float(model_cfg.get("subsample", 0.8)),
        colsample_bytree=float(model_cfg.get("colsample_bytree", 0.8)),
        reg_lambda=float(model_cfg.get("reg_lambda", 1.0)),
        min_child_weight=float(model_cfg.get("min_child_weight", 1.0)),
        gamma=float(model_cfg.get("gamma", 0.0)),
        scale_pos_weight=float(pos_weight),
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=42,
    )


def build_sequence_maps(df: pd.DataFrame) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    scores = {}
    labels = {}
    for sample_id, group in df.sort_values(["sample_id", "beat_idx"]).groupby("sample_id", sort=False):
        scores[sample_id] = group["score"].to_numpy(dtype=np.float32)
        labels[sample_id] = group["boundary_peak"].to_numpy(dtype=np.float32)
    return scores, labels


def select_feature_columns(cfg: dict, columns: list[str]) -> list[str]:
    feature_cfg = cfg.get("features", {})
    include = feature_cfg.get("include")
    exclude = set(feature_cfg.get("exclude", []))
    selected = list(columns)
    if include:
        include_set = set(include)
        selected = [col for col in selected if col in include_set]
    if exclude:
        selected = [col for col in selected if col not in exclude]
    return selected


def save_feature_summary(model_name: str, model, feature_cols: list[str], out_dir: Path):
    if model_name == "logreg":
        coef = model.named_steps["model"].coef_[0]
        frame = pd.DataFrame({"feature": feature_cols, "weight": coef}).sort_values("weight", ascending=False)
        frame.to_csv(out_dir / "feature_weights.csv", index=False)
        return
    if model_name == "xgboost" and hasattr(model, "feature_importances_"):
        frame = pd.DataFrame({"feature": feature_cols, "importance": model.feature_importances_})
        frame = frame.sort_values("importance", ascending=False)
        frame.to_csv(out_dir / "feature_importance.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description="Train logistic regression and XGBoost baselines.")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--table_path", default=None, help="Optional override for beat table path")
    parser.add_argument("--output_dir", default=None, help="Optional override for output directory")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_cfg = cfg.get("data", {})
    baseline_cfg = cfg.get("baseline", {})
    eval_cfg = cfg.get("evaluation", {})

    table_path = resolve_path(cfg, args.table_path or data_cfg["beat_table_path"])
    out_root = resolve_path(cfg, args.output_dir or baseline_cfg["output_dir"])
    out_root.mkdir(parents=True, exist_ok=True)

    df = load_table(table_path)
    df = df[df["split"].isin(["train", "val"])].copy()
    feature_cols = select_feature_columns(cfg, feature_columns(df))

    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()
    if train_df.empty or val_df.empty:
        raise ValueError("Both train and val splits must be non-empty")

    x_train = train_df[feature_cols].to_numpy(dtype=np.float32)
    y_train = train_df["boundary_peak"].to_numpy(dtype=np.int32)
    x_val = val_df[feature_cols].to_numpy(dtype=np.float32)
    y_val = val_df["boundary_peak"].to_numpy(dtype=np.int32)

    pos = int(y_train.sum())
    neg = int((y_train == 0).sum())
    pos_weight = float(neg / max(pos, 1))

    builders = {
        "logreg": lambda: build_logreg(cfg),
        "xgboost": lambda: build_xgboost(cfg, pos_weight=pos_weight),
    }

    thresholds = threshold_grid(cfg)
    min_distance = int(eval_cfg.get("min_distance", 6))
    tolerance = int(eval_cfg.get("event_tolerance", 1))
    prominence = float(eval_cfg.get("prominence", 0.0))

    for model_name in baseline_cfg.get("models", ["logreg", "xgboost"]):
        model = builders[model_name]()
        model.fit(x_train, y_train)

        train_scores = model.predict_proba(x_train)[:, 1].astype(np.float32)
        val_scores = model.predict_proba(x_val)[:, 1].astype(np.float32)

        val_pred = val_df[["sample_id", "piece_id", "performer_id", "level", "beat_idx", "boundary_peak"]].copy()
        val_pred["score"] = val_scores
        sequence_scores, sequence_labels = build_sequence_maps(val_pred)
        best = search_best_threshold(
            sequence_scores=sequence_scores,
            sequence_labels=sequence_labels,
            thresholds=thresholds,
            tolerance=tolerance,
            min_distance=min_distance,
            prominence=prominence,
        )

        train_ap = float(0.0)
        if np.any(y_train > 0):
            from sklearn.metrics import average_precision_score

            train_ap = float(average_precision_score(y_train, train_scores))

        model_out = out_root / model_name
        model_out.mkdir(parents=True, exist_ok=True)

        metrics = {
            "model": model_name,
            "table_path": str(table_path),
            "train_rows": int(len(train_df)),
            "val_rows": int(len(val_df)),
            "num_features": int(len(feature_cols)),
            "train_positive_rate": float(train_df["boundary_peak"].mean()),
            "val_positive_rate": float(val_df["boundary_peak"].mean()),
            "train_average_precision": train_ap,
            "val_average_precision": best.average_precision,
            "best_threshold": best.threshold,
            "event_precision": best.precision,
            "event_recall": best.recall,
            "event_f1": best.f1,
            "mean_offset": best.mean_offset,
            "matches": best.matches,
            "pred_events": best.pred_events,
            "true_events": best.true_events,
        }
        (model_out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

        val_pred.to_csv(model_out / "val_predictions.csv.gz", index=False, compression="gzip")
        with (model_out / "model.pkl").open("wb") as f:
            pickle.dump({"model": model, "feature_columns": feature_cols, "metrics": metrics}, f)
        save_feature_summary(model_name, model, feature_cols, model_out)

        print(
            f"[{model_name}] AP={best.average_precision:.4f} | "
            f"event_f1={best.f1:.4f} | threshold={best.threshold:.3f}"
        )


if __name__ == "__main__":
    main()
