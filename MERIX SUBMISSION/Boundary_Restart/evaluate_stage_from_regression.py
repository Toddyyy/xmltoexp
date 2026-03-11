#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score

from boundary_restart.config import load_config, resolve_path, threshold_grid


KEYS = ["sample_id", "piece_id", "beat_idx"]


def class_summary(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    report = classification_report(y_true, y_pred, labels=[0, 1, 2, 3], output_dict=True, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "class_f1": {str(i): float(report[str(i)]["f1-score"]) for i in range(4)},
        "class_precision": {str(i): float(report[str(i)]["precision"]) for i in range(4)},
        "class_recall": {str(i): float(report[str(i)]["recall"]) for i in range(4)},
        "class_support": {str(i): int(report[str(i)]["support"]) for i in range(4)},
    }


def assign_stage(score: np.ndarray, threshold: float, centers: dict[int, float]) -> np.ndarray:
    pred = np.zeros(score.shape[0], dtype=np.int64)
    active = score >= threshold
    if np.any(active):
        center_items = sorted(centers.items())
        center_classes = np.asarray([item[0] for item in center_items], dtype=np.int64)
        center_values = np.asarray([item[1] for item in center_items], dtype=np.float32)
        diff = np.abs(score[active, None] - center_values[None, :])
        pred[active] = center_classes[np.argmin(diff, axis=1)]
    return pred


def main():
    parser = argparse.ArgumentParser(description="Map regression scores to stage classes by train-set class centers.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--pred", required=True)
    parser.add_argument("--table_path", default=None)
    parser.add_argument("--target_col", default="boundary_peak")
    parser.add_argument("--output_json", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_cfg = cfg.get("data", {})
    table_path = resolve_path(cfg, args.table_path or data_cfg["beat_table_path"])
    pred_path = resolve_path(cfg, args.pred)

    table = pd.read_csv(table_path)
    pred = pd.read_csv(pred_path)
    merged = pred.merge(table[KEYS + ["split", "stage_class", "boundary_peak"]], on=KEYS, how="left", validate="one_to_one")
    target_col = str(args.target_col)
    if target_col not in table.columns:
        raise ValueError(f"{target_col} not found in {table_path}")
    train = table[table["split"] == "train"].copy()
    centers = (
        train[train["stage_class"] > 0]
        .groupby("stage_class")[target_col]
        .mean()
        .to_dict()
    )
    centers = {int(k): float(v) for k, v in centers.items()}
    thresholds = threshold_grid(cfg)

    best = None
    y_true = merged["stage_class"].to_numpy(dtype=np.int64)
    for threshold in thresholds.tolist():
        y_pred = assign_stage(merged["score"].to_numpy(dtype=np.float32), float(threshold), centers)
        metrics = class_summary(y_true, y_pred)
        key = (metrics["macro_f1"], metrics["weighted_f1"], metrics["accuracy"])
        if best is None or key > best[0]:
            best = (key, float(threshold), metrics)

    if best is None:
        raise RuntimeError("No threshold candidates were evaluated")

    summary = {
        "prediction_path": str(pred_path),
        "table_path": str(table_path),
        "target_col": target_col,
        "class_centers": centers,
        "best_threshold": best[1],
        **best[2],
    }
    if args.output_json:
        out_path = resolve_path(cfg, args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
