#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from boundary_restart.config import load_config, threshold_grid
from boundary_restart.metrics import search_best_threshold


KEY_COLUMNS = ["sample_id", "piece_id", "beat_idx", "boundary_peak"]


def load_prediction_frame(path: Path, member_idx: int) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [col for col in KEY_COLUMNS + ["score"] if col not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    renamed = df[KEY_COLUMNS + ["score"]].copy()
    renamed = renamed.rename(columns={"score": f"score_{member_idx}"})
    return renamed


def build_sequence_maps(df: pd.DataFrame) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    scores = {}
    labels = {}
    ordered = df.sort_values(["sample_id", "beat_idx"])
    for sample_id, group in ordered.groupby("sample_id", sort=False):
        scores[sample_id] = group["score"].to_numpy(dtype=np.float32)
        labels[sample_id] = group["boundary_peak"].to_numpy(dtype=np.float32)
    return scores, labels


def main():
    parser = argparse.ArgumentParser(description="Average multiple sequence-model validation predictions.")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--pred", action="append", required=True, help="Prediction CSV(.gz) path; repeat per member")
    parser.add_argument("--output_json", default=None, help="Optional summary JSON path")
    parser.add_argument("--output_csv", default=None, help="Optional ensembled prediction CSV(.gz) path")
    args = parser.parse_args()

    if len(args.pred) < 2:
        raise ValueError("Ensembling needs at least two prediction files")

    cfg = load_config(args.config)
    eval_cfg = cfg.get("evaluation", {})
    thresholds = threshold_grid(cfg)
    tolerance = int(eval_cfg.get("event_tolerance", 1))
    min_distance = int(eval_cfg.get("min_distance", 6))
    prominence = float(eval_cfg.get("prominence", 0.0))

    merged = None
    pred_paths = [str(Path(path).resolve()) for path in args.pred]
    for member_idx, pred_path in enumerate(pred_paths):
        frame = load_prediction_frame(Path(pred_path), member_idx=member_idx)
        if merged is None:
            merged = frame
        else:
            merged = merged.merge(frame, on=KEY_COLUMNS, how="inner", validate="one_to_one")

    if merged is None:
        raise RuntimeError("No predictions were loaded")

    score_cols = [col for col in merged.columns if col.startswith("score_")]
    merged["score"] = merged[score_cols].mean(axis=1).astype(np.float32)
    ensemble_df = merged[KEY_COLUMNS + ["score"]].copy()

    sequence_scores, sequence_labels = build_sequence_maps(ensemble_df)
    best = search_best_threshold(
        sequence_scores=sequence_scores,
        sequence_labels=sequence_labels,
        thresholds=thresholds,
        tolerance=tolerance,
        min_distance=min_distance,
        prominence=prominence,
    )

    summary = {
        "num_members": len(pred_paths),
        "prediction_paths": pred_paths,
        "best_threshold": best.threshold,
        "val_average_precision": best.average_precision,
        "event_precision": best.precision,
        "event_recall": best.recall,
        "event_f1": best.f1,
        "mean_offset": best.mean_offset,
        "matches": best.matches,
        "pred_events": best.pred_events,
        "true_events": best.true_events,
    }

    if args.output_json:
        out_json = Path(args.output_json).resolve()
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if args.output_csv:
        out_csv = Path(args.output_csv).resolve()
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        compression = "gzip" if out_csv.suffix == ".gz" or out_csv.name.endswith(".csv.gz") else None
        ensemble_df.to_csv(out_csv, index=False, compression=compression)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
