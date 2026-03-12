#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from boundary_restart.config import load_config, threshold_grid
from boundary_restart.metrics import (
    evaluate_labeled_event_sequences,
    search_threshold_with_min_precision,
)


def labeled_metrics_to_dict(metrics) -> dict:
    return {
        "threshold": metrics.threshold,
        "macro_precision": metrics.macro_precision,
        "macro_recall": metrics.macro_recall,
        "macro_f1": metrics.macro_f1,
        "micro_precision": metrics.micro_precision,
        "micro_recall": metrics.micro_recall,
        "micro_f1": metrics.micro_f1,
        "mean_offset": metrics.mean_offset,
        "class_precision": {str(k): float(v) for k, v in metrics.class_precision.items()},
        "class_recall": {str(k): float(v) for k, v in metrics.class_recall.items()},
        "class_f1": {str(k): float(v) for k, v in metrics.class_f1.items()},
        "class_matches": {str(k): int(v) for k, v in metrics.class_matches.items()},
        "class_pred_events": {str(k): int(v) for k, v in metrics.class_pred_events.items()},
        "class_true_events": {str(k): int(v) for k, v in metrics.class_true_events.items()},
    }


def load_prediction_frame(path: Path, model_name: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {
        "sample_id",
        "piece_id",
        "beat_idx",
        "detector_target",
        "stage_class_midhigh",
        "detector_score",
        "pred_midhigh_class",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    rename_map = {
        "detector_score": f"detector_score_{model_name}",
        "pred_midhigh_class": f"pred_midhigh_class_{model_name}",
    }
    if "pred_high_prob" in frame.columns:
        rename_map["pred_high_prob"] = f"pred_high_prob_{model_name}"
    keep_cols = [
        "sample_id",
        "piece_id",
        "beat_idx",
        "detector_target",
        "stage_class_midhigh",
        "detector_score",
        "pred_midhigh_class",
    ]
    if "pred_high_prob" in frame.columns:
        keep_cols.append("pred_high_prob")
    return frame[keep_cols].rename(columns=rename_map)


def merge_prediction_frames(tcn_path: Path, bilstm_path: Path) -> pd.DataFrame:
    tcn = load_prediction_frame(tcn_path, model_name="tcn")
    bilstm = load_prediction_frame(bilstm_path, model_name="bilstm")
    merged = tcn.merge(
        bilstm,
        on=["sample_id", "piece_id", "beat_idx", "detector_target", "stage_class_midhigh"],
        how="inner",
        validate="one_to_one",
    )
    if merged.empty:
        raise ValueError("Merged prediction frame is empty; check that both files are from the same held-out split")
    return merged.sort_values(["sample_id", "beat_idx"]).reset_index(drop=True)


def build_pred_labels(frame: pd.DataFrame, rule: str) -> np.ndarray:
    tcn = frame["pred_midhigh_class_tcn"].to_numpy(dtype=np.int32)
    bilstm = frame["pred_midhigh_class_bilstm"].to_numpy(dtype=np.int32)
    if rule == "tcn":
        return tcn
    if rule == "bilstm":
        return bilstm
    if rule == "agree_else_tcn":
        return np.where(tcn == bilstm, tcn, tcn).astype(np.int32)
    if rule == "agree_else_bilstm":
        return np.where(tcn == bilstm, tcn, bilstm).astype(np.int32)
    if rule == "agree_only":
        return np.where(tcn == bilstm, tcn, 0).astype(np.int32)
    raise ValueError(f"Unsupported label rule: {rule}")


def build_strategy_scores(frame: pd.DataFrame, gate_values: list[float]) -> dict[str, np.ndarray]:
    tcn = frame["detector_score_tcn"].to_numpy(dtype=np.float32)
    bilstm = frame["detector_score_bilstm"].to_numpy(dtype=np.float32)
    strategies: dict[str, np.ndarray] = {
        "tcn": tcn,
        "bilstm": bilstm,
        "mean": (tcn + bilstm) / 2.0,
        "weighted_tcn_0p75": 0.75 * tcn + 0.25 * bilstm,
        "max": np.maximum(tcn, bilstm),
        "min": np.minimum(tcn, bilstm),
        "product": tcn * bilstm,
    }
    for gate in gate_values:
        gate_mask = (bilstm >= gate).astype(np.float32)
        gate_tag = f"{gate:.2f}".replace(".", "p")
        strategies[f"tcn_gated_bilstm_g{gate_tag}"] = tcn * gate_mask
        strategies[f"weighted_tcn_0p75_gated_bilstm_g{gate_tag}"] = (0.75 * tcn + 0.25 * bilstm) * gate_mask
        strategies[f"min_gated_bilstm_g{gate_tag}"] = np.minimum(tcn, bilstm) * gate_mask
    return strategies


def sequence_maps(
    frame: pd.DataFrame,
    score_column: str,
    label_array: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    sequence_scores: dict[str, np.ndarray] = {}
    sequence_detector_labels: dict[str, np.ndarray] = {}
    sequence_stage_labels: dict[str, np.ndarray] = {}
    sequence_pred_labels: dict[str, np.ndarray] = {}
    ordered = frame.sort_values(["sample_id", "beat_idx"]).copy()
    ordered["ensemble_pred_label"] = label_array.astype(np.int32)
    for sample_id, group in ordered.groupby("sample_id", sort=False):
        sequence_scores[sample_id] = group[score_column].to_numpy(dtype=np.float32)
        sequence_detector_labels[sample_id] = group["detector_target"].to_numpy(dtype=np.float32)
        sequence_stage_labels[sample_id] = group["stage_class_midhigh"].to_numpy(dtype=np.int32)
        sequence_pred_labels[sample_id] = group["ensemble_pred_label"].to_numpy(dtype=np.int32)
    return sequence_scores, sequence_detector_labels, sequence_stage_labels, sequence_pred_labels


def evaluate_strategy(
    frame: pd.DataFrame,
    score_name: str,
    score_values: np.ndarray,
    label_rule: str,
    thresholds: np.ndarray,
    tolerance: int,
    min_distance: int,
    min_precision: float,
    prominence: float,
) -> dict:
    eval_frame = frame.copy()
    eval_frame["ensemble_score"] = score_values.astype(np.float32)
    pred_labels = build_pred_labels(eval_frame, rule=label_rule)
    sequence_scores, sequence_detector_labels, sequence_stage_labels, sequence_pred_labels = sequence_maps(
        eval_frame,
        score_column="ensemble_score",
        label_array=pred_labels,
    )
    detector_metrics = search_threshold_with_min_precision(
        sequence_scores=sequence_scores,
        sequence_labels=sequence_detector_labels,
        thresholds=thresholds,
        tolerance=tolerance,
        min_distance=min_distance,
        min_precision=min_precision,
        prominence=prominence,
    )
    labeled_metrics = evaluate_labeled_event_sequences(
        sequence_scores=sequence_scores,
        sequence_pred_labels=sequence_pred_labels,
        sequence_true_labels=sequence_stage_labels,
        positive_classes=(1, 2),
        threshold=float(detector_metrics.threshold),
        tolerance=tolerance,
        min_distance=min_distance,
        prominence=prominence,
    )
    return {
        "strategy": score_name,
        "label_rule": label_rule,
        "precision_floor_met": bool(detector_metrics.precision >= min_precision),
        "event_precision": float(detector_metrics.precision),
        "event_recall": float(detector_metrics.recall),
        "event_f1": float(detector_metrics.f1),
        "event_ap": float(detector_metrics.average_precision),
        "best_threshold": float(detector_metrics.threshold),
        "mean_offset": None if detector_metrics.mean_offset is None else float(detector_metrics.mean_offset),
        "matches": int(detector_metrics.matches),
        "pred_events": int(detector_metrics.pred_events),
        "true_events": int(detector_metrics.true_events),
        "end_to_end_midhigh": labeled_metrics_to_dict(labeled_metrics),
    }


def rank_key(record: dict) -> tuple:
    end_to_end = record["end_to_end_midhigh"]
    return (
        int(record["precision_floor_met"]),
        float(record["event_recall"]),
        float(record["event_precision"]),
        float(end_to_end["macro_f1"]),
        float(record["event_f1"]),
        -float(record["best_threshold"]),
    )


def main():
    parser = argparse.ArgumentParser(description="Evaluate TCN/BiLSTM high-precision ensemble strategies.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--tcn_pred", required=True)
    parser.add_argument("--bilstm_pred", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--min_precision", type=float, default=0.95)
    parser.add_argument("--label_rules", nargs="+", default=["tcn", "agree_else_tcn", "agree_only"])
    parser.add_argument("--gate_values", nargs="*", type=float, default=[0.60, 0.70, 0.80, 0.85, 0.90, 0.95])
    args = parser.parse_args()

    cfg = load_config(args.config)
    eval_cfg = cfg.get("evaluation", {})
    thresholds = threshold_grid(cfg)
    tolerance = int(eval_cfg.get("event_tolerance", 1))
    min_distance = int(eval_cfg.get("min_distance", 6))
    prominence = float(eval_cfg.get("prominence", 0.0))

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    merged = merge_prediction_frames(Path(args.tcn_pred).resolve(), Path(args.bilstm_pred).resolve())
    strategies = build_strategy_scores(merged, gate_values=[float(v) for v in args.gate_values])

    results = []
    for score_name, score_values in strategies.items():
        for label_rule in args.label_rules:
            result = evaluate_strategy(
                frame=merged,
                score_name=score_name,
                score_values=score_values,
                label_rule=label_rule,
                thresholds=thresholds,
                tolerance=tolerance,
                min_distance=min_distance,
                min_precision=float(args.min_precision),
                prominence=prominence,
            )
            results.append(result)

    ranked = sorted(results, key=rank_key, reverse=True)
    best_by_recall = ranked[0] if ranked else None
    best_by_macro = max(
        ranked,
        key=lambda item: (
            int(item["precision_floor_met"]),
            float(item["end_to_end_midhigh"]["macro_f1"]),
            float(item["event_precision"]),
            float(item["event_recall"]),
        ),
        default=None,
    )

    summary = {
        "config": str(Path(args.config).resolve()),
        "tcn_pred": str(Path(args.tcn_pred).resolve()),
        "bilstm_pred": str(Path(args.bilstm_pred).resolve()),
        "min_precision": float(args.min_precision),
        "label_rules": list(args.label_rules),
        "gate_values": [float(v) for v in args.gate_values],
        "rows_merged": int(len(merged)),
        "unique_samples": int(merged["sample_id"].nunique()),
        "best_by_recall_under_floor": best_by_recall,
        "best_by_macro_f1_under_floor": best_by_macro,
        "all_results": ranked,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pd.DataFrame(ranked).to_json(out_dir / "results.jsonl", orient="records", lines=True)
    pd.DataFrame(
        [
            {
                "strategy": item["strategy"],
                "label_rule": item["label_rule"],
                "precision_floor_met": item["precision_floor_met"],
                "event_precision": item["event_precision"],
                "event_recall": item["event_recall"],
                "event_f1": item["event_f1"],
                "event_ap": item["event_ap"],
                "best_threshold": item["best_threshold"],
                "macro_event_f1": item["end_to_end_midhigh"]["macro_f1"],
                "micro_event_f1": item["end_to_end_midhigh"]["micro_f1"],
            }
            for item in ranked
        ]
    ).to_csv(out_dir / "leaderboard.csv", index=False)
    print(f"Wrote ensemble evaluation to {out_dir}")
    if best_by_recall is not None:
        print(
            f"Best recall under floor: {best_by_recall['strategy']} + {best_by_recall['label_rule']} | "
            f"precision={best_by_recall['event_precision']:.4f} recall={best_by_recall['event_recall']:.4f} "
            f"macro_event_f1={best_by_recall['end_to_end_midhigh']['macro_f1']:.4f}"
        )
    if best_by_macro is not None:
        print(
            f"Best macro_event_f1 under floor: {best_by_macro['strategy']} + {best_by_macro['label_rule']} | "
            f"precision={best_by_macro['event_precision']:.4f} recall={best_by_macro['event_recall']:.4f} "
            f"macro_event_f1={best_by_macro['end_to_end_midhigh']['macro_f1']:.4f}"
        )


if __name__ == "__main__":
    main()
