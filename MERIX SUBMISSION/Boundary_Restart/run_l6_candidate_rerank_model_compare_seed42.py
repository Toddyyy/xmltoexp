#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from run_l6_candidate_rerank_multipiece_seed42 import (
    CANDIDATE_RADIUS,
    DEFAULT_PIECES,
    DEVICE,
    EARLY_STOP_PATIENCE,
    MAX_EPOCHS,
    MIN_UNION_PRECISION,
    SEED,
    TARGET_SPECS,
    TRAIN_FREQ_FLOOR,
    assemble_candidate_frame,
    build_full_sequence_scores,
    candidate_coverage,
    event_mask_from_scores,
    load_detector_predictions,
    load_threshold,
    row_from_metrics,
    search_threshold_strict,
    sequence_lookup,
    train_detector,
)
from boundary_restart.config import load_config, threshold_grid


def build_reranker(model_name: str):
    if model_name == "logreg":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=4000,
                        class_weight="balanced",
                        random_state=SEED,
                    ),
                ),
            ]
        )
    if model_name == "mlp":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    MLPClassifier(
                        hidden_layer_sizes=(32, 16),
                        activation="relu",
                        solver="adam",
                        alpha=1e-3,
                        batch_size=64,
                        learning_rate_init=1e-3,
                        max_iter=600,
                        early_stopping=True,
                        n_iter_no_change=20,
                        random_state=SEED,
                    ),
                ),
            ]
        )
    if model_name == "xgboost":
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            min_child_weight=1.0,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=SEED,
            n_jobs=1,
        )
    raise ValueError(model_name)


def fit_predict_reranker(model_name: str, train_df: pd.DataFrame, val_df: pd.DataFrame, model_features: list[str]) -> np.ndarray:
    x_train = train_df[model_features].to_numpy(dtype=np.float32)
    y_train = train_df["rerank_train_label"].to_numpy(dtype=np.int64)
    sample_weight = 1.0 + train_df["frequency_target"].to_numpy(dtype=np.float32) * 4.0
    x_val = val_df[model_features].to_numpy(dtype=np.float32)
    model = build_reranker(model_name)
    if model_name == "xgboost":
        model.fit(x_train, y_train, sample_weight=sample_weight)
        scores = model.predict_proba(x_val)[:, 1]
    else:
        fit_kwargs = {}
        if model_name == "logreg":
            fit_kwargs["clf__sample_weight"] = sample_weight
        model.fit(x_train, y_train, **fit_kwargs)
        scores = model.predict_proba(x_val)[:, 1]
    return np.asarray(scores, dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["logreg", "mlp", "xgboost"])
    parser.add_argument("--tag_suffix", default="")
    parser.add_argument("--hard_exit", action="store_true")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    cfg = load_config(project_root / "configs/salience_grouped3_hi8_score_only_xml_curated.yaml")
    train_script = project_root / "train_piece_union_protocol.py"
    config_path = project_root / "configs/salience_grouped3_hi8_score_only_xml_curated.yaml"

    run_root = project_root / "outputs/local_runs/l6_candidate_rerank_multipiece_seed42_t0p05_p70"
    suffix = f"_{args.tag_suffix}" if args.tag_suffix else ""
    report_root = project_root / f"reports/l6_candidate_rerank_model_compare_seed42_t0p05_p70{suffix}"
    run_root.mkdir(parents=True, exist_ok=True)
    report_root.mkdir(parents=True, exist_ok=True)

    eval_cfg = cfg.get("evaluation", {})
    thresholds = threshold_grid(cfg)
    tolerance = int(eval_cfg.get("event_tolerance", 1))
    min_distance = int(eval_cfg.get("min_distance", 6))
    prominence = float(eval_cfg.get("prominence", 0.0))
    consensus_threshold = float(eval_cfg.get("consensus_threshold", 0.5))

    rows: list[dict[str, object]] = []
    for outer_piece in DEFAULT_PIECES:
        piece_run_root = run_root / outer_piece
        piece_report_root = report_root / outer_piece
        piece_run_root.mkdir(parents=True, exist_ok=True)
        piece_report_root.mkdir(parents=True, exist_ok=True)

        for label, (target, cumulative_merge_tolerance) in TARGET_SPECS.items():
            train_detector(
                train_script,
                config_path,
                outer_piece=outer_piece,
                seed=SEED,
                label=label,
                target=target,
                cumulative_merge_tolerance=cumulative_merge_tolerance,
                output_dir=piece_run_root / target,
            )

        l6_dir = piece_run_root / TARGET_SPECS["L6"][0]
        l5_dir = piece_run_root / TARGET_SPECS["L5+"][0]
        l4_dir = piece_run_root / TARGET_SPECS["L4+"][0]

        l6_piece_df, l6_pred_df, feature_cols = load_detector_predictions(
            project_root,
            cfg,
            outer_piece=outer_piece,
            detector_target=TARGET_SPECS["L6"][0],
            checkpoint_dir=l6_dir,
            cumulative_merge_tolerance=TARGET_SPECS["L6"][1],
        )
        _, l5_pred_df, _ = load_detector_predictions(
            project_root,
            cfg,
            outer_piece=outer_piece,
            detector_target=TARGET_SPECS["L5+"][0],
            checkpoint_dir=l5_dir,
            cumulative_merge_tolerance=TARGET_SPECS["L5+"][1],
        )
        _, l4_pred_df, _ = load_detector_predictions(
            project_root,
            cfg,
            outer_piece=outer_piece,
            detector_target=TARGET_SPECS["L4+"][0],
            checkpoint_dir=l4_dir,
            cumulative_merge_tolerance=TARGET_SPECS["L4+"][1],
        )

        l5_threshold = load_threshold(l5_dir / "summary.json")
        l4_threshold = load_threshold(l4_dir / "summary.json")
        l6_scores = sequence_lookup(l6_pred_df)
        l5_scores = sequence_lookup(l5_pred_df)
        l4_scores = sequence_lookup(l4_pred_df)
        l5_mask = event_mask_from_scores(
            l5_pred_df,
            l5_threshold,
            radius=CANDIDATE_RADIUS,
            min_distance=min_distance,
            prominence=prominence,
        )
        l4_mask = event_mask_from_scores(
            l4_pred_df,
            l4_threshold,
            radius=CANDIDATE_RADIUS,
            min_distance=min_distance,
            prominence=prominence,
        )

        candidate_df = assemble_candidate_frame(
            l6_piece_df,
            l6_scores=l6_scores,
            l5_scores=l5_scores,
            l4_scores=l4_scores,
            l5_mask=l5_mask,
            l4_mask=l4_mask,
        )
        candidate_df.to_csv(piece_report_root / "candidate_frame.csv.gz", index=False, compression="gzip")

        val_candidate_df = candidate_df[candidate_df["protocol_split"] == "val"].copy()
        coverage_matches, coverage_true = candidate_coverage(val_candidate_df, l6_piece_df)
        coverage_ratio = float(coverage_matches / coverage_true) if coverage_true > 0 else 0.0

        direct_summary = json.loads((l6_dir / "summary.json").read_text(encoding="utf-8"))["union_metrics"]
        rows.append(
            {
                "piece_id": outer_piece,
                "method": "direct_l6",
                "candidate_matches": 0,
                "candidate_true_events": 0,
                "candidate_coverage": 0.0,
                "meets_union_floor": float(direct_summary["union_precision"]) >= MIN_UNION_PRECISION,
                "threshold": float(direct_summary["threshold"]),
                "union_precision": float(direct_summary["union_precision"]),
                "frequency_weighted_precision": float(direct_summary["frequency_weighted_precision"]),
                "consensus_precision": float(direct_summary["consensus_precision"]),
                "union_recall": float(direct_summary["union_recall"]),
                "weighted_recall": float(direct_summary["weighted_recall"]),
                "consensus_recall": float(direct_summary["consensus_recall"]),
                "pred_events": int(direct_summary["pred_events"]),
                "matches": int(direct_summary["matches"]),
                "note": "",
            }
        )

        train_df = candidate_df[candidate_df["protocol_split"] == "train"].copy()
        val_df = candidate_df[candidate_df["protocol_split"] == "val"].copy()
        model_features = list(feature_cols) + [
            "l6_base_score",
            "l5_score",
            "l4_score",
            "candidate_from_l5",
            "candidate_from_l4",
        ]
        if train_df.empty or val_df.empty or np.unique(train_df["rerank_train_label"].to_numpy(dtype=np.int64)).size < 2:
            for model_name in ("logreg", "xgboost", "mlp"):
                rows.append(
                    row_from_metrics(
                        piece_id=outer_piece,
                        method=f"rerank_{model_name}",
                        candidate_matches=coverage_matches,
                        candidate_true_events=coverage_true,
                        candidate_coverage=coverage_ratio,
                        metrics=None,
                        meets_union_floor=False,
                        threshold=None,
                        note="invalid_candidate_split",
                    )
                )
            continue

        for model_name in args.models:
            val_scored_df = val_df.copy()
            val_scored_df["rerank_score"] = fit_predict_reranker(model_name, train_df, val_df, model_features)
            val_scored_df.to_csv(piece_report_root / f"val_candidates_{model_name}.csv.gz", index=False, compression="gzip")

            sequence_scores = build_full_sequence_scores(l6_piece_df, val_scored_df)
            sequence_union = {}
            sequence_frequency = {}
            ordered_truth = l6_piece_df[l6_piece_df["protocol_split"] == "val"].sort_values(["piece_sample_id", "beat_idx"]).copy()
            for sample_id, group in ordered_truth.groupby("piece_sample_id", sort=False):
                sequence_union[str(sample_id)] = group["union_target"].to_numpy(dtype=np.float32)
                sequence_frequency[str(sample_id)] = group["frequency_target"].to_numpy(dtype=np.float32)

            strict_metrics = search_threshold_strict(
                sequence_scores=sequence_scores,
                sequence_union_labels=sequence_union,
                sequence_frequency_targets=sequence_frequency,
                thresholds=thresholds,
                tolerance=tolerance,
                min_distance=min_distance,
                consensus_threshold=consensus_threshold,
                prominence=prominence,
                min_union_precision=MIN_UNION_PRECISION,
            )
            rows.append(
                row_from_metrics(
                    piece_id=outer_piece,
                    method=f"rerank_{model_name}",
                    candidate_matches=coverage_matches,
                    candidate_true_events=coverage_true,
                    candidate_coverage=coverage_ratio,
                    metrics=strict_metrics,
                    meets_union_floor=strict_metrics is not None,
                    threshold=None if strict_metrics is None else float(strict_metrics.threshold),
                    note="" if strict_metrics is not None else "no_valid_threshold",
                )
            )

        piece_df = pd.DataFrame([row for row in rows if row["piece_id"] == outer_piece])
        piece_df.to_csv(piece_report_root / "leaderboard.csv", index=False)

    all_results = pd.DataFrame(rows)
    all_results.to_csv(report_root / "all_results.csv", index=False)
    summary = (
        all_results.groupby("method", as_index=False)
        .agg(
            pieces=("piece_id", "count"),
            valid_count=("meets_union_floor", "sum"),
            mean_union_precision=("union_precision", "mean"),
            mean_weighted_recall=("weighted_recall", "mean"),
            mean_consensus_recall=("consensus_recall", "mean"),
            mean_candidate_coverage=("candidate_coverage", "mean"),
        )
    )
    summary.to_csv(report_root / "summary_mean.csv", index=False)

    with (report_root / "config.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "pieces": DEFAULT_PIECES,
                "seed": SEED,
                "device": DEVICE,
                "train_frequency_floor": TRAIN_FREQ_FLOOR,
                "min_union_precision": MIN_UNION_PRECISION,
                "candidate_mode": "L5+L4 -> L6 rerank",
                "rerank_models": args.models,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(report_root / "all_results.csv")
    print(report_root / "summary_mean.csv")
    print(summary.to_csv(index=False))
    if args.hard_exit:
        os._exit(0)


if __name__ == "__main__":
    main()
