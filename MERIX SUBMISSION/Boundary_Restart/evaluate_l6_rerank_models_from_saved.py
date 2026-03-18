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

from boundary_restart.config import load_config, threshold_grid
from run_l6_candidate_rerank_multipiece_seed42 import (
    DEFAULT_PIECES,
    MIN_UNION_PRECISION,
    TARGET_SPECS,
    build_full_sequence_scores,
    load_detector_predictions,
    row_from_metrics,
    search_threshold_strict,
)


def build_model(model_name: str, seed: int):
    if model_name == "logreg":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=4000, class_weight="balanced", random_state=seed)),
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
                        random_state=seed,
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
            random_state=seed,
            n_jobs=1,
        )
    raise ValueError(model_name)


def fit_predict(model_name: str, train_df: pd.DataFrame, val_df: pd.DataFrame, model_features: list[str], seed: int) -> np.ndarray:
    model = build_model(model_name, seed)
    x_train = train_df[model_features].to_numpy(dtype=np.float32)
    y_train = train_df["rerank_train_label"].to_numpy(dtype=np.int64)
    x_val = val_df[model_features].to_numpy(dtype=np.float32)
    sample_weight = 1.0 + train_df["frequency_target"].to_numpy(dtype=np.float32) * 4.0
    if model_name == "xgboost":
        model.fit(x_train, y_train, sample_weight=sample_weight)
    elif model_name == "logreg":
        model.fit(x_train, y_train, clf__sample_weight=sample_weight)
    else:
        model.fit(x_train, y_train)
    return np.asarray(model.predict_proba(x_val)[:, 1], dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["logreg", "mlp", "xgboost"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hard_exit", action="store_true")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    cfg = load_config(project_root / "configs/salience_grouped3_hi8_score_only_xml_curated.yaml")
    eval_cfg = cfg.get("evaluation", {})
    thresholds = threshold_grid(cfg)
    tolerance = int(eval_cfg.get("event_tolerance", 1))
    min_distance = int(eval_cfg.get("min_distance", 6))
    prominence = float(eval_cfg.get("prominence", 0.0))
    consensus_threshold = float(eval_cfg.get("consensus_threshold", 0.5))

    base_run_root = project_root / "outputs/local_runs/l6_candidate_rerank_multipiece_seed42_t0p05_p70"
    base_report_root = project_root / "reports/l6_candidate_rerank_multipiece_seed42_t0p05_p70"
    report_root = project_root / f"reports/l6_rerank_saved_{args.model}_seed{args.seed}_p70"
    report_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for outer_piece in DEFAULT_PIECES:
        piece_report_root = report_root / outer_piece
        piece_report_root.mkdir(parents=True, exist_ok=True)

        l6_dir = base_run_root / outer_piece / TARGET_SPECS["L6"][0]
        l6_piece_df, _, feature_cols = load_detector_predictions(
            project_root,
            cfg,
            outer_piece=outer_piece,
            detector_target=TARGET_SPECS["L6"][0],
            checkpoint_dir=l6_dir,
            cumulative_merge_tolerance=TARGET_SPECS["L6"][1],
        )

        candidate_df = pd.read_csv(base_report_root / outer_piece / "candidate_frame.csv.gz")
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
        cov = pd.read_csv(base_report_root / outer_piece / "leaderboard.csv")
        cov_row = cov[cov["method"] == "rerank_l5_l4_to_l6_strict"].iloc[0]
        coverage_matches = int(cov_row["candidate_matches"])
        coverage_true = int(cov_row["candidate_true_events"])
        coverage_ratio = float(cov_row["candidate_coverage"])

        model_features = list(feature_cols) + [
            "l6_base_score",
            "l5_score",
            "l4_score",
            "candidate_from_l5",
            "candidate_from_l4",
        ]

        if train_df.empty or val_df.empty or np.unique(train_df["rerank_train_label"].to_numpy(dtype=np.int64)).size < 2:
            rows.append(
                row_from_metrics(
                    piece_id=outer_piece,
                    method=f"rerank_{args.model}",
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

        val_df = val_df.copy()
        val_df["rerank_score"] = fit_predict(args.model, train_df, val_df, model_features, args.seed)
        val_df.to_csv(piece_report_root / f"val_candidates_{args.model}.csv.gz", index=False, compression="gzip")

        sequence_scores = build_full_sequence_scores(l6_piece_df, val_df)
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
                method=f"rerank_{args.model}",
                candidate_matches=coverage_matches,
                candidate_true_events=coverage_true,
                candidate_coverage=coverage_ratio,
                metrics=strict_metrics,
                meets_union_floor=strict_metrics is not None,
                threshold=None if strict_metrics is None else float(strict_metrics.threshold),
                note="" if strict_metrics is not None else "no_valid_threshold",
            )
        )

        pd.DataFrame([r for r in rows if r["piece_id"] == outer_piece]).to_csv(piece_report_root / "leaderboard.csv", index=False)

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
    print(report_root / "all_results.csv")
    print(report_root / "summary_mean.csv")
    print(summary.to_csv(index=False))
    if args.hard_exit:
        os._exit(0)


if __name__ == "__main__":
    main()
