#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from run_weighted_hierarchical_rerank_seed42 import (
    CUMULATIVE_MERGE_TOLERANCE,
    DEVICE,
    OUTER_PIECES,
    PEAK_RADIUS,
    SEED,
    TARGET_SPECS,
    TRAIN_FREQ_FLOOR,
    build_full_sequence_scores,
    candidate_coverage,
    event_mask_from_sequence_scores,
    load_config,
    load_detector_predictions,
    resolve_path,
    search_threshold_strict_or_best,
    sequence_lookup,
    threshold_grid,
    truth_maps,
)


BASE_TAG = "seed42_t0p05_stage708085_widegap"
PREV_STAGES = ["L1+", "L2+", "L3+", "L4+", "L5+"]
PREV_CANDIDATE_STAGES = ["L3+", "L4+", "L5+"]
PREV_WEIGHTS = {
    "L1+": 0.20,
    "L2+": 0.40,
    "L3+": 0.60,
    "L4+": 0.80,
    "L5+": 1.00,
}


def load_threshold(summary_path: Path) -> float:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return float(summary["union_metrics"]["threshold"])


def assemble_candidate_frame_all_prev(
    weighted_piece_df: pd.DataFrame,
    current_scores: dict[str, np.ndarray],
    prev_scores: dict[str, dict[str, np.ndarray]],
    prev_masks: dict[str, dict[str, np.ndarray]],
) -> pd.DataFrame:
    rows = []
    ordered = weighted_piece_df.sort_values(["piece_sample_id", "beat_idx"]).copy()
    for sample_id, group in ordered.groupby("piece_sample_id", sort=False):
        sample_id = str(sample_id)
        group = group.reset_index(drop=True)
        candidate = np.zeros(len(group), dtype=bool)
        for stage in PREV_CANDIDATE_STAGES:
            candidate |= prev_masks[stage].get(sample_id, np.zeros(len(group), dtype=bool))
        if not np.any(candidate):
            continue
        sub = group.loc[candidate].copy()
        sub["current_base_score"] = current_scores[sample_id][candidate]
        for stage in PREV_STAGES:
            stage_scores = prev_scores[stage][sample_id][candidate]
            sub[f"{stage}_score"] = stage_scores
            sub[f"{stage}_score_weighted"] = (stage_scores * float(PREV_WEIGHTS[stage])).astype(np.float32)
            sub[f"candidate_from_{stage}"] = prev_masks[stage][sample_id][candidate].astype(np.float32)
        sub["rerank_train_label"] = (
            sub["train_frequency_target"].to_numpy(dtype=np.float32) >= float(TRAIN_FREQ_FLOOR)
        ).astype(np.int64)
        rows.append(sub)
    return pd.concat(rows, axis=0, ignore_index=True) if rows else pd.DataFrame()


def main() -> None:
    project_root = Path(__file__).resolve().parent
    cfg = load_config(project_root / "configs/salience_grouped3_hi8_score_only_xml_curated.yaml")
    eval_cfg = cfg.get("evaluation", {})
    thresholds = threshold_grid(cfg)
    tolerance = int(eval_cfg.get("event_tolerance", 1))
    min_distance = int(eval_cfg.get("min_distance", 6))
    prominence = float(eval_cfg.get("prominence", 0.0))
    consensus_threshold = float(eval_cfg.get("consensus_threshold", 0.5))

    run_root = project_root / f"outputs/local_runs/weighted_hierarchical_rerank_{BASE_TAG}"
    source_report_root = project_root / f"reports/weighted_hierarchical_rerank_{BASE_TAG}"
    report_root = project_root / f"reports/l6_rerank_all_prev_{BASE_TAG}"
    report_root.mkdir(parents=True, exist_ok=True)

    current_output = load_detector_predictions(
        cfg,
        detector_target=TARGET_SPECS["L6"][0],
        checkpoint_dir=run_root / TARGET_SPECS["L6"][0],
        cumulative_merge_tolerance=TARGET_SPECS["L6"][1],
        train_level_weights=json.loads((source_report_root / "config.json").read_text(encoding="utf-8"))["train_level_weights"],
    )
    weighted_piece_df, eval_piece_df, pred_df, feature_cols = current_output
    current_scores = sequence_lookup(pred_df)

    prev_scores = {}
    prev_masks = {}
    for stage in PREV_STAGES:
        target, cumulative_merge_tolerance = TARGET_SPECS[stage]
        _, _, stage_pred_df, _ = load_detector_predictions(
            cfg,
            detector_target=target,
            checkpoint_dir=run_root / target,
            cumulative_merge_tolerance=cumulative_merge_tolerance,
            train_level_weights=json.loads((source_report_root / "config.json").read_text(encoding="utf-8"))["train_level_weights"],
        )
        seq_scores = sequence_lookup(stage_pred_df)
        prev_scores[stage] = seq_scores
        prev_threshold = load_threshold(run_root / target / "summary.json")
        prev_masks[stage] = event_mask_from_sequence_scores(
            seq_scores,
            prev_threshold,
            radius=PEAK_RADIUS,
            min_distance=min_distance,
            prominence=prominence,
        )

    candidate_df = assemble_candidate_frame_all_prev(
        weighted_piece_df,
        current_scores=current_scores,
        prev_scores=prev_scores,
        prev_masks=prev_masks,
    )
    candidate_df.to_csv(report_root / "candidate_frame.csv.gz", index=False, compression="gzip")

    val_candidate_df = candidate_df[candidate_df["protocol_split"] == "val"].copy()
    coverage_matches, coverage_true = candidate_coverage(val_candidate_df, eval_piece_df)
    coverage_ratio = float(coverage_matches / coverage_true) if coverage_true > 0 else 0.0

    train_df = candidate_df[candidate_df["protocol_split"] == "train"].copy()
    val_df = candidate_df[candidate_df["protocol_split"] == "val"].copy()
    model_features = list(feature_cols) + ["current_base_score"]
    for stage in PREV_STAGES:
        model_features.extend(
            [f"{stage}_score", f"{stage}_score_weighted", f"candidate_from_{stage}"]
        )

    reranker = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=4000, class_weight="balanced", random_state=SEED)),
        ]
    )
    x_train = train_df[model_features].to_numpy(dtype=np.float32)
    y_train = train_df["rerank_train_label"].to_numpy(dtype=np.int64)
    sample_weight = 1.0 + train_df["train_frequency_target"].to_numpy(dtype=np.float32) * 4.0
    reranker.fit(x_train, y_train, clf__sample_weight=sample_weight)

    val_df = val_df.copy()
    val_df["rerank_score"] = reranker.predict_proba(val_df[model_features].to_numpy(dtype=np.float32))[:, 1].astype(
        np.float32
    )
    val_df.to_csv(report_root / "val_candidates.csv.gz", index=False, compression="gzip")

    sequence_scores = build_full_sequence_scores(weighted_piece_df, val_df)
    val_union, val_frequency = truth_maps(eval_piece_df, split="val")
    val_keys = set(val_union)
    val_scores = {k: sequence_scores[k] for k in val_keys}
    strict_metrics, met_floor = search_threshold_strict_or_best(
        sequence_scores=val_scores,
        sequence_union_labels=val_union,
        sequence_frequency_targets=val_frequency,
        thresholds=thresholds,
        tolerance=tolerance,
        min_distance=min_distance,
        consensus_threshold=consensus_threshold,
        prominence=prominence,
        min_union_precision=0.70,
    )

    base_df = pd.read_csv(source_report_root / "all_results.csv")
    l6_rows = base_df[base_df["stage"] == "L6"].copy()
    l6_rows = l6_rows[l6_rows["method"].isin(["baseline_direct", "direct", "weighted_topdown_rerank"])]

    new_row = pd.DataFrame(
        [
            {
                "stage": "L6",
                "method": "rerank_all_prev_levels",
                "threshold": float(strict_metrics.threshold),
                "required_union_precision": 0.70,
                "meets_union_floor": bool(met_floor),
                "union_precision": float(strict_metrics.union_precision),
                "frequency_weighted_precision": float(strict_metrics.frequency_weighted_precision),
                "consensus_precision": float(strict_metrics.consensus_precision),
                "union_recall": float(strict_metrics.union_recall),
                "weighted_recall": float(strict_metrics.weighted_recall),
                "consensus_recall": float(strict_metrics.consensus_recall),
                "pred_events": int(strict_metrics.pred_events),
                "matches": int(strict_metrics.matches),
                "candidate_matches": int(coverage_matches),
                "candidate_true_events": int(coverage_true),
                "candidate_coverage": float(coverage_ratio),
                "prev_stage": "L1..L5",
                "prev_stage_weight": np.nan,
            }
        ]
    )

    result_df = pd.concat([l6_rows, new_row], ignore_index=True)
    result_df.to_csv(report_root / "l6_compare.csv", index=False)
    (report_root / "config.json").write_text(
        json.dumps(
            {
                "base_tag": BASE_TAG,
                "outer_pieces": OUTER_PIECES,
                "seed": SEED,
                "prev_stages": PREV_STAGES,
                "candidate_stages": PREV_CANDIDATE_STAGES,
                "prev_weights": PREV_WEIGHTS,
                "train_frequency_floor": TRAIN_FREQ_FLOOR,
                "required_union_precision": 0.70,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(report_root / "l6_compare.csv")
    print(result_df.to_csv(index=False))


if __name__ == "__main__":
    main()
