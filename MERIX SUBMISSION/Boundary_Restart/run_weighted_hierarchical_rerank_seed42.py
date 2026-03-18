#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from boundary_restart.config import load_config, resolve_path, threshold_grid
from boundary_restart.features import PeakConfig
from boundary_restart.metrics import evaluate_union_frequency_sequences, extract_events, greedy_match_pairs
from boundary_restart.models import build_sequence_model
from boundary_restart.table_io import feature_columns, load_table
from train_piece_union_protocol import (
    PieceUnionDataset,
    apply_piece_protocol_split,
    apply_rest_span_training_labels,
    build_piece_union_frame,
    collate_piece_union,
    detector_sequence_maps,
    piece_samples_from_frame,
    predict_detector,
    select_feature_columns,
)


OUTER_PIECES = ["M06-1", "M06-2", "M06-3"]
SEED = 42
TRAIN_FREQ_FLOOR = 0.05
MAX_EPOCHS = 60
EARLY_STOP_PATIENCE = 10
DEVICE = "mps"
PEAK_RADIUS = 1
CUMULATIVE_MERGE_TOLERANCE = 2

DEFAULT_TRAIN_LEVEL_WEIGHTS = {
    "level6": 1.00,
    "level5": 0.95,
    "level4": 0.90,
    "level3": 0.85,
    "level2": 0.80,
    "level1": 0.75,
}

DEFAULT_RERANK_PREV_WEIGHTS = {
    "L2+": 0.75,
    "L3+": 0.80,
    "L4+": 0.85,
    "L5+": 0.90,
    "L6": 0.95,
}

DEFAULT_STAGE_MIN_UNION_PRECISION = {
    "L1+": 0.85,
    "L2+": 0.85,
    "L3+": 0.85,
    "L4+": 0.85,
    "L5+": 0.80,
    "L6": 0.70,
}

TARGET_SPECS = {
    "L6": ("level6_boundary", 0),
    "L5+": ("level5plus_split56_boundary", CUMULATIVE_MERGE_TOLERANCE),
    "L4+": ("level4plus_split56_boundary", CUMULATIVE_MERGE_TOLERANCE),
    "L3+": ("level3plus_split56_boundary", CUMULATIVE_MERGE_TOLERANCE),
    "L2+": ("level2plus_split56_boundary", CUMULATIVE_MERGE_TOLERANCE),
    "L1+": ("level1plus_split56_boundary", CUMULATIVE_MERGE_TOLERANCE),
}

TOPDOWN_ORDER = ["L6", "L5+", "L4+", "L3+", "L2+", "L1+"]
BOTTOMUP_RERANK_ORDER = ["L2+", "L3+", "L4+", "L5+", "L6"]
PREV_STAGE = {
    "L2+": "L1+",
    "L3+": "L2+",
    "L4+": "L3+",
    "L5+": "L4+",
    "L6": "L5+",
}


def train_detector(
    train_script: Path,
    config_path: Path,
    *,
    label: str,
    backbone: str,
    target: str,
    cumulative_merge_tolerance: int,
    output_dir: Path,
    train_level_weights: dict[str, float],
    min_union_precision: float,
) -> None:
    summary_path = output_dir / "summary.json"
    if summary_path.exists() and (output_dir / "detector_best.pt").exists():
        return
    cmd = [
        sys.executable,
        str(train_script),
        "--config",
        str(config_path),
        "--heldout_piece",
        *OUTER_PIECES,
        "--model",
        backbone,
        "--device",
        DEVICE,
        "--seed",
        str(SEED),
        "--detector_target",
        target,
        "--selection_metric",
        "weighted_recall",
        "--precision_metric",
        "union_precision",
        "--min_precision",
        str(min_union_precision),
        "--epochs",
        str(MAX_EPOCHS),
        "--early_stop_patience",
        str(EARLY_STOP_PATIENCE),
        "--skip_stage_grading",
        "--min_train_frequency_target",
        str(TRAIN_FREQ_FLOOR),
        "--output_dir",
        str(output_dir),
        "--cumulative_component_weights_json",
        json.dumps(train_level_weights, sort_keys=True),
    ]
    if cumulative_merge_tolerance > 0:
        cmd.extend(["--cumulative_merge_tolerance", str(cumulative_merge_tolerance)])
    print("TRAIN", label, backbone)
    subprocess.run(cmd, check=True)


def build_piece_frames(
    cfg: dict,
    feature_cols: list[str],
    *,
    detector_target: str,
    cumulative_merge_tolerance: int,
    train_level_weights: dict[str, float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    table_path = resolve_path(cfg, cfg["data"]["beat_table_path"])
    df = load_table(table_path)
    df = apply_piece_protocol_split(df, heldout_pieces=OUTER_PIECES)
    peak_cfg = PeakConfig(
        distance=int(cfg.get("data", {}).get("peak_distance", 6)),
        height=float(cfg.get("data", {}).get("peak_height", 0.15)),
        prominence=float(cfg.get("data", {}).get("peak_prominence", 0.05)),
    )
    beat_unit_fallback = float(cfg.get("data", {}).get("beat_unit_fallback", 1.0))
    weighted_piece = build_piece_union_frame(
        df,
        feature_cols=feature_cols,
        target_mode=detector_target,
        peak_cfg=peak_cfg,
        beat_unit_fallback=beat_unit_fallback,
        cumulative_merge_tolerance=int(cumulative_merge_tolerance),
        cumulative_component_weights=train_level_weights,
    )
    weighted_piece = apply_rest_span_training_labels(
        weighted_piece,
        mode="none",
        min_len=2,
        source_col="xml_rest_duration_norm",
        source_threshold=1e-8,
        tolerance_negative_weight=1.0,
        min_train_frequency_target=float(TRAIN_FREQ_FLOOR),
    )
    eval_piece = build_piece_union_frame(
        df,
        feature_cols=feature_cols,
        target_mode=detector_target,
        peak_cfg=peak_cfg,
        beat_unit_fallback=beat_unit_fallback,
        cumulative_merge_tolerance=int(cumulative_merge_tolerance),
        cumulative_component_weights=None,
    )
    eval_piece = apply_rest_span_training_labels(
        eval_piece,
        mode="none",
        min_len=2,
        source_col="xml_rest_duration_norm",
        source_threshold=1e-8,
        tolerance_negative_weight=1.0,
        min_train_frequency_target=0.0,
    )
    return weighted_piece, eval_piece


def load_detector_predictions(
    cfg: dict,
    *,
    detector_target: str,
    checkpoint_dir: Path,
    cumulative_merge_tolerance: int,
    train_level_weights: dict[str, float],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    checkpoint = torch.load(checkpoint_dir / "detector_best.pt", map_location="cpu", weights_only=False)
    feature_cols = list(checkpoint["feature_columns"])
    weighted_piece_df, eval_piece_df = build_piece_frames(
        cfg,
        feature_cols,
        detector_target=detector_target,
        cumulative_merge_tolerance=cumulative_merge_tolerance,
        train_level_weights=train_level_weights,
    )

    mean = np.asarray(checkpoint["mean"], dtype=np.float32)
    std = np.asarray(checkpoint["std"], dtype=np.float32)
    samples = piece_samples_from_frame(weighted_piece_df, feature_cols, split="train") + piece_samples_from_frame(
        weighted_piece_df,
        feature_cols,
        split="val",
    )
    ds = PieceUnionDataset(samples, mean=mean, std=std)
    loader = DataLoader(ds, batch_size=64, shuffle=False, collate_fn=collate_piece_union)

    model = build_sequence_model(
        checkpoint["model_type"],
        input_dim=len(feature_cols),
        cfg=cfg,
        output_dim=1,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(torch.device("cpu"))
    pred_df = predict_detector(model, loader, device=torch.device("cpu"))
    return weighted_piece_df, eval_piece_df, pred_df, feature_cols


def sequence_lookup(pred_df: pd.DataFrame) -> dict[str, np.ndarray]:
    seq_scores, _, _ = detector_sequence_maps(pred_df)
    return {str(k): np.asarray(v, dtype=np.float32) for k, v in seq_scores.items()}


def truth_maps(piece_df: pd.DataFrame, *, split: str | None = None) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    union_map = {}
    freq_map = {}
    work = piece_df.copy()
    if split is not None:
        work = work[work["protocol_split"] == split].copy()
    ordered = work.sort_values(["piece_sample_id", "beat_idx"]).copy()
    for sample_id, group in ordered.groupby("piece_sample_id", sort=False):
        union_map[str(sample_id)] = group["union_target"].to_numpy(dtype=np.float32)
        freq_map[str(sample_id)] = group["frequency_target"].to_numpy(dtype=np.float32)
    return union_map, freq_map


def search_threshold_strict_or_best(
    *,
    sequence_scores: dict[str, np.ndarray],
    sequence_union_labels: dict[str, np.ndarray],
    sequence_frequency_targets: dict[str, np.ndarray],
    thresholds: np.ndarray,
    tolerance: int,
    min_distance: int,
    consensus_threshold: float,
    prominence: float,
    min_union_precision: float,
):
    best_meeting = None
    best_any = None

    def key(metrics):
        return (
            float(metrics.weighted_recall),
            float(metrics.union_precision),
            float(metrics.frequency_weighted_precision),
            float(metrics.consensus_recall),
            -float(metrics.mean_offset or 1e9),
            -float(metrics.threshold),
        )

    for threshold in thresholds.tolist():
        metrics = evaluate_union_frequency_sequences(
            sequence_scores=sequence_scores,
            sequence_union_labels=sequence_union_labels,
            sequence_frequency_targets=sequence_frequency_targets,
            threshold=float(threshold),
            tolerance=int(tolerance),
            min_distance=int(min_distance),
            consensus_threshold=float(consensus_threshold),
            prominence=float(prominence),
        )
        if metrics.union_precision >= float(min_union_precision):
            if best_meeting is None or key(metrics) > key(best_meeting):
                best_meeting = metrics
        if best_any is None or key(metrics) > key(best_any):
            best_any = metrics
    if best_meeting is not None:
        return best_meeting, True
    if best_any is None:
        raise ValueError("threshold grid is empty")
    return best_any, False


def event_mask_from_sequence_scores(
    sequence_scores: dict[str, np.ndarray],
    threshold: float,
    *,
    radius: int,
    min_distance: int,
    prominence: float,
) -> dict[str, np.ndarray]:
    masks = {}
    for sample_id, scores in sequence_scores.items():
        scores = np.asarray(scores, dtype=np.float32)
        events = extract_events(scores, threshold=float(threshold), min_distance=int(min_distance), prominence=float(prominence))
        mask = np.zeros(scores.shape[0], dtype=bool)
        for event in events.tolist():
            start = max(0, int(event) - int(radius))
            end = min(scores.shape[0], int(event) + int(radius) + 1)
            mask[start:end] = True
        masks[str(sample_id)] = mask
    return masks


def assemble_stage_candidate_frame(
    weighted_piece_df: pd.DataFrame,
    current_scores: dict[str, np.ndarray],
    prev_scores: dict[str, np.ndarray],
    prev_masks: dict[str, np.ndarray],
    *,
    prev_weight: float,
    train_freq_floor: float,
) -> pd.DataFrame:
    rows = []
    ordered = weighted_piece_df.sort_values(["piece_sample_id", "beat_idx"]).copy()
    for sample_id, group in ordered.groupby("piece_sample_id", sort=False):
        sample_id = str(sample_id)
        group = group.reset_index(drop=True)
        candidate = prev_masks.get(sample_id)
        if candidate is None or not np.any(candidate):
            continue
        sub = group.loc[candidate].copy()
        sub["current_base_score"] = current_scores[sample_id][candidate]
        sub["prev_stage_score"] = prev_scores[sample_id][candidate]
        sub["prev_stage_score_weighted"] = (prev_scores[sample_id][candidate] * float(prev_weight)).astype(np.float32)
        sub["candidate_from_prev"] = 1.0
        sub["rerank_train_label"] = (sub["train_frequency_target"].to_numpy(dtype=np.float32) >= float(train_freq_floor)).astype(
            np.int64
        )
        rows.append(sub)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, axis=0, ignore_index=True)


def build_full_sequence_scores(
    weighted_piece_df: pd.DataFrame,
    scored_df: pd.DataFrame,
) -> dict[str, np.ndarray]:
    score_map = {}
    if not scored_df.empty:
        for sample_id, group in scored_df.groupby("piece_sample_id", sort=False):
            score_map[str(sample_id)] = {
                int(row.beat_idx): float(row.rerank_score)
                for row in group.itertuples(index=False)
            }
    full_scores = {}
    ordered = weighted_piece_df.sort_values(["piece_sample_id", "beat_idx"]).copy()
    for sample_id, group in ordered.groupby("piece_sample_id", sort=False):
        sample_id = str(sample_id)
        seq = np.zeros(len(group), dtype=np.float32)
        beat_to_score = score_map.get(sample_id, {})
        for idx, beat_idx in enumerate(group["beat_idx"].astype(int).tolist()):
            if beat_idx in beat_to_score:
                seq[idx] = beat_to_score[beat_idx]
        full_scores[sample_id] = seq
    return full_scores


def candidate_coverage(candidate_df: pd.DataFrame, eval_piece_df: pd.DataFrame) -> tuple[int, int]:
    candidate_map = {
        str(sample_id): group["beat_idx"].to_numpy(dtype=np.int32)
        for sample_id, group in candidate_df.groupby("piece_sample_id", sort=False)
    }
    total_matches = 0
    total_true = 0
    val_truth = eval_piece_df[eval_piece_df["protocol_split"] == "val"].sort_values(["piece_sample_id", "beat_idx"]).copy()
    for sample_id, group in val_truth.groupby("piece_sample_id", sort=False):
        sample_id = str(sample_id)
        true_events = np.flatnonzero(group["union_target"].to_numpy(dtype=np.float32) > 0.5).astype(np.int32)
        cand_events = candidate_map.get(sample_id, np.empty(0, dtype=np.int32))
        match_pairs = greedy_match_pairs(cand_events, true_events, tolerance=1)
        total_matches += len(match_pairs)
        total_true += int(true_events.size)
    return total_matches, total_true


def evaluate_direct_stage(
    label: str,
    eval_piece_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    *,
    thresholds: np.ndarray,
    tolerance: int,
    min_distance: int,
    consensus_threshold: float,
    prominence: float,
    min_union_precision: float,
) -> dict:
    sequence_scores = sequence_lookup(pred_df)
    val_union, val_frequency = truth_maps(eval_piece_df, split="val")
    val_scores = {k: v for k, v in sequence_scores.items() if k in val_union}
    metrics, met_floor = search_threshold_strict_or_best(
        sequence_scores=val_scores,
        sequence_union_labels=val_union,
        sequence_frequency_targets=val_frequency,
        thresholds=thresholds,
        tolerance=tolerance,
        min_distance=min_distance,
        consensus_threshold=consensus_threshold,
        prominence=prominence,
        min_union_precision=min_union_precision,
    )
    return {
        "stage": label,
        "method": "direct",
        "threshold": float(metrics.threshold),
        "required_union_precision": float(min_union_precision),
        "meets_union_floor": bool(met_floor),
        "union_precision": float(metrics.union_precision),
        "frequency_weighted_precision": float(metrics.frequency_weighted_precision),
        "consensus_precision": float(metrics.consensus_precision),
        "union_recall": float(metrics.union_recall),
        "weighted_recall": float(metrics.weighted_recall),
        "consensus_recall": float(metrics.consensus_recall),
        "pred_events": int(metrics.pred_events),
        "matches": int(metrics.matches),
        "all_sequence_scores": sequence_scores,
        "val_union": val_union,
        "val_frequency": val_frequency,
    }


def load_existing_baseline() -> pd.DataFrame:
    root = Path(__file__).resolve().parent / "outputs/local_runs/frequency_pruned_hierarchy_seed42_e60"
    stage_to_dir = {
        "L6": "M06_outer_level6_boundary_freqfloor_0p05_seed42",
        "L5+": "M06_outer_level5plus_split56_boundary_freqfloor_0p05_seed42",
        "L4+": "M06_outer_level4plus_split56_boundary_freqfloor_0p05_seed42",
        "L3+": "M06_outer_level3plus_split56_boundary_freqfloor_0p05_seed42",
        "L2+": "M06_outer_level2plus_split56_boundary_freqfloor_0p05_seed42",
        "L1+": "M06_outer_level1plus_split56_boundary_freqfloor_0p05_seed42",
    }
    rows = []
    for stage, dirname in stage_to_dir.items():
        summary = json.loads((root / dirname / "summary.json").read_text(encoding="utf-8"))
        metrics = summary["union_metrics"]
        rows.append(
            {
                "stage": stage,
                "method": "baseline_direct",
                "threshold": float(metrics["threshold"]),
                "required_union_precision": float(DEFAULT_STAGE_MIN_UNION_PRECISION[stage]),
                "meets_union_floor": float(metrics["union_precision"]) >= float(DEFAULT_STAGE_MIN_UNION_PRECISION[stage]),
                "union_precision": float(metrics["union_precision"]),
                "frequency_weighted_precision": float(metrics["frequency_weighted_precision"]),
                "consensus_precision": float(metrics["consensus_precision"]),
                "union_recall": float(metrics["union_recall"]),
                "weighted_recall": float(metrics["weighted_recall"]),
                "consensus_recall": float(metrics["consensus_recall"]),
                "pred_events": int(metrics["pred_events"]),
                "matches": int(metrics["matches"]),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Weighted top-down cumulative training + bottom-up rerank prototype.")
    parser.add_argument("--tag", default="seed42_t0p05_p70")
    parser.add_argument("--backbone", default="tcn", choices=["tcn", "bilstm"])
    parser.add_argument("--train_level_weights_json", default=None)
    parser.add_argument("--rerank_prev_weights_json", default=None)
    parser.add_argument("--stage_min_union_precision_json", default=None)
    parser.add_argument("--include_tcn_baseline", action="store_true")
    args = parser.parse_args()

    train_level_weights = dict(DEFAULT_TRAIN_LEVEL_WEIGHTS)
    rerank_prev_weights = dict(DEFAULT_RERANK_PREV_WEIGHTS)
    stage_min_union_precision = dict(DEFAULT_STAGE_MIN_UNION_PRECISION)
    if args.train_level_weights_json:
        train_level_weights.update({str(k): float(v) for k, v in json.loads(str(args.train_level_weights_json)).items()})
    if args.rerank_prev_weights_json:
        rerank_prev_weights.update({str(k): float(v) for k, v in json.loads(str(args.rerank_prev_weights_json)).items()})
    if args.stage_min_union_precision_json:
        stage_min_union_precision.update(
            {str(k): float(v) for k, v in json.loads(str(args.stage_min_union_precision_json)).items()}
        )

    project_root = Path(__file__).resolve().parent
    cfg = load_config(project_root / "configs/salience_grouped3_hi8_score_only_xml_curated.yaml")
    train_script = project_root / "train_piece_union_protocol.py"
    config_path = project_root / "configs/salience_grouped3_hi8_score_only_xml_curated.yaml"
    run_root = project_root / f"outputs/local_runs/weighted_hierarchical_rerank_{args.tag}"
    report_root = project_root / f"reports/weighted_hierarchical_rerank_{args.tag}"
    run_root.mkdir(parents=True, exist_ok=True)
    report_root.mkdir(parents=True, exist_ok=True)

    table_path = resolve_path(cfg, cfg["data"]["beat_table_path"])
    df = load_table(table_path)
    feature_cols = select_feature_columns(cfg, feature_columns(df))
    eval_cfg = cfg.get("evaluation", {})
    thresholds = threshold_grid(cfg)
    tolerance = int(eval_cfg.get("event_tolerance", 1))
    min_distance = int(eval_cfg.get("min_distance", 6))
    prominence = float(eval_cfg.get("prominence", 0.0))
    consensus_threshold = float(eval_cfg.get("consensus_threshold", 0.5))

    direct_outputs = {}
    direct_stage_metrics = {}
    for label in TOPDOWN_ORDER:
        target, cumulative_merge_tolerance = TARGET_SPECS[label]
        output_dir = run_root / target
        train_detector(
            train_script,
            config_path,
            label=label,
            backbone=args.backbone,
            target=target,
            cumulative_merge_tolerance=cumulative_merge_tolerance,
            output_dir=output_dir,
            train_level_weights=train_level_weights,
            min_union_precision=float(stage_min_union_precision[label]),
        )
        weighted_piece_df, eval_piece_df, pred_df, loaded_feature_cols = load_detector_predictions(
            cfg,
            detector_target=target,
            checkpoint_dir=output_dir,
            cumulative_merge_tolerance=cumulative_merge_tolerance,
            train_level_weights=train_level_weights,
        )
        direct_outputs[label] = {
            "weighted_piece_df": weighted_piece_df,
            "eval_piece_df": eval_piece_df,
            "pred_df": pred_df,
            "feature_cols": loaded_feature_cols,
        }

    rows = []
    final_stage_outputs = {}
    if args.include_tcn_baseline:
        baseline_df = load_existing_baseline()
        rows.extend(baseline_df.to_dict(orient="records"))

    for label in reversed(TOPDOWN_ORDER):
        direct_metrics = evaluate_direct_stage(
            label,
            direct_outputs[label]["eval_piece_df"],
            direct_outputs[label]["pred_df"],
            thresholds=thresholds,
            tolerance=tolerance,
            min_distance=min_distance,
            consensus_threshold=consensus_threshold,
            prominence=prominence,
            min_union_precision=float(stage_min_union_precision[label]),
        )
        direct_stage_metrics[label] = direct_metrics
        rows.append({k: v for k, v in direct_metrics.items() if k not in {"all_sequence_scores", "val_union", "val_frequency"}})
        if label == "L1+":
            final_stage_outputs[label] = direct_metrics

    for label in BOTTOMUP_RERANK_ORDER:
        prev_label = PREV_STAGE[label]
        prev_output = final_stage_outputs[prev_label]
        current_output = direct_outputs[label]
        current_direct_metrics = direct_stage_metrics[label]
        prev_masks = event_mask_from_sequence_scores(
            prev_output["all_sequence_scores"],
            float(prev_output["threshold"]),
            radius=PEAK_RADIUS,
            min_distance=min_distance,
            prominence=prominence,
        )
        candidate_df = assemble_stage_candidate_frame(
            current_output["weighted_piece_df"],
            current_scores=sequence_lookup(current_output["pred_df"]),
            prev_scores=prev_output["all_sequence_scores"],
            prev_masks=prev_masks,
            prev_weight=float(rerank_prev_weights[label]),
            train_freq_floor=float(TRAIN_FREQ_FLOOR),
        )
        if candidate_df.empty:
            raise RuntimeError(f"No candidates for {label} from {prev_label}")
        candidate_df.to_csv(report_root / f"{label}_candidates.csv.gz", index=False, compression="gzip")

        model_features = list(current_output["feature_cols"]) + [
            "current_base_score",
            "prev_stage_score",
            "prev_stage_score_weighted",
            "candidate_from_prev",
        ]
        train_df = candidate_df[candidate_df["protocol_split"] == "train"].copy()
        val_df = candidate_df[candidate_df["protocol_split"] == "val"].copy()
        reranker = Pipeline(
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
        x_train = train_df[model_features].to_numpy(dtype=np.float32)
        y_train = train_df["rerank_train_label"].to_numpy(dtype=np.int64)
        sample_weight = 1.0 + train_df["train_frequency_target"].to_numpy(dtype=np.float32) * 4.0
        reranker.fit(x_train, y_train, clf__sample_weight=sample_weight)

        scored_df = candidate_df.copy()
        scored_df["rerank_score"] = reranker.predict_proba(candidate_df[model_features].to_numpy(dtype=np.float32))[:, 1].astype(
            np.float32
        )
        scored_df.to_csv(report_root / f"{label}_rerank_scores.csv.gz", index=False, compression="gzip")
        rerank_scores = build_full_sequence_scores(current_output["weighted_piece_df"], scored_df)

        val_keys = set(
            current_output["eval_piece_df"].loc[current_output["eval_piece_df"]["protocol_split"] == "val", "piece_sample_id"].astype(str).tolist()
        )
        val_scores = {k: v for k, v in rerank_scores.items() if k in val_keys}
        val_union = {k: current_direct_metrics["val_union"][k] for k in val_keys}
        val_frequency = {k: current_direct_metrics["val_frequency"][k] for k in val_keys}
        rerank_metrics, met_floor = search_threshold_strict_or_best(
            sequence_scores=val_scores,
            sequence_union_labels=val_union,
            sequence_frequency_targets=val_frequency,
            thresholds=thresholds,
            tolerance=tolerance,
            min_distance=min_distance,
            consensus_threshold=consensus_threshold,
            prominence=prominence,
            min_union_precision=float(stage_min_union_precision[label]),
        )
        coverage_matches, coverage_true = candidate_coverage(val_df, current_output["eval_piece_df"])
        rows.append(
            {
                "stage": label,
                "method": "weighted_topdown_rerank",
                "threshold": float(rerank_metrics.threshold),
                "required_union_precision": float(stage_min_union_precision[label]),
                "meets_union_floor": bool(met_floor),
                "union_precision": float(rerank_metrics.union_precision),
                "frequency_weighted_precision": float(rerank_metrics.frequency_weighted_precision),
                "consensus_precision": float(rerank_metrics.consensus_precision),
                "union_recall": float(rerank_metrics.union_recall),
                "weighted_recall": float(rerank_metrics.weighted_recall),
                "consensus_recall": float(rerank_metrics.consensus_recall),
                "pred_events": int(rerank_metrics.pred_events),
                "matches": int(rerank_metrics.matches),
                "candidate_matches": int(coverage_matches),
                "candidate_true_events": int(coverage_true),
                "candidate_coverage": float(coverage_matches / coverage_true) if coverage_true > 0 else 0.0,
                "prev_stage": prev_label,
                "prev_stage_weight": float(rerank_prev_weights[label]),
            }
        )
        final_stage_outputs[label] = {
            "stage": label,
            "method": "weighted_topdown_rerank",
            "threshold": float(rerank_metrics.threshold),
            "all_sequence_scores": rerank_scores,
            "val_union": current_direct_metrics["val_union"],
            "val_frequency": current_direct_metrics["val_frequency"],
        }

    result_df = pd.DataFrame(rows)
    result_df.to_csv(report_root / "all_results.csv", index=False)
    summary_df = (
        result_df.groupby(["stage", "method"], as_index=False)
        .agg(
            union_precision=("union_precision", "mean"),
            frequency_weighted_precision=("frequency_weighted_precision", "mean"),
            consensus_precision=("consensus_precision", "mean"),
            union_recall=("union_recall", "mean"),
            weighted_recall=("weighted_recall", "mean"),
            consensus_recall=("consensus_recall", "mean"),
            pred_events=("pred_events", "mean"),
            matches=("matches", "mean"),
        )
    )
    summary_df.to_csv(report_root / "summary_by_stage.csv", index=False)
    (report_root / "config.json").write_text(
        json.dumps(
            {
                "tag": args.tag,
                "backbone": args.backbone,
                "train_level_weights": train_level_weights,
                "rerank_prev_weights": rerank_prev_weights,
                "stage_min_union_precision": stage_min_union_precision,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    allowed_methods = ["direct", "weighted_topdown_rerank"]
    if args.include_tcn_baseline:
        allowed_methods.insert(0, "baseline_direct")
    direct_vs_rerank = (
        result_df[result_df["method"].isin(allowed_methods)]
        .sort_values(["stage", "method"])
        .reset_index(drop=True)
    )
    direct_vs_rerank.to_csv(report_root / "direct_vs_rerank.csv", index=False)
    print(report_root / "all_results.csv")
    print(report_root / "summary_by_stage.csv")
    print(direct_vs_rerank.to_csv(index=False))


if __name__ == "__main__":
    main()
