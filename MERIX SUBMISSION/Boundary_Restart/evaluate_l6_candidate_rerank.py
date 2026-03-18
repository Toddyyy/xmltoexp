#!/usr/bin/env python3

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from boundary_restart.config import load_config, resolve_path
from boundary_restart.features import PeakConfig
from boundary_restart.metrics import extract_events, greedy_match_pairs, search_union_frequency_threshold
from boundary_restart.models import build_sequence_model
from boundary_restart.table_io import load_table
from train_piece_union_protocol import (
    PieceUnionDataset,
    apply_piece_protocol_split,
    build_piece_union_frame,
    collate_piece_union,
    detector_sequence_maps,
    piece_samples_from_frame,
    predict_detector,
)


OUTER_PIECES = ["M06-1", "M06-2", "M06-3"]
SEED = 42
TRAIN_FREQ_FLOOR = 0.05
DEVICE = "cpu"
MIN_DISTANCE = 6
PROMINENCE = 0.0
CONSENSUS_THRESHOLD = 0.5
MATCH_TOLERANCE = 1
CANDIDATE_RADIUS = 1


def load_detector_bundle(
    project_root: Path,
    cfg: dict,
    *,
    detector_target: str,
    checkpoint_dir: Path,
    cumulative_merge_tolerance: int,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], float]:
    checkpoint = torch.load(checkpoint_dir / "detector_best.pt", map_location="cpu", weights_only=False)
    feature_cols = list(checkpoint["feature_columns"])
    table_path = resolve_path(cfg, cfg["data"]["beat_table_path"])
    df = load_table(table_path)
    df = apply_piece_protocol_split(df, heldout_pieces=OUTER_PIECES)
    peak_cfg = PeakConfig(
        distance=int(cfg.get("data", {}).get("peak_distance", 6)),
        height=float(cfg.get("data", {}).get("peak_height", 0.15)),
        prominence=float(cfg.get("data", {}).get("peak_prominence", 0.05)),
    )
    beat_unit_fallback = float(cfg.get("data", {}).get("beat_unit_fallback", 1.0))
    piece_df = build_piece_union_frame(
        df,
        feature_cols=feature_cols,
        target_mode=detector_target,
        peak_cfg=peak_cfg,
        beat_unit_fallback=beat_unit_fallback,
        cumulative_merge_tolerance=int(cumulative_merge_tolerance),
    )

    mean = np.asarray(checkpoint["mean"], dtype=np.float32)
    std = np.asarray(checkpoint["std"], dtype=np.float32)
    samples = []
    for split in ("train", "val"):
        samples.extend(piece_samples_from_frame(piece_df, feature_cols, split=split))
    ds = PieceUnionDataset(samples, mean=mean, std=std)
    loader = DataLoader(ds, batch_size=64, shuffle=False, collate_fn=collate_piece_union)

    model = build_sequence_model(
        checkpoint["model_type"],
        input_dim=len(feature_cols),
        cfg=cfg,
        output_dim=1,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    pred_df = predict_detector(model, loader, device=torch.device(DEVICE))
    threshold = float(checkpoint.get("best_threshold", checkpoint_dir.joinpath("summary.json").read_text()))
    if isinstance(threshold, str):
        threshold = float(json.loads((checkpoint_dir / "summary.json").read_text())["union_metrics"]["threshold"])
    return piece_df, pred_df, feature_cols, float(threshold)


def load_threshold(summary_path: Path) -> float:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return float(summary["union_metrics"]["threshold"])


def event_mask_from_scores(
    pred_df: pd.DataFrame,
    threshold: float,
    *,
    radius: int,
) -> dict[str, np.ndarray]:
    masks = {}
    ordered = pred_df.sort_values(["sample_id", "beat_idx"])
    for sample_id, group in ordered.groupby("sample_id", sort=False):
        scores = group["detector_score"].to_numpy(dtype=np.float32)
        events = extract_events(scores, threshold=threshold, min_distance=MIN_DISTANCE, prominence=PROMINENCE)
        mask = np.zeros(scores.shape[0], dtype=bool)
        for event in events.tolist():
            start = max(0, int(event) - int(radius))
            end = min(scores.shape[0], int(event) + int(radius) + 1)
            mask[start:end] = True
        masks[str(sample_id)] = mask
    return masks


def sequence_lookup(pred_df: pd.DataFrame) -> dict[str, np.ndarray]:
    seq_scores, _, _ = detector_sequence_maps(pred_df)
    return {str(k): np.asarray(v, dtype=np.float32) for k, v in seq_scores.items()}


def assemble_candidate_frame(
    l6_piece_df: pd.DataFrame,
    l6_scores: dict[str, np.ndarray],
    l5_scores: dict[str, np.ndarray],
    l4_scores: dict[str, np.ndarray],
    l5_mask: dict[str, np.ndarray],
    l4_mask: dict[str, np.ndarray],
    feature_cols: list[str],
    candidate_mode: str,
) -> pd.DataFrame:
    rows = []
    ordered = l6_piece_df.sort_values(["piece_sample_id", "beat_idx"]).copy()
    for sample_id, group in ordered.groupby("piece_sample_id", sort=False):
        sample_id = str(sample_id)
        group = group.reset_index(drop=True)
        l5_candidate = l5_mask[sample_id]
        l4_candidate = l4_mask[sample_id]
        if candidate_mode == "l5":
            candidate = l5_candidate
        elif candidate_mode == "l5_l4":
            candidate = np.logical_or(l5_candidate, l4_candidate)
        else:
            raise ValueError(candidate_mode)
        if not np.any(candidate):
            continue
        sub = group.loc[candidate].copy()
        sub["l6_base_score"] = l6_scores[sample_id][candidate]
        sub["l5_score"] = l5_scores[sample_id][candidate]
        sub["l4_score"] = l4_scores[sample_id][candidate]
        sub["candidate_from_l5"] = l5_candidate[candidate].astype(np.float32)
        sub["candidate_from_l4"] = l4_candidate[candidate].astype(np.float32)
        sub["candidate_mode"] = candidate_mode
        rows.append(sub)
    return pd.concat(rows, axis=0, ignore_index=True) if rows else pd.DataFrame()


def candidate_coverage(
    candidate_df: pd.DataFrame,
    l6_piece_df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    ordered_truth = l6_piece_df.sort_values(["piece_sample_id", "beat_idx"]).copy()
    candidate_map = {
        str(sample_id): group["beat_idx"].to_numpy(dtype=np.int32)
        for sample_id, group in candidate_df.groupby("piece_sample_id", sort=False)
    }
    for sample_id, group in ordered_truth.groupby("piece_sample_id", sort=False):
        sample_id = str(sample_id)
        true_events = np.flatnonzero(group["union_target"].to_numpy(dtype=np.float32) > 0.5).astype(np.int32)
        cand_events = candidate_map.get(sample_id, np.empty(0, dtype=np.int32))
        match_pairs = greedy_match_pairs(cand_events, true_events, tolerance=MATCH_TOLERANCE)
        rows.append(
            {
                "piece_id": group["piece_id"].iloc[0],
                "matches": len(match_pairs),
                "candidate_events": int(cand_events.size),
                "true_union_events": int(true_events.size),
            }
        )
    return pd.DataFrame(rows)


def build_full_sequence_scores(
    piece_df: pd.DataFrame,
    rerank_df: pd.DataFrame,
) -> dict[str, np.ndarray]:
    full_scores = {}
    ordered = piece_df.sort_values(["piece_sample_id", "beat_idx"]).copy()
    rerank_map = {}
    if not rerank_df.empty:
        for sample_id, group in rerank_df.groupby("piece_sample_id", sort=False):
            rerank_map[str(sample_id)] = {
                int(row.beat_idx): float(row.rerank_score)
                for row in group.itertuples(index=False)
            }
    for sample_id, group in ordered.groupby("piece_sample_id", sort=False):
        sample_id = str(sample_id)
        seq = np.zeros(len(group), dtype=np.float32)
        beat_map = rerank_map.get(sample_id, {})
        for idx, beat_idx in enumerate(group["beat_idx"].astype(int).tolist()):
            if beat_idx in beat_map:
                seq[idx] = beat_map[beat_idx]
        full_scores[sample_id] = seq
    return full_scores


def main() -> None:
    project_root = Path(__file__).resolve().parent
    cfg = load_config(project_root / "configs/salience_grouped3_hi8_score_only_xml_curated.yaml")

    base_dir = project_root / "outputs/local_runs/frequency_pruned_hierarchy_seed42_e60"
    l6_dir = base_dir / "M06_outer_level6_boundary_freqfloor_0p05_seed42"
    l5_dir = base_dir / "M06_outer_level5plus_split56_boundary_freqfloor_0p05_seed42"
    l4_dir = base_dir / "M06_outer_level4plus_split56_boundary_freqfloor_0p05_seed42"

    l6_piece_df, l6_pred_df, feature_cols, _ = load_detector_bundle(
        project_root, cfg, detector_target="level6_boundary", checkpoint_dir=l6_dir, cumulative_merge_tolerance=0
    )
    _, l5_pred_df, _, _ = load_detector_bundle(
        project_root, cfg, detector_target="level5plus_split56_boundary", checkpoint_dir=l5_dir, cumulative_merge_tolerance=2
    )
    _, l4_pred_df, _, _ = load_detector_bundle(
        project_root, cfg, detector_target="level4plus_split56_boundary", checkpoint_dir=l4_dir, cumulative_merge_tolerance=2
    )

    l5_threshold = load_threshold(l5_dir / "summary.json")
    l4_threshold = load_threshold(l4_dir / "summary.json")

    l6_scores = sequence_lookup(l6_pred_df)
    l5_scores = sequence_lookup(l5_pred_df)
    l4_scores = sequence_lookup(l4_pred_df)
    l5_mask = event_mask_from_scores(l5_pred_df, l5_threshold, radius=CANDIDATE_RADIUS)
    l4_mask = event_mask_from_scores(l4_pred_df, l4_threshold, radius=CANDIDATE_RADIUS)

    thresholds = np.asarray(cfg.get("evaluation", {}).get("thresholds", np.linspace(0.05, 0.95, 19)), dtype=np.float32)
    if thresholds.ndim == 0:
        thresholds = np.linspace(0.05, 0.95, 19, dtype=np.float32)
    tolerance = int(cfg.get("evaluation", {}).get("event_tolerance", MATCH_TOLERANCE))
    min_distance = int(cfg.get("evaluation", {}).get("min_distance", MIN_DISTANCE))
    prominence = float(cfg.get("evaluation", {}).get("prominence", PROMINENCE))
    consensus_threshold = float(cfg.get("evaluation", {}).get("consensus_threshold", CONSENSUS_THRESHOLD))

    sequence_union = {}
    sequence_frequency = {}
    ordered_l6 = l6_piece_df.sort_values(["piece_sample_id", "beat_idx"]).copy()
    for sample_id, group in ordered_l6.groupby("piece_sample_id", sort=False):
        sequence_union[str(sample_id)] = group["union_target"].to_numpy(dtype=np.float32)
        sequence_frequency[str(sample_id)] = group["frequency_target"].to_numpy(dtype=np.float32)

    direct_baseline = json.loads((l6_dir / "summary.json").read_text(encoding="utf-8"))["union_metrics"]

    report_dir = project_root / "reports/l6_candidate_rerank_m06_seed42"
    report_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for candidate_mode in ("l5", "l5_l4"):
        candidate_df = assemble_candidate_frame(
            l6_piece_df,
            l6_scores=l6_scores,
            l5_scores=l5_scores,
            l4_scores=l4_scores,
            l5_mask=l5_mask,
            l4_mask=l4_mask,
            feature_cols=feature_cols,
            candidate_mode=candidate_mode,
        )
        coverage_df = candidate_coverage(candidate_df, l6_piece_df)
        coverage_df.to_csv(report_dir / f"{candidate_mode}_candidate_coverage.csv", index=False)

        train_df = candidate_df[candidate_df["protocol_split"] == "train"].copy()
        val_df = candidate_df[candidate_df["protocol_split"] == "val"].copy()
        if train_df.empty or val_df.empty:
            continue

        model_features = list(feature_cols) + [
            "l6_base_score",
            "l5_score",
            "l4_score",
            "candidate_from_l5",
            "candidate_from_l4",
        ]
        x_train = train_df[model_features].to_numpy(dtype=np.float32)
        y_train = train_df["union_target"].to_numpy(dtype=np.int64)
        sample_weight = 1.0 + train_df["frequency_target"].to_numpy(dtype=np.float32) * 4.0
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
        reranker.fit(x_train, y_train, clf__sample_weight=sample_weight)

        val_probs = reranker.predict_proba(val_df[model_features].to_numpy(dtype=np.float32))[:, 1].astype(np.float32)
        val_df = val_df.copy()
        val_df["rerank_score"] = val_probs
        val_df.to_csv(report_dir / f"{candidate_mode}_val_candidates.csv.gz", index=False, compression="gzip")

        sequence_scores = build_full_sequence_scores(l6_piece_df[l6_piece_df["protocol_split"] == "val"], val_df)
        for min_precision in (0.85, 0.60):
            metrics = search_union_frequency_threshold(
                sequence_scores=sequence_scores,
                sequence_union_labels={k: sequence_union[k] for k in sequence_scores},
                sequence_frequency_targets={k: sequence_frequency[k] for k in sequence_scores},
                thresholds=thresholds,
                tolerance=tolerance,
                min_distance=min_distance,
                min_precision=float(min_precision),
                consensus_threshold=consensus_threshold,
                prominence=prominence,
                primary_metric="weighted_recall",
                precision_metric="union_precision",
                min_union_precision=float(min_precision),
            )
            rows.append(
                {
                    "method": f"rerank_{candidate_mode}",
                    "candidate_mode": candidate_mode,
                    "min_union_precision_floor": float(min_precision),
                    "threshold": float(metrics.threshold),
                    "union_precision": float(metrics.union_precision),
                    "frequency_weighted_precision": float(metrics.frequency_weighted_precision),
                    "consensus_precision": float(metrics.consensus_precision),
                    "union_recall": float(metrics.union_recall),
                    "weighted_recall": float(metrics.weighted_recall),
                    "consensus_recall": float(metrics.consensus_recall),
                    "pred_events": int(metrics.pred_events),
                    "matches": int(metrics.matches),
                    "candidate_events_val": int(len(val_df)),
                }
            )

    rows.append(
        {
            "method": "direct_l6_baseline",
            "candidate_mode": "full_sequence",
            "min_union_precision_floor": 0.85,
            "threshold": float(direct_baseline["threshold"]),
            "union_precision": float(direct_baseline["union_precision"]),
            "frequency_weighted_precision": float(direct_baseline["frequency_weighted_precision"]),
            "consensus_precision": float(direct_baseline["consensus_precision"]),
            "union_recall": float(direct_baseline["union_recall"]),
            "weighted_recall": float(direct_baseline["weighted_recall"]),
            "consensus_recall": float(direct_baseline["consensus_recall"]),
            "pred_events": int(direct_baseline["pred_events"]),
            "matches": int(direct_baseline["matches"]),
            "candidate_events_val": 0,
        }
    )

    leaderboard = pd.DataFrame(rows).sort_values(
        ["min_union_precision_floor", "weighted_recall", "union_precision"],
        ascending=[False, False, False],
    )
    leaderboard.to_csv(report_dir / "leaderboard.csv", index=False)
    print(report_dir / "leaderboard.csv")
    print(leaderboard.to_csv(index=False))


if __name__ == "__main__":
    main()
