#!/usr/bin/env python3

from __future__ import annotations

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
SEEDS = [42, 43, 44]
TRAIN_FREQ_FLOOR = 0.05
MIN_UNION_PRECISION = 0.70
MAX_EPOCHS = 60
EARLY_STOP_PATIENCE = 10
DEVICE = "mps"
CANDIDATE_RADIUS = 1

TARGET_SPECS = {
    "L6": ("level6_boundary", 0),
    "L5+": ("level5plus_split56_boundary", 2),
    "L4+": ("level4plus_split56_boundary", 2),
}


def train_detector(
    train_script: Path,
    config_path: Path,
    *,
    seed: int,
    label: str,
    target: str,
    cumulative_merge_tolerance: int,
    output_dir: Path,
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
        "tcn",
        "--device",
        DEVICE,
        "--seed",
        str(seed),
        "--detector_target",
        target,
        "--selection_metric",
        "weighted_recall",
        "--precision_metric",
        "union_precision",
        "--min_precision",
        str(MIN_UNION_PRECISION),
        "--epochs",
        str(MAX_EPOCHS),
        "--early_stop_patience",
        str(EARLY_STOP_PATIENCE),
        "--skip_stage_grading",
        "--min_train_frequency_target",
        str(TRAIN_FREQ_FLOOR),
        "--output_dir",
        str(output_dir),
    ]
    if cumulative_merge_tolerance > 0:
        cmd.extend(["--cumulative_merge_tolerance", str(cumulative_merge_tolerance)])
    print("TRAIN", label, "seed", seed)
    subprocess.run(cmd, check=True)


def load_threshold(summary_path: Path) -> float:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return float(summary["union_metrics"]["threshold"])


def load_detector_predictions(
    project_root: Path,
    cfg: dict,
    *,
    detector_target: str,
    checkpoint_dir: Path,
    cumulative_merge_tolerance: int,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
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
    samples = piece_samples_from_frame(piece_df, feature_cols, split="train") + piece_samples_from_frame(
        piece_df, feature_cols, split="val"
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
    return piece_df, pred_df, feature_cols


def event_mask_from_scores(pred_df: pd.DataFrame, threshold: float, *, radius: int, min_distance: int, prominence: float):
    masks = {}
    ordered = pred_df.sort_values(["sample_id", "beat_idx"])
    for sample_id, group in ordered.groupby("sample_id", sort=False):
        scores = group["detector_score"].to_numpy(dtype=np.float32)
        events = extract_events(scores, threshold=float(threshold), min_distance=int(min_distance), prominence=float(prominence))
        mask = np.zeros(len(group), dtype=bool)
        for event in events.tolist():
            start = max(0, int(event) - int(radius))
            end = min(len(group), int(event) + int(radius) + 1)
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
) -> pd.DataFrame:
    rows = []
    ordered = l6_piece_df.sort_values(["piece_sample_id", "beat_idx"]).copy()
    for sample_id, group in ordered.groupby("piece_sample_id", sort=False):
        sample_id = str(sample_id)
        group = group.reset_index(drop=True)
        candidate = np.logical_or(l5_mask[sample_id], l4_mask[sample_id])
        if not np.any(candidate):
            continue
        sub = group.loc[candidate].copy()
        sub["l6_base_score"] = l6_scores[sample_id][candidate]
        sub["l5_score"] = l5_scores[sample_id][candidate]
        sub["l4_score"] = l4_scores[sample_id][candidate]
        sub["candidate_from_l5"] = l5_mask[sample_id][candidate].astype(np.float32)
        sub["candidate_from_l4"] = l4_mask[sample_id][candidate].astype(np.float32)
        sub["rerank_train_label"] = (sub["frequency_target"].to_numpy(dtype=np.float32) >= TRAIN_FREQ_FLOOR).astype(np.int64)
        rows.append(sub)
    return pd.concat(rows, axis=0, ignore_index=True) if rows else pd.DataFrame()


def candidate_coverage(candidate_df: pd.DataFrame, l6_piece_df: pd.DataFrame) -> tuple[int, int]:
    candidate_map = {
        str(sample_id): group["beat_idx"].to_numpy(dtype=np.int32)
        for sample_id, group in candidate_df.groupby("piece_sample_id", sort=False)
    }
    total_matches = 0
    total_true = 0
    val_truth = l6_piece_df[l6_piece_df["protocol_split"] == "val"].sort_values(["piece_sample_id", "beat_idx"]).copy()
    for sample_id, group in val_truth.groupby("piece_sample_id", sort=False):
        sample_id = str(sample_id)
        true_events = np.flatnonzero(group["union_target"].to_numpy(dtype=np.float32) > 0.5).astype(np.int32)
        cand_events = candidate_map.get(sample_id, np.empty(0, dtype=np.int32))
        match_pairs = greedy_match_pairs(cand_events, true_events, tolerance=1)
        total_matches += len(match_pairs)
        total_true += int(true_events.size)
    return total_matches, total_true


def build_full_sequence_scores(piece_df: pd.DataFrame, rerank_df: pd.DataFrame) -> dict[str, np.ndarray]:
    rerank_map = {}
    if not rerank_df.empty:
        for sample_id, group in rerank_df.groupby("piece_sample_id", sort=False):
            rerank_map[str(sample_id)] = {
                int(row.beat_idx): float(row.rerank_score)
                for row in group.itertuples(index=False)
            }
    scores = {}
    ordered = piece_df[piece_df["protocol_split"] == "val"].sort_values(["piece_sample_id", "beat_idx"]).copy()
    for sample_id, group in ordered.groupby("piece_sample_id", sort=False):
        sample_id = str(sample_id)
        seq = np.zeros(len(group), dtype=np.float32)
        beat_to_score = rerank_map.get(sample_id, {})
        for idx, beat_idx in enumerate(group["beat_idx"].astype(int).tolist()):
            if beat_idx in beat_to_score:
                seq[idx] = beat_to_score[beat_idx]
        scores[sample_id] = seq
    return scores


def main() -> None:
    project_root = Path(__file__).resolve().parent
    cfg = load_config(project_root / "configs/salience_grouped3_hi8_score_only_xml_curated.yaml")
    train_script = project_root / "train_piece_union_protocol.py"
    config_path = project_root / "configs/salience_grouped3_hi8_score_only_xml_curated.yaml"
    run_root = project_root / "outputs/local_runs/l6_candidate_rerank_multiseed_t0p05_p70"
    report_root = project_root / "reports/l6_candidate_rerank_multiseed_t0p05_p70"
    run_root.mkdir(parents=True, exist_ok=True)
    report_root.mkdir(parents=True, exist_ok=True)

    eval_cfg = cfg.get("evaluation", {})
    thresholds = threshold_grid(cfg)
    tolerance = int(eval_cfg.get("event_tolerance", 1))
    min_distance = int(eval_cfg.get("min_distance", 6))
    prominence = float(eval_cfg.get("prominence", 0.0))
    consensus_threshold = float(eval_cfg.get("consensus_threshold", 0.5))

    rows = []
    for seed in SEEDS:
        seed_run_root = run_root / f"seed{seed}"
        seed_report_root = report_root / f"seed{seed}"
        seed_run_root.mkdir(parents=True, exist_ok=True)
        seed_report_root.mkdir(parents=True, exist_ok=True)

        for label, (target, cumulative_merge_tolerance) in TARGET_SPECS.items():
            train_detector(
                train_script,
                config_path,
                seed=seed,
                label=label,
                target=target,
                cumulative_merge_tolerance=cumulative_merge_tolerance,
                output_dir=seed_run_root / target,
            )

        l6_dir = seed_run_root / TARGET_SPECS["L6"][0]
        l5_dir = seed_run_root / TARGET_SPECS["L5+"][0]
        l4_dir = seed_run_root / TARGET_SPECS["L4+"][0]

        l6_piece_df, l6_pred_df, feature_cols = load_detector_predictions(
            project_root,
            cfg,
            detector_target=TARGET_SPECS["L6"][0],
            checkpoint_dir=l6_dir,
            cumulative_merge_tolerance=TARGET_SPECS["L6"][1],
        )
        _, l5_pred_df, _ = load_detector_predictions(
            project_root,
            cfg,
            detector_target=TARGET_SPECS["L5+"][0],
            checkpoint_dir=l5_dir,
            cumulative_merge_tolerance=TARGET_SPECS["L5+"][1],
        )
        _, l4_pred_df, _ = load_detector_predictions(
            project_root,
            cfg,
            detector_target=TARGET_SPECS["L4+"][0],
            checkpoint_dir=l4_dir,
            cumulative_merge_tolerance=TARGET_SPECS["L4+"][1],
        )

        l5_threshold = load_threshold(l5_dir / "summary.json")
        l4_threshold = load_threshold(l4_dir / "summary.json")

        l6_scores = sequence_lookup(l6_pred_df)
        l5_scores = sequence_lookup(l5_pred_df)
        l4_scores = sequence_lookup(l4_pred_df)
        l5_mask = event_mask_from_scores(l5_pred_df, l5_threshold, radius=CANDIDATE_RADIUS, min_distance=min_distance, prominence=prominence)
        l4_mask = event_mask_from_scores(l4_pred_df, l4_threshold, radius=CANDIDATE_RADIUS, min_distance=min_distance, prominence=prominence)

        candidate_df = assemble_candidate_frame(
            l6_piece_df,
            l6_scores=l6_scores,
            l5_scores=l5_scores,
            l4_scores=l4_scores,
            l5_mask=l5_mask,
            l4_mask=l4_mask,
        )
        candidate_df.to_csv(seed_report_root / "candidate_frame.csv.gz", index=False, compression="gzip")

        val_candidate_df = candidate_df[candidate_df["protocol_split"] == "val"].copy()
        coverage_matches, coverage_true = candidate_coverage(val_candidate_df, l6_piece_df)

        train_df = candidate_df[candidate_df["protocol_split"] == "train"].copy()
        val_df = candidate_df[candidate_df["protocol_split"] == "val"].copy()
        if train_df.empty or val_df.empty:
            raise RuntimeError(f"Empty train/val candidate set for seed {seed}")

        model_features = list(feature_cols) + [
            "l6_base_score",
            "l5_score",
            "l4_score",
            "candidate_from_l5",
            "candidate_from_l4",
        ]
        reranker = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=4000,
                        class_weight="balanced",
                        random_state=seed,
                    ),
                ),
            ]
        )
        x_train = train_df[model_features].to_numpy(dtype=np.float32)
        y_train = train_df["rerank_train_label"].to_numpy(dtype=np.int64)
        sample_weight = 1.0 + train_df["frequency_target"].to_numpy(dtype=np.float32) * 4.0
        reranker.fit(x_train, y_train, clf__sample_weight=sample_weight)

        val_df = val_df.copy()
        val_df["rerank_score"] = reranker.predict_proba(val_df[model_features].to_numpy(dtype=np.float32))[:, 1].astype(np.float32)
        val_df.to_csv(seed_report_root / "val_candidates.csv.gz", index=False, compression="gzip")

        sequence_scores = build_full_sequence_scores(l6_piece_df, val_df)
        sequence_union = {}
        sequence_frequency = {}
        ordered_truth = l6_piece_df[l6_piece_df["protocol_split"] == "val"].sort_values(["piece_sample_id", "beat_idx"]).copy()
        for sample_id, group in ordered_truth.groupby("piece_sample_id", sort=False):
            sequence_union[str(sample_id)] = group["union_target"].to_numpy(dtype=np.float32)
            sequence_frequency[str(sample_id)] = group["frequency_target"].to_numpy(dtype=np.float32)

        rerank_metrics = search_union_frequency_threshold(
            sequence_scores=sequence_scores,
            sequence_union_labels=sequence_union,
            sequence_frequency_targets=sequence_frequency,
            thresholds=thresholds,
            tolerance=tolerance,
            min_distance=min_distance,
            min_precision=float(MIN_UNION_PRECISION),
            consensus_threshold=consensus_threshold,
            prominence=prominence,
            primary_metric="weighted_recall",
            precision_metric="union_precision",
            min_union_precision=float(MIN_UNION_PRECISION),
        )
        direct_summary = json.loads((l6_dir / "summary.json").read_text(encoding="utf-8"))["union_metrics"]

        seed_rows = [
            {
                "seed": seed,
                "method": "direct_l6",
                "candidate_mode": "full_sequence",
                "candidate_matches": 0,
                "candidate_true_events": 0,
                "candidate_coverage": 0.0,
                "best_epoch": json.loads((l6_dir / "summary.json").read_text(encoding="utf-8")).get("best_epoch"),
                "threshold": float(direct_summary["threshold"]),
                "union_precision": float(direct_summary["union_precision"]),
                "frequency_weighted_precision": float(direct_summary["frequency_weighted_precision"]),
                "consensus_precision": float(direct_summary["consensus_precision"]),
                "union_recall": float(direct_summary["union_recall"]),
                "weighted_recall": float(direct_summary["weighted_recall"]),
                "consensus_recall": float(direct_summary["consensus_recall"]),
                "pred_events": int(direct_summary["pred_events"]),
                "matches": int(direct_summary["matches"]),
            },
            {
                "seed": seed,
                "method": "rerank_l5_l4_to_l6",
                "candidate_mode": "l5_l4",
                "candidate_matches": int(coverage_matches),
                "candidate_true_events": int(coverage_true),
                "candidate_coverage": float(coverage_matches / coverage_true) if coverage_true > 0 else 0.0,
                "best_epoch": None,
                "threshold": float(rerank_metrics.threshold),
                "union_precision": float(rerank_metrics.union_precision),
                "frequency_weighted_precision": float(rerank_metrics.frequency_weighted_precision),
                "consensus_precision": float(rerank_metrics.consensus_precision),
                "union_recall": float(rerank_metrics.union_recall),
                "weighted_recall": float(rerank_metrics.weighted_recall),
                "consensus_recall": float(rerank_metrics.consensus_recall),
                "pred_events": int(rerank_metrics.pred_events),
                "matches": int(rerank_metrics.matches),
            },
        ]
        pd.DataFrame(seed_rows).to_csv(seed_report_root / "leaderboard.csv", index=False)
        rows.extend(seed_rows)

    detail_path = report_root / "all_seed_results.csv"
    with detail_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    avg = (
        pd.DataFrame(rows)
        .groupby("method", as_index=False)
        .agg(
            mean_union_precision=("union_precision", "mean"),
            std_union_precision=("union_precision", "std"),
            mean_frequency_weighted_precision=("frequency_weighted_precision", "mean"),
            mean_weighted_recall=("weighted_recall", "mean"),
            std_weighted_recall=("weighted_recall", "std"),
            mean_consensus_recall=("consensus_recall", "mean"),
            mean_candidate_coverage=("candidate_coverage", "mean"),
        )
    )
    avg.to_csv(report_root / "summary_mean_std.csv", index=False)
    print(detail_path)
    print(report_root / "summary_mean_std.csv")
    print(avg.to_csv(index=False))


if __name__ == "__main__":
    main()
