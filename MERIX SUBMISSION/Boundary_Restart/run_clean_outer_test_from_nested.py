#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from boundary_restart.config import load_config, resolve_path
from boundary_restart.derived_features import add_highlevel_derived_features
from boundary_restart.features import PeakConfig
from boundary_restart.metrics import search_union_frequency_threshold
from boundary_restart.models import build_sequence_model
from boundary_restart.table_io import feature_columns, load_table
from train_piece_union_protocol import (
    PieceUnionDataset,
    apply_piece_protocol_split,
    apply_rest_span_training_labels,
    build_predicted_event_frame,
    build_piece_union_frame,
    collate_piece_union,
    compute_normalizer,
    detector_sequence_maps,
    piece_samples_from_frame,
    predict_detector,
    resolve_device,
    select_feature_columns,
    set_seed,
    train_one_epoch,
    union_metrics_to_dict,
)


def read_nested_selection(nested_dir: Path) -> tuple[dict, float, int]:
    summary = json.loads((nested_dir / "summary.json").read_text(encoding="utf-8"))
    best_candidate = summary["best_candidate"]
    candidate_slug = str(best_candidate["candidate_slug"])

    inner_results = []
    with (nested_dir / "inner_fold_results.csv").open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row["candidate_slug"] == candidate_slug:
                inner_results.append(row)
    if not inner_results:
        raise ValueError(f"No inner fold rows found for candidate {candidate_slug}")

    thresholds = [float(row["threshold"]) for row in inner_results]

    epoch_values = []
    inner_root = nested_dir / "inner_cv" / candidate_slug
    for fold_dir in sorted(inner_root.glob("fold_*")):
        summary_path = fold_dir / "summary.json"
        if not summary_path.exists():
            continue
        fold_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        epoch_values.append(int(fold_summary["best_epoch"]))
    if not epoch_values:
        raise ValueError(f"No inner fold summary best_epoch found under {inner_root}")

    fixed_threshold = float(sum(thresholds) / len(thresholds))
    fixed_epochs = int(round(sum(epoch_values) / len(epoch_values)))
    fixed_epochs = max(fixed_epochs, 1)
    return best_candidate["candidate"], fixed_threshold, fixed_epochs


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean outer test using frozen params from nested CV.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--nested_report_dir", required=True)
    parser.add_argument("--outer_heldout_piece", nargs="+", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--fixed_threshold", type=float, default=None)
    parser.add_argument("--fixed_epochs", type=int, default=None)
    parser.add_argument("--add_derived_highlevel_features", action="store_true")
    parser.add_argument("--derived_feature_include", nargs="*", default=None)
    args = parser.parse_args()

    nested_dir = Path(args.nested_report_dir).resolve()
    out_root = Path(args.output_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    cfg = load_config(args.config)
    seq_cfg = cfg.get("sequence", {})
    eval_cfg = cfg.get("evaluation", {})
    data_cfg = cfg.get("data", {})

    candidate, inferred_threshold, inferred_epochs = read_nested_selection(nested_dir)
    fixed_threshold = float(args.fixed_threshold if args.fixed_threshold is not None else inferred_threshold)
    fixed_epochs = int(args.fixed_epochs if args.fixed_epochs is not None else inferred_epochs)

    seed = int(candidate.get("seed", seq_cfg.get("seed", 42)))
    set_seed(seed)
    device = resolve_device(args.device if args.device != "auto" else str(candidate.get("device", "auto")))

    table_path = resolve_path(cfg, data_cfg["beat_table_path"])
    df = load_table(table_path)
    original_columns = set(df.columns)
    if args.add_derived_highlevel_features:
        df = add_highlevel_derived_features(df)

    heldout = sorted(set(args.outer_heldout_piece))
    df = apply_piece_protocol_split(df, heldout_pieces=heldout)
    df = df[df["protocol_split"].isin(["train", "val"])].copy()

    all_feature_cols = feature_columns(df)
    feature_cols = select_feature_columns(cfg, all_feature_cols)
    if args.add_derived_highlevel_features and args.derived_feature_include:
        allowed = set(args.derived_feature_include)
        derived_cols = [col for col in feature_cols if col not in original_columns]
        feature_cols = [col for col in feature_cols if col not in derived_cols or col in allowed]

    peak_cfg = PeakConfig(
        distance=int(data_cfg.get("peak_distance", 6)),
        height=float(data_cfg.get("peak_height", 0.15)),
        prominence=float(data_cfg.get("peak_prominence", 0.05)),
    )
    beat_unit_fallback = float(data_cfg.get("beat_unit_fallback", 1.0))
    piece_df = build_piece_union_frame(
        df,
        feature_cols=feature_cols,
        target_mode=str(candidate["detector_target"]),
        peak_cfg=peak_cfg,
        beat_unit_fallback=beat_unit_fallback,
    )
    piece_df = apply_rest_span_training_labels(
        piece_df,
        min_train_frequency_target=float(candidate.get("min_train_frequency_target", 0.0)),
        mode=str(candidate.get("rest_span_label_mode", "none")),
        min_len=int(candidate.get("rest_span_min_len", 2)),
        source_col=str(candidate.get("rest_span_source_col", "xml_rest_duration_norm")),
        source_threshold=float(candidate.get("rest_span_source_threshold", 1e-8)),
        tolerance_negative_weight=float(candidate.get("rest_span_tolerance_negative_weight", 1.0)),
    )

    train_samples = piece_samples_from_frame(piece_df, feature_cols, split="train")
    val_samples = piece_samples_from_frame(piece_df, feature_cols, split="val")
    if not train_samples or not val_samples:
        raise ValueError("Protocol split produced an empty train or val set")

    mean, std = compute_normalizer(train_samples)
    train_ds = PieceUnionDataset(
        train_samples,
        mean=mean,
        std=std,
        hard_negative_radius=int(candidate.get("hard_negative_radius", 0)),
        hard_negative_weight=float(candidate.get("hard_negative_weight", 1.0)),
        easy_negative_weight=float(candidate.get("easy_negative_weight", 1.0)),
    )
    val_ds = PieceUnionDataset(val_samples, mean=mean, std=std)

    batch_size = int(args.batch_size or candidate.get("batch_size") or seq_cfg.get("batch_size", 64))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_piece_union)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_piece_union)

    model = build_sequence_model(str(candidate["model"]), input_dim=len(feature_cols), cfg=cfg, output_dim=1).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(seq_cfg.get("lr", 1e-3)),
        weight_decay=float(seq_cfg.get("weight_decay", 1e-4)),
    )

    train_labels = np.concatenate([sample["train_frequency_target"] for sample in train_samples], axis=0)
    loss_type = str(candidate.get("loss_type", "bce"))
    if loss_type in {"bce", "bce_freq_weighted"}:
        pos = float(train_labels.sum())
        neg = float(train_labels.shape[0] - pos)
        pos_weight = torch.tensor([neg / max(pos, 1.0)], device=device, dtype=torch.float32)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")
    elif loss_type == "huber":
        loss_fn = nn.SmoothL1Loss(reduction="none")
    elif loss_type == "mse":
        loss_fn = nn.MSELoss(reduction="none")
    else:
        raise ValueError(f"Unsupported loss_type: {loss_type}")

    grad_clip = float(seq_cfg.get("grad_clip", 1.0))
    history = []
    for epoch in range(1, fixed_epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            loss_fn=loss_fn,
            loss_type=loss_type,
            grad_clip=grad_clip,
            log_interval=0,
        )
        history.append({"epoch": epoch, "train_loss": float(train_loss)})
        print(f"Epoch {epoch}/{fixed_epochs} | train_loss {train_loss:.4f}")

    pred_df = predict_detector(model, val_loader, device=device)
    sequence_scores, sequence_union, sequence_frequency = detector_sequence_maps(pred_df)
    tolerance = int(eval_cfg.get("event_tolerance", 1))
    min_distance = int(eval_cfg.get("min_distance", 6))
    prominence = float(eval_cfg.get("prominence", 0.0))
    consensus_threshold = float(eval_cfg.get("consensus_threshold", 0.5))
    metrics = search_union_frequency_threshold(
        sequence_scores=sequence_scores,
        sequence_union_labels=sequence_union,
        sequence_frequency_targets=sequence_frequency,
        thresholds=np.asarray([fixed_threshold], dtype=np.float32),
        tolerance=tolerance,
        min_distance=min_distance,
        min_precision=float(candidate.get("min_precision", 0.0)),
        consensus_threshold=consensus_threshold,
        prominence=prominence,
    )

    pred_df.to_csv(out_root / "outer_predictions.csv.gz", index=False, compression="gzip")
    predicted_events = build_predicted_event_frame(
        pred_df=pred_df,
        threshold=float(fixed_threshold),
        min_distance=min_distance,
        prominence=prominence,
        tolerance=tolerance,
    )
    predicted_events.to_csv(out_root / "predicted_events.csv.gz", index=False, compression="gzip")

    summary = {
        "config": str(Path(args.config).resolve()),
        "nested_report_dir": str(nested_dir),
        "outer_heldout_pieces": heldout,
        "frozen_candidate": candidate,
        "frozen_threshold": float(fixed_threshold),
        "frozen_epochs": int(fixed_epochs),
        "seed": seed,
        "device": str(device),
        "train_piece_count": int(piece_df[piece_df["protocol_split"] == "train"]["piece_id"].nunique()),
        "outer_piece_count": int(piece_df[piece_df["protocol_split"] == "val"]["piece_id"].nunique()),
        "feature_columns": feature_cols,
        "history": history,
        "union_metrics": union_metrics_to_dict(metrics),
    }
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_root / "frozen_selection.json").write_text(
        json.dumps(
            {
                "candidate": candidate,
                "fixed_threshold": float(fixed_threshold),
                "fixed_epochs": int(fixed_epochs),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        f"Clean outer { '__'.join(heldout) } | union_precision={metrics.union_precision:.4f} | "
        f"union_recall={metrics.union_recall:.4f} | weighted_recall={metrics.weighted_recall:.4f} | "
        f"consensus_recall={metrics.consensus_recall:.4f} | threshold={fixed_threshold:.3f} | epochs={fixed_epochs}"
    )


if __name__ == "__main__":
    main()
