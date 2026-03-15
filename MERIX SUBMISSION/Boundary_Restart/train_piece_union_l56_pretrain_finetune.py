#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from boundary_restart.config import load_config, resolve_path, threshold_grid
from boundary_restart.features import PeakConfig
from boundary_restart.models import build_sequence_model
from boundary_restart.table_io import feature_columns, load_table
from train_piece_union_fourgroup_shared import build_fourgroup_piece_frame
from train_piece_union_protocol import (
    PieceUnionDataset,
    apply_piece_protocol_split,
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
from boundary_restart.metrics import search_union_frequency_threshold


LOWER_GROUPS = ("L1", "L2", "L34")


class LowerGroupDataset(Dataset):
    def __init__(self, samples: list[dict], mean: np.ndarray, std: np.ndarray):
        self.samples = samples
        self.mean = mean.astype(np.float32)
        self.std = std.astype(np.float32)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        features = (sample["features"] - self.mean) / self.std
        return {
            "sample_id": sample["sample_id"],
            "piece_id": sample["piece_id"],
            "beat_idx": sample["beat_idx"].astype(np.int32),
            "features": features.astype(np.float32),
            "targets": sample["targets"].astype(np.float32),
            "unions": sample["unions"].astype(np.float32),
            "length": int(sample["targets"].shape[0]),
        }


def collate_lower_groups(batch: list[dict]) -> dict:
    lengths = [item["length"] for item in batch]
    max_len = max(lengths)
    feat_dim = batch[0]["features"].shape[1]
    num_heads = batch[0]["targets"].shape[1]
    features = torch.zeros(len(batch), max_len, feat_dim, dtype=torch.float32)
    targets = torch.zeros(len(batch), max_len, num_heads, dtype=torch.float32)
    unions = torch.zeros(len(batch), max_len, num_heads, dtype=torch.float32)
    beat_idx = torch.zeros(len(batch), max_len, dtype=torch.int64)
    mask = torch.zeros(len(batch), max_len, dtype=torch.bool)
    sample_ids = []
    piece_ids = []
    for idx, item in enumerate(batch):
        length = item["length"]
        features[idx, :length] = torch.from_numpy(item["features"])
        targets[idx, :length] = torch.from_numpy(item["targets"])
        unions[idx, :length] = torch.from_numpy(item["unions"])
        beat_idx[idx, :length] = torch.from_numpy(item["beat_idx"])
        mask[idx, :length] = True
        sample_ids.append(item["sample_id"])
        piece_ids.append(item["piece_id"])
    return {
        "features": features,
        "targets": targets,
        "unions": unions,
        "beat_idx": beat_idx,
        "mask": mask,
        "lengths": torch.tensor(lengths, dtype=torch.int64),
        "sample_ids": sample_ids,
        "piece_ids": piece_ids,
    }


def lower_samples_from_frame(df: pd.DataFrame, feature_cols: list[str], split: str) -> list[dict]:
    subset = df[df["protocol_split"] == split].copy().sort_values(["piece_sample_id", "beat_idx"])
    target_cols = [f"{group}_frequency" for group in LOWER_GROUPS]
    union_cols = [f"{group}_union" for group in LOWER_GROUPS]
    samples = []
    for sample_id, group in subset.groupby("piece_sample_id", sort=False):
        samples.append(
            {
                "sample_id": sample_id,
                "piece_id": group["piece_id"].iloc[0],
                "beat_idx": group["beat_idx"].to_numpy(dtype=np.int32),
                "features": group[feature_cols].to_numpy(dtype=np.float32),
                "targets": group[target_cols].to_numpy(dtype=np.float32),
                "unions": group[union_cols].to_numpy(dtype=np.float32),
            }
        )
    return samples


def train_lower_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    loss_fn: nn.Module,
    grad_clip: float,
) -> float:
    model.train()
    total_loss = 0.0
    total_tokens = 0
    for batch in loader:
        features = batch["features"].to(device)
        targets = batch["targets"].to(device)
        mask = batch["mask"].to(device)
        lengths = batch["lengths"].to(device)

        optimizer.zero_grad()
        logits = model(features, lengths=lengths)
        per_token_head = loss_fn(logits, targets)
        weights = mask.float().unsqueeze(-1).expand_as(per_token_head)
        loss = (per_token_head * weights).sum() / weights.sum().clamp(min=1.0)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += float(loss.item()) * int(mask.sum().item())
        total_tokens += int(mask.sum().item())
    return total_loss / max(total_tokens, 1)


@torch.no_grad()
def predict_lower(model: nn.Module, loader: DataLoader, device: torch.device) -> pd.DataFrame:
    model.eval()
    rows = []
    for batch in loader:
        features = batch["features"].to(device)
        targets = batch["targets"].to(device)
        unions = batch["unions"].to(device)
        beat_idx = batch["beat_idx"].to(device)
        mask = batch["mask"].to(device)
        lengths = batch["lengths"].to(device)
        probs = torch.sigmoid(model(features, lengths=lengths))
        for batch_idx_i, sample_id in enumerate(batch["sample_ids"]):
            length = int(mask[batch_idx_i].sum().item())
            for pos in range(length):
                row = {
                    "sample_id": sample_id,
                    "piece_id": batch["piece_ids"][batch_idx_i],
                    "beat_idx": int(beat_idx[batch_idx_i, pos].item()),
                }
                for head_idx, group_name in enumerate(LOWER_GROUPS):
                    row[f"{group_name}_frequency"] = float(targets[batch_idx_i, pos, head_idx].item())
                    row[f"{group_name}_union"] = float(unions[batch_idx_i, pos, head_idx].item())
                    row[f"{group_name}_score"] = float(probs[batch_idx_i, pos, head_idx].item())
                rows.append(row)
    return pd.DataFrame(rows)


def lower_sequence_maps(pred_df: pd.DataFrame, group_name: str) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    sequence_scores = {}
    sequence_union = {}
    sequence_frequency = {}
    ordered = pred_df.sort_values(["sample_id", "beat_idx"])
    for sample_id, group in ordered.groupby("sample_id", sort=False):
        sequence_scores[sample_id] = group[f"{group_name}_score"].to_numpy(dtype=np.float32)
        sequence_union[sample_id] = group[f"{group_name}_union"].to_numpy(dtype=np.float32)
        sequence_frequency[sample_id] = group[f"{group_name}_frequency"].to_numpy(dtype=np.float32)
    return sequence_scores, sequence_union, sequence_frequency


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--heldout_piece", nargs="+", required=True)
    parser.add_argument("--train_pieces", nargs="*", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--pretrain_epochs", type=int, default=20)
    parser.add_argument("--finetune_epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--early_stop_patience", type=int, default=5)
    parser.add_argument("--min_precision", type=float, default=0.85)
    parser.add_argument("--selection_metric", choices=["weighted_recall", "union_recall", "consensus_recall"], default="union_recall")
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    seq_cfg = cfg.get("sequence", {})
    eval_cfg = cfg.get("evaluation", {})
    data_cfg = cfg.get("data", {})
    seed = int(args.seed if args.seed is not None else seq_cfg.get("seed", 42))
    set_seed(seed)
    device = resolve_device(args.device)

    table_path = resolve_path(cfg, data_cfg["beat_table_path"])
    heldout_slug = "__".join(args.heldout_piece)
    if args.output_dir:
        out_root = Path(args.output_dir).resolve()
    else:
        out_root = resolve_path(
            cfg,
            f"../outputs/local_runs/{heldout_slug}_l56_pretrain_finetune",
        )
    out_root.mkdir(parents=True, exist_ok=True)

    df = load_table(table_path)
    df = apply_piece_protocol_split(df, heldout_pieces=args.heldout_piece, train_pieces=args.train_pieces)
    df = df[df["protocol_split"].isin(["train", "val"])].copy()
    all_feature_cols = feature_columns(df)
    feature_cols = select_feature_columns(cfg, all_feature_cols)

    peak_cfg = PeakConfig(
        distance=int(data_cfg.get("peak_distance", 6)),
        height=float(data_cfg.get("peak_height", 0.15)),
        prominence=float(data_cfg.get("peak_prominence", 0.05)),
    )
    beat_unit_fallback = float(data_cfg.get("beat_unit_fallback", 1.0))

    fourgroup_df = build_fourgroup_piece_frame(
        df=df,
        feature_cols=feature_cols,
        peak_cfg=peak_cfg,
        beat_unit_fallback=beat_unit_fallback,
    )
    lower_train_samples = lower_samples_from_frame(fourgroup_df, feature_cols, split="train")
    lower_val_samples = lower_samples_from_frame(fourgroup_df, feature_cols, split="val")
    mean, std = compute_normalizer(lower_train_samples)

    batch_size = int(args.batch_size or seq_cfg.get("batch_size", 64))
    lower_train_ds = LowerGroupDataset(lower_train_samples, mean=mean, std=std)
    lower_val_ds = LowerGroupDataset(lower_val_samples, mean=mean, std=std)
    lower_train_loader = DataLoader(lower_train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_lower_groups)
    lower_val_loader = DataLoader(lower_val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_lower_groups)

    lower_model = build_sequence_model("tcn", input_dim=len(feature_cols), cfg=cfg, output_dim=len(LOWER_GROUPS)).to(device)
    lower_optimizer = torch.optim.AdamW(
        lower_model.parameters(),
        lr=float(seq_cfg.get("lr", 1e-3)),
        weight_decay=float(seq_cfg.get("weight_decay", 1e-4)),
    )
    lower_loss_fn = nn.BCEWithLogitsLoss(reduction="none")
    thresholds = threshold_grid(cfg)
    tolerance = int(eval_cfg.get("event_tolerance", 1))
    min_distance = int(eval_cfg.get("min_distance", 6))
    prominence = float(eval_cfg.get("prominence", 0.0))
    consensus_threshold = float(eval_cfg.get("consensus_threshold", 0.5))
    grad_clip = float(seq_cfg.get("grad_clip", 1.0))

    best_lower_key = None
    best_encoder_state = None
    best_lower_metrics = None
    bad_epochs = 0
    pretrain_history = []
    for epoch in range(1, int(args.pretrain_epochs) + 1):
        train_loss = train_lower_one_epoch(
            model=lower_model,
            loader=lower_train_loader,
            optimizer=lower_optimizer,
            device=device,
            loss_fn=lower_loss_fn,
            grad_clip=grad_clip,
        )
        val_pred = predict_lower(lower_model, lower_val_loader, device=device)
        head_metrics = {}
        for group_name in LOWER_GROUPS:
            seq_scores, seq_union, seq_freq = lower_sequence_maps(val_pred, group_name)
            head_metrics[group_name] = search_union_frequency_threshold(
                sequence_scores=seq_scores,
                sequence_union_labels=seq_union,
                sequence_frequency_targets=seq_freq,
                thresholds=thresholds,
                tolerance=tolerance,
                min_distance=min_distance,
                min_precision=float(args.min_precision),
                consensus_threshold=consensus_threshold,
                prominence=prominence,
            )
        mean_precision = float(np.mean([head_metrics[g].union_precision for g in LOWER_GROUPS]))
        mean_weighted_recall = float(np.mean([head_metrics[g].weighted_recall for g in LOWER_GROUPS]))
        mean_union_recall = float(np.mean([head_metrics[g].union_recall for g in LOWER_GROUPS]))
        mean_consensus_recall = float(np.mean([head_metrics[g].consensus_recall for g in LOWER_GROUPS]))
        pretrain_history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "mean_precision": mean_precision,
                "mean_union_recall": mean_union_recall,
                "mean_weighted_recall": mean_weighted_recall,
                "mean_consensus_recall": mean_consensus_recall,
            }
        )
        print(
            f"Pretrain epoch {epoch}/{int(args.pretrain_epochs)} | train_loss {train_loss:.4f} | "
            f"mean_precision {mean_precision:.4f} | mean_union_recall {mean_union_recall:.4f} | "
            f"mean_weighted_recall {mean_weighted_recall:.4f}"
        )
        if args.selection_metric == "union_recall":
            primary_metric = mean_union_recall
        elif args.selection_metric == "consensus_recall":
            primary_metric = mean_consensus_recall
        else:
            primary_metric = mean_weighted_recall
        precision_floor_met = mean_precision >= float(args.min_precision)
        current_key = (
            float(precision_floor_met),
            primary_metric if precision_floor_met else mean_precision,
            mean_precision,
            mean_weighted_recall,
            mean_consensus_recall,
        )
        if best_lower_key is None or current_key > best_lower_key:
            best_lower_key = current_key
            best_encoder_state = {k: v.detach().cpu().clone() for k, v in lower_model.network.state_dict().items()}
            best_lower_metrics = {
                "mean_precision": mean_precision,
                "mean_union_recall": mean_union_recall,
                "mean_weighted_recall": mean_weighted_recall,
                "mean_consensus_recall": mean_consensus_recall,
                "head_metrics": {g: union_metrics_to_dict(head_metrics[g]) for g in LOWER_GROUPS},
                "epoch": epoch,
            }
            bad_epochs = 0
        else:
            bad_epochs += 1
            if int(args.early_stop_patience) > 0 and bad_epochs >= int(args.early_stop_patience):
                print(f"Lower pretraining early stopping at epoch {epoch}")
                break

    if best_encoder_state is None or best_lower_metrics is None:
        raise RuntimeError("Pretraining did not produce a best encoder")

    piece_df = build_piece_union_frame(
        df,
        feature_cols=feature_cols,
        target_mode="level56_boundary",
        peak_cfg=peak_cfg,
        beat_unit_fallback=beat_unit_fallback,
    )
    train_samples = piece_samples_from_frame(piece_df, feature_cols, split="train")
    val_samples = piece_samples_from_frame(piece_df, feature_cols, split="val")
    train_ds = PieceUnionDataset(train_samples, mean=mean, std=std)
    val_ds = PieceUnionDataset(val_samples, mean=mean, std=std)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_piece_union)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_piece_union)

    finetune_model = build_sequence_model("tcn", input_dim=len(feature_cols), cfg=cfg, output_dim=1).to(device)
    finetune_model.network.load_state_dict(best_encoder_state, strict=True)
    finetune_optimizer = torch.optim.AdamW(
        finetune_model.parameters(),
        lr=float(seq_cfg.get("lr", 1e-3)) * 0.5,
        weight_decay=float(seq_cfg.get("weight_decay", 1e-4)),
    )

    train_labels = np.concatenate([sample["frequency_target"] for sample in train_samples], axis=0)
    pos = float(train_labels.sum())
    neg = float(train_labels.shape[0] - pos)
    pos_weight = torch.tensor([neg / max(pos, 1.0)], device=device, dtype=torch.float32)
    finetune_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")

    best_key = None
    best_metrics = None
    best_val_pred = None
    finetune_history = []
    bad_epochs = 0
    best_epoch = 0
    for epoch in range(1, int(args.finetune_epochs) + 1):
        train_loss = train_one_epoch(
            model=finetune_model,
            loader=train_loader,
            optimizer=finetune_optimizer,
            device=device,
            loss_fn=finetune_loss_fn,
            loss_type="bce",
            grad_clip=grad_clip,
            log_interval=0,
        )
        val_pred = predict_detector(finetune_model, val_loader, device=device)
        sequence_scores, sequence_union, sequence_frequency = detector_sequence_maps(val_pred)
        metrics = search_union_frequency_threshold(
            sequence_scores=sequence_scores,
            sequence_union_labels=sequence_union,
            sequence_frequency_targets=sequence_frequency,
            thresholds=thresholds,
            tolerance=tolerance,
            min_distance=min_distance,
            min_precision=float(args.min_precision),
            consensus_threshold=consensus_threshold,
            prominence=prominence,
        )
        finetune_history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "union_precision": metrics.union_precision,
                "union_recall": metrics.union_recall,
                "weighted_recall": metrics.weighted_recall,
                "consensus_recall": metrics.consensus_recall,
                "threshold": metrics.threshold,
            }
        )
        print(
            f"Finetune epoch {epoch}/{int(args.finetune_epochs)} | train_loss {train_loss:.4f} | "
            f"union_precision {metrics.union_precision:.4f} | union_recall {metrics.union_recall:.4f} | "
            f"weighted_recall {metrics.weighted_recall:.4f}"
        )
        if args.selection_metric == "union_recall":
            primary_metric = metrics.union_recall
        elif args.selection_metric == "consensus_recall":
            primary_metric = metrics.consensus_recall
        else:
            primary_metric = metrics.weighted_recall
        precision_floor_met = metrics.union_precision >= float(args.min_precision)
        current_key = (
            float(precision_floor_met),
            primary_metric if precision_floor_met else metrics.union_precision,
            metrics.union_precision,
            metrics.weighted_recall,
            metrics.consensus_recall,
            metrics.union_f1,
            -float(metrics.mean_offset or 1e9),
        )
        if best_key is None or current_key > best_key:
            best_key = current_key
            best_metrics = metrics
            best_val_pred = val_pred.copy()
            best_epoch = epoch
            bad_epochs = 0
        else:
            bad_epochs += 1
            if int(args.early_stop_patience) > 0 and bad_epochs >= int(args.early_stop_patience):
                print(f"L56 finetune early stopping at epoch {epoch}")
                break

    if best_metrics is None or best_val_pred is None:
        raise RuntimeError("Finetuning did not produce validation metrics")

    best_val_pred.to_csv(out_root / "val_predictions.csv.gz", index=False, compression="gzip")
    torch.save(
        {
            "feature_columns": feature_cols,
            "mean": mean,
            "std": std,
            "pretrained_encoder_state_dict": best_encoder_state,
            "finetuned_model_state_dict": finetune_model.state_dict(),
            "best_epoch": best_epoch,
        },
        out_root / "detector_best.pt",
    )
    summary = {
        "table_path": str(table_path),
        "heldout_pieces": list(args.heldout_piece),
        "model_type": "tcn_l56_pretrain_finetune",
        "device": str(device),
        "seed": seed,
        "feature_columns": feature_cols,
        "selection_metric": args.selection_metric,
        "precision_floor": float(args.min_precision),
        "pretrain": {
            "best_lower_metrics": best_lower_metrics,
            "history": pretrain_history,
        },
        "finetune": {
            "best_epoch": best_epoch,
            "union_metrics": union_metrics_to_dict(best_metrics),
            "history": finetune_history,
        },
    }
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(
        f"Held-out {heldout_slug} | best_epoch={best_epoch} | "
        f"precision={best_metrics.union_precision:.4f} | "
        f"union_recall={best_metrics.union_recall:.4f} | "
        f"weighted_recall={best_metrics.weighted_recall:.4f} | "
        f"consensus_recall={best_metrics.consensus_recall:.4f}"
    )


if __name__ == "__main__":
    main()
