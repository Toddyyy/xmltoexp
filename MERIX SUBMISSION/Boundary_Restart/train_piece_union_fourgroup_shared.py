#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from boundary_restart.config import load_config, resolve_path, threshold_grid
from boundary_restart.features import PeakConfig, boundary_probs_to_binary, load_boundary_npz, replace_level_suffix
from boundary_restart.metrics import search_union_frequency_threshold
from boundary_restart.models import TemporalBlock
from boundary_restart.table_io import feature_columns, load_table


GROUP_LEVELS = {
    "L1": (1,),
    "L2": (2,),
    "L34": (3, 4),
    "L56": (5, 6),
}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def select_feature_columns(cfg: dict, columns: list[str]) -> list[str]:
    feature_cfg = cfg.get("features", {})
    include = feature_cfg.get("include")
    exclude = set(feature_cfg.get("exclude", []))
    selected = list(columns)
    if include:
        include_set = set(include)
        selected = [col for col in selected if col in include_set]
    if exclude:
        selected = [col for col in selected if col not in exclude]
    return [col for col in selected if col != "protocol_split"]


def apply_piece_protocol_split(
    df: pd.DataFrame,
    heldout_pieces: list[str],
    train_pieces: list[str] | None = None,
) -> pd.DataFrame:
    frame = df.copy()
    heldout_set = set(heldout_pieces)
    all_pieces = set(frame["piece_id"].unique().tolist())
    missing = sorted((heldout_set | set(train_pieces or [])) - all_pieces)
    if missing:
        raise ValueError(f"Unknown pieces in protocol split: {missing}")
    train_set = set(train_pieces) if train_pieces else all_pieces - heldout_set
    if heldout_set & train_set:
        raise ValueError("heldout_pieces and train_pieces must be disjoint")
    frame["protocol_split"] = "unused"
    frame.loc[frame["piece_id"].isin(train_set), "protocol_split"] = "train"
    frame.loc[frame["piece_id"].isin(heldout_set), "protocol_split"] = "val"
    return frame


def build_fourgroup_piece_frame(
    df: pd.DataFrame,
    feature_cols: list[str],
    peak_cfg: PeakConfig,
    beat_unit_fallback: float,
) -> pd.DataFrame:
    frame = df.copy()
    beat_idx = frame["beat_idx"].to_numpy(dtype=np.int32)
    freq_cols = {}
    for group_name, raw_levels in GROUP_LEVELS.items():
        detector_binary = np.zeros(len(frame), dtype=np.float32)
        for source_path, positions in frame.groupby("source_path", sort=False).indices.items():
            pos = np.asarray(positions, dtype=np.int64)
            union_binary = None
            for raw_level in raw_levels:
                level_path = replace_level_suffix(Path(str(source_path)), level=raw_level)
                loaded = load_boundary_npz(level_path, beat_unit_fallback=beat_unit_fallback)
                binary = boundary_probs_to_binary(
                    np.asarray(loaded["boundary_probs"], dtype=np.float32),
                    peak_cfg,
                ).astype(np.float32)
                union_binary = binary if union_binary is None else np.maximum(union_binary, binary)
            sample_beat_idx = beat_idx[pos]
            detector_binary[pos] = union_binary[sample_beat_idx]
        freq_col = f"{group_name}_binary"
        frame[freq_col] = detector_binary
        freq_cols[group_name] = freq_col

    agg_spec: dict[str, str] = {
        "protocol_split": "first",
        "num_beats": "first",
        "sample_id": pd.Series.nunique,
    }
    for freq_col in freq_cols.values():
        agg_spec[freq_col] = "mean"
    for col in feature_cols:
        agg_spec[col] = "first"

    piece = (
        frame.sort_values(["piece_id", "beat_idx", "sample_id"])
        .groupby(["piece_id", "beat_idx"], sort=False)
        .agg(agg_spec)
        .rename(columns={"sample_id": "performer_count"})
        .reset_index()
    )
    for group_name, freq_col in freq_cols.items():
        piece[f"{group_name}_frequency"] = piece.pop(freq_col).astype(np.float32)
        piece[f"{group_name}_union"] = (piece[f"{group_name}_frequency"] > 0.0).astype(np.float32)
    piece["piece_sample_id"] = piece["piece_id"]
    return piece


class FourGroupDataset(Dataset):
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


def collate_fourgroup(batch: list[dict]) -> dict:
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


def compute_normalizer(samples: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    stacked = np.concatenate([sample["features"] for sample in samples], axis=0)
    mean = stacked.mean(axis=0).astype(np.float32)
    std = stacked.std(axis=0).astype(np.float32)
    std[std < 1e-6] = 1.0
    return mean, std


def piece_samples_from_frame(df: pd.DataFrame, feature_cols: list[str], split: str) -> list[dict]:
    subset = df[df["protocol_split"] == split].copy().sort_values(["piece_sample_id", "beat_idx"])
    target_cols = [f"{group}_frequency" for group in GROUP_LEVELS]
    union_cols = [f"{group}_union" for group in GROUP_LEVELS]
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


class FourHeadTCN(nn.Module):
    def __init__(self, input_dim: int, channels: list[int], kernel_size: int = 3, dropout: float = 0.2):
        super().__init__()
        blocks = []
        in_channels = input_dim
        for block_idx, out_channels in enumerate(channels):
            blocks.append(
                TemporalBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    dilation=2 ** block_idx,
                    dropout=dropout,
                )
            )
            in_channels = out_channels
        self.encoder = nn.Sequential(*blocks)
        self.head = nn.Conv1d(in_channels, len(GROUP_LEVELS), 1)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        del lengths
        x = x.transpose(1, 2)
        h = self.encoder(x)
        return self.head(h).transpose(1, 2)


def build_model(input_dim: int, cfg: dict) -> nn.Module:
    seq_cfg = cfg.get("sequence", {})
    channels = [int(v) for v in seq_cfg.get("tcn_channels", [64, 64, 64])]
    return FourHeadTCN(
        input_dim=input_dim,
        channels=channels,
        kernel_size=int(seq_cfg.get("kernel_size", 3)),
        dropout=float(seq_cfg.get("dropout", 0.2)),
    )


def train_one_epoch(
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

        optimizer.zero_grad()
        logits = model(features)
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
def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> pd.DataFrame:
    model.eval()
    rows = []
    for batch in loader:
        features = batch["features"].to(device)
        targets = batch["targets"].to(device)
        unions = batch["unions"].to(device)
        beat_idx = batch["beat_idx"].to(device)
        mask = batch["mask"].to(device)

        probs = torch.sigmoid(model(features))
        for batch_idx_i, sample_id in enumerate(batch["sample_ids"]):
            length = int(mask[batch_idx_i].sum().item())
            for pos in range(length):
                row = {
                    "sample_id": sample_id,
                    "piece_id": batch["piece_ids"][batch_idx_i],
                    "beat_idx": int(beat_idx[batch_idx_i, pos].item()),
                }
                for head_idx, group_name in enumerate(GROUP_LEVELS):
                    row[f"{group_name}_frequency"] = float(targets[batch_idx_i, pos, head_idx].item())
                    row[f"{group_name}_union"] = float(unions[batch_idx_i, pos, head_idx].item())
                    row[f"{group_name}_score"] = float(probs[batch_idx_i, pos, head_idx].item())
                rows.append(row)
    return pd.DataFrame(rows)


def head_sequence_maps(pred_df: pd.DataFrame, group_name: str) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    sequence_scores = {}
    sequence_union = {}
    sequence_frequency = {}
    ordered = pred_df.sort_values(["sample_id", "beat_idx"])
    for sample_id, group in ordered.groupby("sample_id", sort=False):
        sequence_scores[sample_id] = group[f"{group_name}_score"].to_numpy(dtype=np.float32)
        sequence_union[sample_id] = group[f"{group_name}_union"].to_numpy(dtype=np.float32)
        sequence_frequency[sample_id] = group[f"{group_name}_frequency"].to_numpy(dtype=np.float32)
    return sequence_scores, sequence_union, sequence_frequency


def union_metrics_to_dict(metrics) -> dict:
    return {
        "threshold": metrics.threshold,
        "union_precision": metrics.union_precision,
        "union_recall": metrics.union_recall,
        "union_f1": metrics.union_f1,
        "weighted_recall": metrics.weighted_recall,
        "consensus_recall": metrics.consensus_recall,
        "mean_offset": metrics.mean_offset,
        "matches": metrics.matches,
        "pred_events": metrics.pred_events,
        "true_union_events": metrics.true_union_events,
        "true_consensus_events": metrics.true_consensus_events,
        "matched_weight": metrics.matched_weight,
        "total_weight": metrics.total_weight,
    }


def primary_metric(metrics, selection_metric: str) -> float:
    if selection_metric == "union_recall":
        return metrics.union_recall
    if selection_metric == "consensus_recall":
        return metrics.consensus_recall
    return metrics.weighted_recall


def aggregate_epoch_metrics(head_metrics: dict[str, object], min_precision: float, selection_metric: str) -> tuple:
    values = []
    for group_name in GROUP_LEVELS:
        metrics = head_metrics[group_name]
        floor_met = metrics.union_precision >= min_precision
        score = primary_metric(metrics, selection_metric) if floor_met else metrics.union_precision
        values.append(
            (
                float(floor_met),
                score,
                metrics.union_precision,
                metrics.weighted_recall,
                metrics.consensus_recall,
                metrics.union_f1,
                -float(metrics.mean_offset or 1e9),
            )
        )
    values_arr = np.asarray(values, dtype=np.float32)
    return tuple(values_arr.mean(axis=0).tolist())


def main():
    parser = argparse.ArgumentParser(description="Shared-encoder four-head direct TCN for L1/L2/L34/L56.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--heldout_piece", nargs="+", required=True)
    parser.add_argument("--train_pieces", nargs="*", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--early_stop_patience", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--min_precision", type=float, default=0.85)
    parser.add_argument(
        "--selection_metric",
        choices=["weighted_recall", "union_recall", "consensus_recall"],
        default="union_recall",
    )
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
        out_root = resolve_path(cfg, f"../outputs/piece_union_protocol/{heldout_slug}/tcn_fourgroup_shared_{args.selection_metric}_p{int(round(args.min_precision * 100))}")
    out_root.mkdir(parents=True, exist_ok=True)

    df = load_table(table_path)
    df = apply_piece_protocol_split(df, heldout_pieces=args.heldout_piece, train_pieces=args.train_pieces)
    df = df[df["protocol_split"].isin(["train", "val"])].copy()
    feature_cols = select_feature_columns(cfg, feature_columns(df))
    peak_cfg = PeakConfig(
        distance=int(data_cfg.get("peak_distance", 6)),
        height=float(data_cfg.get("peak_height", 0.15)),
        prominence=float(data_cfg.get("peak_prominence", 0.05)),
    )
    piece_df = build_fourgroup_piece_frame(
        df=df,
        feature_cols=feature_cols,
        peak_cfg=peak_cfg,
        beat_unit_fallback=float(data_cfg.get("beat_unit_fallback", 1.0)),
    )

    train_samples = piece_samples_from_frame(piece_df, feature_cols, split="train")
    val_samples = piece_samples_from_frame(piece_df, feature_cols, split="val")
    if not train_samples or not val_samples:
        raise ValueError("Protocol split produced an empty train or val set")

    mean, std = compute_normalizer(train_samples)
    train_ds = FourGroupDataset(train_samples, mean=mean, std=std)
    val_ds = FourGroupDataset(val_samples, mean=mean, std=std)
    batch_size = int(args.batch_size or seq_cfg.get("batch_size", 64))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fourgroup)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fourgroup)

    model = build_model(input_dim=len(feature_cols), cfg=cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(seq_cfg.get("lr", 1e-3)),
        weight_decay=float(seq_cfg.get("weight_decay", 1e-4)),
    )

    train_targets = np.concatenate([sample["targets"] for sample in train_samples], axis=0)
    pos = train_targets.sum(axis=0)
    neg = train_targets.shape[0] - pos
    pos_weight = torch.tensor(neg / np.clip(pos, 1.0, None), device=device, dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")

    thresholds = threshold_grid(cfg)
    tolerance = int(eval_cfg.get("event_tolerance", 1))
    min_distance = int(eval_cfg.get("min_distance", 6))
    prominence = float(eval_cfg.get("prominence", 0.0))
    consensus_threshold = float(eval_cfg.get("consensus_threshold", 0.5))
    epochs = int(args.epochs or seq_cfg.get("epochs", 30))
    patience = int(args.early_stop_patience if args.early_stop_patience is not None else seq_cfg.get("early_stop_patience", 5))
    grad_clip = float(seq_cfg.get("grad_clip", 1.0))

    best_epoch = 0
    best_key = None
    best_head_metrics = None
    best_val_pred = None
    history = []
    bad_epochs = 0

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            loss_fn=loss_fn,
            grad_clip=grad_clip,
        )
        val_pred = predict(model, val_loader, device=device)
        epoch_head_metrics = {}
        for group_name in GROUP_LEVELS:
            sequence_scores, sequence_union, sequence_frequency = head_sequence_maps(val_pred, group_name)
            epoch_head_metrics[group_name] = search_union_frequency_threshold(
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
        current_key = aggregate_epoch_metrics(epoch_head_metrics, float(args.min_precision), args.selection_metric)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "head_metrics": {group: union_metrics_to_dict(epoch_head_metrics[group]) for group in GROUP_LEVELS},
                "mean_precision": float(np.mean([epoch_head_metrics[g].union_precision for g in GROUP_LEVELS])),
                "mean_weighted_recall": float(np.mean([epoch_head_metrics[g].weighted_recall for g in GROUP_LEVELS])),
            }
        )
        print(
            f"Epoch {epoch}/{epochs} | train_loss {train_loss:.4f} | "
            f"mean_precision {history[-1]['mean_precision']:.4f} | "
            f"mean_weighted_recall {history[-1]['mean_weighted_recall']:.4f}"
        )
        if best_key is None or current_key > best_key:
            best_key = current_key
            best_epoch = epoch
            best_head_metrics = epoch_head_metrics
            best_val_pred = val_pred.copy()
            bad_epochs = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "feature_columns": feature_cols,
                    "mean": mean,
                    "std": std,
                    "best_epoch": best_epoch,
                    "selection_metric": args.selection_metric,
                    "min_precision": args.min_precision,
                    "head_thresholds": {group: float(epoch_head_metrics[group].threshold) for group in GROUP_LEVELS},
                },
                out_root / "detector_best.pt",
            )
        else:
            bad_epochs += 1
            if patience > 0 and bad_epochs >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    if best_head_metrics is None or best_val_pred is None:
        raise RuntimeError("Training did not produce validation metrics")

    best_val_pred.to_csv(out_root / "val_predictions.csv.gz", index=False, compression="gzip")
    np.savez(out_root / "detector_scaler_stats.npz", mean=mean, std=std)

    summary = {
        "table_path": str(table_path),
        "heldout_pieces": list(args.heldout_piece),
        "train_piece_count": int(piece_df[piece_df["protocol_split"] == "train"]["piece_id"].nunique()),
        "val_piece_count": int(piece_df[piece_df["protocol_split"] == "val"]["piece_id"].nunique()),
        "train_sequence_count": int(piece_df[piece_df["protocol_split"] == "train"]["piece_sample_id"].nunique()),
        "val_sequence_count": int(piece_df[piece_df["protocol_split"] == "val"]["piece_sample_id"].nunique()),
        "model_type": "tcn_fourgroup_shared",
        "seed": seed,
        "device": str(device),
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "early_stop_patience": patience,
        "precision_floor": float(args.min_precision),
        "selection_metric": args.selection_metric,
        "head_metrics": {group: union_metrics_to_dict(best_head_metrics[group]) for group in GROUP_LEVELS},
        "feature_columns": feature_cols,
        "history": history,
    }
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Held-out {heldout_slug} | best_epoch={best_epoch}")
    for group_name in GROUP_LEVELS:
        metrics = best_head_metrics[group_name]
        print(
            f"  {group_name}: precision={metrics.union_precision:.4f} | "
            f"union_recall={metrics.union_recall:.4f} | "
            f"weighted_recall={metrics.weighted_recall:.4f} | "
            f"consensus_recall={metrics.consensus_recall:.4f}"
        )


if __name__ == "__main__":
    main()
