#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset

from boundary_restart.config import load_config, resolve_path, threshold_grid
from boundary_restart.metrics import evaluate_labeled_event_sequences, search_union_frequency_threshold
from boundary_restart.models import TemporalBlock
from boundary_restart.table_io import feature_columns, load_table


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
    if train_pieces:
        train_set = set(train_pieces)
    else:
        train_set = all_pieces - heldout_set
    if heldout_set & train_set:
        raise ValueError("heldout_pieces and train_pieces must be disjoint")
    frame["protocol_split"] = "unused"
    frame.loc[frame["piece_id"].isin(train_set), "protocol_split"] = "train"
    frame.loc[frame["piece_id"].isin(heldout_set), "protocol_split"] = "val"
    return frame


def build_loss_weights(
    union_labels: np.ndarray,
    hard_negative_radius: int,
    hard_negative_weight: float,
    easy_negative_weight: float,
) -> np.ndarray:
    union_labels = np.asarray(union_labels, dtype=np.float32)
    weights = np.ones_like(union_labels, dtype=np.float32)
    neg_mask = union_labels < 0.5
    if np.any(neg_mask):
        weights[neg_mask] = float(easy_negative_weight)
    if hard_negative_radius <= 0 or hard_negative_weight <= easy_negative_weight:
        return weights
    pos_idx = np.flatnonzero(union_labels > 0.5)
    if pos_idx.size == 0:
        return weights
    hard_mask = np.zeros_like(union_labels, dtype=bool)
    radius = int(hard_negative_radius)
    for center in pos_idx.tolist():
        start = max(0, center - radius)
        end = min(union_labels.shape[0], center + radius + 1)
        hard_mask[start:end] = True
    hard_mask &= neg_mask
    weights[hard_mask] = float(hard_negative_weight)
    return weights.astype(np.float32)


def build_piece_union_frame(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    frame = df.copy()
    frame["any_binary"] = (frame["stage_class"].to_numpy(dtype=np.int64) > 0).astype(np.float32)
    frame["low_binary"] = (frame["stage_class"].to_numpy(dtype=np.int64) == 1).astype(np.float32)
    frame["mid_binary"] = (frame["stage_class"].to_numpy(dtype=np.int64) == 2).astype(np.float32)
    frame["high_binary"] = (frame["stage_class"].to_numpy(dtype=np.int64) >= 3).astype(np.float32)

    agg_spec: dict[str, str] = {
        "protocol_split": "first",
        "num_beats": "first",
        "any_binary": "mean",
        "low_binary": "mean",
        "mid_binary": "mean",
        "high_binary": "mean",
        "sample_id": pd.Series.nunique,
    }
    for col in feature_cols:
        agg_spec[col] = "first"

    piece = (
        frame.sort_values(["piece_id", "beat_idx", "sample_id"])
        .groupby(["piece_id", "beat_idx"], sort=False)
        .agg(agg_spec)
        .rename(
            columns={
                "any_binary": "any_frequency",
                "low_binary": "low_frequency",
                "mid_binary": "mid_frequency",
                "high_binary": "high_frequency",
                "sample_id": "performer_count",
            }
        )
        .reset_index()
    )
    piece["union_target"] = (piece["any_frequency"] > 0.0).astype(np.float32)
    stage_freq = piece[["low_frequency", "mid_frequency", "high_frequency"]].to_numpy(dtype=np.float32)
    dominant_idx = np.argmax(stage_freq, axis=1) + 1
    dominant_idx = np.where(piece["union_target"].to_numpy(dtype=np.float32) > 0.0, dominant_idx, 0)
    piece["dominant_stage"] = dominant_idx.astype(np.int64)
    piece["piece_sample_id"] = piece["piece_id"]
    return piece


class MultiHeadBiLSTM(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        stage_output_dim: int = 3,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.encoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.detector_head = nn.Linear(hidden_dim * 2, 1)
        self.stage_head = nn.Linear(hidden_dim * 2, stage_output_dim)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.input_proj(x)
        if lengths is not None:
            packed = pack_padded_sequence(
                x,
                lengths.detach().to(device="cpu", dtype=torch.int64),
                batch_first=True,
                enforce_sorted=False,
            )
            packed_out, _ = self.encoder(packed)
            x, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=x.size(1))
        else:
            x, _ = self.encoder(x)
        x = self.dropout(x)
        detector_logits = self.detector_head(x).squeeze(-1)
        stage_logits = self.stage_head(x)
        return detector_logits, stage_logits


class MultiHeadTCN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        channels: list[int],
        kernel_size: int = 3,
        dropout: float = 0.2,
        stage_output_dim: int = 3,
    ):
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
        self.detector_head = nn.Conv1d(in_channels, 1, 1)
        self.stage_head = nn.Conv1d(in_channels, stage_output_dim, 1)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        del lengths
        x = x.transpose(1, 2)
        h = self.encoder(x)
        detector_logits = self.detector_head(h).squeeze(1)
        stage_logits = self.stage_head(h).transpose(1, 2)
        return detector_logits, stage_logits


def build_multitask_model(model_type: str, input_dim: int, cfg: dict) -> nn.Module:
    seq_cfg = cfg.get("sequence", {})
    model_type = model_type.lower()
    if model_type == "bilstm":
        return MultiHeadBiLSTM(
            input_dim=input_dim,
            hidden_dim=int(seq_cfg.get("hidden_dim", 64)),
            num_layers=int(seq_cfg.get("num_layers", 2)),
            dropout=float(seq_cfg.get("dropout", 0.2)),
        )
    if model_type == "tcn":
        channels = [int(v) for v in seq_cfg.get("tcn_channels", [64, 64, 64])]
        return MultiHeadTCN(
            input_dim=input_dim,
            channels=channels,
            kernel_size=int(seq_cfg.get("kernel_size", 3)),
            dropout=float(seq_cfg.get("dropout", 0.2)),
        )
    raise ValueError(f"Unsupported model_type: {model_type}")


class PieceUnionMultitaskDataset(Dataset):
    def __init__(
        self,
        samples: list[dict],
        mean: np.ndarray,
        std: np.ndarray,
        hard_negative_radius: int = 0,
        hard_negative_weight: float = 1.0,
        easy_negative_weight: float = 1.0,
    ):
        self.samples = samples
        self.mean = mean.astype(np.float32)
        self.std = std.astype(np.float32)
        self.hard_negative_radius = int(hard_negative_radius)
        self.hard_negative_weight = float(hard_negative_weight)
        self.easy_negative_weight = float(easy_negative_weight)

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
            "any_frequency": sample["any_frequency"].astype(np.float32),
            "union_target": sample["union_target"].astype(np.float32),
            "stage_frequency": sample["stage_frequency"].astype(np.float32),
            "dominant_stage": sample["dominant_stage"].astype(np.int64),
            "loss_weights": build_loss_weights(
                sample["union_target"],
                hard_negative_radius=self.hard_negative_radius,
                hard_negative_weight=self.hard_negative_weight,
                easy_negative_weight=self.easy_negative_weight,
            ),
            "length": int(sample["any_frequency"].shape[0]),
        }


def collate_multitask(batch: list[dict]) -> dict:
    lengths = [item["length"] for item in batch]
    max_len = max(lengths)
    feat_dim = batch[0]["features"].shape[1]
    features = torch.zeros(len(batch), max_len, feat_dim, dtype=torch.float32)
    any_frequency = torch.zeros(len(batch), max_len, dtype=torch.float32)
    union_target = torch.zeros(len(batch), max_len, dtype=torch.float32)
    stage_frequency = torch.zeros(len(batch), max_len, 3, dtype=torch.float32)
    dominant_stage = torch.zeros(len(batch), max_len, dtype=torch.int64)
    loss_weights = torch.ones(len(batch), max_len, dtype=torch.float32)
    beat_idx = torch.zeros(len(batch), max_len, dtype=torch.int64)
    mask = torch.zeros(len(batch), max_len, dtype=torch.bool)
    sample_ids = []
    piece_ids = []

    for idx, item in enumerate(batch):
        length = item["length"]
        features[idx, :length] = torch.from_numpy(item["features"])
        any_frequency[idx, :length] = torch.from_numpy(item["any_frequency"])
        union_target[idx, :length] = torch.from_numpy(item["union_target"])
        stage_frequency[idx, :length] = torch.from_numpy(item["stage_frequency"])
        dominant_stage[idx, :length] = torch.from_numpy(item["dominant_stage"])
        loss_weights[idx, :length] = torch.from_numpy(item["loss_weights"])
        beat_idx[idx, :length] = torch.from_numpy(item["beat_idx"])
        mask[idx, :length] = True
        sample_ids.append(item["sample_id"])
        piece_ids.append(item["piece_id"])

    return {
        "features": features,
        "any_frequency": any_frequency,
        "union_target": union_target,
        "stage_frequency": stage_frequency,
        "dominant_stage": dominant_stage,
        "loss_weights": loss_weights,
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
    samples = []
    for sample_id, group in subset.groupby("piece_sample_id", sort=False):
        samples.append(
            {
                "sample_id": sample_id,
                "piece_id": group["piece_id"].iloc[0],
                "beat_idx": group["beat_idx"].to_numpy(dtype=np.int32),
                "features": group[feature_cols].to_numpy(dtype=np.float32),
                "any_frequency": group["any_frequency"].to_numpy(dtype=np.float32),
                "union_target": group["union_target"].to_numpy(dtype=np.float32),
                "stage_frequency": group[["low_frequency", "mid_frequency", "high_frequency"]].to_numpy(dtype=np.float32),
                "dominant_stage": group["dominant_stage"].to_numpy(dtype=np.int64),
            }
        )
    return samples


def soft_cross_entropy_from_distribution(
    logits: torch.Tensor,
    stage_frequency: torch.Tensor,
    positive_mask: torch.Tensor,
) -> torch.Tensor:
    stage_sum = stage_frequency.sum(dim=-1, keepdim=True).clamp(min=1e-6)
    stage_dist = stage_frequency / stage_sum
    log_probs = torch.log_softmax(logits, dim=-1)
    per_token = -(stage_dist * log_probs).sum(dim=-1)
    valid = positive_mask.float()
    return (per_token * valid).sum() / valid.sum().clamp(min=1.0)


def compute_mass_loss(
    detector_logits: torch.Tensor,
    stage_logits: torch.Tensor,
    stage_frequency: torch.Tensor,
    positive_mask: torch.Tensor,
    class_weights: torch.Tensor,
) -> torch.Tensor:
    detector_prob = torch.sigmoid(detector_logits).unsqueeze(-1)
    stage_prob = torch.softmax(stage_logits, dim=-1)
    stage_mass_pred = detector_prob * stage_prob
    per_class = torch.nn.functional.smooth_l1_loss(stage_mass_pred, stage_frequency, reduction="none")
    per_class = per_class * class_weights.view(1, 1, -1)
    per_token = per_class.sum(dim=-1)
    valid = positive_mask.float()
    return (per_token * valid).sum() / valid.sum().clamp(min=1.0)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    detector_loss_fn: nn.Module,
    mass_class_weights: torch.Tensor,
    stage_soft_loss_weight: float,
    stage_mass_loss_weight: float,
    grad_clip: float,
    log_interval: int = 0,
) -> tuple[float, float, float, float]:
    model.train()
    total_loss = 0.0
    total_detector_loss = 0.0
    total_stage_soft_loss = 0.0
    total_stage_mass_loss = 0.0
    total_tokens = 0
    total_positive_tokens = 0
    for batch_idx, batch in enumerate(loader, start=1):
        features = batch["features"].to(device)
        any_frequency = batch["any_frequency"].to(device)
        union_target = batch["union_target"].to(device)
        stage_frequency = batch["stage_frequency"].to(device)
        loss_weights = batch["loss_weights"].to(device)
        mask = batch["mask"].to(device)
        lengths = batch["lengths"].to(device)
        positive_mask = (union_target > 0.5) & mask

        optimizer.zero_grad()
        detector_logits, stage_logits = model(features, lengths=lengths)

        detector_loss = detector_loss_fn(detector_logits, any_frequency)
        token_weights = mask.float() * loss_weights
        detector_loss = (detector_loss * token_weights).sum() / token_weights.sum().clamp(min=1.0)

        stage_soft_loss = soft_cross_entropy_from_distribution(stage_logits, stage_frequency, positive_mask)
        stage_mass_loss = compute_mass_loss(
            detector_logits=detector_logits,
            stage_logits=stage_logits,
            stage_frequency=stage_frequency,
            positive_mask=positive_mask,
            class_weights=mass_class_weights,
        )

        total = detector_loss + float(stage_soft_loss_weight) * stage_soft_loss + float(stage_mass_loss_weight) * stage_mass_loss
        total.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        token_count = int(mask.sum().item())
        positive_count = int(positive_mask.sum().item())
        total_loss += float(total.item()) * token_count
        total_detector_loss += float(detector_loss.item()) * token_count
        total_stage_soft_loss += float(stage_soft_loss.item()) * max(positive_count, 1)
        total_stage_mass_loss += float(stage_mass_loss.item()) * max(positive_count, 1)
        total_tokens += token_count
        total_positive_tokens += positive_count
        if log_interval > 0 and batch_idx % log_interval == 0:
            running = total_loss / max(total_tokens, 1)
            print(f"  step {batch_idx}/{len(loader)} | running_loss {running:.4f}")

    return (
        total_loss / max(total_tokens, 1),
        total_detector_loss / max(total_tokens, 1),
        total_stage_soft_loss / max(total_positive_tokens, 1),
        total_stage_mass_loss / max(total_positive_tokens, 1),
    )


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> pd.DataFrame:
    model.eval()
    rows = []
    for batch in loader:
        features = batch["features"].to(device)
        any_frequency = batch["any_frequency"].to(device)
        union_target = batch["union_target"].to(device)
        stage_frequency = batch["stage_frequency"].to(device)
        dominant_stage = batch["dominant_stage"].to(device)
        beat_idx = batch["beat_idx"].to(device)
        mask = batch["mask"].to(device)
        lengths = batch["lengths"].to(device)

        detector_logits, stage_logits = model(features, lengths=lengths)
        detector_scores = torch.sigmoid(detector_logits)
        stage_probs = torch.softmax(stage_logits, dim=-1)
        pred_stage = stage_probs.argmax(dim=-1) + 1

        for batch_idx_i, sample_id in enumerate(batch["sample_ids"]):
            length = int(mask[batch_idx_i].sum().item())
            for pos in range(length):
                rows.append(
                    {
                        "sample_id": sample_id,
                        "piece_id": batch["piece_ids"][batch_idx_i],
                        "beat_idx": int(beat_idx[batch_idx_i, pos].item()),
                        "any_frequency": float(any_frequency[batch_idx_i, pos].item()),
                        "union_target": float(union_target[batch_idx_i, pos].item()),
                        "dominant_stage": int(dominant_stage[batch_idx_i, pos].item()),
                        "detector_score": float(detector_scores[batch_idx_i, pos].item()),
                        "pred_stage_class": int(pred_stage[batch_idx_i, pos].item()),
                        "low_frequency": float(stage_frequency[batch_idx_i, pos, 0].item()),
                        "mid_frequency": float(stage_frequency[batch_idx_i, pos, 1].item()),
                        "high_frequency": float(stage_frequency[batch_idx_i, pos, 2].item()),
                        "pred_stage_prob_1": float(stage_probs[batch_idx_i, pos, 0].item()),
                        "pred_stage_prob_2": float(stage_probs[batch_idx_i, pos, 1].item()),
                        "pred_stage_prob_3": float(stage_probs[batch_idx_i, pos, 2].item()),
                    }
                )
    return pd.DataFrame(rows)


def detector_sequence_maps(pred_df: pd.DataFrame) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    sequence_scores = {}
    sequence_union = {}
    sequence_frequency = {}
    ordered = pred_df.sort_values(["sample_id", "beat_idx"])
    for sample_id, group in ordered.groupby("sample_id", sort=False):
        sequence_scores[sample_id] = group["detector_score"].to_numpy(dtype=np.float32)
        sequence_union[sample_id] = group["union_target"].to_numpy(dtype=np.float32)
        sequence_frequency[sample_id] = group["any_frequency"].to_numpy(dtype=np.float32)
    return sequence_scores, sequence_union, sequence_frequency


def grading_report(y_true: np.ndarray, y_pred: np.ndarray, labels: list[int]) -> dict:
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "class_f1": {str(label): float(report[str(label)]["f1-score"]) for label in labels},
        "class_precision": {str(label): float(report[str(label)]["precision"]) for label in labels},
        "class_recall": {str(label): float(report[str(label)]["recall"]) for label in labels},
        "class_support": {str(label): int(report[str(label)]["support"]) for label in labels},
    }


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


def main():
    parser = argparse.ArgumentParser(description="Piece-level union/frequency multitask detector+stage model.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--heldout_piece", nargs="+", required=True)
    parser.add_argument("--train_pieces", nargs="*", default=None)
    parser.add_argument("--model", choices=["bilstm", "tcn"], default="tcn")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--early_stop_patience", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--log_interval", type=int, default=0)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--min_precision", type=float, default=0.85)
    parser.add_argument("--consensus_threshold", type=float, default=None)
    parser.add_argument("--hard_negative_radius", type=int, default=0)
    parser.add_argument("--hard_negative_weight", type=float, default=1.0)
    parser.add_argument("--easy_negative_weight", type=float, default=1.0)
    parser.add_argument("--stage_soft_loss_weight", type=float, default=1.0)
    parser.add_argument("--stage_mass_loss_weight", type=float, default=1.0)
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
    hardneg_suffix = ""
    if int(args.hard_negative_radius) > 0 and float(args.hard_negative_weight) != float(args.easy_negative_weight):
        easy_tag = str(args.easy_negative_weight).replace(".", "p")
        hard_tag = str(args.hard_negative_weight).replace(".", "p")
        hardneg_suffix = f"_hnr{int(args.hard_negative_radius)}_hw{hard_tag}_ew{easy_tag}"
    if args.output_dir:
        out_root = Path(args.output_dir).resolve()
    else:
        out_root = resolve_path(
            cfg,
            f"../outputs/piece_union_multitask/{heldout_slug}/{args.model}_any_boundary_xml_curated_p{int(round(args.min_precision * 100))}{hardneg_suffix}",
        )
    out_root.mkdir(parents=True, exist_ok=True)

    df = load_table(table_path)
    df = apply_piece_protocol_split(df, heldout_pieces=args.heldout_piece, train_pieces=args.train_pieces)
    df = df[df["protocol_split"].isin(["train", "val"])].copy()
    feature_cols = select_feature_columns(cfg, feature_columns(df))
    piece_df = build_piece_union_frame(df, feature_cols=feature_cols)

    train_samples = piece_samples_from_frame(piece_df, feature_cols, split="train")
    val_samples = piece_samples_from_frame(piece_df, feature_cols, split="val")
    if not train_samples or not val_samples:
        raise ValueError("Protocol split produced an empty train or val set")

    mean, std = compute_normalizer(train_samples)
    train_ds = PieceUnionMultitaskDataset(
        train_samples,
        mean=mean,
        std=std,
        hard_negative_radius=int(args.hard_negative_radius),
        hard_negative_weight=float(args.hard_negative_weight),
        easy_negative_weight=float(args.easy_negative_weight),
    )
    val_ds = PieceUnionMultitaskDataset(val_samples, mean=mean, std=std)
    batch_size = int(args.batch_size or seq_cfg.get("batch_size", 64))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_multitask)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_multitask)

    model = build_multitask_model(args.model, input_dim=len(feature_cols), cfg=cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(seq_cfg.get("lr", 1e-3)),
        weight_decay=float(seq_cfg.get("weight_decay", 1e-4)),
    )

    train_any = np.concatenate([sample["any_frequency"] for sample in train_samples], axis=0)
    pos = float(train_any.sum())
    neg = float(train_any.shape[0] - pos)
    detector_pos_weight = torch.tensor([neg / max(pos, 1.0)], device=device, dtype=torch.float32)
    detector_loss_fn = nn.BCEWithLogitsLoss(pos_weight=detector_pos_weight, reduction="none")

    stage_mass = np.concatenate([sample["stage_frequency"] for sample in train_samples], axis=0)
    class_mass = stage_mass.sum(axis=0).astype(np.float32)
    class_mass[class_mass < 1e-6] = 1.0
    mass_class_weights = torch.tensor(class_mass.sum() / class_mass, device=device, dtype=torch.float32)
    mass_class_weights = mass_class_weights / mass_class_weights.mean().clamp(min=1e-6)

    thresholds = threshold_grid(cfg)
    tolerance = int(eval_cfg.get("event_tolerance", 1))
    min_distance = int(eval_cfg.get("min_distance", 6))
    prominence = float(eval_cfg.get("prominence", 0.0))
    consensus_threshold = float(args.consensus_threshold if args.consensus_threshold is not None else eval_cfg.get("consensus_threshold", 0.5))
    epochs = int(args.epochs or seq_cfg.get("epochs", 30))
    patience = int(args.early_stop_patience if args.early_stop_patience is not None else seq_cfg.get("early_stop_patience", 5))
    grad_clip = float(seq_cfg.get("grad_clip", 1.0))

    best_epoch = 0
    best_key = None
    best_metrics = None
    best_pred = None
    best_oracle = None
    best_stage_event = None
    history = []
    bad_epochs = 0

    for epoch in range(1, epochs + 1):
        train_loss, train_detector_loss, train_stage_soft_loss, train_stage_mass_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            detector_loss_fn=detector_loss_fn,
            mass_class_weights=mass_class_weights,
            stage_soft_loss_weight=float(args.stage_soft_loss_weight),
            stage_mass_loss_weight=float(args.stage_mass_loss_weight),
            grad_clip=grad_clip,
            log_interval=max(int(args.log_interval), 0),
        )
        val_pred = predict(model, val_loader, device=device)
        sequence_scores, sequence_union, sequence_frequency = detector_sequence_maps(val_pred)
        union_metrics = search_union_frequency_threshold(
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
        precision_floor_met = union_metrics.union_precision >= float(args.min_precision)

        pos_val = val_pred[val_pred["dominant_stage"] > 0].copy()
        oracle_grading = grading_report(
            y_true=pos_val["dominant_stage"].to_numpy(dtype=np.int64),
            y_pred=pos_val["pred_stage_class"].to_numpy(dtype=np.int64),
            labels=[1, 2, 3],
        )

        sequence_pred_labels = {}
        sequence_true_labels = {}
        for sample_id, group in val_pred.sort_values(["sample_id", "beat_idx"]).groupby("sample_id", sort=False):
            sequence_pred_labels[sample_id] = group["pred_stage_class"].to_numpy(dtype=np.int32)
            sequence_true_labels[sample_id] = group["dominant_stage"].to_numpy(dtype=np.int32)
        stage_event_metrics = evaluate_labeled_event_sequences(
            sequence_scores=sequence_scores,
            sequence_pred_labels=sequence_pred_labels,
            sequence_true_labels=sequence_true_labels,
            positive_classes=(1, 2, 3),
            threshold=float(union_metrics.threshold),
            tolerance=tolerance,
            min_distance=min_distance,
            prominence=prominence,
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_detector_loss": train_detector_loss,
                "train_stage_soft_loss": train_stage_soft_loss,
                "train_stage_mass_loss": train_stage_mass_loss,
                "union_precision": union_metrics.union_precision,
                "weighted_recall": union_metrics.weighted_recall,
                "consensus_recall": union_metrics.consensus_recall,
                "oracle_stage_macro_f1": oracle_grading["macro_f1"],
                "end_to_end_stage_macro_f1": stage_event_metrics.macro_f1,
                "best_threshold": union_metrics.threshold,
                "precision_floor_met": precision_floor_met,
            }
        )
        print(
            f"Epoch {epoch}/{epochs} | train_loss {train_loss:.4f} | "
            f"union_precision {union_metrics.union_precision:.4f} | weighted_recall {union_metrics.weighted_recall:.4f} | "
            f"oracle_f1 {oracle_grading['macro_f1']:.4f} | e2e_f1 {stage_event_metrics.macro_f1:.4f}"
        )

        current_key = (
            float(precision_floor_met),
            union_metrics.weighted_recall if precision_floor_met else union_metrics.union_precision,
            union_metrics.union_precision,
            stage_event_metrics.macro_f1,
            oracle_grading["macro_f1"],
            union_metrics.consensus_recall,
            -float(union_metrics.mean_offset or 1e9),
        )
        if best_key is None or current_key > best_key:
            best_key = current_key
            best_epoch = epoch
            best_metrics = union_metrics
            best_pred = val_pred.copy()
            best_oracle = oracle_grading
            best_stage_event = stage_event_metrics
            bad_epochs = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_type": args.model,
                    "feature_columns": feature_cols,
                    "mean": mean,
                    "std": std,
                    "best_epoch": best_epoch,
                    "best_threshold": union_metrics.threshold,
                    "min_precision": args.min_precision,
                    "consensus_threshold": consensus_threshold,
                    "stage_soft_loss_weight": args.stage_soft_loss_weight,
                    "stage_mass_loss_weight": args.stage_mass_loss_weight,
                },
                out_root / "best.pt",
            )
        else:
            bad_epochs += 1
            if patience > 0 and bad_epochs >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    if best_metrics is None or best_pred is None or best_oracle is None or best_stage_event is None:
        raise RuntimeError("Multitask training did not produce validation metrics")

    best_pred.to_csv(out_root / "val_predictions.csv.gz", index=False, compression="gzip")
    np.savez(out_root / "scaler_stats.npz", mean=mean, std=std)

    train_df = piece_df[piece_df["protocol_split"] == "train"].copy()
    val_df = piece_df[piece_df["protocol_split"] == "val"].copy()
    summary = {
        "table_path": str(table_path),
        "heldout_pieces": list(args.heldout_piece),
        "train_piece_count": int(train_df["piece_id"].nunique()),
        "val_piece_count": int(val_df["piece_id"].nunique()),
        "train_sequence_count": int(train_df["piece_sample_id"].nunique()),
        "val_sequence_count": int(val_df["piece_sample_id"].nunique()),
        "model_type": args.model,
        "seed": seed,
        "device": str(device),
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "early_stop_patience": patience,
        "precision_floor": float(args.min_precision),
        "consensus_threshold": consensus_threshold,
        "hard_negative_radius": int(args.hard_negative_radius),
        "hard_negative_weight": float(args.hard_negative_weight),
        "easy_negative_weight": float(args.easy_negative_weight),
        "stage_soft_loss_weight": float(args.stage_soft_loss_weight),
        "stage_mass_loss_weight": float(args.stage_mass_loss_weight),
        "precision_floor_met": bool(best_metrics.union_precision >= float(args.min_precision)),
        "union_metrics": union_metrics_to_dict(best_metrics),
        "oracle_stage_grading": {
            "target": "dominant_stage",
            **best_oracle,
        },
        "end_to_end_stage": labeled_metrics_to_dict(best_stage_event),
        "feature_columns": feature_cols,
        "history": history,
    }
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        f"Held-out {heldout_slug} | union_precision={best_metrics.union_precision:.4f} | "
        f"weighted_recall={best_metrics.weighted_recall:.4f} | consensus_recall={best_metrics.consensus_recall:.4f}"
    )
    print(
        f"  oracle_stage_macro_f1={best_oracle['macro_f1']:.4f} | "
        f"end_to_end_stage_macro_f1={best_stage_event.macro_f1:.4f}"
    )
    print(
        "  end_to_end_stage_class_f1="
        f"low:{best_stage_event.class_f1.get(1, 0.0):.4f},"
        f"mid:{best_stage_event.class_f1.get(2, 0.0):.4f},"
        f"high:{best_stage_event.class_f1.get(3, 0.0):.4f}"
    )


if __name__ == "__main__":
    main()
