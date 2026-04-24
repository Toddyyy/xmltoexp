#!/usr/bin/env python3

from __future__ import annotations

import argparse
import copy
import json
import random
import re
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset

from boundary_restart.config import load_config, resolve_path, threshold_grid
from boundary_restart.cumulative_targets import (
    COMPONENT_RAW_LEVELS,
    build_piece_frequency_for_raw_levels,
    build_topdown_cumulative_frequency,
    cumulative_components_for_target,
)
from boundary_restart.derived_features import add_highlevel_derived_features
from boundary_restart.features import PeakConfig, boundary_probs_to_binary, load_boundary_npz, replace_level_suffix
from boundary_restart.metrics import (
    evaluate_labeled_event_sequences,
    extract_events,
    greedy_match_pairs,
    search_union_frequency_threshold,
)
from boundary_restart.rest_spans import (
    build_rest_span_arrays,
    build_rest_span_tolerance_weights,
    canonicalize_frequency_with_ignore,
    expand_frequency_over_rest_spans,
)
from boundary_restart.models import build_sequence_model
from boundary_restart.table_io import feature_columns, load_table

RAW_LEVEL_TARGET_RE = re.compile(r"^level([1-6])_boundary$")
RAW_LEVEL_GROUP_TARGETS = {
    "level1plus_boundary": (1, 2, 3, 4, 5, 6),
    "level2plus_boundary": (2, 3, 4, 5, 6),
    "level3plus_boundary": (3, 4, 5, 6),
    "level4plus_boundary": (4, 5, 6),
    "level5plus_split56_boundary": (5, 6),
    "level1plus_split56_boundary": (1, 2, 3, 4, 5, 6),
    "level2plus_split56_boundary": (2, 3, 4, 5, 6),
    "level3plus_split56_boundary": (3, 4, 5, 6),
    "level4plus_split56_boundary": (4, 5, 6),
    "level34_boundary": (3, 4),
    "level56_boundary": (5, 6),
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


def select_grader_feature_columns(cfg: dict, detector_cols: list[str], all_columns: list[str]) -> list[str]:
    feature_cfg = cfg.get("features", {})
    grader_include = feature_cfg.get("grader_include")
    grader_extra_include = feature_cfg.get("grader_extra_include")
    if grader_include:
        include_set = set(grader_include)
        return [col for col in all_columns if col in include_set and col != "protocol_split"]
    if grader_extra_include:
        selected = list(detector_cols)
        existing = set(selected)
        for col in grader_extra_include:
            if col in all_columns and col not in existing and col != "protocol_split":
                selected.append(col)
                existing.add(col)
        return selected
    return list(detector_cols)


def detector_labels(stage_class: np.ndarray, mode: str) -> np.ndarray:
    stage_class = np.asarray(stage_class, dtype=np.int64)
    if mode == "any_boundary":
        return (stage_class > 0).astype(np.float32)
    if mode == "midhigh_boundary":
        return (stage_class >= 2).astype(np.float32)
    if mode == "low_boundary":
        return (stage_class == 1).astype(np.float32)
    if mode == "mid_boundary":
        return (stage_class == 2).astype(np.float32)
    if mode == "high_boundary":
        return (stage_class >= 3).astype(np.float32)
    raw_level = parse_raw_level_target(mode)
    if raw_level is not None:
        raise ValueError(
            f"detector_labels received raw level target {mode} without per-level target augmentation"
        )
    raise ValueError(f"Unsupported detector target mode: {mode}")


def parse_raw_level_target(mode: str) -> int | None:
    match = RAW_LEVEL_TARGET_RE.match(str(mode))
    return int(match.group(1)) if match else None


def parse_raw_level_targets(mode: str) -> tuple[int, ...] | None:
    single = parse_raw_level_target(mode)
    if single is not None:
        return (single,)
    grouped = RAW_LEVEL_GROUP_TARGETS.get(str(mode))
    if grouped is not None:
        return tuple(int(x) for x in grouped)
    return None


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


def build_piece_union_frame(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_mode: str,
    peak_cfg: PeakConfig,
    beat_unit_fallback: float,
    cumulative_merge_tolerance: int = 0,
    cumulative_component_weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    frame = df.copy()
    cumulative_components = cumulative_components_for_target(target_mode)
    raw_levels = parse_raw_level_targets(target_mode)
    if cumulative_components is not None and int(cumulative_merge_tolerance) > 0:
        pass
    elif raw_levels is not None:
        if "source_path" not in frame.columns:
            raise ValueError("raw level targets require source_path in the beat table")
        detector_binary = np.zeros(len(frame), dtype=np.float32)
        beat_idx = frame["beat_idx"].to_numpy(dtype=np.int32)
        for source_path, positions in frame.groupby("source_path", sort=False).indices.items():
            pos = np.asarray(positions, dtype=np.int64)
            boundary_binary = None
            for raw_level in raw_levels:
                level_path = replace_level_suffix(Path(str(source_path)), level=raw_level)
                loaded = load_boundary_npz(level_path, beat_unit_fallback=beat_unit_fallback)
                current_binary = boundary_probs_to_binary(
                    np.asarray(loaded["boundary_probs"], dtype=np.float32),
                    peak_cfg,
                ).astype(np.float32)
                boundary_binary = current_binary if boundary_binary is None else np.maximum(boundary_binary, current_binary)
            sample_beat_idx = beat_idx[pos]
            if sample_beat_idx.size and sample_beat_idx.max() >= boundary_binary.shape[0]:
                raise ValueError(
                    f"beat_idx out of range for {source_path} {raw_levels}: max beat {sample_beat_idx.max()} >= {boundary_binary.shape[0]}"
                )
            detector_binary[pos] = boundary_binary[sample_beat_idx].astype(np.float32)
        frame["detector_binary"] = detector_binary.astype(np.float32)
    else:
        frame["detector_binary"] = detector_labels(frame["stage_class"].to_numpy(dtype=np.int64), mode=target_mode)
    frame["low_binary"] = (frame["stage_class"].to_numpy(dtype=np.int64) == 1).astype(np.float32)
    frame["mid_binary"] = (frame["stage_class"].to_numpy(dtype=np.int64) == 2).astype(np.float32)
    frame["high_binary"] = (frame["stage_class"].to_numpy(dtype=np.int64) >= 3).astype(np.float32)
    agg_spec: dict[str, str] = {
        "protocol_split": "first",
        "num_beats": "first",
        "low_binary": "mean",
        "mid_binary": "mean",
        "high_binary": "mean",
        "sample_id": pd.Series.nunique,
    }
    if "detector_binary" in frame.columns:
        agg_spec["detector_binary"] = "mean"
    for col in feature_cols:
        agg_spec[col] = "first"

    piece = (
        frame.sort_values(["piece_id", "beat_idx", "sample_id"])
        .groupby(["piece_id", "beat_idx"], sort=False)
        .agg(agg_spec)
        .rename(columns={"sample_id": "performer_count"})
        .reset_index()
    )
    if cumulative_components is not None and int(cumulative_merge_tolerance) > 0:
        component_map = {
            component_name: build_piece_frequency_for_raw_levels(
                frame,
                raw_levels=COMPONENT_RAW_LEVELS[component_name],
                peak_cfg=peak_cfg,
                beat_unit_fallback=beat_unit_fallback,
            )
            for component_name in cumulative_components
        }
        merged = build_topdown_cumulative_frequency(
            piece[["piece_id", "beat_idx"]],
            component_map=component_map,
            component_order=cumulative_components,
            tolerance=int(cumulative_merge_tolerance),
            component_weights=cumulative_component_weights,
        )
        piece = piece.merge(merged, on=["piece_id", "beat_idx"], how="left")
        piece["frequency_target"] = piece["frequency_target"].fillna(0.0).astype(np.float32)
    else:
        piece = piece.rename(columns={"detector_binary": "frequency_target"})
    piece["union_target"] = (piece["frequency_target"] > 0.0).astype(np.float32)
    piece = piece.rename(
        columns={
            "low_binary": "low_frequency",
            "mid_binary": "mid_frequency",
            "high_binary": "high_frequency",
        }
    )
    stage_freq = piece[["low_frequency", "mid_frequency", "high_frequency"]].to_numpy(dtype=np.float32)
    dominant_idx = np.argmax(stage_freq, axis=1) + 1
    dominant_idx = np.where(piece["union_target"].to_numpy(dtype=np.float32) > 0.0, dominant_idx, 0)
    piece["dominant_stage"] = dominant_idx.astype(np.int64)
    piece["piece_sample_id"] = piece["piece_id"]
    return piece


def apply_rest_span_training_labels(
    piece_df: pd.DataFrame,
    mode: str,
    min_len: int,
    source_col: str,
    source_threshold: float,
    tolerance_negative_weight: float,
    min_train_frequency_target: float,
) -> pd.DataFrame:
    frame = piece_df.copy()
    frame["train_frequency_target"] = frame["frequency_target"].astype(np.float32)
    frame["train_union_target"] = frame["union_target"].astype(np.float32)
    frame["train_loss_factor"] = np.ones(len(frame), dtype=np.float32)
    if float(min_train_frequency_target) > 0.0:
        keep_mask = frame["train_frequency_target"].to_numpy(dtype=np.float32) >= float(min_train_frequency_target)
        frame.loc[~keep_mask, "train_frequency_target"] = 0.0
        frame.loc[~keep_mask, "train_union_target"] = 0.0
    if mode == "none" and float(tolerance_negative_weight) >= 1.0:
        return frame

    if source_col not in frame.columns:
        raise ValueError(f"rest-span training labels require {source_col} in the piece-level frame")

    updated_groups = []
    for piece_id, group in frame.sort_values(["piece_id", "beat_idx"]).groupby("piece_id", sort=False):
        group = group.copy().reset_index(drop=True)
        empty_mask = group[source_col].to_numpy(dtype=np.float32) > float(source_threshold)
        span_id, _, _ = build_rest_span_arrays(empty_mask, min_len=int(min_len))
        train_freq = group["frequency_target"].to_numpy(dtype=np.float32).copy()
        train_union = group["union_target"].to_numpy(dtype=np.float32).copy()
        train_loss_factor = np.ones_like(train_freq, dtype=np.float32)
        if mode == "expand_max":
            train_freq = expand_frequency_over_rest_spans(
                train_freq,
                span_id=span_id,
                agg="max",
            )
            train_union = (train_freq > 0.0).astype(np.float32)
        elif mode == "canonical_ignore":
            train_freq, train_union, ignore_loss = canonicalize_frequency_with_ignore(
                train_freq,
                span_id=span_id,
                agg="max",
            )
            train_loss_factor *= ignore_loss
        elif mode == "none":
            pass
        else:
            raise ValueError(f"Unsupported rest_span_label_mode: {mode}")
        if float(tolerance_negative_weight) < 1.0:
            train_loss_factor *= build_rest_span_tolerance_weights(
                group["frequency_target"].to_numpy(dtype=np.float32),
                span_id=span_id,
                negative_weight=float(tolerance_negative_weight),
            )
        group["train_frequency_target"] = train_freq.astype(np.float32)
        group["train_union_target"] = train_union.astype(np.float32)
        group["train_loss_factor"] = train_loss_factor.astype(np.float32)
        updated_groups.append(group)
    return pd.concat(updated_groups, axis=0, ignore_index=True)


class PieceUnionDataset(Dataset):
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
            "labels": sample["train_frequency_target"].astype(np.float32),
            "union_labels": sample["union_target"].astype(np.float32),
            "frequency_target": sample["frequency_target"].astype(np.float32),
            "performer_count": sample["performer_count"].astype(np.int32),
            "loss_weights": build_loss_weights(
                sample["train_union_target"],
                hard_negative_radius=self.hard_negative_radius,
                hard_negative_weight=self.hard_negative_weight,
                easy_negative_weight=self.easy_negative_weight,
            )
            * sample["train_loss_factor"].astype(np.float32),
            "length": int(sample["train_frequency_target"].shape[0]),
        }


def collate_piece_union(batch: list[dict]) -> dict:
    lengths = [item["length"] for item in batch]
    max_len = max(lengths)
    feat_dim = batch[0]["features"].shape[1]
    features = torch.zeros(len(batch), max_len, feat_dim, dtype=torch.float32)
    labels = torch.zeros(len(batch), max_len, dtype=torch.float32)
    union_labels = torch.zeros(len(batch), max_len, dtype=torch.float32)
    frequency_target = torch.zeros(len(batch), max_len, dtype=torch.float32)
    performer_count = torch.zeros(len(batch), max_len, dtype=torch.int64)
    loss_weights = torch.ones(len(batch), max_len, dtype=torch.float32)
    beat_idx = torch.zeros(len(batch), max_len, dtype=torch.int64)
    mask = torch.zeros(len(batch), max_len, dtype=torch.bool)
    sample_ids = []
    piece_ids = []

    for idx, item in enumerate(batch):
        length = item["length"]
        features[idx, :length] = torch.from_numpy(item["features"])
        labels[idx, :length] = torch.from_numpy(item["labels"])
        union_labels[idx, :length] = torch.from_numpy(item["union_labels"])
        frequency_target[idx, :length] = torch.from_numpy(item["frequency_target"])
        performer_count[idx, :length] = torch.from_numpy(item["performer_count"])
        loss_weights[idx, :length] = torch.from_numpy(item["loss_weights"])
        beat_idx[idx, :length] = torch.from_numpy(item["beat_idx"])
        mask[idx, :length] = True
        sample_ids.append(item["sample_id"])
        piece_ids.append(item["piece_id"])

    return {
        "features": features,
        "labels": labels,
        "union_labels": union_labels,
        "frequency_target": frequency_target,
        "performer_count": performer_count,
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


def piece_samples_from_frame(
    df: pd.DataFrame,
    feature_cols: list[str],
    split: str,
) -> list[dict]:
    subset = df[df["protocol_split"] == split].copy().sort_values(["piece_sample_id", "beat_idx"])
    samples = []
    for sample_id, group in subset.groupby("piece_sample_id", sort=False):
        samples.append(
            {
                "sample_id": sample_id,
                "piece_id": group["piece_id"].iloc[0],
                "beat_idx": group["beat_idx"].to_numpy(dtype=np.int32),
                "features": group[feature_cols].to_numpy(dtype=np.float32),
                "union_target": group["union_target"].to_numpy(dtype=np.float32),
                "frequency_target": group["frequency_target"].to_numpy(dtype=np.float32),
                "train_union_target": group["train_union_target"].to_numpy(dtype=np.float32)
                if "train_union_target" in group.columns
                else group["union_target"].to_numpy(dtype=np.float32),
                "train_frequency_target": group["train_frequency_target"].to_numpy(dtype=np.float32)
                if "train_frequency_target" in group.columns
                else group["frequency_target"].to_numpy(dtype=np.float32),
                "performer_count": group["performer_count"].to_numpy(dtype=np.int32),
                "train_loss_factor": group["train_loss_factor"].to_numpy(dtype=np.float32)
                if "train_loss_factor" in group.columns
                else np.ones(len(group), dtype=np.float32),
            }
        )
    return samples


def compute_detector_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    loss_weights: torch.Tensor,
    loss_type: str,
    loss_fn: nn.Module,
) -> torch.Tensor:
    if loss_type in {"bce", "bce_freq_weighted"}:
        per_token = loss_fn(logits, labels)
    elif loss_type in {"huber", "mse"}:
        preds = torch.sigmoid(logits)
        per_token = loss_fn(preds, labels)
    else:
        raise ValueError(f"Unsupported loss_type: {loss_type}")
    token_weights = mask.float() * loss_weights
    if loss_type == "bce_freq_weighted":
        positive_mask = labels > 0.0
        if torch.any(positive_mask):
            pos_mean = labels[positive_mask].mean().clamp(min=1e-3)
            freq_factor = torch.ones_like(labels)
            freq_factor[positive_mask] = (labels[positive_mask] / pos_mean).clamp(min=0.25, max=4.0)
            token_weights = token_weights * freq_factor
    return (per_token * token_weights).sum() / token_weights.sum().clamp(min=1.0)


def primary_metric_value(metrics, metric_name: str) -> float:
    if metric_name == "union_recall":
        return float(metrics.union_recall)
    if metric_name == "consensus_recall":
        return float(metrics.consensus_recall)
    return float(metrics.weighted_recall)


def precision_metric_value(metrics, metric_name: str) -> float:
    if metric_name == "frequency_weighted_precision":
        return float(metrics.frequency_weighted_precision)
    if metric_name == "consensus_precision":
        return float(metrics.consensus_precision)
    return float(metrics.union_precision)


def resolve_precision_floors(args) -> dict[str, float]:
    floors = {
        "union_precision": 0.0,
        "frequency_weighted_precision": 0.0,
        "consensus_precision": 0.0,
    }
    floors[str(args.precision_metric)] = float(args.min_precision)
    if args.min_union_precision_floor is not None:
        floors["union_precision"] = max(floors["union_precision"], float(args.min_union_precision_floor))
    if args.min_frequency_weighted_precision_floor is not None:
        floors["frequency_weighted_precision"] = max(
            floors["frequency_weighted_precision"], float(args.min_frequency_weighted_precision_floor)
        )
    if args.min_consensus_precision_floor is not None:
        floors["consensus_precision"] = max(
            floors["consensus_precision"], float(args.min_consensus_precision_floor)
        )
    return floors


def precision_floors_met(metrics, floors: dict[str, float]) -> bool:
    return (
        metrics.union_precision >= float(floors["union_precision"])
        and metrics.frequency_weighted_precision >= float(floors["frequency_weighted_precision"])
        and metrics.consensus_precision >= float(floors["consensus_precision"])
    )


def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
    loss_fn,
    loss_type: str,
    grad_clip: float,
    ema_state: dict[str, torch.Tensor] | None = None,
    ema_decay: float = 0.0,
    log_interval: int = 0,
) -> float:
    model.train()
    total_loss = 0.0
    total_tokens = 0
    for batch_idx, batch in enumerate(loader, start=1):
        features = batch["features"].to(device)
        labels = batch["labels"].to(device)
        loss_weights = batch["loss_weights"].to(device)
        mask = batch["mask"].to(device)
        lengths = batch["lengths"].to(device)

        optimizer.zero_grad()
        logits = model(features, lengths=lengths)
        loss = compute_detector_loss(
            logits=logits,
            labels=labels,
            mask=mask,
            loss_weights=loss_weights,
            loss_type=loss_type,
            loss_fn=loss_fn,
        )
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        if ema_state is not None:
            update_ema_state(ema_state, model, decay=float(ema_decay))

        total_loss += float(loss.item()) * int(mask.sum().item())
        total_tokens += int(mask.sum().item())
        if log_interval > 0 and batch_idx % log_interval == 0:
            running_loss = total_loss / max(total_tokens, 1)
            print(f"  step {batch_idx}/{len(loader)} | running_loss {running_loss:.4f}")
    return total_loss / max(total_tokens, 1)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str,
    epochs: int,
    factor: float,
    patience: int,
    min_lr: float,
    eta_min: float,
):
    if scheduler_type == "none":
        return None
    if scheduler_type == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=float(factor),
            patience=int(patience),
            min_lr=float(min_lr),
        )
    if scheduler_type == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(int(epochs), 1),
            eta_min=float(eta_min),
        )
    raise ValueError(f"Unsupported scheduler_type: {scheduler_type}")


def snapshot_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def average_state_dicts(state_dicts: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    if not state_dicts:
        raise ValueError("state_dicts must not be empty")
    avg_state = {}
    keys = state_dicts[0].keys()
    for key in keys:
        tensors = [sd[key].float() for sd in state_dicts]
        avg = torch.stack(tensors, dim=0).mean(dim=0)
        avg_state[key] = avg.to(dtype=state_dicts[0][key].dtype)
    return avg_state


def init_ema_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return snapshot_state_dict(model)


def update_ema_state(ema_state: dict[str, torch.Tensor], model: nn.Module, decay: float) -> None:
    model_state = model.state_dict()
    for key, value in model_state.items():
        detached = value.detach().cpu()
        if not torch.is_floating_point(detached):
            ema_state[key] = detached.clone()
            continue
        ema_state[key].mul_(float(decay)).add_(detached, alpha=float(1.0 - decay))


@torch.no_grad()
def predict_detector(model, loader, device) -> pd.DataFrame:
    model.eval()
    rows = []
    for batch in loader:
        features = batch["features"].to(device)
        labels = batch["labels"].to(device)
        union_labels = batch["union_labels"].to(device)
        frequency_target = batch["frequency_target"].to(device)
        performer_count = batch["performer_count"].to(device)
        beat_idx = batch["beat_idx"].to(device)
        mask = batch["mask"].to(device)
        lengths = batch["lengths"].to(device)
        logits = model(features, lengths=lengths)
        probs = torch.sigmoid(logits)
        for batch_idx_i, sample_id in enumerate(batch["sample_ids"]):
            length = int(mask[batch_idx_i].sum().item())
            for pos in range(length):
                rows.append(
                    {
                        "sample_id": sample_id,
                        "piece_id": batch["piece_ids"][batch_idx_i],
                        "beat_idx": int(beat_idx[batch_idx_i, pos].item()),
                        "union_target": float(union_labels[batch_idx_i, pos].item()),
                        "frequency_target": float(frequency_target[batch_idx_i, pos].item()),
                        "performer_count": int(performer_count[batch_idx_i, pos].item()),
                        "detector_score": float(probs[batch_idx_i, pos].item()),
                        "train_label": float(labels[batch_idx_i, pos].item()),
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
        sequence_frequency[sample_id] = group["frequency_target"].to_numpy(dtype=np.float32)
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


def train_stage_grader(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: list[str],
    seed: int,
) -> tuple[dict, pd.DataFrame]:
    labels = [1, 2, 3]
    train_pos = train_df[train_df["dominant_stage"] > 0].copy()
    val_pos = val_df[val_df["dominant_stage"] > 0].copy()
    if train_pos.empty or val_pos.empty:
        raise ValueError("Stage grading requires positive low/mid/high labels in both train and val splits")

    scaler = StandardScaler()
    x_train = scaler.fit_transform(train_pos[feature_cols].to_numpy(dtype=np.float32))
    y_train = train_pos["dominant_stage"].to_numpy(dtype=np.int64)
    x_val_pos = scaler.transform(val_pos[feature_cols].to_numpy(dtype=np.float32))
    y_val_pos = val_pos["dominant_stage"].to_numpy(dtype=np.int64)

    clf = LogisticRegression(
        max_iter=4000,
        class_weight="balanced",
        random_state=seed,
    )
    clf.fit(x_train, y_train)

    oracle_pred = clf.predict(x_val_pos).astype(np.int64)
    oracle_metrics = grading_report(y_true=y_val_pos, y_pred=oracle_pred, labels=labels)

    val_all = val_df.copy()
    x_val_all = scaler.transform(val_all[feature_cols].to_numpy(dtype=np.float32))
    pred_all = clf.predict(x_val_all).astype(np.int64)
    prob_all = clf.predict_proba(x_val_all)
    val_all["pred_stage_class"] = pred_all.astype(np.int64)
    for class_idx, label in enumerate(clf.classes_.tolist()):
        val_all[f"pred_stage_prob_{int(label)}"] = prob_all[:, class_idx].astype(np.float32)
    return oracle_metrics, val_all


def union_metrics_to_dict(metrics) -> dict:
    return {
        "threshold": metrics.threshold,
        "union_precision": metrics.union_precision,
        "frequency_weighted_precision": metrics.frequency_weighted_precision,
        "consensus_precision": metrics.consensus_precision,
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


def build_predicted_event_frame(
    pred_df: pd.DataFrame,
    threshold: float,
    min_distance: int,
    prominence: float,
    tolerance: int,
) -> pd.DataFrame:
    rows = []
    ordered = pred_df.sort_values(["sample_id", "beat_idx"]).copy()
    prob_cols = [col for col in ordered.columns if col.startswith("pred_stage_prob_")]

    for sample_id, group in ordered.groupby("sample_id", sort=False):
        group = group.reset_index(drop=True)
        scores = group["detector_score"].to_numpy(dtype=np.float32)
        pred_events = extract_events(
            scores,
            threshold=float(threshold),
            min_distance=int(min_distance),
            prominence=float(prominence),
        )
        true_union_events = np.flatnonzero(group["union_target"].to_numpy(dtype=np.float32) > 0.5).astype(np.int32)
        match_pairs = greedy_match_pairs(pred_events, true_union_events, tolerance=int(tolerance))
        match_map = {pred_idx: (true_idx, offset) for pred_idx, true_idx, offset in match_pairs}

        for event_rank, pred_idx in enumerate(pred_events.tolist(), start=1):
            row = group.iloc[int(pred_idx)]
            true_match = match_map.get(int(event_rank - 1))
            event_row = {
                "sample_id": str(sample_id),
                "piece_id": str(row["piece_id"]),
                "event_rank": int(event_rank),
                "beat_idx": int(row["beat_idx"]),
                "detector_score": float(row["detector_score"]),
                "threshold": float(threshold),
                "union_target_at_beat": float(row["union_target"]),
                "frequency_target_at_beat": float(row["frequency_target"]),
                "performer_count": int(row["performer_count"]),
                "matched_union": bool(true_match is not None),
                "match_offset": int(true_match[1]) if true_match is not None else None,
                "matched_true_beat_idx": int(true_union_events[true_match[0]]) if true_match is not None else None,
            }
            if "dominant_stage" in group.columns:
                event_row["dominant_stage_at_beat"] = int(row["dominant_stage"])
            if "pred_stage_class" in group.columns:
                event_row["pred_stage_class"] = int(row["pred_stage_class"])
            for col in prob_cols:
                event_row[col] = float(row[col])
            rows.append(event_row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Piece-level union/frequency detector protocol.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--heldout_piece", nargs="+", required=True)
    parser.add_argument("--train_pieces", nargs="*", default=None)
    parser.add_argument("--model", choices=["bilstm", "tcn", "cnn", "transformer"], default="tcn")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--early_stop_patience", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--log_interval", type=int, default=0)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument(
        "--detector_target",
        choices=[
            "any_boundary",
            "midhigh_boundary",
            "low_boundary",
            "mid_boundary",
            "high_boundary",
            "level1_boundary",
            "level2_boundary",
            "level3_boundary",
            "level4_boundary",
            "level5_boundary",
            "level6_boundary",
            "level1plus_boundary",
            "level2plus_boundary",
            "level3plus_boundary",
            "level4plus_boundary",
            "level5plus_split56_boundary",
            "level1plus_split56_boundary",
            "level2plus_split56_boundary",
            "level3plus_split56_boundary",
            "level4plus_split56_boundary",
            "level34_boundary",
            "level56_boundary",
        ],
        default="midhigh_boundary",
    )
    parser.add_argument("--min_precision", type=float, default=0.85)
    parser.add_argument("--hard_negative_radius", type=int, default=0)
    parser.add_argument("--hard_negative_weight", type=float, default=1.0)
    parser.add_argument("--easy_negative_weight", type=float, default=1.0)
    parser.add_argument(
        "--selection_metric",
        choices=["weighted_recall", "union_recall", "consensus_recall"],
        default="weighted_recall",
    )
    parser.add_argument(
        "--precision_metric",
        choices=["union_precision", "frequency_weighted_precision", "consensus_precision"],
        default="union_precision",
    )
    parser.add_argument("--min_union_precision_floor", type=float, default=None)
    parser.add_argument("--min_frequency_weighted_precision_floor", type=float, default=None)
    parser.add_argument("--min_consensus_precision_floor", type=float, default=None)
    parser.add_argument("--loss_type", choices=["bce", "bce_freq_weighted", "huber", "mse"], default="bce")
    parser.add_argument("--rest_span_label_mode", choices=["none", "expand_max", "canonical_ignore"], default="none")
    parser.add_argument("--rest_span_min_len", type=int, default=2)
    parser.add_argument("--rest_span_source_col", default="xml_rest_duration_norm")
    parser.add_argument("--rest_span_source_threshold", type=float, default=1e-8)
    parser.add_argument("--rest_span_tolerance_negative_weight", type=float, default=1.0)
    parser.add_argument("--min_train_frequency_target", type=float, default=0.0)
    parser.add_argument("--skip_stage_grading", action="store_true")
    parser.add_argument("--add_derived_highlevel_features", action="store_true")
    parser.add_argument("--derived_feature_include", nargs="*", default=None)
    parser.add_argument("--transformer_dim", type=int, default=None)
    parser.add_argument("--transformer_heads", type=int, default=None)
    parser.add_argument("--transformer_layers", type=int, default=None)
    parser.add_argument("--transformer_ff_dim", type=int, default=None)
    parser.add_argument("--scheduler_type", choices=["none", "plateau", "cosine"], default=None)
    parser.add_argument("--scheduler_factor", type=float, default=None)
    parser.add_argument("--scheduler_patience", type=int, default=None)
    parser.add_argument("--scheduler_min_lr", type=float, default=None)
    parser.add_argument("--scheduler_eta_min", type=float, default=None)
    parser.add_argument("--checkpoint_avg_last_k", type=int, default=0)
    parser.add_argument("--ema_decay", type=float, default=0.0)
    parser.add_argument("--cumulative_merge_tolerance", type=int, default=0)
    parser.add_argument("--cumulative_component_weights_json", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    seq_cfg = cfg.get("sequence", {})
    if args.transformer_dim is not None:
        seq_cfg["transformer_dim"] = int(args.transformer_dim)
    if args.transformer_heads is not None:
        seq_cfg["transformer_heads"] = int(args.transformer_heads)
    if args.transformer_layers is not None:
        seq_cfg["transformer_layers"] = int(args.transformer_layers)
    if args.transformer_ff_dim is not None:
        seq_cfg["transformer_ff_dim"] = int(args.transformer_ff_dim)
    eval_cfg = cfg.get("evaluation", {})
    data_cfg = cfg.get("data", {})
    cumulative_component_weights = None
    if args.cumulative_component_weights_json:
        cumulative_component_weights = {
            str(key): float(value)
            for key, value in json.loads(str(args.cumulative_component_weights_json)).items()
        }

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
            f"../outputs/piece_union_protocol/{heldout_slug}/{args.model}_{args.detector_target}_xml_curated_p{int(round(args.min_precision * 100))}{hardneg_suffix}",
        )
    out_root.mkdir(parents=True, exist_ok=True)

    df = load_table(table_path)
    original_columns = set(df.columns)
    if args.add_derived_highlevel_features:
        df = add_highlevel_derived_features(df)
    df = apply_piece_protocol_split(df, heldout_pieces=args.heldout_piece, train_pieces=args.train_pieces)
    df = df[df["protocol_split"].isin(["train", "val"])].copy()
    all_feature_cols = feature_columns(df)
    feature_cols = select_feature_columns(cfg, all_feature_cols)
    if args.add_derived_highlevel_features and args.derived_feature_include:
        allowed = set(args.derived_feature_include)
        derived_cols = [col for col in feature_cols if col not in original_columns]
        feature_cols = [col for col in feature_cols if col not in derived_cols or col in allowed]
    grader_feature_cols = select_grader_feature_columns(cfg, feature_cols, all_feature_cols)
    peak_cfg = PeakConfig(
        distance=int(data_cfg.get("peak_distance", 6)),
        height=float(data_cfg.get("peak_height", 0.15)),
        prominence=float(data_cfg.get("peak_prominence", 0.05)),
    )
    beat_unit_fallback = float(data_cfg.get("beat_unit_fallback", 1.0))
    piece_df = build_piece_union_frame(
        df,
        feature_cols=feature_cols,
        target_mode=args.detector_target,
        peak_cfg=peak_cfg,
        beat_unit_fallback=beat_unit_fallback,
        cumulative_merge_tolerance=int(args.cumulative_merge_tolerance),
        cumulative_component_weights=cumulative_component_weights,
    )
    piece_df = apply_rest_span_training_labels(
        piece_df,
        mode=str(args.rest_span_label_mode),
        min_len=int(args.rest_span_min_len),
        source_col=str(args.rest_span_source_col),
        source_threshold=float(args.rest_span_source_threshold),
        tolerance_negative_weight=float(args.rest_span_tolerance_negative_weight),
        min_train_frequency_target=float(args.min_train_frequency_target),
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
        hard_negative_radius=int(args.hard_negative_radius),
        hard_negative_weight=float(args.hard_negative_weight),
        easy_negative_weight=float(args.easy_negative_weight),
    )
    val_ds = PieceUnionDataset(val_samples, mean=mean, std=std)
    batch_size = int(args.batch_size or seq_cfg.get("batch_size", 64))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_piece_union)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_piece_union)

    model = build_sequence_model(args.model, input_dim=len(feature_cols), cfg=cfg, output_dim=1).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(seq_cfg.get("lr", 1e-3)),
        weight_decay=float(seq_cfg.get("weight_decay", 1e-4)),
    )

    train_labels = np.concatenate([sample["train_frequency_target"] for sample in train_samples], axis=0)
    if args.loss_type in {"bce", "bce_freq_weighted"}:
        pos = float(train_labels.sum())
        neg = float(train_labels.shape[0] - pos)
        pos_weight = torch.tensor([neg / max(pos, 1.0)], device=device, dtype=torch.float32)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")
    elif args.loss_type == "huber":
        loss_fn = nn.SmoothL1Loss(reduction="none")
    elif args.loss_type == "mse":
        loss_fn = nn.MSELoss(reduction="none")
    else:
        raise ValueError(f"Unsupported loss_type: {args.loss_type}")

    thresholds = threshold_grid(cfg)
    tolerance = int(eval_cfg.get("event_tolerance", 1))
    min_distance = int(eval_cfg.get("min_distance", 6))
    prominence = float(eval_cfg.get("prominence", 0.0))
    consensus_threshold = float(eval_cfg.get("consensus_threshold", 0.5))
    epochs = int(args.epochs or seq_cfg.get("epochs", 30))
    patience = int(args.early_stop_patience if args.early_stop_patience is not None else seq_cfg.get("early_stop_patience", 5))
    grad_clip = float(seq_cfg.get("grad_clip", 1.0))
    scheduler_type = str(args.scheduler_type or seq_cfg.get("scheduler_type", "none"))
    scheduler = build_scheduler(
        optimizer=optimizer,
        scheduler_type=scheduler_type,
        epochs=epochs,
        factor=float(args.scheduler_factor if args.scheduler_factor is not None else seq_cfg.get("scheduler_factor", 0.5)),
        patience=int(args.scheduler_patience if args.scheduler_patience is not None else seq_cfg.get("scheduler_patience", 2)),
        min_lr=float(args.scheduler_min_lr if args.scheduler_min_lr is not None else seq_cfg.get("scheduler_min_lr", 1e-5)),
        eta_min=float(args.scheduler_eta_min if args.scheduler_eta_min is not None else seq_cfg.get("scheduler_eta_min", 1e-5)),
    )
    checkpoint_avg_last_k = max(int(args.checkpoint_avg_last_k), 0)
    ema_decay = float(args.ema_decay)
    use_ema = ema_decay > 0.0
    ema_state = init_ema_state(model) if use_ema else None
    ema_model = build_sequence_model(args.model, input_dim=len(feature_cols), cfg=cfg, output_dim=1).to(device) if use_ema else None
    precision_floors = resolve_precision_floors(args)

    best_epoch = 0
    best_key = None
    best_metrics = None
    best_val_pred = None
    best_validation_model = "raw"
    history = []
    bad_epochs = 0
    recent_state_dicts: deque[tuple[int, dict[str, torch.Tensor]]] = deque(maxlen=checkpoint_avg_last_k or None)

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            loss_fn=loss_fn,
            loss_type=args.loss_type,
            grad_clip=grad_clip,
            ema_state=ema_state,
            ema_decay=ema_decay,
            log_interval=max(int(args.log_interval), 0),
        )
        eval_model = model
        validation_model_name = "raw"
        if use_ema and ema_model is not None and ema_state is not None:
            ema_model.load_state_dict(ema_state)
            eval_model = ema_model
            validation_model_name = "ema"
        val_pred = predict_detector(eval_model, val_loader, device=device)
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
            primary_metric=str(args.selection_metric),
            precision_metric=str(args.precision_metric),
            min_union_precision=float(precision_floors["union_precision"]),
            min_frequency_weighted_precision=float(precision_floors["frequency_weighted_precision"]),
            min_consensus_precision=float(precision_floors["consensus_precision"]),
        )
        selected_precision = precision_metric_value(metrics, str(args.precision_metric))
        precision_floor_met = precision_floors_met(metrics, precision_floors)
        primary_metric = primary_metric_value(metrics, str(args.selection_metric))

        if checkpoint_avg_last_k > 0:
            recent_state_dicts.append((epoch, snapshot_state_dict(eval_model)))

        if scheduler is not None:
            if scheduler_type == "plateau":
                scheduler.step(primary_metric)
            else:
                scheduler.step()
        current_lr = float(optimizer.param_groups[0]["lr"])
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "union_precision": metrics.union_precision,
                "frequency_weighted_precision": metrics.frequency_weighted_precision,
                "consensus_precision": metrics.consensus_precision,
                "union_recall": metrics.union_recall,
                "union_f1": metrics.union_f1,
                "weighted_recall": metrics.weighted_recall,
                "consensus_recall": metrics.consensus_recall,
                "best_threshold": metrics.threshold,
                "precision_floor_met": precision_floor_met,
                "selected_precision_value": selected_precision,
                "precision_floors": {key: float(value) for key, value in precision_floors.items()},
                "validation_model": validation_model_name,
                "lr": current_lr,
            }
        )
        print(
            f"Epoch {epoch}/{epochs} | train_loss {train_loss:.4f} | "
            f"{args.precision_metric} {selected_precision:.4f} | "
            f"{args.selection_metric} {primary_metric:.4f} | "
            f"union_precision {metrics.union_precision:.4f} | threshold {metrics.threshold:.3f} | "
            f"val_model {validation_model_name} | lr {current_lr:.6f}"
        )

        current_key = (
            float(precision_floor_met),
            primary_metric if precision_floor_met else selected_precision,
            selected_precision,
            metrics.union_precision,
            metrics.weighted_recall,
            metrics.consensus_recall,
            metrics.union_f1,
            -float(metrics.mean_offset or 1e9),
        )
        if best_key is None or current_key > best_key:
            best_key = current_key
            best_epoch = epoch
            best_metrics = metrics
            best_val_pred = val_pred.copy()
            best_validation_model = validation_model_name
            bad_epochs = 0
            torch.save(
                {
                    "model_state_dict": snapshot_state_dict(eval_model),
                    "model_type": args.model,
                    "feature_columns": feature_cols,
                    "mean": mean,
                    "std": std,
                    "best_epoch": best_epoch,
                    "detector_target": args.detector_target,
                    "best_threshold": metrics.threshold,
                    "min_precision": args.min_precision,
                    "precision_metric": args.precision_metric,
                    "selection_metric": args.selection_metric,
                    "consensus_threshold": consensus_threshold,
                    "loss_type": args.loss_type,
                    "validation_model": validation_model_name,
                    "ema_decay": ema_decay,
                },
                out_root / "detector_best.pt",
            )
        else:
            bad_epochs += 1
            if patience > 0 and bad_epochs >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    if best_metrics is None or best_val_pred is None:
        raise RuntimeError("Detector training did not produce validation metrics")

    checkpoint_average_summary = None
    if checkpoint_avg_last_k > 1 and len(recent_state_dicts) >= 2:
        avg_epochs = [epoch_idx for epoch_idx, _ in recent_state_dicts]
        avg_state_dict = average_state_dicts([state for _, state in recent_state_dicts])
        averaged_model = build_sequence_model(args.model, input_dim=len(feature_cols), cfg=cfg, output_dim=1).to(device)
        averaged_model.load_state_dict(avg_state_dict)
        avg_val_pred = predict_detector(averaged_model, val_loader, device=device)
        avg_sequence_scores, avg_sequence_union, avg_sequence_frequency = detector_sequence_maps(avg_val_pred)
        avg_metrics = search_union_frequency_threshold(
            sequence_scores=avg_sequence_scores,
            sequence_union_labels=avg_sequence_union,
            sequence_frequency_targets=avg_sequence_frequency,
            thresholds=thresholds,
            tolerance=tolerance,
            min_distance=min_distance,
            min_precision=float(args.min_precision),
            consensus_threshold=consensus_threshold,
            prominence=prominence,
            primary_metric=str(args.selection_metric),
            precision_metric=str(args.precision_metric),
            min_union_precision=float(precision_floors["union_precision"]),
            min_frequency_weighted_precision=float(precision_floors["frequency_weighted_precision"]),
            min_consensus_precision=float(precision_floors["consensus_precision"]),
        )
        avg_selected_precision = precision_metric_value(avg_metrics, str(args.precision_metric))
        avg_precision_floor_met = precision_floors_met(avg_metrics, precision_floors)
        avg_primary_metric = primary_metric_value(avg_metrics, str(args.selection_metric))
        avg_key = (
            float(avg_precision_floor_met),
            avg_primary_metric if avg_precision_floor_met else avg_selected_precision,
            avg_selected_precision,
            avg_metrics.union_precision,
            avg_metrics.weighted_recall,
            avg_metrics.consensus_recall,
            avg_metrics.union_f1,
            -float(avg_metrics.mean_offset or 1e9),
        )
        checkpoint_average_summary = {
            "epochs": avg_epochs,
            "union_metrics": union_metrics_to_dict(avg_metrics),
            "precision_floor_met": bool(avg_precision_floor_met),
            "selected": bool(avg_key > best_key),
        }
        torch.save(
            {
                "model_state_dict": avg_state_dict,
                "model_type": args.model,
                "feature_columns": feature_cols,
                "mean": mean,
                "std": std,
                "averaged_epochs": avg_epochs,
                "detector_target": args.detector_target,
                "best_threshold": avg_metrics.threshold,
                "min_precision": args.min_precision,
                "consensus_threshold": consensus_threshold,
                "loss_type": args.loss_type,
                "scheduler_type": scheduler_type,
                "checkpoint_avg_last_k": checkpoint_avg_last_k,
                "precision_metric": args.precision_metric,
                "selection_metric": args.selection_metric,
                "ema_decay": ema_decay,
            },
            out_root / "detector_lastk_avg.pt",
        )
        if avg_key > best_key:
            best_key = avg_key
            best_epoch = int(avg_epochs[-1])
            best_metrics = avg_metrics
            best_val_pred = avg_val_pred.copy()
            best_validation_model = "lastk_average"
            torch.save(
                {
                    "model_state_dict": avg_state_dict,
                    "model_type": args.model,
                    "feature_columns": feature_cols,
                    "mean": mean,
                    "std": std,
                    "best_epoch": best_epoch,
                    "detector_target": args.detector_target,
                    "best_threshold": avg_metrics.threshold,
                    "min_precision": args.min_precision,
                    "consensus_threshold": consensus_threshold,
                    "loss_type": args.loss_type,
                    "scheduler_type": scheduler_type,
                    "checkpoint_avg_last_k": checkpoint_avg_last_k,
                    "precision_metric": args.precision_metric,
                    "selection_metric": args.selection_metric,
                    "ema_decay": ema_decay,
                    "selected_from_lastk_average": True,
                },
                out_root / "detector_best.pt",
            )

    train_df = piece_df[piece_df["protocol_split"] == "train"].copy()
    val_df = piece_df[piece_df["protocol_split"] == "val"].copy()
    oracle_stage_grading = None
    class_event_metrics = None
    if args.skip_stage_grading:
        merged_val = best_val_pred.copy()
    else:
        oracle_stage_grading, graded_val = train_stage_grader(
            train_df=train_df,
            val_df=val_df,
            feature_cols=grader_feature_cols,
            seed=seed,
        )
        prob_cols = [col for col in graded_val.columns if col.startswith("pred_stage_prob_")]
        merged_val = best_val_pred.merge(
            graded_val[["piece_sample_id", "piece_id", "beat_idx", "dominant_stage", "pred_stage_class", *prob_cols]].rename(
                columns={"piece_sample_id": "sample_id"}
            ),
            on=["sample_id", "piece_id", "beat_idx"],
            how="left",
            validate="one_to_one",
        )

        sequence_scores = {}
        sequence_pred_labels = {}
        sequence_true_labels = {}
        for sample_id, group in merged_val.sort_values(["sample_id", "beat_idx"]).groupby("sample_id", sort=False):
            sequence_scores[sample_id] = group["detector_score"].to_numpy(dtype=np.float32)
            sequence_pred_labels[sample_id] = group["pred_stage_class"].fillna(0).to_numpy(dtype=np.int32)
            sequence_true_labels[sample_id] = group["dominant_stage"].fillna(0).to_numpy(dtype=np.int32)

        class_event_metrics = evaluate_labeled_event_sequences(
            sequence_scores=sequence_scores,
            sequence_pred_labels=sequence_pred_labels,
            sequence_true_labels=sequence_true_labels,
            positive_classes=(1, 2, 3),
            threshold=float(best_metrics.threshold),
            tolerance=tolerance,
            min_distance=min_distance,
            prominence=prominence,
        )

    merged_val.to_csv(out_root / "val_predictions.csv.gz", index=False, compression="gzip")
    np.savez(out_root / "detector_scaler_stats.npz", mean=mean, std=std)

    summary = {
        "table_path": str(table_path),
        "heldout_pieces": list(args.heldout_piece),
        "train_piece_count": int(train_df["piece_id"].nunique()),
        "val_piece_count": int(val_df["piece_id"].nunique()),
        "train_sequence_count": int(train_df["piece_sample_id"].nunique()),
        "val_sequence_count": int(val_df["piece_sample_id"].nunique()),
        "model_type": args.model,
        "detector_target": args.detector_target,
        "model_hparams": {
            "transformer_dim": seq_cfg.get("transformer_dim"),
            "transformer_heads": seq_cfg.get("transformer_heads"),
            "transformer_layers": seq_cfg.get("transformer_layers"),
            "transformer_ff_dim": seq_cfg.get("transformer_ff_dim"),
        },
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
        "selection_metric": str(args.selection_metric),
        "precision_metric": str(args.precision_metric),
        "precision_floors": {key: float(value) for key, value in precision_floors.items()},
        "loss_type": str(args.loss_type),
        "scheduler_type": str(scheduler_type),
        "checkpoint_avg_last_k": int(checkpoint_avg_last_k),
        "ema_decay": float(ema_decay),
        "cumulative_merge_tolerance": int(args.cumulative_merge_tolerance),
        "cumulative_component_weights": cumulative_component_weights,
        "rest_span_label_mode": str(args.rest_span_label_mode),
        "rest_span_min_len": int(args.rest_span_min_len),
        "rest_span_source_col": str(args.rest_span_source_col),
        "rest_span_source_threshold": float(args.rest_span_source_threshold),
        "rest_span_tolerance_negative_weight": float(args.rest_span_tolerance_negative_weight),
        "min_train_frequency_target": float(args.min_train_frequency_target),
        "skip_stage_grading": bool(args.skip_stage_grading),
        "precision_floor_met": bool(precision_floors_met(best_metrics, precision_floors)),
        "best_validation_model": str(best_validation_model),
        "union_metrics": union_metrics_to_dict(best_metrics),
        "feature_columns": feature_cols,
        "grader_feature_columns": grader_feature_cols,
        "history": history,
    }
    if oracle_stage_grading is not None:
        summary["oracle_stage_grading"] = {
            "target": "dominant_stage",
            **oracle_stage_grading,
        }
    if checkpoint_average_summary is not None:
        summary["checkpoint_average"] = checkpoint_average_summary
    if class_event_metrics is not None:
        summary["end_to_end_stage"] = labeled_metrics_to_dict(class_event_metrics)
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    predicted_events = build_predicted_event_frame(
        pred_df=merged_val,
        threshold=float(best_metrics.threshold),
        min_distance=min_distance,
        prominence=prominence,
        tolerance=tolerance,
    )
    predicted_events.to_csv(out_root / "predicted_events.csv.gz", index=False, compression="gzip")
    print(
        f"Held-out {heldout_slug} | union_precision={best_metrics.union_precision:.4f} | "
        f"weighted_recall={best_metrics.weighted_recall:.4f} | consensus_recall={best_metrics.consensus_recall:.4f}"
    )
    if oracle_stage_grading is not None and class_event_metrics is not None:
        print(
            f"  oracle_stage_macro_f1={oracle_stage_grading['macro_f1']:.4f} | "
            f"end_to_end_stage_macro_f1={class_event_metrics.macro_f1:.4f}"
        )
        print(
            "  end_to_end_stage_class_f1="
            f"low:{class_event_metrics.class_f1.get(1, 0.0):.4f},"
            f"mid:{class_event_metrics.class_f1.get(2, 0.0):.4f},"
            f"high:{class_event_metrics.class_f1.get(3, 0.0):.4f}"
        )


if __name__ == "__main__":
    main()
