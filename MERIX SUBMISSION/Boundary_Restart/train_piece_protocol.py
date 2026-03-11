#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
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
from boundary_restart.metrics import (
    evaluate_labeled_event_sequences,
    search_threshold_with_min_precision,
)
from boundary_restart.models import build_sequence_model
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
    raise ValueError(f"Unsupported detector target mode: {mode}")


def build_loss_weights(
    labels: np.ndarray,
    hard_negative_radius: int,
    hard_negative_weight: float,
    easy_negative_weight: float,
) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.float32)
    weights = np.ones_like(labels, dtype=np.float32)
    neg_mask = labels < 0.5
    if np.any(neg_mask):
        weights[neg_mask] = float(easy_negative_weight)
    if hard_negative_radius <= 0 or hard_negative_weight <= easy_negative_weight:
        return weights
    pos_idx = np.flatnonzero(labels > 0.5)
    if pos_idx.size == 0:
        return weights
    hard_mask = np.zeros_like(labels, dtype=bool)
    radius = int(hard_negative_radius)
    for center in pos_idx.tolist():
        start = max(0, center - radius)
        end = min(labels.shape[0], center + radius + 1)
        hard_mask[start:end] = True
    hard_mask &= neg_mask
    weights[hard_mask] = float(hard_negative_weight)
    return weights.astype(np.float32)


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


class DetectorDataset(Dataset):
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
            "labels": sample["labels"].astype(np.float32),
            "loss_weights": build_loss_weights(
                sample["labels"],
                hard_negative_radius=self.hard_negative_radius,
                hard_negative_weight=self.hard_negative_weight,
                easy_negative_weight=self.easy_negative_weight,
            ),
            "stage_class": sample["stage_class"].astype(np.int64),
            "stage_class_midhigh": sample["stage_class_midhigh"].astype(np.int64),
            "length": int(sample["labels"].shape[0]),
        }


def collate_detector(batch: list[dict]) -> dict:
    lengths = [item["length"] for item in batch]
    max_len = max(lengths)
    feat_dim = batch[0]["features"].shape[1]
    features = torch.zeros(len(batch), max_len, feat_dim, dtype=torch.float32)
    labels = torch.zeros(len(batch), max_len, dtype=torch.float32)
    loss_weights = torch.ones(len(batch), max_len, dtype=torch.float32)
    stage_class = torch.zeros(len(batch), max_len, dtype=torch.int64)
    stage_class_midhigh = torch.zeros(len(batch), max_len, dtype=torch.int64)
    beat_idx = torch.zeros(len(batch), max_len, dtype=torch.int64)
    mask = torch.zeros(len(batch), max_len, dtype=torch.bool)
    sample_ids = []
    piece_ids = []

    for idx, item in enumerate(batch):
        length = item["length"]
        features[idx, :length] = torch.from_numpy(item["features"])
        labels[idx, :length] = torch.from_numpy(item["labels"])
        loss_weights[idx, :length] = torch.from_numpy(item["loss_weights"])
        stage_class[idx, :length] = torch.from_numpy(item["stage_class"])
        stage_class_midhigh[idx, :length] = torch.from_numpy(item["stage_class_midhigh"])
        beat_idx[idx, :length] = torch.from_numpy(item["beat_idx"])
        mask[idx, :length] = True
        sample_ids.append(item["sample_id"])
        piece_ids.append(item["piece_id"])

    return {
        "features": features,
        "labels": labels,
        "loss_weights": loss_weights,
        "stage_class": stage_class,
        "stage_class_midhigh": stage_class_midhigh,
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


def detector_samples_from_table(
    df: pd.DataFrame,
    feature_cols: list[str],
    split: str,
    target_mode: str,
) -> list[dict]:
    subset = df[df["protocol_split"] == split].copy().sort_values(["sample_id", "beat_idx"])
    samples = []
    for sample_id, group in subset.groupby("sample_id", sort=False):
        stage = group["stage_class"].to_numpy(dtype=np.int64)
        samples.append(
            {
                "sample_id": sample_id,
                "piece_id": group["piece_id"].iloc[0],
                "beat_idx": group["beat_idx"].to_numpy(dtype=np.int32),
                "features": group[feature_cols].to_numpy(dtype=np.float32),
                "labels": detector_labels(stage, mode=target_mode),
                "stage_class": stage,
                "stage_class_midhigh": group["stage_class_midhigh"].to_numpy(dtype=np.int64),
            }
        )
    return samples


def train_one_epoch(model, loader, optimizer, device, loss_fn, grad_clip: float, log_interval: int = 0) -> float:
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
        loss = loss_fn(logits, labels)
        token_weights = mask.float() * loss_weights
        loss = (loss * token_weights).sum() / token_weights.sum().clamp(min=1.0)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += float(loss.item()) * int(mask.sum().item())
        total_tokens += int(mask.sum().item())
        if log_interval > 0 and batch_idx % log_interval == 0:
            running_loss = total_loss / max(total_tokens, 1)
            print(f"  step {batch_idx}/{len(loader)} | running_loss {running_loss:.4f}")
    return total_loss / max(total_tokens, 1)


@torch.no_grad()
def predict_detector(model, loader, device) -> pd.DataFrame:
    model.eval()
    rows = []
    for batch in loader:
        features = batch["features"].to(device)
        labels = batch["labels"].to(device)
        stage_class = batch["stage_class"].to(device)
        stage_class_midhigh = batch["stage_class_midhigh"].to(device)
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
                        "detector_target": float(labels[batch_idx_i, pos].item()),
                        "stage_class": int(stage_class[batch_idx_i, pos].item()),
                        "stage_class_midhigh": int(stage_class_midhigh[batch_idx_i, pos].item()),
                        "detector_score": float(probs[batch_idx_i, pos].item()),
                    }
                )
    return pd.DataFrame(rows)


def detector_sequence_maps(pred_df: pd.DataFrame) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    sequence_scores = {}
    sequence_labels = {}
    ordered = pred_df.sort_values(["sample_id", "beat_idx"])
    for sample_id, group in ordered.groupby("sample_id", sort=False):
        sequence_scores[sample_id] = group["detector_score"].to_numpy(dtype=np.float32)
        sequence_labels[sample_id] = group["detector_target"].to_numpy(dtype=np.float32)
    return sequence_scores, sequence_labels


def grading_report(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    report = classification_report(y_true, y_pred, labels=[1, 2], output_dict=True, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=[1, 2], average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=[1, 2], average="weighted", zero_division=0)),
        "class_f1": {label: float(report[str(label)]["f1-score"]) for label in [1, 2]},
        "class_precision": {label: float(report[str(label)]["precision"]) for label in [1, 2]},
        "class_recall": {label: float(report[str(label)]["recall"]) for label in [1, 2]},
        "class_support": {label: int(report[str(label)]["support"]) for label in [1, 2]},
    }


def train_oracle_grader(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: list[str],
    seed: int,
) -> tuple[dict, pd.DataFrame]:
    train_pos = train_df[train_df["stage_class_midhigh"] > 0].copy()
    val_pos = val_df[val_df["stage_class_midhigh"] > 0].copy()
    if train_pos.empty or val_pos.empty:
        raise ValueError("Oracle grading requires positive mid/high labels in both train and val splits")

    scaler = StandardScaler()
    x_train = scaler.fit_transform(train_pos[feature_cols].to_numpy(dtype=np.float32))
    y_train = (train_pos["stage_class_midhigh"].to_numpy(dtype=np.int64) - 1).astype(np.int64)
    x_val_pos = scaler.transform(val_pos[feature_cols].to_numpy(dtype=np.float32))
    y_val_pos = val_pos["stage_class_midhigh"].to_numpy(dtype=np.int64)

    clf = LogisticRegression(
        max_iter=4000,
        class_weight="balanced",
        random_state=seed,
    )
    clf.fit(x_train, y_train)

    oracle_pred = clf.predict(x_val_pos).astype(np.int64) + 1
    oracle_metrics = grading_report(y_true=y_val_pos, y_pred=oracle_pred)

    val_all = val_df.copy()
    x_val_all = scaler.transform(val_all[feature_cols].to_numpy(dtype=np.float32))
    pred_all = clf.predict(x_val_all).astype(np.int64) + 1
    prob_all = clf.predict_proba(x_val_all)
    val_all["pred_midhigh_class"] = pred_all.astype(np.int64)
    if prob_all.shape[1] == 2:
        val_all["pred_high_prob"] = prob_all[:, 1].astype(np.float32)
    else:
        val_all["pred_high_prob"] = 0.0
    return oracle_metrics, val_all


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


def main():
    parser = argparse.ArgumentParser(description="Piece-level high-precision boundary protocol with oracle and end-to-end mid/high grading.")
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
    parser.add_argument("--detector_target", choices=["any_boundary", "midhigh_boundary"], default="any_boundary")
    parser.add_argument("--min_precision", type=float, default=0.95)
    parser.add_argument("--hard_negative_radius", type=int, default=0)
    parser.add_argument("--hard_negative_weight", type=float, default=1.0)
    parser.add_argument("--easy_negative_weight", type=float, default=1.0)
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
            f"../outputs/piece_protocol/{heldout_slug}/{args.model}_{args.detector_target}_p{int(round(args.min_precision * 100))}{hardneg_suffix}",
        )
    out_root.mkdir(parents=True, exist_ok=True)

    df = load_table(table_path)
    df = apply_piece_protocol_split(df, heldout_pieces=args.heldout_piece, train_pieces=args.train_pieces)
    df = df[df["protocol_split"].isin(["train", "val"])].copy()
    all_feature_cols = feature_columns(df)
    feature_cols = select_feature_columns(cfg, all_feature_cols)
    grader_feature_cols = select_grader_feature_columns(cfg, feature_cols, all_feature_cols)

    train_samples = detector_samples_from_table(df, feature_cols, split="train", target_mode=args.detector_target)
    val_samples = detector_samples_from_table(df, feature_cols, split="val", target_mode=args.detector_target)
    if not train_samples or not val_samples:
        raise ValueError("Protocol split produced an empty train or val set")

    mean, std = compute_normalizer(train_samples)
    train_ds = DetectorDataset(
        train_samples,
        mean=mean,
        std=std,
        hard_negative_radius=int(args.hard_negative_radius),
        hard_negative_weight=float(args.hard_negative_weight),
        easy_negative_weight=float(args.easy_negative_weight),
    )
    val_ds = DetectorDataset(val_samples, mean=mean, std=std)
    batch_size = int(args.batch_size or seq_cfg.get("batch_size", 64))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_detector)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_detector)

    model = build_sequence_model(args.model, input_dim=len(feature_cols), cfg=cfg, output_dim=1).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(seq_cfg.get("lr", 1e-3)),
        weight_decay=float(seq_cfg.get("weight_decay", 1e-4)),
    )

    train_labels = np.concatenate([sample["labels"] for sample in train_samples], axis=0)
    pos = float(train_labels.sum())
    neg = float(train_labels.shape[0] - pos)
    pos_weight = torch.tensor([neg / max(pos, 1.0)], device=device, dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")

    thresholds = threshold_grid(cfg)
    tolerance = int(eval_cfg.get("event_tolerance", 1))
    min_distance = int(eval_cfg.get("min_distance", 6))
    prominence = float(eval_cfg.get("prominence", 0.0))
    epochs = int(args.epochs or seq_cfg.get("epochs", 30))
    patience = int(args.early_stop_patience if args.early_stop_patience is not None else seq_cfg.get("early_stop_patience", 5))
    grad_clip = float(seq_cfg.get("grad_clip", 1.0))

    best_epoch = 0
    best_key = None
    best_detector = None
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
            log_interval=max(int(args.log_interval), 0),
        )
        val_pred = predict_detector(model, val_loader, device=device)
        sequence_scores, sequence_labels = detector_sequence_maps(val_pred)
        detector_metrics = search_threshold_with_min_precision(
            sequence_scores=sequence_scores,
            sequence_labels=sequence_labels,
            thresholds=thresholds,
            tolerance=tolerance,
            min_distance=min_distance,
            min_precision=float(args.min_precision),
            prominence=prominence,
        )
        precision_floor_met = detector_metrics.precision >= float(args.min_precision)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "event_precision": detector_metrics.precision,
                "event_recall": detector_metrics.recall,
                "event_f1": detector_metrics.f1,
                "event_ap": detector_metrics.average_precision,
                "best_threshold": detector_metrics.threshold,
                "precision_floor_met": precision_floor_met,
            }
        )
        print(
            f"Epoch {epoch}/{epochs} | train_loss {train_loss:.4f} | "
            f"precision {detector_metrics.precision:.4f} | recall {detector_metrics.recall:.4f} | "
            f"event_f1 {detector_metrics.f1:.4f} | threshold {detector_metrics.threshold:.3f}"
        )

        current_key = (
            float(precision_floor_met),
            detector_metrics.recall if precision_floor_met else detector_metrics.precision,
            detector_metrics.precision,
            detector_metrics.f1,
            -float(detector_metrics.mean_offset or 1e9),
        )
        if best_key is None or current_key > best_key:
            best_key = current_key
            best_epoch = epoch
            best_detector = detector_metrics
            best_val_pred = val_pred.copy()
            bad_epochs = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_type": args.model,
                    "feature_columns": feature_cols,
                    "mean": mean,
                    "std": std,
                    "best_epoch": best_epoch,
                    "detector_target": args.detector_target,
                    "best_threshold": detector_metrics.threshold,
                    "min_precision": args.min_precision,
                },
                out_root / "detector_best.pt",
            )
        else:
            bad_epochs += 1
            if patience > 0 and bad_epochs >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    if best_detector is None or best_val_pred is None:
        raise RuntimeError("Detector training did not produce validation metrics")

    train_df = df[df["protocol_split"] == "train"].copy()
    val_df = df[df["protocol_split"] == "val"].copy()
    oracle_grading, graded_val = train_oracle_grader(
        train_df=train_df,
        val_df=val_df,
        feature_cols=grader_feature_cols,
        seed=seed,
    )

    merged_val = best_val_pred.merge(
        graded_val[
            [
                "sample_id",
                "piece_id",
                "beat_idx",
                "stage_class_midhigh",
                "pred_midhigh_class",
                "pred_high_prob",
            ]
        ],
        on=["sample_id", "piece_id", "beat_idx", "stage_class_midhigh"],
        how="left",
        validate="one_to_one",
    )
    merged_val.to_csv(out_root / "val_predictions.csv.gz", index=False, compression="gzip")

    sequence_scores = {}
    sequence_pred_labels = {}
    sequence_true_labels = {}
    for sample_id, group in merged_val.sort_values(["sample_id", "beat_idx"]).groupby("sample_id", sort=False):
        sequence_scores[sample_id] = group["detector_score"].to_numpy(dtype=np.float32)
        sequence_pred_labels[sample_id] = group["pred_midhigh_class"].to_numpy(dtype=np.int32)
        sequence_true_labels[sample_id] = group["stage_class_midhigh"].to_numpy(dtype=np.int32)

    class_event_metrics = evaluate_labeled_event_sequences(
        sequence_scores=sequence_scores,
        sequence_pred_labels=sequence_pred_labels,
        sequence_true_labels=sequence_true_labels,
        positive_classes=(1, 2),
        threshold=float(best_detector.threshold),
        tolerance=tolerance,
        min_distance=min_distance,
        prominence=prominence,
    )

    summary = {
        "table_path": str(table_path),
        "heldout_pieces": list(args.heldout_piece),
        "train_piece_count": int(train_df["piece_id"].nunique()),
        "val_piece_count": int(val_df["piece_id"].nunique()),
        "train_sample_count": int(train_df["sample_id"].nunique()),
        "val_sample_count": int(val_df["sample_id"].nunique()),
        "model_type": args.model,
        "detector_target": args.detector_target,
        "seed": seed,
        "device": str(device),
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "early_stop_patience": patience,
        "precision_floor": float(args.min_precision),
        "hard_negative_radius": int(args.hard_negative_radius),
        "hard_negative_weight": float(args.hard_negative_weight),
        "easy_negative_weight": float(args.easy_negative_weight),
        "precision_floor_met": bool(best_detector.precision >= float(args.min_precision)),
        "event_precision": best_detector.precision,
        "event_recall": best_detector.recall,
        "event_f1": best_detector.f1,
        "event_ap": best_detector.average_precision,
        "best_threshold": best_detector.threshold,
        "mean_offset": best_detector.mean_offset,
        "matches": best_detector.matches,
        "pred_events": best_detector.pred_events,
        "true_events": best_detector.true_events,
        "oracle_grading": {
            "target": "stage_class_midhigh",
            **oracle_grading,
        },
        "end_to_end_midhigh": labeled_metrics_to_dict(class_event_metrics),
        "feature_columns": feature_cols,
        "grader_feature_columns": grader_feature_cols,
        "history": history,
    }
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    np.savez(out_root / "detector_scaler_stats.npz", mean=mean, std=std)
    print(
        f"Held-out {heldout_slug} | precision={best_detector.precision:.4f} | "
        f"recall={best_detector.recall:.4f} | mid/high macro_event_f1={class_event_metrics.macro_f1:.4f}"
    )


if __name__ == "__main__":
    main()
