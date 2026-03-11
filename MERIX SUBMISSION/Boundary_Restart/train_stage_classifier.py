#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import random

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

from boundary_restart.config import load_config, resolve_path, threshold_grid
from boundary_restart.metrics import search_best_threshold
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
    return selected


class StageDataset(Dataset):
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
            "features": features.astype(np.float32),
            "labels": sample["labels"].astype(np.int64),
            "length": int(sample["labels"].shape[0]),
        }


def collate_stage(batch: list[dict]) -> dict:
    lengths = [item["length"] for item in batch]
    max_len = max(lengths)
    feat_dim = batch[0]["features"].shape[1]
    features = torch.zeros(len(batch), max_len, feat_dim, dtype=torch.float32)
    labels = torch.zeros(len(batch), max_len, dtype=torch.int64)
    mask = torch.zeros(len(batch), max_len, dtype=torch.bool)
    sample_ids = []
    piece_ids = []

    for idx, item in enumerate(batch):
        length = item["length"]
        features[idx, :length] = torch.from_numpy(item["features"])
        labels[idx, :length] = torch.from_numpy(item["labels"])
        mask[idx, :length] = True
        sample_ids.append(item["sample_id"])
        piece_ids.append(item["piece_id"])

    return {
        "features": features,
        "labels": labels,
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


def samples_from_stage_table(df: pd.DataFrame, feature_cols: list[str], split: str, target_col: str) -> list[dict]:
    subset = df[df["split"] == split].copy().sort_values(["sample_id", "beat_idx"])
    samples = []
    for sample_id, group in subset.groupby("sample_id", sort=False):
        samples.append(
            {
                "sample_id": sample_id,
                "piece_id": group["piece_id"].iloc[0],
                "features": group[feature_cols].to_numpy(dtype=np.float32),
                "labels": group[target_col].to_numpy(dtype=np.int64),
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
        mask = batch["mask"].to(device)
        lengths = batch["lengths"].to(device)

        optimizer.zero_grad()
        logits = model(features, lengths=lengths)
        loss = loss_fn(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
        loss = loss.reshape(labels.shape)
        loss = (loss * mask.float()).sum() / mask.float().sum().clamp(min=1.0)
        loss.backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += float(loss.item()) * int(mask.sum().item())
        total_tokens += int(mask.sum().item())
        if log_interval > 0 and batch_idx % log_interval == 0:
            running_loss = total_loss / max(total_tokens, 1)
            print(f"  step {batch_idx}/{len(loader)} | running_loss {running_loss:.4f}")
    return total_loss / max(total_tokens, 1)


@torch.no_grad()
def predict(model, loader, device) -> pd.DataFrame:
    model.eval()
    rows = []
    for batch in loader:
        features = batch["features"].to(device)
        labels = batch["labels"].to(device)
        mask = batch["mask"].to(device)
        lengths = batch["lengths"].to(device)
        logits = model(features, lengths=lengths)
        probs = torch.softmax(logits, dim=-1)
        pred_class = probs.argmax(dim=-1)
        boundary_score = 1.0 - probs[..., 0]
        for batch_idx, sample_id in enumerate(batch["sample_ids"]):
            length = int(mask[batch_idx].sum().item())
            sample_probs = probs[batch_idx, :length].cpu().numpy()
            sample_labels = labels[batch_idx, :length].cpu().numpy()
            sample_pred = pred_class[batch_idx, :length].cpu().numpy()
            sample_boundary_score = boundary_score[batch_idx, :length].cpu().numpy()
            for beat_idx in range(length):
                row = {
                    "sample_id": sample_id,
                    "piece_id": batch["piece_ids"][batch_idx],
                    "beat_idx": beat_idx,
                    "stage_class": int(sample_labels[beat_idx]),
                    "pred_class": int(sample_pred[beat_idx]),
                    "boundary_score": float(sample_boundary_score[beat_idx]),
                }
                for class_idx in range(sample_probs.shape[1]):
                    row[f"prob_{class_idx}"] = float(sample_probs[beat_idx, class_idx])
                rows.append(row)
    return pd.DataFrame(rows)


def stage_metrics(pred_df: pd.DataFrame, labels: list[int]) -> dict:
    y_true = pred_df["stage_class"].to_numpy(dtype=np.int64)
    y_pred = pred_df["pred_class"].to_numpy(dtype=np.int64)
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "class_f1": {str(i): float(report[str(i)]["f1-score"]) for i in labels},
        "class_precision": {str(i): float(report[str(i)]["precision"]) for i in labels},
        "class_recall": {str(i): float(report[str(i)]["recall"]) for i in labels},
        "class_support": {str(i): int(report[str(i)]["support"]) for i in labels},
    }


def build_boundary_sequence_maps(pred_df: pd.DataFrame) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    scores = {}
    labels = {}
    ordered = pred_df.sort_values(["sample_id", "beat_idx"])
    for sample_id, group in ordered.groupby("sample_id", sort=False):
        scores[sample_id] = group["boundary_score"].to_numpy(dtype=np.float32)
        labels[sample_id] = (group["stage_class"].to_numpy(dtype=np.int64) > 0).astype(np.float32)
    return scores, labels


def main():
    parser = argparse.ArgumentParser(description="Train a 4-class stage classifier.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", choices=["bilstm", "tcn"], default="tcn")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--log_interval", type=int, default=0)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--target_col", default="stage_class")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seq_cfg = cfg.get("sequence", {})
    eval_cfg = cfg.get("evaluation", {})
    data_cfg = cfg.get("data", {})

    seed = int(args.seed if args.seed is not None else seq_cfg.get("seed", 42))
    set_seed(seed)
    device = resolve_device(args.device)
    table_path = resolve_path(cfg, data_cfg["beat_table_path"])
    out_root = resolve_path(cfg, args.output_dir or seq_cfg["output_dir"]) / f"{args.model}_{args.target_col}"
    out_root.mkdir(parents=True, exist_ok=True)

    df = load_table(table_path)
    df = df[df["split"].isin(["train", "val"])].copy()
    feature_cols = select_feature_columns(cfg, feature_columns(df))
    train_samples = samples_from_stage_table(df, feature_cols, split="train", target_col=args.target_col)
    val_samples = samples_from_stage_table(df, feature_cols, split="val", target_col=args.target_col)

    mean, std = compute_normalizer(train_samples)
    train_ds = StageDataset(train_samples, mean=mean, std=std)
    val_ds = StageDataset(val_samples, mean=mean, std=std)
    batch_size = int(args.batch_size or seq_cfg.get("batch_size", 64))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_stage)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_stage)

    train_labels = np.concatenate([sample["labels"] for sample in train_samples], axis=0)
    num_classes = int(train_labels.max()) + 1
    labels = list(range(num_classes))

    model = build_sequence_model(args.model, input_dim=len(feature_cols), cfg=cfg, output_dim=num_classes).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(seq_cfg.get("lr", 1e-3)),
        weight_decay=float(seq_cfg.get("weight_decay", 1e-4)),
    )

    counts = np.bincount(train_labels, minlength=num_classes).astype(np.float32)
    counts[counts < 1] = 1.0
    class_weights = torch.tensor(counts.sum() / counts, device=device, dtype=torch.float32)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights, reduction="none")

    thresholds = threshold_grid(cfg)
    tolerance = int(eval_cfg.get("event_tolerance", 1))
    min_distance = int(eval_cfg.get("min_distance", 6))
    prominence = float(eval_cfg.get("prominence", 0.0))
    epochs = int(args.epochs or seq_cfg.get("epochs", 30))
    patience = int(seq_cfg.get("early_stop_patience", 5))
    grad_clip = float(seq_cfg.get("grad_clip", 1.0))

    best_epoch = 0
    best_key = None
    best_summary = None
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
        val_pred = predict(model, val_loader, device=device)
        cls_metrics = stage_metrics(val_pred, labels=labels)
        sequence_scores, sequence_labels = build_boundary_sequence_maps(val_pred)
        event_metrics = search_best_threshold(
            sequence_scores=sequence_scores,
            sequence_labels=sequence_labels,
            thresholds=thresholds,
            tolerance=tolerance,
            min_distance=min_distance,
            prominence=prominence,
        )
        history_item = {
            "epoch": epoch,
            "train_loss": train_loss,
            "accuracy": cls_metrics["accuracy"],
            "macro_f1": cls_metrics["macro_f1"],
            "weighted_f1": cls_metrics["weighted_f1"],
            "boundary_event_f1": event_metrics.f1,
        }
        history.append(history_item)
        print(
            f"Epoch {epoch}/{epochs} | train_loss {train_loss:.4f} | "
            f"macro_f1 {cls_metrics['macro_f1']:.4f} | boundary_event_f1 {event_metrics.f1:.4f}"
        )

        current_key = (cls_metrics["macro_f1"], cls_metrics["weighted_f1"], event_metrics.f1)
        if best_key is None or current_key > best_key:
            best_key = current_key
            best_epoch = epoch
            bad_epochs = 0
            best_summary = {
                **cls_metrics,
                "boundary_event_precision": event_metrics.precision,
                "boundary_event_recall": event_metrics.recall,
                "boundary_event_f1": event_metrics.f1,
                "boundary_event_ap": event_metrics.average_precision,
                "boundary_best_threshold": event_metrics.threshold,
            }
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_type": args.model,
                    "feature_columns": feature_cols,
                    "mean": mean,
                    "std": std,
                    "best_epoch": best_epoch,
                    "target_col": args.target_col,
                },
                out_root / "best.pt",
            )
            val_pred.to_csv(out_root / "val_predictions.csv.gz", index=False, compression="gzip")
        else:
            bad_epochs += 1
            if patience > 0 and bad_epochs >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    if best_summary is None:
        raise RuntimeError("No validation metrics were produced")

    summary = {
        "model_type": args.model,
        "table_path": str(table_path),
        "target_col": args.target_col,
        "device": str(device),
        "seed": seed,
        "num_classes": num_classes,
        "epochs_run": len(history),
        "best_epoch": best_epoch,
        **best_summary,
        "feature_columns": feature_cols,
        "history": history,
    }
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        f"Best epoch: {best_epoch} | macro_f1={best_summary['macro_f1']:.4f} | "
        f"weighted_f1={best_summary['weighted_f1']:.4f}"
    )


if __name__ == "__main__":
    main()
