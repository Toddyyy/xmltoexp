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
from boundary_restart.metrics import search_best_threshold
from boundary_restart.models import build_sequence_model
from boundary_restart.table_io import feature_columns, load_table, samples_from_table


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_soft_target_weights(seq_cfg: dict) -> np.ndarray | None:
    weights = seq_cfg.get("soft_target_weights")
    if not weights:
        return None
    arr = np.asarray(weights, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return None
    if not np.isclose(arr[0], 1.0):
        arr[0] = 1.0
    return arr


def soften_binary_labels(labels: np.ndarray, weights: np.ndarray | None) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.float32)
    if weights is None or weights.size <= 1:
        return labels.astype(np.float32, copy=True)

    peak_idx = np.flatnonzero(labels > 0.5)
    if peak_idx.size == 0:
        return labels.astype(np.float32, copy=True)

    soft = np.zeros_like(labels, dtype=np.float32)
    radius = int(weights.size) - 1
    for center in peak_idx.tolist():
        start = max(0, center - radius)
        end = min(labels.shape[0], center + radius + 1)
        for idx in range(start, end):
            offset = abs(idx - center)
            soft[idx] = max(soft[idx], float(weights[offset]))
    return np.clip(soft, 0.0, 1.0).astype(np.float32)


class SequenceDataset(Dataset):
    def __init__(
        self,
        samples: list[dict],
        mean: np.ndarray,
        std: np.ndarray,
        soft_target_weights: np.ndarray | None = None,
    ):
        self.samples = samples
        self.mean = mean.astype(np.float32)
        self.std = std.astype(np.float32)
        self.soft_target_weights = soft_target_weights

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        features = (sample["features"] - self.mean) / self.std
        labels = sample["labels"].astype(np.float32)
        return {
            "sample_id": sample["sample_id"],
            "piece_id": sample["piece_id"],
            "features": features.astype(np.float32),
            "labels": labels,
            "targets": soften_binary_labels(labels, self.soft_target_weights),
            "length": int(sample["features"].shape[0]),
        }


def collate_sequences(batch: list[dict]) -> dict:
    lengths = [item["length"] for item in batch]
    max_len = max(lengths)
    feat_dim = batch[0]["features"].shape[1]
    features = torch.zeros(len(batch), max_len, feat_dim, dtype=torch.float32)
    labels = torch.zeros(len(batch), max_len, dtype=torch.float32)
    targets = torch.zeros(len(batch), max_len, dtype=torch.float32)
    mask = torch.zeros(len(batch), max_len, dtype=torch.bool)
    lengths = torch.tensor(lengths, dtype=torch.int64)
    sample_ids = []
    piece_ids = []

    for idx, item in enumerate(batch):
        length = item["length"]
        features[idx, :length] = torch.from_numpy(item["features"])
        labels[idx, :length] = torch.from_numpy(item["labels"])
        targets[idx, :length] = torch.from_numpy(item["targets"])
        mask[idx, :length] = True
        sample_ids.append(item["sample_id"])
        piece_ids.append(item["piece_id"])

    return {
        "features": features,
        "labels": labels,
        "targets": targets,
        "mask": mask,
        "lengths": lengths,
        "sample_ids": sample_ids,
        "piece_ids": piece_ids,
    }


def compute_normalizer(samples: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    stacked = np.concatenate([sample["features"] for sample in samples], axis=0)
    mean = stacked.mean(axis=0).astype(np.float32)
    std = stacked.std(axis=0).astype(np.float32)
    std[std < 1e-6] = 1.0
    return mean, std


def train_one_epoch(model, loader, optimizer, device, loss_fn, grad_clip, log_interval: int = 0):
    model.train()
    total_loss = 0.0
    total_tokens = 0
    for batch_idx, batch in enumerate(loader, start=1):
        features = batch["features"].to(device)
        targets = batch["targets"].to(device)
        mask = batch["mask"].to(device)
        lengths = batch["lengths"].to(device)

        optimizer.zero_grad()
        logits = model(features, lengths=lengths)
        loss = loss_fn(logits, targets)
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
def predict_sequences(model, loader, device) -> pd.DataFrame:
    model.eval()
    rows = []
    for batch in loader:
        features = batch["features"].to(device)
        labels = batch["labels"].to(device)
        mask = batch["mask"].to(device)
        lengths = batch["lengths"].to(device)
        logits = model(features, lengths=lengths)
        probs = torch.sigmoid(logits)
        for batch_idx, sample_id in enumerate(batch["sample_ids"]):
            length = int(mask[batch_idx].sum().item())
            sample_scores = probs[batch_idx, :length].cpu().numpy()
            sample_labels = labels[batch_idx, :length].cpu().numpy()
            for beat_idx, (score, label) in enumerate(zip(sample_scores.tolist(), sample_labels.tolist())):
                rows.append(
                    {
                        "sample_id": sample_id,
                        "piece_id": batch["piece_ids"][batch_idx],
                        "beat_idx": beat_idx,
                        "boundary_peak": float(label),
                        "score": float(score),
                    }
                )
    return pd.DataFrame(rows)


def to_sequence_maps(pred_df: pd.DataFrame) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    scores = {}
    labels = {}
    for sample_id, group in pred_df.sort_values(["sample_id", "beat_idx"]).groupby("sample_id", sort=False):
        scores[sample_id] = group["score"].to_numpy(dtype=np.float32)
        labels[sample_id] = group["boundary_peak"].to_numpy(dtype=np.float32)
    return scores, labels


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


def main():
    parser = argparse.ArgumentParser(description="Train a small beat-level sequence model.")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--table_path", default=None, help="Optional override for beat table path")
    parser.add_argument("--output_dir", default=None, help="Optional override for output directory")
    parser.add_argument("--model", choices=["bilstm", "tcn"], default=None, help="Optional model override")
    parser.add_argument("--device", default="auto", help="cpu|cuda|auto")
    parser.add_argument("--epochs", type=int, default=None, help="Optional epoch override")
    parser.add_argument("--batch_size", type=int, default=None, help="Optional batch size override")
    parser.add_argument("--seed", type=int, default=None, help="Optional seed override")
    parser.add_argument("--log_interval", type=int, default=0, help="Print running loss every N training batches")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seq_cfg = cfg.get("sequence", {})
    eval_cfg = cfg.get("evaluation", {})
    data_cfg = cfg.get("data", {})

    model_type = args.model or seq_cfg.get("model_type", "bilstm")
    epochs = int(args.epochs or seq_cfg.get("epochs", 30))
    seed = int(args.seed if args.seed is not None else seq_cfg.get("seed", 42))
    set_seed(seed)
    soft_target_weights = parse_soft_target_weights(seq_cfg)

    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    table_path = resolve_path(cfg, args.table_path or data_cfg["beat_table_path"])
    out_root = resolve_path(cfg, args.output_dir or seq_cfg["output_dir"]) / model_type
    out_root.mkdir(parents=True, exist_ok=True)

    df = load_table(table_path)
    df = df[df["split"].isin(["train", "val"])].copy()
    cols = select_feature_columns(cfg, feature_columns(df))

    train_samples = samples_from_table(df, cols, split="train")
    val_samples = samples_from_table(df, cols, split="val")
    if not train_samples or not val_samples:
        raise ValueError("Both train and val splits must be non-empty")

    mean, std = compute_normalizer(train_samples)
    train_ds = SequenceDataset(
        train_samples,
        mean=mean,
        std=std,
        soft_target_weights=soft_target_weights,
    )
    val_ds = SequenceDataset(
        val_samples,
        mean=mean,
        std=std,
        soft_target_weights=soft_target_weights,
    )

    batch_size = int(args.batch_size or seq_cfg.get("batch_size", 8))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_sequences)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_sequences)

    model = build_sequence_model(model_type=model_type, input_dim=len(cols), cfg=cfg).to(device)
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
    patience = int(seq_cfg.get("early_stop_patience", 5))
    grad_clip = float(seq_cfg.get("grad_clip", 1.0))

    best_f1 = -1.0
    best_epoch = 0
    best_metrics = None
    epochs_without_improve = 0
    history = []

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
        val_pred = predict_sequences(model, val_loader, device=device)
        sequence_scores, sequence_labels = to_sequence_maps(val_pred)
        best = search_best_threshold(
            sequence_scores=sequence_scores,
            sequence_labels=sequence_labels,
            thresholds=thresholds,
            tolerance=tolerance,
            min_distance=min_distance,
            prominence=prominence,
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_average_precision": best.average_precision,
                "event_f1": best.f1,
                "best_threshold": best.threshold,
            }
        )
        print(
            f"Epoch {epoch}/{epochs} | train_loss {train_loss:.4f} | "
            f"val_ap {best.average_precision:.4f} | val_event_f1 {best.f1:.4f}"
        )

        if best.f1 > best_f1:
            best_f1 = best.f1
            best_epoch = epoch
            best_metrics = best
            epochs_without_improve = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_type": model_type,
                    "feature_columns": cols,
                    "mean": mean,
                    "std": std,
                    "best_epoch": best_epoch,
                    "best_threshold": best.threshold,
                },
                out_root / "best.pt",
            )
            val_pred.to_csv(out_root / "val_predictions.csv.gz", index=False, compression="gzip")
        else:
            epochs_without_improve += 1
            if patience > 0 and epochs_without_improve >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    if best_metrics is None:
        raise RuntimeError("No validation metrics were produced")

    summary = {
        "model_type": model_type,
        "table_path": str(table_path),
        "device": str(device),
        "seed": seed,
        "epochs_run": len(history),
        "best_epoch": best_epoch,
        "best_threshold": best_metrics.threshold,
        "val_average_precision": best_metrics.average_precision,
        "event_precision": best_metrics.precision,
        "event_recall": best_metrics.recall,
        "event_f1": best_metrics.f1,
        "mean_offset": best_metrics.mean_offset,
        "matches": best_metrics.matches,
        "pred_events": best_metrics.pred_events,
        "true_events": best_metrics.true_events,
        "soft_target_weights": None if soft_target_weights is None else soft_target_weights.tolist(),
        "feature_columns": cols,
        "history": history,
    }
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    np.savez(out_root / "scaler_stats.npz", mean=mean, std=std)
    print(f"Best epoch: {best_epoch} | event_f1={best_metrics.f1:.4f} | threshold={best_metrics.threshold:.3f}")


if __name__ == "__main__":
    main()
