#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
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


def resolve_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class RegressionDataset(Dataset):
    def __init__(self, samples: list[dict], mean: np.ndarray, std: np.ndarray):
        self.samples = samples
        self.mean = mean.astype(np.float32)
        self.std = std.astype(np.float32)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        features = (sample["features"] - self.mean) / self.std
        targets = sample["labels"].astype(np.float32)
        return {
            "sample_id": sample["sample_id"],
            "piece_id": sample["piece_id"],
            "features": features.astype(np.float32),
            "targets": targets,
            "length": int(targets.shape[0]),
        }


def collate_regression(batch: list[dict]) -> dict:
    lengths = [item["length"] for item in batch]
    max_len = max(lengths)
    feat_dim = batch[0]["features"].shape[1]
    features = torch.zeros(len(batch), max_len, feat_dim, dtype=torch.float32)
    targets = torch.zeros(len(batch), max_len, dtype=torch.float32)
    mask = torch.zeros(len(batch), max_len, dtype=torch.bool)
    sample_ids = []
    piece_ids = []

    for idx, item in enumerate(batch):
        length = item["length"]
        features[idx, :length] = torch.from_numpy(item["features"])
        targets[idx, :length] = torch.from_numpy(item["targets"])
        mask[idx, :length] = True
        sample_ids.append(item["sample_id"])
        piece_ids.append(item["piece_id"])

    return {
        "features": features,
        "targets": targets,
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


def regression_loss(pred: torch.Tensor, target: torch.Tensor, loss_name: str) -> torch.Tensor:
    if loss_name == "mse":
        return F.mse_loss(pred, target, reduction="none")
    if loss_name == "huber":
        return F.huber_loss(pred, target, reduction="none", delta=0.1)
    raise ValueError(f"Unsupported loss: {loss_name}")


def train_one_epoch(model, loader, optimizer, device, loss_name: str, grad_clip: float, log_interval: int = 0) -> float:
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
        preds = torch.sigmoid(logits)
        loss = regression_loss(preds, targets, loss_name=loss_name)
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
        targets = batch["targets"].to(device)
        mask = batch["mask"].to(device)
        lengths = batch["lengths"].to(device)
        logits = model(features, lengths=lengths)
        preds = torch.sigmoid(logits)
        for batch_idx, sample_id in enumerate(batch["sample_ids"]):
            length = int(mask[batch_idx].sum().item())
            sample_scores = preds[batch_idx, :length].cpu().numpy()
            sample_targets = targets[batch_idx, :length].cpu().numpy()
            for beat_idx, (score, target) in enumerate(zip(sample_scores.tolist(), sample_targets.tolist())):
                rows.append(
                    {
                        "sample_id": sample_id,
                        "piece_id": batch["piece_ids"][batch_idx],
                        "beat_idx": beat_idx,
                        "target": float(target),
                        "binary_target": float(target > 0.0),
                        "score": float(score),
                    }
                )
    return pd.DataFrame(rows)


def safe_corr(fn, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size < 2:
        return 0.0
    if np.allclose(y_true, y_true[0]) or np.allclose(y_pred, y_pred[0]):
        return 0.0
    value = float(fn(y_true, y_pred).statistic)
    return 0.0 if np.isnan(value) else value


def compute_regression_metrics(target: np.ndarray, score: np.ndarray) -> dict[str, float | int | None]:
    mask = target > 0.0
    metrics = {
        "target_mean": float(target.mean()),
        "pred_mean": float(score.mean()),
        "mae": float(mean_absolute_error(target, score)),
        "rmse": float(np.sqrt(mean_squared_error(target, score))),
        "r2": float(r2_score(target, score)),
        "spearman": safe_corr(spearmanr, target, score),
        "pearson": safe_corr(pearsonr, target, score),
        "positive_rows": int(mask.sum()),
        "positive_mae": None,
        "positive_rmse": None,
        "positive_spearman": None,
        "positive_pearson": None,
    }
    if mask.any():
        metrics["positive_mae"] = float(mean_absolute_error(target[mask], score[mask]))
        metrics["positive_rmse"] = float(np.sqrt(mean_squared_error(target[mask], score[mask])))
        metrics["positive_spearman"] = safe_corr(spearmanr, target[mask], score[mask])
        metrics["positive_pearson"] = safe_corr(pearsonr, target[mask], score[mask])
    return metrics


def build_sequence_maps(pred_df: pd.DataFrame) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    scores = {}
    labels = {}
    for sample_id, group in pred_df.sort_values(["sample_id", "beat_idx"]).groupby("sample_id", sort=False):
        scores[sample_id] = group["score"].to_numpy(dtype=np.float32)
        labels[sample_id] = group["binary_target"].to_numpy(dtype=np.float32)
    return scores, labels


def main():
    parser = argparse.ArgumentParser(description="Train a sequence regressor for salience targets.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", choices=["bilstm", "tcn"], default="tcn")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--loss", choices=["mse", "huber"], default="huber")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--log_interval", type=int, default=0)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--target_col", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    seq_cfg = cfg.get("sequence", {})
    eval_cfg = cfg.get("evaluation", {})
    data_cfg = cfg.get("data", {})

    seed = int(args.seed if args.seed is not None else seq_cfg.get("seed", 42))
    set_seed(seed)
    device = resolve_device(args.device)
    target_col = str(args.target_col or data_cfg.get("target_column", "boundary_peak"))
    table_path = resolve_path(cfg, data_cfg["beat_table_path"])
    out_root = resolve_path(cfg, args.output_dir or seq_cfg["output_dir"]) / f"{args.model}_{args.loss}"
    out_root.mkdir(parents=True, exist_ok=True)

    df = load_table(table_path)
    df = df[df["split"].isin(["train", "val"])].copy()
    cols = select_feature_columns(cfg, feature_columns(df))

    train_samples = samples_from_table(df, cols, split="train", target_col=target_col, score_col=target_col)
    val_samples = samples_from_table(df, cols, split="val", target_col=target_col, score_col=target_col)
    if not train_samples or not val_samples:
        raise ValueError("Both train and val splits must be non-empty")

    mean, std = compute_normalizer(train_samples)
    train_ds = RegressionDataset(train_samples, mean=mean, std=std)
    val_ds = RegressionDataset(val_samples, mean=mean, std=std)
    batch_size = int(args.batch_size or seq_cfg.get("batch_size", 64))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_regression)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_regression)

    model = build_sequence_model(args.model, input_dim=len(cols), cfg=cfg, output_dim=1).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(seq_cfg.get("lr", 1e-3)),
        weight_decay=float(seq_cfg.get("weight_decay", 1e-4)),
    )

    thresholds = threshold_grid(cfg)
    tolerance = int(eval_cfg.get("event_tolerance", 1))
    min_distance = int(eval_cfg.get("min_distance", 6))
    prominence = float(eval_cfg.get("prominence", 0.0))
    patience = int(seq_cfg.get("early_stop_patience", 5))
    epochs = int(args.epochs or seq_cfg.get("epochs", 30))
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
            loss_name=args.loss,
            grad_clip=grad_clip,
            log_interval=max(int(args.log_interval), 0),
        )
        val_pred = predict_sequences(model, val_loader, device=device)
        reg_metrics = compute_regression_metrics(
            val_pred["target"].to_numpy(dtype=np.float32),
            val_pred["score"].to_numpy(dtype=np.float32),
        )
        sequence_scores, sequence_labels = build_sequence_maps(val_pred)
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
            "mae": reg_metrics["mae"],
            "rmse": reg_metrics["rmse"],
            "spearman": reg_metrics["spearman"],
            "positive_spearman": reg_metrics["positive_spearman"],
            "event_f1": event_metrics.f1,
            "event_ap": event_metrics.average_precision,
        }
        history.append(history_item)
        print(
            f"Epoch {epoch}/{epochs} | train_loss {train_loss:.4f} | "
            f"spearman {reg_metrics['spearman']:.4f} | rmse {reg_metrics['rmse']:.4f} | "
            f"event_f1 {event_metrics.f1:.4f}"
        )

        current_key = (
            float(reg_metrics["spearman"]),
            float(reg_metrics["positive_spearman"] or -1.0),
            -float(reg_metrics["rmse"]),
        )
        if best_key is None or current_key > best_key:
            best_key = current_key
            best_epoch = epoch
            bad_epochs = 0
            best_summary = {
                **reg_metrics,
                "event_precision": event_metrics.precision,
                "event_recall": event_metrics.recall,
                "event_f1": event_metrics.f1,
                "event_ap": event_metrics.average_precision,
                "best_threshold": event_metrics.threshold,
                "mean_offset": event_metrics.mean_offset,
                "matches": event_metrics.matches,
                "pred_events": event_metrics.pred_events,
                "true_events": event_metrics.true_events,
            }
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_type": args.model,
                    "feature_columns": cols,
                    "mean": mean,
                    "std": std,
                    "best_epoch": best_epoch,
                    "target_col": target_col,
                    "loss": args.loss,
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
        "loss": args.loss,
        "table_path": str(table_path),
        "target_col": target_col,
        "device": str(device),
        "seed": seed,
        "epochs_run": len(history),
        "best_epoch": best_epoch,
        **best_summary,
        "feature_columns": cols,
        "history": history,
    }
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        f"Best epoch: {best_epoch} | spearman={best_summary['spearman']:.4f} | "
        f"rmse={best_summary['rmse']:.4f} | event_f1={best_summary['event_f1']:.4f}"
    )


if __name__ == "__main__":
    main()
