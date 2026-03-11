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
from torch.utils.data import DataLoader, Dataset

from boundary_restart.config import load_config, resolve_path, threshold_grid
from boundary_restart.metrics import (
    evaluate_labeled_event_sequences,
    search_threshold_with_min_precision,
)
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


def build_detector_labels(stage_class: np.ndarray) -> np.ndarray:
    return (np.asarray(stage_class, dtype=np.int64) >= 2).astype(np.float32)


class DualHeadTCN(nn.Module):
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
        self.detector_head = nn.Conv1d(in_channels, 1, 1)
        self.grade_head = nn.Conv1d(in_channels, 2, 1)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        del lengths
        x = x.transpose(1, 2)
        h = self.encoder(x)
        detector_logits = self.detector_head(h).squeeze(1)
        grade_logits = self.grade_head(h).transpose(1, 2)
        return detector_logits, grade_logits


class DualHeadDataset(Dataset):
    def __init__(self, samples: list[dict], mean: np.ndarray, std: np.ndarray):
        self.samples = samples
        self.mean = mean.astype(np.float32)
        self.std = std.astype(np.float32)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        features = (sample["features"] - self.mean) / self.std
        grade_labels = sample["grade_labels"].astype(np.int64)
        ce_targets = grade_labels.copy()
        ce_targets[ce_targets > 0] -= 1
        ce_targets[ce_targets == 0] = -100
        return {
            "sample_id": sample["sample_id"],
            "piece_id": sample["piece_id"],
            "beat_idx": sample["beat_idx"].astype(np.int32),
            "features": features.astype(np.float32),
            "detector_labels": sample["detector_labels"].astype(np.float32),
            "grade_labels": grade_labels,
            "ce_targets": ce_targets.astype(np.int64),
            "length": int(sample["features"].shape[0]),
        }


def collate_dual_head(batch: list[dict]) -> dict:
    lengths = [item["length"] for item in batch]
    max_len = max(lengths)
    feat_dim = batch[0]["features"].shape[1]
    features = torch.zeros(len(batch), max_len, feat_dim, dtype=torch.float32)
    detector_labels = torch.zeros(len(batch), max_len, dtype=torch.float32)
    grade_labels = torch.zeros(len(batch), max_len, dtype=torch.int64)
    ce_targets = torch.full((len(batch), max_len), -100, dtype=torch.int64)
    beat_idx = torch.zeros(len(batch), max_len, dtype=torch.int64)
    mask = torch.zeros(len(batch), max_len, dtype=torch.bool)
    sample_ids = []
    piece_ids = []

    for idx, item in enumerate(batch):
        length = item["length"]
        features[idx, :length] = torch.from_numpy(item["features"])
        detector_labels[idx, :length] = torch.from_numpy(item["detector_labels"])
        grade_labels[idx, :length] = torch.from_numpy(item["grade_labels"])
        ce_targets[idx, :length] = torch.from_numpy(item["ce_targets"])
        beat_idx[idx, :length] = torch.from_numpy(item["beat_idx"])
        mask[idx, :length] = True
        sample_ids.append(item["sample_id"])
        piece_ids.append(item["piece_id"])

    return {
        "features": features,
        "detector_labels": detector_labels,
        "grade_labels": grade_labels,
        "ce_targets": ce_targets,
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


def dual_head_samples_from_table(df: pd.DataFrame, feature_cols: list[str], split: str) -> list[dict]:
    subset = df[df["protocol_split"] == split].copy().sort_values(["sample_id", "beat_idx"])
    samples = []
    for sample_id, group in subset.groupby("sample_id", sort=False):
        samples.append(
            {
                "sample_id": sample_id,
                "piece_id": group["piece_id"].iloc[0],
                "beat_idx": group["beat_idx"].to_numpy(dtype=np.int32),
                "features": group[feature_cols].to_numpy(dtype=np.float32),
                "detector_labels": build_detector_labels(group["stage_class"].to_numpy(dtype=np.int64)),
                "grade_labels": group["stage_class_midhigh"].to_numpy(dtype=np.int64),
            }
        )
    return samples


def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
    detector_loss_fn,
    grade_loss_fn,
    grade_loss_weight: float,
    grad_clip: float,
    log_interval: int = 0,
) -> tuple[float, float, float]:
    model.train()
    total_loss = 0.0
    total_detector_loss = 0.0
    total_grade_loss = 0.0
    total_tokens = 0
    total_grade_tokens = 0
    for batch_idx, batch in enumerate(loader, start=1):
        features = batch["features"].to(device)
        detector_labels = batch["detector_labels"].to(device)
        ce_targets = batch["ce_targets"].to(device)
        mask = batch["mask"].to(device)
        lengths = batch["lengths"].to(device)

        optimizer.zero_grad()
        detector_logits, grade_logits = model(features, lengths=lengths)

        detector_loss = detector_loss_fn(detector_logits, detector_labels)
        detector_loss = (detector_loss * mask.float()).sum() / mask.float().sum().clamp(min=1.0)

        flat_grade = grade_logits.reshape(-1, grade_logits.size(-1))
        flat_targets = ce_targets.reshape(-1)
        raw_grade_loss = grade_loss_fn(flat_grade, flat_targets).reshape(ce_targets.shape)
        grade_valid = (ce_targets >= 0).float()
        grade_loss = (raw_grade_loss * grade_valid).sum() / grade_valid.sum().clamp(min=1.0)

        total = detector_loss + float(grade_loss_weight) * grade_loss
        total.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += float(total.item()) * int(mask.sum().item())
        total_detector_loss += float(detector_loss.item()) * int(mask.sum().item())
        total_grade_loss += float(grade_loss.item()) * int(grade_valid.sum().item())
        total_tokens += int(mask.sum().item())
        total_grade_tokens += int(grade_valid.sum().item())
        if log_interval > 0 and batch_idx % log_interval == 0:
            running = total_loss / max(total_tokens, 1)
            print(f"  step {batch_idx}/{len(loader)} | running_loss {running:.4f}")

    return (
        total_loss / max(total_tokens, 1),
        total_detector_loss / max(total_tokens, 1),
        total_grade_loss / max(total_grade_tokens, 1),
    )


@torch.no_grad()
def predict(model, loader, device) -> pd.DataFrame:
    model.eval()
    rows = []
    for batch in loader:
        features = batch["features"].to(device)
        detector_labels = batch["detector_labels"].to(device)
        grade_labels = batch["grade_labels"].to(device)
        beat_idx = batch["beat_idx"].to(device)
        mask = batch["mask"].to(device)
        lengths = batch["lengths"].to(device)

        detector_logits, grade_logits = model(features, lengths=lengths)
        detector_scores = torch.sigmoid(detector_logits)
        grade_probs = torch.softmax(grade_logits, dim=-1)
        pred_grade = grade_probs.argmax(dim=-1) + 1

        for batch_idx_i, sample_id in enumerate(batch["sample_ids"]):
            length = int(mask[batch_idx_i].sum().item())
            for pos in range(length):
                rows.append(
                    {
                        "sample_id": sample_id,
                        "piece_id": batch["piece_ids"][batch_idx_i],
                        "beat_idx": int(beat_idx[batch_idx_i, pos].item()),
                        "detector_target": float(detector_labels[batch_idx_i, pos].item()),
                        "stage_class_midhigh": int(grade_labels[batch_idx_i, pos].item()),
                        "detector_score": float(detector_scores[batch_idx_i, pos].item()),
                        "pred_midhigh_class": int(pred_grade[batch_idx_i, pos].item()),
                        "pred_high_prob": float(grade_probs[batch_idx_i, pos, 1].item()),
                    }
                )
    return pd.DataFrame(rows)


def detector_sequence_maps(pred_df: pd.DataFrame) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    scores = {}
    labels = {}
    for sample_id, group in pred_df.sort_values(["sample_id", "beat_idx"]).groupby("sample_id", sort=False):
        scores[sample_id] = group["detector_score"].to_numpy(dtype=np.float32)
        labels[sample_id] = group["detector_target"].to_numpy(dtype=np.float32)
    return scores, labels


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
    parser = argparse.ArgumentParser(description="Piece-level dual-head detector+grader for mid/high boundaries.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--heldout_piece", nargs="+", required=True)
    parser.add_argument("--train_pieces", nargs="*", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--log_interval", type=int, default=0)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--min_precision", type=float, default=0.95)
    parser.add_argument("--grade_loss_weight", type=float, default=1.0)
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
            f"../outputs/piece_protocol/{heldout_slug}/tcn_dual_head_p{int(round(args.min_precision * 100))}",
        )
    out_root.mkdir(parents=True, exist_ok=True)

    df = load_table(table_path)
    df = apply_piece_protocol_split(df, heldout_pieces=args.heldout_piece, train_pieces=args.train_pieces)
    df = df[df["protocol_split"].isin(["train", "val"])].copy()
    feature_cols = select_feature_columns(cfg, feature_columns(df))

    train_samples = dual_head_samples_from_table(df, feature_cols, split="train")
    val_samples = dual_head_samples_from_table(df, feature_cols, split="val")
    if not train_samples or not val_samples:
        raise ValueError("Protocol split produced an empty train or val set")

    mean, std = compute_normalizer(train_samples)
    train_ds = DualHeadDataset(train_samples, mean=mean, std=std)
    val_ds = DualHeadDataset(val_samples, mean=mean, std=std)
    batch_size = int(args.batch_size or seq_cfg.get("batch_size", 64))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_dual_head)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_dual_head)

    channels = [int(v) for v in seq_cfg.get("tcn_channels", [64, 64, 64])]
    model = DualHeadTCN(
        input_dim=len(feature_cols),
        channels=channels,
        kernel_size=int(seq_cfg.get("kernel_size", 3)),
        dropout=float(seq_cfg.get("dropout", 0.2)),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(seq_cfg.get("lr", 1e-3)),
        weight_decay=float(seq_cfg.get("weight_decay", 1e-4)),
    )

    detector_train_labels = np.concatenate([sample["detector_labels"] for sample in train_samples], axis=0)
    pos = float(detector_train_labels.sum())
    neg = float(detector_train_labels.shape[0] - pos)
    detector_pos_weight = torch.tensor([neg / max(pos, 1.0)], device=device, dtype=torch.float32)
    detector_loss_fn = nn.BCEWithLogitsLoss(pos_weight=detector_pos_weight, reduction="none")

    grade_labels = np.concatenate([sample["grade_labels"] for sample in train_samples], axis=0)
    grade_labels = grade_labels[grade_labels > 0] - 1
    grade_counts = np.bincount(grade_labels, minlength=2).astype(np.float32)
    grade_counts[grade_counts < 1] = 1.0
    grade_weights = torch.tensor(grade_counts.sum() / grade_counts, device=device, dtype=torch.float32)
    grade_loss_fn = nn.CrossEntropyLoss(weight=grade_weights, ignore_index=-100, reduction="none")

    thresholds = threshold_grid(cfg)
    tolerance = int(eval_cfg.get("event_tolerance", 1))
    min_distance = int(eval_cfg.get("min_distance", 6))
    prominence = float(eval_cfg.get("prominence", 0.0))
    epochs = int(args.epochs or seq_cfg.get("epochs", 30))
    patience = int(seq_cfg.get("early_stop_patience", 5))
    grad_clip = float(seq_cfg.get("grad_clip", 1.0))

    best_epoch = 0
    best_key = None
    best_detector = None
    best_pred = None
    history = []
    bad_epochs = 0

    for epoch in range(1, epochs + 1):
        train_loss, train_detector_loss, train_grade_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            detector_loss_fn=detector_loss_fn,
            grade_loss_fn=grade_loss_fn,
            grade_loss_weight=float(args.grade_loss_weight),
            grad_clip=grad_clip,
            log_interval=max(int(args.log_interval), 0),
        )
        val_pred = predict(model, val_loader, device=device)
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

        pos_val = val_pred[val_pred["stage_class_midhigh"] > 0].copy()
        oracle_grading = grading_report(
            y_true=pos_val["stage_class_midhigh"].to_numpy(dtype=np.int64),
            y_pred=pos_val["pred_midhigh_class"].to_numpy(dtype=np.int64),
        )

        sequence_pred_labels = {}
        sequence_true_labels = {}
        for sample_id, group in val_pred.sort_values(["sample_id", "beat_idx"]).groupby("sample_id", sort=False):
            sequence_pred_labels[sample_id] = group["pred_midhigh_class"].to_numpy(dtype=np.int32)
            sequence_true_labels[sample_id] = group["stage_class_midhigh"].to_numpy(dtype=np.int32)
        class_event_metrics = evaluate_labeled_event_sequences(
            sequence_scores=sequence_scores,
            sequence_pred_labels=sequence_pred_labels,
            sequence_true_labels=sequence_true_labels,
            positive_classes=(1, 2),
            threshold=float(detector_metrics.threshold),
            tolerance=tolerance,
            min_distance=min_distance,
            prominence=prominence,
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_detector_loss": train_detector_loss,
                "train_grade_loss": train_grade_loss,
                "event_precision": detector_metrics.precision,
                "event_recall": detector_metrics.recall,
                "event_f1": detector_metrics.f1,
                "oracle_macro_f1": oracle_grading["macro_f1"],
                "end_to_end_macro_f1": class_event_metrics.macro_f1,
                "best_threshold": detector_metrics.threshold,
                "precision_floor_met": precision_floor_met,
            }
        )
        print(
            f"Epoch {epoch}/{epochs} | train_loss {train_loss:.4f} | "
            f"precision {detector_metrics.precision:.4f} | recall {detector_metrics.recall:.4f} | "
            f"oracle_f1 {oracle_grading['macro_f1']:.4f} | e2e_f1 {class_event_metrics.macro_f1:.4f}"
        )

        current_key = (
            float(precision_floor_met),
            detector_metrics.recall if precision_floor_met else detector_metrics.precision,
            class_event_metrics.macro_f1,
            detector_metrics.precision,
            oracle_grading["macro_f1"],
            -float(detector_metrics.mean_offset or 1e9),
        )
        if best_key is None or current_key > best_key:
            best_key = current_key
            best_epoch = epoch
            best_detector = detector_metrics
            best_pred = val_pred.copy()
            best_oracle = oracle_grading
            best_class_event = class_event_metrics
            bad_epochs = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "feature_columns": feature_cols,
                    "mean": mean,
                    "std": std,
                    "best_epoch": best_epoch,
                    "best_threshold": detector_metrics.threshold,
                    "min_precision": args.min_precision,
                    "grade_loss_weight": args.grade_loss_weight,
                },
                out_root / "best.pt",
            )
        else:
            bad_epochs += 1
            if patience > 0 and bad_epochs >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    if best_detector is None or best_pred is None:
        raise RuntimeError("Dual-head training did not produce validation metrics")

    best_pred.to_csv(out_root / "val_predictions.csv.gz", index=False, compression="gzip")

    summary = {
        "table_path": str(table_path),
        "heldout_pieces": list(args.heldout_piece),
        "train_piece_count": int(df[df["protocol_split"] == "train"]["piece_id"].nunique()),
        "val_piece_count": int(df[df["protocol_split"] == "val"]["piece_id"].nunique()),
        "train_sample_count": int(df[df["protocol_split"] == "train"]["sample_id"].nunique()),
        "val_sample_count": int(df[df["protocol_split"] == "val"]["sample_id"].nunique()),
        "model_type": "tcn_dual_head",
        "seed": seed,
        "device": str(device),
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "precision_floor": float(args.min_precision),
        "grade_loss_weight": float(args.grade_loss_weight),
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
        "oracle_grading": best_oracle,
        "end_to_end_midhigh": labeled_metrics_to_dict(best_class_event),
        "feature_columns": feature_cols,
        "history": history,
    }
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    np.savez(out_root / "scaler_stats.npz", mean=mean, std=std)
    print(
        f"Held-out {heldout_slug} | precision={best_detector.precision:.4f} | "
        f"recall={best_detector.recall:.4f} | mid/high macro_event_f1={best_class_event.macro_f1:.4f}"
    )


if __name__ == "__main__":
    main()
