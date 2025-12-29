import argparse
import os
import random
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
import yaml

from dataset_beat import BeatBoundaryDataset, collate_beat
from model.model_beat import BeatBoundaryModel, BeatBoundaryConfig


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def create_dataloaders(cfg):
    dataset = BeatBoundaryDataset(
        data_dir=cfg["data"]["data_dir"],
        file_ext=cfg["data"]["file_ext"],
        max_len=cfg["data"]["max_len"],
    )

    total = len(dataset)
    if total < 2:
        # tiny dataset: use the same data for train/val
        train_ds = dataset
        val_ds = dataset
    else:
        train_size = int(cfg["data"]["train_split"] * total)
        train_size = max(1, min(train_size, total - 1))
        val_size = total - train_size
        train_ds, val_ds = random_split(
            dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(cfg["training"]["seed"]),
        )

    pad_to = cfg["data"]["max_len"]
    collate_fn = lambda batch: collate_beat(batch, pad_to=pad_to)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["data"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"]["num_workers"],
        collate_fn=collate_fn,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["data"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        collate_fn=collate_fn,
        pin_memory=True,
    )
    return train_loader, val_loader, dataset.feature_dim


def build_model(cfg, input_dim):
    model_cfg = BeatBoundaryConfig(
        input_dim=input_dim,
        d_model=cfg["model"]["d_model"],
        nhead=cfg["model"]["nhead"],
        num_layers=cfg["model"]["num_layers"],
        dim_feedforward=cfg["model"]["dim_feedforward"],
        dropout=cfg["model"]["dropout"],
        max_len=cfg["model"]["max_len"],
    )
    return BeatBoundaryModel(model_cfg)


def train_one_epoch(model, loader, optimizer, device, grad_clip):
    model.train()
    total_loss = 0.0
    total_tokens = 0
    for batch in loader:
        score = batch["score_feats"].to(device)
        labels = batch["labels"].to(device)
        mask = batch["attn_mask"].to(device)

        optimizer.zero_grad()
        _, loss = model(score, attn_mask=mask, labels=labels)
        loss.backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        tokens = mask.sum().item()
        total_loss += loss.item() * tokens
        total_tokens += tokens
    return total_loss / max(total_tokens, 1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for batch in loader:
        score = batch["score_feats"].to(device)
        labels = batch["labels"].to(device)
        mask = batch["attn_mask"].to(device)

        _, loss = model(score, attn_mask=mask, labels=labels)
        tokens = mask.sum().item()
        total_loss += loss.item() * tokens
        total_tokens += tokens
    return total_loss / max(total_tokens, 1)


def main():
    parser = argparse.ArgumentParser(description="Train beat-level boundary model")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument("--device", default=None, help="cpu|cuda|auto")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Set device
    if args.device:
        device = args.device
    else:
        device = cfg["training"].get("device", "auto")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    print(f"Using device: {device}")

    set_seed(cfg["training"]["seed"])

    train_loader, val_loader, input_dim = create_dataloaders(cfg)
    model = build_model(cfg, input_dim=input_dim).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    # Prepare save dir
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"{cfg['trainer']['experiment_name']}_{ts}"
    save_dir = Path(cfg["trainer"]["save_dir"]) / exp_name
    save_dir.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")
    epochs = cfg["training"]["epochs"]
    grad_clip = cfg["training"].get("grad_clip", None)

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, grad_clip)
        val_loss = evaluate(model, val_loader, device)
        print(f"Epoch {epoch}/{epochs} | train_loss {train_loss:.4f} | val_loss {val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), save_dir / "best.pt")
        torch.save(model.state_dict(), save_dir / "last.pt")


if __name__ == "__main__":
    main()
