import argparse
import os
import random
import re
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
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


def print_batch_sanity(batch):
    labels = batch["labels"]
    beat_ids = batch["beat_ids"]
    mask = batch["attn_mask"]
    max_beats = int(batch["max_beats"].item())

    labels_mean = labels.mean().item()
    labels_std = labels.std().item()
    labels_max = labels.max().item()
    labels_sum = labels.sum().item()

    valid = (beat_ids >= 0) & mask
    valid_ids = beat_ids[valid]
    beat_min = int(valid_ids.min().item()) if valid_ids.numel() > 0 else None
    beat_max = int(valid_ids.max().item()) if valid_ids.numel() > 0 else None
    note_count = int(mask.sum().item())

    b0 = None
    if beat_ids.size(0) > 0:
        valid0 = (beat_ids[0] >= 0) & mask[0]
        if valid0.any():
            bc = torch.bincount(beat_ids[0][valid0], minlength=max_beats).tolist()
            if len(bc) > 50:
                b0 = (bc[:50], len(bc))
            else:
                b0 = (bc, len(bc))

    print("Sanity A:")
    print(
        f"labels mean/std/max/sum = {labels_mean:.6f} / {labels_std:.6f} / {labels_max:.6f} / {labels_sum:.2f}"
    )
    print(f"beat_ids min/max = {beat_min} / {beat_max} | max_beats = {max_beats} | labels_len = {labels.shape[1]}")
    print(f"valid note count (mask.sum) = {note_count}")
    if b0 is None:
        print("bincount(sample0) = <empty>")
    elif b0[1] > 50:
        print(f"bincount(sample0, first 50 of {b0[1]} beats) = {b0[0]}")
    else:
        print(f"bincount(sample0) = {b0[0]}")


def set_bias_only(model):
    for p in model.parameters():
        p.requires_grad = False
    if hasattr(model, "head") and model.head.bias is not None:
        model.head.bias.requires_grad = True
    if hasattr(model, "head") and model.head.weight is not None:
        model.head.weight.data.zero_()
        model.head.weight.requires_grad = False
    return [p for p in model.parameters() if p.requires_grad]


def load_piece_split(path: str):
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Split file must be a mapping, got: {type(data).__name__}")
    train_pieces = set(data.get("train", []) or [])
    val_pieces = set(data.get("val", []) or [])
    test_pieces = set(data.get("test", []) or [])
    if not train_pieces or not val_pieces:
        raise ValueError("Split file must define non-empty 'train' and 'val' lists.")
    if train_pieces & val_pieces or train_pieces & test_pieces or val_pieces & test_pieces:
        raise ValueError("train/val/test piece ids in split file must be disjoint.")
    return {
        "train": train_pieces,
        "val": val_pieces,
        "test": test_pieces,
    }


def load_aux_split(path: str):
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Aux split file must be a mapping, got: {type(data).__name__}")
    return data


def recording_id_from_path(path: Path) -> str | None:
    stem = path.stem
    parts = stem.split("_")
    for part in parts[1:]:
        if part.startswith("pid"):
            return part
    m = re.search(r"(pid[^_]+)", stem)
    if m:
        return m.group(1)
    return None


def piece_id_from_path(path: Path, cfg) -> str:
    stem = path.stem
    regex = cfg.get("data", {}).get("piece_id_regex")
    if regex:
        m = re.search(regex, stem)
        if not m and "\\\\" in regex:
            try:
                m = re.search(regex.encode("utf-8").decode("unicode_escape"), stem)
            except Exception:
                m = None
        if m:
            return m.group(1) if m.groups() else m.group(0)
    delim = cfg.get("data", {}).get("piece_id_delim")
    if delim and delim in stem:
        return stem.split(delim)[0]
    return stem


def build_aux_filters(aux_data, aux_mode=None, aux_targets=None):
    if not aux_mode:
        return None
    targets = set(aux_targets or [])
    if aux_mode == "heldout_pianists":
        entries = aux_data.get("heldout_pianists", []) or []
        if targets:
            entries = [e for e in entries if e.get("name") in targets]
        excluded_ids = {rid for e in entries for rid in (e.get("recording_ids", []) or [])}
        return {
            "mode": aux_mode,
            "excluded_ids": excluded_ids,
            "selected_targets": sorted(e.get("name") for e in entries),
        }
    if aux_mode == "same_piece_80":
        entries = aux_data.get("same_piece_80_percent", []) or []
        if targets:
            entries = [e for e in entries if e.get("piece") in targets]
        keep_map = {
            e.get("piece"): set(e.get("keep_recording_ids", []) or [])
            for e in entries
            if e.get("piece")
        }
        holdout_map = {
            e.get("piece"): set(e.get("holdout_recording_ids", []) or [])
            for e in entries
            if e.get("piece")
        }
        return {
            "mode": aux_mode,
            "keep_map": keep_map,
            "holdout_map": holdout_map,
            "selected_targets": sorted(keep_map.keys()),
        }
    raise ValueError(f"Unsupported aux_mode: {aux_mode}")


def build_dataset(cfg, beat_sequence_length=None, beat_stride=None):
    data_cfg = cfg["data"]
    return BeatBoundaryDataset(
        data_dir=cfg["data"]["data_dir"],
        file_ext=cfg["data"]["file_ext"],
        max_len=cfg["data"]["max_len"],
        sequence_length=cfg["data"].get("sequence_length"),
        stride=cfg["data"].get("stride"),
        beat_sequence_length=(
            data_cfg.get("beat_sequence_length")
            if beat_sequence_length is None
            else beat_sequence_length
        ),
        beat_stride=(
            data_cfg.get("beat_stride")
            if beat_stride is None
            else beat_stride
        ),
        drop_short=cfg["data"].get("drop_short", True),
        position_mode=cfg["data"].get("position_mode", "absolute"),
        use_base_features_only=cfg["data"].get("use_base_features_only", False),
        performer_id_regex=cfg["data"].get("performer_id_regex"),
        label_mode=cfg["data"].get("label_mode", "ratio"),
        dist_min_dist=cfg["data"].get("dist_min_dist", 6),
        dist_height=cfg["data"].get("dist_height", 0.15),
        dist_prominence=cfg["data"].get("dist_prominence", 0.05),
        dist_tau=cfg["data"].get("dist_tau", 4.0),
        add_beat_pos=cfg["data"].get("add_beat_pos", False),
        max_samples=cfg["data"].get("max_samples"),
        value_ranges=cfg["data"].get("value_ranges"),
        label_binarize_threshold=cfg["data"].get("label_binarize_threshold"),
    )


def apply_dataset_filters(dataset, cfg, level=None, aux_split_file=None, aux_mode=None, aux_targets=None):
    if level is not None:
        level_tag = f"_L{int(level)}"
        filtered = [s for s in dataset.samples if Path(s["path"]).stem.endswith(level_tag)]
        if not filtered:
            raise ValueError(f"No samples found for level {level} (suffix {level_tag}) in {dataset.data_dir}")
        dataset.samples = filtered

    aux_summary = {"mode": aux_mode, "selected_targets": [], "excluded_ids": 0, "restricted_pieces": []}
    if aux_split_file and aux_mode:
        aux_data = load_aux_split(aux_split_file)
        aux_filter = build_aux_filters(aux_data, aux_mode=aux_mode, aux_targets=aux_targets)
        filtered_samples = []
        if aux_filter["mode"] == "heldout_pianists":
            excluded_ids = set(aux_filter["excluded_ids"])
            for s in dataset.samples:
                rid = recording_id_from_path(Path(s["path"]))
                if rid not in excluded_ids:
                    filtered_samples.append(s)
            aux_summary = {
                "mode": aux_mode,
                "selected_targets": aux_filter["selected_targets"],
                "excluded_ids": len(excluded_ids),
                "restricted_pieces": [],
            }
        elif aux_filter["mode"] == "same_piece_80":
            keep_map = aux_filter["keep_map"]
            restricted_pieces = sorted(keep_map.keys())
            for s in dataset.samples:
                piece = piece_id_from_path(Path(s["path"]))
                rid = recording_id_from_path(Path(s["path"]))
                if piece in keep_map:
                    if rid in keep_map[piece]:
                        filtered_samples.append(s)
                else:
                    filtered_samples.append(s)
            aux_summary = {
                "mode": aux_mode,
                "selected_targets": aux_filter["selected_targets"],
                "excluded_ids": 0,
                "restricted_pieces": restricted_pieces,
            }
        else:
            filtered_samples = dataset.samples
        if not filtered_samples:
            raise ValueError("Auxiliary filtering removed all samples.")
        dataset.samples = filtered_samples

    return aux_summary


def create_dataloaders(cfg, level=None, split_file=None, aux_split_file=None, aux_mode=None, aux_targets=None):
    data_cfg = cfg["data"]
    train_beat_sequence_length = data_cfg.get(
        "train_beat_sequence_length",
        data_cfg.get("beat_sequence_length"),
    )
    train_beat_stride = data_cfg.get(
        "train_beat_stride",
        data_cfg.get("beat_stride"),
    )
    eval_beat_sequence_length = data_cfg.get(
        "eval_beat_sequence_length",
        data_cfg.get("beat_sequence_length"),
    )
    eval_beat_stride = data_cfg.get(
        "eval_beat_stride",
        data_cfg.get("beat_stride"),
    )

    train_dataset = build_dataset(
        cfg,
        beat_sequence_length=train_beat_sequence_length,
        beat_stride=train_beat_stride,
    )
    eval_dataset = build_dataset(
        cfg,
        beat_sequence_length=eval_beat_sequence_length,
        beat_stride=eval_beat_stride,
    )

    aux_summary = apply_dataset_filters(
        train_dataset,
        cfg,
        level=level,
        aux_split_file=aux_split_file,
        aux_mode=aux_mode,
        aux_targets=aux_targets,
    )
    apply_dataset_filters(
        eval_dataset,
        cfg,
        level=level,
        aux_split_file=aux_split_file,
        aux_mode=aux_mode,
        aux_targets=aux_targets,
    )

    train_piece_ids = [piece_id_from_path(s["path"], cfg) for s in train_dataset.samples]
    eval_piece_ids = [piece_id_from_path(s["path"], cfg) for s in eval_dataset.samples]
    unique_pieces = list(set(train_piece_ids) | set(eval_piece_ids))
    split_meta = None
    if split_file:
        split_meta = load_piece_split(split_file)
        train_known = set(train_piece_ids)
        eval_known = set(eval_piece_ids)
        missing_train = sorted(split_meta["train"] - train_known)
        missing_eval = sorted((split_meta["val"] | split_meta["test"]) - eval_known)
        missing = sorted(set(missing_train) | set(missing_eval))
        if missing:
            raise ValueError(
                "Split file references pieces not found in dataset: "
                f"{missing}"
            )
        train_indices = [i for i, pid in enumerate(train_piece_ids) if pid in split_meta["train"]]
        val_indices = [i for i, pid in enumerate(eval_piece_ids) if pid in split_meta["val"]]
        test_indices = [i for i, pid in enumerate(eval_piece_ids) if pid in split_meta["test"]]
        train_ds = Subset(train_dataset, train_indices)
        val_ds = Subset(eval_dataset, val_indices)
        test_ds = Subset(eval_dataset, test_indices) if test_indices else None
    elif len(unique_pieces) < 2:
        train_ds = train_dataset
        val_ds = eval_dataset
        test_ds = None
        train_indices = list(range(len(train_dataset.samples)))
        val_indices = list(range(len(eval_dataset.samples)))
        test_indices = []
    else:
        rng = random.Random(cfg["training"]["seed"])
        rng.shuffle(unique_pieces)
        train_count = int(cfg["data"]["train_split"] * len(unique_pieces))
        train_count = max(1, min(train_count, len(unique_pieces) - 1))
        train_pieces = set(unique_pieces[:train_count])
        val_pieces = set(unique_pieces[train_count:])

        train_indices = [i for i, pid in enumerate(train_piece_ids) if pid in train_pieces]
        val_indices = [i for i, pid in enumerate(eval_piece_ids) if pid in val_pieces]

        train_ds = Subset(train_dataset, train_indices)
        val_ds = Subset(eval_dataset, val_indices)
        test_ds = None
        test_indices = []

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
    test_loader = None
    if test_ds is not None:
        test_loader = DataLoader(
            test_ds,
            batch_size=cfg["data"]["batch_size"],
            shuffle=False,
            num_workers=cfg["data"]["num_workers"],
            collate_fn=collate_fn,
            pin_memory=True,
        )
    split_summary = {
        "all_pieces": sorted(unique_pieces),
        "train_pieces": sorted({train_piece_ids[i] for i in train_indices}),
        "val_pieces": sorted({eval_piece_ids[i] for i in val_indices}),
        "test_pieces": sorted({eval_piece_ids[i] for i in test_indices}) if split_meta and split_meta["test"] else [],
        "aux_filter": aux_summary,
        "train_window": {
            "beat_sequence_length": train_beat_sequence_length,
            "beat_stride": train_beat_stride,
        },
        "eval_window": {
            "beat_sequence_length": eval_beat_sequence_length,
            "beat_stride": eval_beat_stride,
        },
    }
    return train_loader, val_loader, test_loader, train_dataset.feature_dim, train_dataset, split_summary


def build_model(cfg, input_dim):
    label_mode = cfg.get("data", {}).get("label_mode")
    dual_head = cfg["model"].get("dual_head")
    if dual_head is None:
        dual_head = label_mode == "dual"
    model_cfg = BeatBoundaryConfig(
        input_dim=input_dim,
        d_model=cfg["model"]["d_model"],
        nhead=cfg["model"]["nhead"],
        num_layers=cfg["model"]["num_layers"],
        dim_feedforward=cfg["model"]["dim_feedforward"],
        dropout=cfg["model"]["dropout"],
        max_len=cfg["model"]["max_len"],
        fixed_beats=cfg["model"].get("fixed_beats"),
        include_empty_beats=cfg["model"].get("include_empty_beats", False),
        dual_head=dual_head,
        note_rnn_hidden=cfg["model"].get("note_rnn_hidden"),
        note_rnn_layers=cfg["model"].get("note_rnn_layers", 1),
        note_rnn_dropout=cfg["model"].get("note_rnn_dropout", cfg["model"]["dropout"]),
        performer_cond=cfg["model"].get("performer_cond", False),
        performer_emb_dim=cfg["model"].get("performer_emb_dim", 32),
        performer_vocab_size=cfg["model"].get("performer_vocab_size", 0),
    )
    pos_weight = cfg.get("training", {}).get("pos_weight")
    loss_type = cfg.get("training", {}).get("loss_type", "bce")
    prob_loss_type = cfg.get("training", {}).get("prob_loss_type", "bce")
    prob_pos_weight = cfg.get("training", {}).get("prob_pos_weight")
    prob_loss_weight = cfg.get("training", {}).get("prob_loss_weight", 1.0)
    return BeatBoundaryModel(
        model_cfg,
        pos_weight=pos_weight,
        loss_type=loss_type,
        prob_loss_type=prob_loss_type,
        prob_pos_weight=prob_pos_weight,
        prob_loss_weight=prob_loss_weight,
    )


def train_one_epoch(model, loader, optimizer, device, grad_clip):
    model.train()
    total_loss = 0.0
    total_batches = 0
    for batch in loader:
        note_feats = batch["note_feats"].to(device)
        beat_ids = batch["beat_ids"].to(device)
        labels = batch["labels"].to(device)
        labels_prob = batch.get("labels_prob")
        if labels_prob is not None:
            labels_prob = labels_prob.to(device)
        mask = batch["attn_mask"].to(device)
        max_beats = int(batch["max_beats"].item())
        num_beats = batch["num_beats"].to(device)
        performer_ids = batch.get("performer_ids")
        if performer_ids is not None:
            performer_ids = performer_ids.to(device)

        optimizer.zero_grad()
        _, loss = model(
            note_feats,
            beat_ids=beat_ids,
            num_beats=max_beats,
            num_beats_per_sample=num_beats,
            performer_ids=performer_ids,
            attn_mask=mask,
            labels=labels,
            labels_prob=labels_prob,
        )
        loss.backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item()
        total_batches += 1
    return total_loss / max(total_batches, 1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_batches = 0
    for batch in loader:
        note_feats = batch["note_feats"].to(device)
        beat_ids = batch["beat_ids"].to(device)
        labels = batch["labels"].to(device)
        labels_prob = batch.get("labels_prob")
        if labels_prob is not None:
            labels_prob = labels_prob.to(device)
        mask = batch["attn_mask"].to(device)
        max_beats = int(batch["max_beats"].item())
        num_beats = batch["num_beats"].to(device)
        performer_ids = batch.get("performer_ids")
        if performer_ids is not None:
            performer_ids = performer_ids.to(device)

        _, loss = model(
            note_feats,
            beat_ids=beat_ids,
            num_beats=max_beats,
            num_beats_per_sample=num_beats,
            performer_ids=performer_ids,
            attn_mask=mask,
            labels=labels,
            labels_prob=labels_prob,
        )
        total_loss += loss.item()
        total_batches += 1
    return total_loss / max(total_batches, 1)


def main():
    parser = argparse.ArgumentParser(description="Train beat-level boundary model")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument("--device", default=None, help="cpu|cuda|auto")
    parser.add_argument("--sanity_batch", action="store_true", help="Print one batch sanity stats and exit")
    parser.add_argument("--bias_only", action="store_true", help="Train only head bias (all other params frozen)")
    parser.add_argument(
        "--freeze_base",
        action="store_true",
        help="Freeze backbone and train only performer conditioning params",
    )
    parser.add_argument("--level", type=int, default=None, help="Use only samples with suffix _L{level}.npz")
    parser.add_argument("--pos_weight", type=float, default=None, help="Override training.pos_weight")
    parser.add_argument("--split_file", default=None, help="YAML file with explicit piece-level train/val/test splits")
    parser.add_argument("--aux_split_file", default=None, help="YAML file with auxiliary performer holdout definitions")
    parser.add_argument(
        "--aux_mode",
        default=None,
        choices=["heldout_pianists", "same_piece_80"],
        help="Apply auxiliary filtering on top of piece split",
    )
    parser.add_argument(
        "--aux_targets",
        default=None,
        help="Comma-separated pianist names or piece ids to select a subset from aux split file",
    )
    parser.add_argument("--early_stop_patience", type=int, default=8, help="Stop if val loss does not improve for N epochs")
    parser.add_argument("--early_stop_min_delta", type=float, default=0.0, help="Minimum val-loss improvement to reset patience")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.pos_weight is not None:
        cfg.setdefault("training", {})["pos_weight"] = float(args.pos_weight)

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

    aux_targets = [x.strip() for x in args.aux_targets.split(",")] if args.aux_targets else None
    train_loader, val_loader, test_loader, input_dim, dataset, split_summary = create_dataloaders(
        cfg,
        level=args.level,
        split_file=args.split_file,
        aux_split_file=args.aux_split_file,
        aux_mode=args.aux_mode,
        aux_targets=aux_targets,
    )
    if args.sanity_batch:
        batch = next(iter(train_loader))
        print_batch_sanity(batch)
        return
    if cfg.get("model", {}).get("performer_cond"):
        if dataset.num_performers <= 0:
            raise ValueError("performer_cond is enabled but no performer IDs were found in filenames.")
        cfg["model"]["performer_vocab_size"] = int(dataset.num_performers) + 1
    model = build_model(cfg, input_dim=input_dim).to(device)

    trainable_params = model.parameters()
    weight_decay = cfg["training"]["weight_decay"]
    if args.bias_only:
        trainable_params = set_bias_only(model)
        if not trainable_params:
            raise RuntimeError("Bias-only mode requested, but no trainable parameters found.")
        weight_decay = 0.0
        print("Bias-only training: head.weight zeroed; only head.bias is trainable.")
    if args.freeze_base or cfg["training"].get("freeze_base", False):
        for name, param in model.named_parameters():
            param.requires_grad = name.startswith("performer_")
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        if not trainable_params:
            raise RuntimeError("Freeze-base requested, but no performer conditioning params found.")
        print("Freeze-base: only performer conditioning params are trainable.")

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=cfg["training"]["lr"],
        weight_decay=weight_decay,
    )

    # Prepare save dir
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = cfg["trainer"]["experiment_name"]
    if args.level is not None:
        exp_name = f"{exp_name}_L{args.level}"
    exp_name = f"{exp_name}_{ts}"

    base_save_dir = Path(cfg["trainer"]["save_dir"])
    if args.level is not None:
        base_save_dir = base_save_dir / f"level_{args.level}"
    save_dir = base_save_dir / exp_name
    save_dir.mkdir(parents=True, exist_ok=True)
    with (save_dir / "split_summary.yaml").open("w") as f:
        yaml.safe_dump(split_summary, f, sort_keys=False)
    if cfg.get("model", {}).get("performer_cond") and getattr(dataset, "performer_map", None):
        with (save_dir / "performer_map.yaml").open("w") as f:
            yaml.safe_dump(dataset.performer_map, f)

    best_val = float("inf")
    best_epoch = 0
    epochs = cfg["training"]["epochs"]
    grad_clip = cfg["training"].get("grad_clip", None)
    patience = max(int(args.early_stop_patience), 0)
    min_delta = float(args.early_stop_min_delta)
    bad_epochs = 0

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, grad_clip)
        val_loss = evaluate(model, val_loader, device)
        print(f"Epoch {epoch}/{epochs} | train_loss {train_loss:.4f} | val_loss {val_loss:.4f}")

        if val_loss < (best_val - min_delta):
            best_val = val_loss
            best_epoch = epoch
            bad_epochs = 0
            torch.save(model.state_dict(), save_dir / "best.pt")
            with (save_dir / "best_metrics.yaml").open("w") as f:
                yaml.safe_dump(
                    {
                        "best_epoch": int(best_epoch),
                        "best_val_loss": float(best_val),
                    },
                    f,
                    sort_keys=False,
                )
        else:
            bad_epochs += 1
        torch.save(model.state_dict(), save_dir / "last.pt")
        if patience > 0 and bad_epochs >= patience:
            print(
                f"Early stopping at epoch {epoch}: no val improvement for {bad_epochs} epochs "
                f"(best_epoch={best_epoch}, best_val_loss={best_val:.4f})"
            )
            break

    if test_loader is not None:
        best_model = build_model(cfg, input_dim=input_dim).to(device)
        best_model.load_state_dict(torch.load(save_dir / "best.pt", map_location=device))
        best_test_loss = evaluate(best_model, test_loader, device)

        last_model = build_model(cfg, input_dim=input_dim).to(device)
        last_model.load_state_dict(torch.load(save_dir / "last.pt", map_location=device))
        last_test_loss = evaluate(last_model, test_loader, device)

        print(f"Test | best_loss {best_test_loss:.4f} | last_loss {last_test_loss:.4f} | best_epoch {best_epoch}")
        with (save_dir / "test_metrics.yaml").open("w") as f:
            yaml.safe_dump(
                {
                    "best_test_loss": float(best_test_loss),
                    "last_test_loss": float(last_test_loss),
                    "best_epoch": int(best_epoch),
                    "best_val_loss": float(best_val),
                    "test_pieces": split_summary["test_pieces"],
                },
                f,
                sort_keys=False,
            )


if __name__ == "__main__":
    main()
