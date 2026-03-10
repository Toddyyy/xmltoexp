import argparse
import random
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import yaml

from dataset_beat import BeatBoundaryDataset, collate_beat
from train_beat import (
    build_model,
    evaluate,
    load_config,
    print_batch_sanity,
    set_bias_only,
    set_seed,
    train_one_epoch,
)


class BeatSalienceDataset(BeatBoundaryDataset):
    def __init__(
        self,
        data_dir: str,
        levels,
        level_weights,
        file_ext: str = "npz",
        max_len=None,
        sequence_length=None,
        stride=None,
        beat_sequence_length=None,
        beat_stride=None,
        drop_short: bool = True,
        position_mode: str = "absolute",
        use_base_features_only: bool = False,
        add_beat_pos: bool = False,
        max_samples=None,
        value_ranges=None,
        label_binarize_threshold=None,
        performer_id_regex=None,
    ):
        self.levels = [int(x) for x in levels]
        if not self.levels:
            raise ValueError("levels must not be empty")
        weights = np.asarray(level_weights, dtype=np.float32)
        if weights.ndim != 1 or weights.size != len(self.levels):
            raise ValueError("level_weights must match levels in length")
        if np.any(weights < 0):
            raise ValueError("level_weights must be non-negative")
        if float(weights.sum()) <= 0:
            raise ValueError("level_weights must sum to > 0")
        self.level_weights = (weights / weights.sum()).astype(np.float32)

        # Mirror BeatBoundaryDataset init without file scanning, because salience
        # groups multiple *_L{n}.npz files into one logical sample.
        self.data_dir = Path(data_dir)
        self.file_ext = file_ext
        self.max_len = max_len
        self.sequence_length = sequence_length
        self.stride = stride
        self.beat_sequence_length = beat_sequence_length
        self.beat_stride = beat_stride
        self.drop_short = drop_short
        self.position_mode = position_mode
        self.use_base_features_only = bool(use_base_features_only)
        self.label_mode = "ratio"
        self.dist_min_dist = 6
        self.dist_height = 0.15
        self.dist_prominence = 0.05
        self.dist_tau = 4.0
        self.add_beat_pos = bool(add_beat_pos)
        self.max_samples = max_samples
        self.feature_names = [
            "pitch_midi",
            "duration",
            "position",
            "part_idx",
            "is_accent",
            "is_staccato",
        ]
        self.value_ranges = self._build_value_ranges(value_ranges)
        self.label_binarize_threshold = label_binarize_threshold
        self.performer_id_regex = performer_id_regex
        if self.position_mode not in {"absolute", "window", "zero"}:
            raise ValueError("position_mode must be one of: absolute, window, zero")

        all_files = sorted(self.data_dir.glob(f"*.{self.file_ext}"))
        if not all_files:
            raise FileNotFoundError(f"No *.{self.file_ext} files found in {self.data_dir}")

        groups = {}
        level_re = re.compile(r"^(?P<base>.+)_L(?P<level>\d+)$")
        for path in all_files:
            m = level_re.match(path.stem)
            if not m:
                continue
            level = int(m.group("level"))
            if level not in self.levels:
                continue
            base = m.group("base")
            groups.setdefault(base, {})[level] = path

        self.group_paths = {
            base: level_map
            for base, level_map in groups.items()
            if all(level in level_map for level in self.levels)
        }
        if not self.group_paths:
            raise FileNotFoundError(
                f"No complete salience groups found in {self.data_dir} for levels {self.levels}"
            )

        self.group_keys = sorted(self.group_paths)
        self.files = [self.group_paths[key][self.levels[0]] for key in self.group_keys]
        self.performer_map = self._build_performer_map()
        self.num_performers = len(self.performer_map)

        first = self._load_group(self.group_keys[0])
        base_dim = len(self.feature_names)
        if self.use_base_features_only:
            self.feature_dim = base_dim + (1 if self.add_beat_pos else 0)
        else:
            self.feature_dim = first["note_feats"].shape[-1] + (1 if self.add_beat_pos else 0)

        self.samples = self._build_samples()
        if self.max_samples is not None:
            if self.max_samples <= 0:
                raise ValueError("max_samples must be > 0")
            self.samples = self.samples[: self.max_samples]

    def _load_group(self, group_key: str):
        level_map = self.group_paths[group_key]
        ref = self._load_file(level_map[self.levels[0]])
        labels = np.zeros_like(ref["boundary_probs"], dtype=np.float32)
        ref_feats = ref["note_feats"]
        ref_ids = ref["beat_ids"]
        ref_beats = int(ref["num_beats"])

        for level, weight in zip(self.levels, self.level_weights):
            item = self._load_file(level_map[level])
            if item["note_feats"].shape != ref_feats.shape or item["beat_ids"].shape != ref_ids.shape:
                raise ValueError(f"Feature mismatch across levels for {group_key}")
            if int(item["num_beats"]) != ref_beats or item["boundary_probs"].shape != labels.shape:
                raise ValueError(f"Boundary length mismatch across levels for {group_key}")
            labels += float(weight) * item["boundary_probs"].astype(np.float32)

        labels = np.clip(labels, 0.0, 1.0)
        return {
            "note_feats": ref_feats,
            "beat_ids": ref_ids,
            "boundary_probs": labels,
            "num_beats": ref_beats,
        }

    def __getitem__(self, idx):
        sample = self.samples[idx]
        data = self._load_group(sample["group_key"])
        feats = data["note_feats"]
        beat_ids = data["beat_ids"]
        labels_ratio = data["boundary_probs"]
        num_beats = data["num_beats"]
        performer_id = self._get_performer_id(sample["path"])

        beat_start = sample.get("beat_start")
        if beat_start is not None:
            feats, beat_ids, labels_ratio, num_beats, _, _ = self._slice_by_beats(
                feats, beat_ids, labels_ratio, beat_start, sample["beat_end"]
            )
        elif sample["start"] is not None:
            feats, beat_ids, labels_ratio, num_beats, _, _ = self._window_and_rebase(
                feats, beat_ids, labels_ratio, sample["start"], sample["end"]
            )
        else:
            if self.max_len is not None and feats.shape[0] > self.max_len:
                feats = feats[: self.max_len]
                beat_ids = beat_ids[: self.max_len]
                num_beats = int(np.max(beat_ids) + 1) if len(beat_ids) > 0 else 0
            if labels_ratio.shape[0] > num_beats:
                labels_ratio = labels_ratio[:num_beats]

        if self.label_binarize_threshold is not None:
            labels_ratio = (labels_ratio > self.label_binarize_threshold).astype(np.float32)

        feats = self._apply_position_mode(feats)
        feats = self._normalize_features(feats)

        if self.add_beat_pos:
            denom = max(num_beats - 1, 1)
            beat_pos = np.where(beat_ids >= 0, beat_ids, 0).astype(np.float32) / float(denom)
            feats = np.concatenate([feats, beat_pos[:, None]], axis=1)

        feats_t = torch.tensor(feats, dtype=torch.float32)
        beat_ids_t = torch.tensor(beat_ids, dtype=torch.long)
        labels_t = torch.tensor(labels_ratio, dtype=torch.float32)

        return {
            "note_feats": feats_t,
            "beat_ids": beat_ids_t,
            "labels": labels_t,
            "num_beats": num_beats,
            "length": feats_t.shape[0],
            "performer_id": performer_id,
        }

    def _build_samples(self):
        if self.beat_sequence_length is not None:
            if self.beat_stride is None:
                self.beat_stride = self.beat_sequence_length
            if self.beat_sequence_length <= 0 or self.beat_stride <= 0:
                raise ValueError("beat_sequence_length and beat_stride must be > 0")
            samples = []
            for key in self.group_keys:
                path = self.group_paths[key][self.levels[0]]
                num_beats = self._get_num_beats(path)
                if num_beats <= 0:
                    continue
                if num_beats < self.beat_sequence_length:
                    if not self.drop_short:
                        samples.append(
                            {
                                "group_key": key,
                                "path": path,
                                "start": None,
                                "end": None,
                                "beat_start": 0,
                                "beat_end": num_beats,
                            }
                        )
                    continue
                start_indices = list(range(0, num_beats - self.beat_sequence_length + 1, self.beat_stride))
                last_start = num_beats - self.beat_sequence_length
                if not start_indices or start_indices[-1] != last_start:
                    start_indices.append(last_start)
                for start in start_indices:
                    samples.append(
                        {
                            "group_key": key,
                            "path": path,
                            "start": None,
                            "end": None,
                            "beat_start": start,
                            "beat_end": start + self.beat_sequence_length,
                        }
                    )
            if not samples:
                raise FileNotFoundError("No salience beat samples built.")
            return samples

        if self.sequence_length is None:
            return [
                {
                    "group_key": key,
                    "path": self.group_paths[key][self.levels[0]],
                    "start": None,
                    "end": None,
                    "beat_start": None,
                    "beat_end": None,
                }
                for key in self.group_keys
            ]

        if self.stride is None:
            self.stride = self.sequence_length
        if self.sequence_length <= 0 or self.stride <= 0:
            raise ValueError("sequence_length and stride must be > 0")

        samples = []
        for key in self.group_keys:
            path = self.group_paths[key][self.levels[0]]
            num_tokens = self._get_note_len(path)
            if num_tokens < self.sequence_length:
                if not self.drop_short:
                    samples.append(
                        {
                            "group_key": key,
                            "path": path,
                            "start": 0,
                            "end": num_tokens,
                            "beat_start": None,
                            "beat_end": None,
                        }
                    )
                continue
            start_indices = list(range(0, num_tokens - self.sequence_length + 1, self.stride))
            last_start = num_tokens - self.sequence_length
            if not start_indices or start_indices[-1] != last_start:
                start_indices.append(last_start)
            for start in start_indices:
                samples.append(
                    {
                        "group_key": key,
                        "path": path,
                        "start": start,
                        "end": start + self.sequence_length,
                        "beat_start": None,
                        "beat_end": None,
                    }
                )
        if not samples:
            raise FileNotFoundError("No salience note samples built.")
        return samples


def _parse_int_list(text: str):
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def _parse_float_list(text: str):
    return [float(x.strip()) for x in str(text).split(",") if x.strip()]


def create_dataloaders(cfg, levels, level_weights, split_file=None):
    dataset = BeatSalienceDataset(
        data_dir=cfg["data"]["data_dir"],
        levels=levels,
        level_weights=level_weights,
        file_ext=cfg["data"]["file_ext"],
        max_len=cfg["data"]["max_len"],
        sequence_length=cfg["data"].get("sequence_length"),
        stride=cfg["data"].get("stride"),
        beat_sequence_length=cfg["data"].get("beat_sequence_length"),
        beat_stride=cfg["data"].get("beat_stride"),
        drop_short=cfg["data"].get("drop_short", True),
        position_mode=cfg["data"].get("position_mode", "absolute"),
        use_base_features_only=cfg["data"].get("use_base_features_only", False),
        add_beat_pos=cfg["data"].get("add_beat_pos", False),
        max_samples=cfg["data"].get("max_samples"),
        value_ranges=cfg["data"].get("value_ranges"),
        label_binarize_threshold=cfg["data"].get("label_binarize_threshold"),
        performer_id_regex=cfg["data"].get("performer_id_regex"),
    )

    def piece_id_from_path(path: Path) -> str:
        stem = path.stem
        regex = cfg.get("data", {}).get("piece_id_regex")
        if regex:
            patterns = [regex]
            if "\\\\" in regex:
                try:
                    patterns.append(regex.encode("utf-8").decode("unicode_escape"))
                except UnicodeDecodeError:
                    pass
            for pattern in patterns:
                m = re.search(pattern, stem)
                if m:
                    return m.group(1) if m.groups() else m.group(0)
        delim = cfg.get("data", {}).get("piece_id_delim")
        if delim and delim in stem:
            return stem.split(delim)[0]
        if "_" in stem:
            return stem.split("_")[0]
        return stem

    piece_ids = [piece_id_from_path(s["path"]) for s in dataset.samples]
    unique_pieces = sorted({pid for pid in piece_ids})

    if split_file is not None:
        with open(split_file, "r") as f:
            split_cfg = yaml.safe_load(f) or {}
        train_pieces = set(split_cfg.get("train", []))
        val_pieces = set(split_cfg.get("val", []))
        test_pieces = set(split_cfg.get("test", []))
        known = set(unique_pieces)
        missing = sorted((train_pieces | val_pieces | test_pieces) - known)
        if missing:
            raise ValueError(f"Split file references pieces not found in dataset: {missing}")
        train_indices = [i for i, pid in enumerate(piece_ids) if pid in train_pieces]
        val_indices = [i for i, pid in enumerate(piece_ids) if pid in val_pieces]
        test_indices = [i for i, pid in enumerate(piece_ids) if pid in test_pieces]
        if not train_indices:
            raise ValueError("Split file produced empty train set")
        if not val_indices:
            raise ValueError("Split file produced empty val set")
        train_ds = Subset(dataset, train_indices)
        val_ds = Subset(dataset, val_indices)
        test_ds = Subset(dataset, test_indices) if test_indices else None
        split_summary = {
            "mode": "split_file",
            "split_file": str(split_file),
            "train_pieces": sorted(train_pieces),
            "val_pieces": sorted(val_pieces),
            "test_pieces": sorted(test_pieces),
        }
    elif len(unique_pieces) < 2:
        train_ds = dataset
        val_ds = dataset
        test_ds = None
        split_summary = {
            "mode": "degenerate",
            "train_pieces": unique_pieces,
            "val_pieces": unique_pieces,
            "test_pieces": [],
        }
    else:
        rng = random.Random(cfg["training"]["seed"])
        rng.shuffle(unique_pieces)
        train_count = int(cfg["data"]["train_split"] * len(unique_pieces))
        train_count = max(1, min(train_count, len(unique_pieces) - 1))
        train_pieces = set(unique_pieces[:train_count])
        val_pieces = set(unique_pieces[train_count:])
        train_indices = [i for i, pid in enumerate(piece_ids) if pid in train_pieces]
        val_indices = [i for i, pid in enumerate(piece_ids) if pid in val_pieces]
        train_ds = Subset(dataset, train_indices)
        val_ds = Subset(dataset, val_indices)
        test_ds = None
        split_summary = {
            "mode": "random_piece_split",
            "seed": int(cfg["training"]["seed"]),
            "train_pieces": sorted(train_pieces),
            "val_pieces": sorted(val_pieces),
            "test_pieces": [],
        }

    split_summary["salience"] = {
        "levels": [int(x) for x in levels],
        "raw_level_weights": [float(x) for x in level_weights],
        "normalized_level_weights": [float(x) for x in dataset.level_weights.tolist()],
    }

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
    return train_loader, val_loader, test_loader, dataset.feature_dim, dataset, split_summary


def main():
    parser = argparse.ArgumentParser(description="Train salience beat-level boundary model")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument("--device", default=None, help="cpu|cuda|auto")
    parser.add_argument("--sanity_batch", action="store_true", help="Print one batch sanity stats and exit")
    parser.add_argument("--bias_only", action="store_true", help="Train only head bias")
    parser.add_argument("--freeze_base", action="store_true", help="Freeze backbone and train only performer params")
    parser.add_argument("--split_file", default=None, help="Optional YAML file with fixed train/val/test piece split")
    parser.add_argument("--pos_weight", type=float, default=None, help="Override training.pos_weight")
    parser.add_argument("--levels", default=None, help="Comma-separated levels to aggregate")
    parser.add_argument("--level_weights", default=None, help="Comma-separated aggregation weights")
    parser.add_argument("--early_stop_patience", type=int, default=0, help="Stop after N bad val epochs")
    parser.add_argument("--early_stop_min_delta", type=float, default=0.0, help="Minimum improvement to reset patience")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.pos_weight is not None:
        cfg.setdefault("training", {})["pos_weight"] = float(args.pos_weight)

    salience_cfg = cfg.get("salience", {})
    levels = _parse_int_list(args.levels) if args.levels else salience_cfg.get("levels", [1, 2, 3, 4, 5, 6])
    level_weights = (
        _parse_float_list(args.level_weights)
        if args.level_weights
        else salience_cfg.get("level_weights", [3, 6, 12, 24, 48, 96])
    )

    if args.device:
        device = args.device
    else:
        device = cfg["training"].get("device", "auto")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    print(f"Using device: {device}")

    set_seed(cfg["training"]["seed"])

    train_loader, val_loader, test_loader, input_dim, dataset, split_summary = create_dataloaders(
        cfg,
        levels=levels,
        level_weights=level_weights,
        split_file=args.split_file,
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

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"{cfg['trainer']['experiment_name']}_{ts}"
    save_dir = Path(cfg["trainer"]["save_dir"]) / "salience" / exp_name
    save_dir.mkdir(parents=True, exist_ok=True)
    with (save_dir / "split_summary.yaml").open("w") as f:
        yaml.safe_dump(split_summary, f, sort_keys=False)
    if cfg.get("model", {}).get("performer_cond") and getattr(dataset, "performer_map", None):
        with (save_dir / "performer_map.yaml").open("w") as f:
            yaml.safe_dump(dataset.performer_map, f)

    best_val = float("inf")
    epochs = cfg["training"]["epochs"]
    grad_clip = cfg["training"].get("grad_clip", None)
    patience = max(0, int(args.early_stop_patience))
    min_delta = float(args.early_stop_min_delta)
    bad_epochs = 0
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, grad_clip)
        val_loss = evaluate(model, val_loader, device)
        print(f"Epoch {epoch}/{epochs} | train_loss {train_loss:.4f} | val_loss {val_loss:.4f}")

        if val_loss < (best_val - min_delta):
            best_val = val_loss
            best_epoch = epoch
            bad_epochs = 0
            torch.save(model.state_dict(), save_dir / "best.pt")
        else:
            bad_epochs += 1
        torch.save(model.state_dict(), save_dir / "last.pt")

        if patience > 0 and bad_epochs >= patience:
            print(
                f"Early stopping at epoch {epoch}: no val improvement for {bad_epochs} epochs "
                f"(best_epoch={best_epoch}, best_val_loss={best_val:.4f})"
            )
            break

    if test_loader is not None and (save_dir / "best.pt").exists():
        model.load_state_dict(torch.load(save_dir / "best.pt", map_location=device))
        best_test = evaluate(model, test_loader, device)
        print(f"Test | best_loss {best_test:.4f} | best_epoch {best_epoch}")


if __name__ == "__main__":
    main()
