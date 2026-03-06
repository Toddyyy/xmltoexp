import argparse
import random
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
import yaml

from dataset_beat import collate_beat
from train_beat import (
    build_aux_filters,
    build_model,
    load_aux_split,
    load_config,
    load_piece_split,
    print_batch_sanity,
    recording_id_from_path,
    set_bias_only,
    set_seed,
    train_one_epoch,
    evaluate,
)


class BeatSalienceDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        levels: List[int],
        level_weights: List[float],
        file_ext: str = "npz",
        max_len: Optional[int] = None,
        beat_sequence_length: Optional[int] = None,
        beat_stride: Optional[int] = None,
        drop_short: bool = True,
        position_mode: str = "absolute",
        use_base_features_only: bool = False,
        add_beat_pos: bool = False,
        max_samples: Optional[int] = None,
        value_ranges: Optional[Dict[str, List[float]]] = None,
        performer_id_regex: Optional[str] = None,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.levels = list(levels)
        self.level_weights = np.asarray(level_weights, dtype=np.float32)
        self.file_ext = file_ext
        self.max_len = max_len
        self.beat_sequence_length = beat_sequence_length
        self.beat_stride = beat_stride
        self.drop_short = drop_short
        self.position_mode = position_mode
        self.use_base_features_only = bool(use_base_features_only)
        self.add_beat_pos = bool(add_beat_pos)
        self.max_samples = max_samples
        self.performer_id_regex = performer_id_regex
        self.feature_names = [
            "pitch_midi",
            "duration",
            "position",
            "part_idx",
            "is_accent",
            "is_staccato",
        ]
        self.value_ranges = self._build_value_ranges(value_ranges)
        if self.position_mode not in {"absolute", "window", "zero"}:
            raise ValueError("position_mode must be one of: absolute, window, zero")

        grouped = self._group_level_files()
        if not grouped:
            raise FileNotFoundError(f"No grouped multi-level samples found in {self.data_dir}")
        self.groups = grouped
        self.performer_map = self._build_performer_map()
        self.num_performers = len(self.performer_map)

        first = self._load_group(self.groups[0]["base"])
        base_dim = len(self.feature_names)
        if self.use_base_features_only:
            self.feature_dim = base_dim + (1 if self.add_beat_pos else 0)
        else:
            self.feature_dim = first["note_feats"].shape[-1] + (1 if self.add_beat_pos else 0)
        self.samples = self._build_samples()
        if self.max_samples is not None:
            self.samples = self.samples[: self.max_samples]

    def _group_level_files(self):
        files = sorted(self.data_dir.glob(f"*.{self.file_ext}"))
        groups: Dict[str, Dict[int, Path]] = {}
        for path in files:
            stem = path.stem
            if "_L" not in stem:
                continue
            base, level_text = stem.rsplit("_L", 1)
            try:
                level = int(level_text)
            except ValueError:
                continue
            groups.setdefault(base, {})[level] = path
        out = []
        required = set(self.levels)
        for base, level_map in sorted(groups.items()):
            if required.issubset(level_map.keys()):
                out.append({"base": base, "levels": {lvl: level_map[lvl] for lvl in self.levels}})
        return out

    def _extract_performer_tag(self, stem: str) -> Optional[str]:
        regex = self.performer_id_regex or r"(pid[^_]+)"
        try:
            m = re.search(regex, stem)
        except re.error:
            return None
        if not m:
            return None
        return m.group(1) if m.groups() else m.group(0)

    def _build_performer_map(self):
        tags = []
        for item in self.groups:
            tag = self._extract_performer_tag(item["base"])
            if tag:
                tags.append(tag)
        return {tag: idx + 1 for idx, tag in enumerate(sorted(set(tags)))}

    def _get_performer_id(self, stem: str) -> int:
        tag = self._extract_performer_tag(stem)
        if not tag:
            return 0
        return int(self.performer_map.get(tag, 0))

    def _build_value_ranges(self, value_ranges: Optional[Dict[str, List[float]]]):
        default_ranges = {
            "pitch_midi": (0.0, 127.0),
            "duration": (0.0, 8.0),
            "position": (0.0, 1.0) if self.position_mode == "window" else (0.0, 4096.0),
            "part_idx": (0.0, 7.0),
            "is_accent": (0.0, 1.0),
            "is_staccato": (0.0, 1.0),
        }
        ranges = {}
        if value_ranges:
            for k, v in value_ranges.items():
                if isinstance(v, (list, tuple)) and len(v) == 2:
                    ranges[k] = (float(v[0]), float(v[1]))
        for k in self.feature_names:
            ranges.setdefault(k, default_ranges[k])
        return ranges

    def _load_group(self, base: str):
        item = next(g for g in self.groups if g["base"] == base)
        level_data = []
        for level in self.levels:
            with np.load(item["levels"][level]) as data:
                level_data.append(
                    {
                        "note_feats": data["note_feats"],
                        "beat_ids": data["beat_ids"],
                        "boundary_probs": data["boundary_probs"],
                        "num_beats": int(data["num_beats"]) if "num_beats" in data else None,
                    }
                )
        first = level_data[0]
        note_feats = first["note_feats"]
        beat_ids = first["beat_ids"]
        num_beats = first["num_beats"] if first["num_beats"] is not None else int(np.max(beat_ids) + 1)
        if self.use_base_features_only:
            note_feats = note_feats[:, : len(self.feature_names)]
        labels = np.zeros(num_beats, dtype=np.float32)
        for weight, level_item in zip(self.level_weights, level_data):
            y = np.asarray(level_item["boundary_probs"], dtype=np.float32)
            if y.shape[0] < num_beats:
                y = np.pad(y, (0, num_beats - y.shape[0]))
            elif y.shape[0] > num_beats:
                y = y[:num_beats]
            labels += float(weight) * y
        labels = np.clip(labels, 0.0, 1.0).astype(np.float32)
        return {
            "note_feats": note_feats,
            "beat_ids": beat_ids,
            "labels": labels,
            "num_beats": num_beats,
        }

    def _build_samples(self):
        samples = []
        for item in self.groups:
            stem = item["base"]
            group = self._load_group(stem)
            num_beats = int(group["num_beats"])
            if self.beat_sequence_length is None:
                samples.append({"base": stem, "beat_start": None, "beat_end": None})
                continue
            stride = self.beat_stride or self.beat_sequence_length
            if num_beats < self.beat_sequence_length:
                if not self.drop_short:
                    samples.append({"base": stem, "beat_start": 0, "beat_end": num_beats})
                continue
            starts = list(range(0, num_beats - self.beat_sequence_length + 1, stride))
            last_start = num_beats - self.beat_sequence_length
            if not starts or starts[-1] != last_start:
                starts.append(last_start)
            for start in starts:
                samples.append({"base": stem, "beat_start": start, "beat_end": start + self.beat_sequence_length})
        if not samples:
            raise FileNotFoundError("No salience samples built. Check beat_sequence_length/beat_stride.")
        return samples

    def __len__(self):
        return len(self.samples)

    def _apply_position_mode(self, feats: np.ndarray) -> np.ndarray:
        if feats.size == 0 or feats.shape[1] < 3:
            return feats
        if self.position_mode == "absolute":
            return feats
        feats = feats.copy()
        if self.position_mode == "zero":
            feats[:, 2] = 0.0
            return feats
        pos = feats[:, 2]
        pmin = float(pos.min())
        pmax = float(pos.max())
        feats[:, 2] = (pos - pmin) / max(pmax - pmin, 1e-6)
        return feats

    def _normalize_features(self, feats: np.ndarray) -> np.ndarray:
        feats = feats.copy()
        for i, name in enumerate(self.feature_names):
            min_v, max_v = self.value_ranges[name]
            vals = 2.0 * (feats[:, i] - min_v) / max(max_v - min_v, 1e-6) - 1.0
            feats[:, i] = np.clip(vals, -1.0, 1.0)
        return feats

    def _slice_by_beats(self, feats, beat_ids, labels, beat_start, beat_end):
        mask = (beat_ids >= beat_start) & (beat_ids < beat_end)
        feats = feats[mask]
        beat_ids = beat_ids[mask] - beat_start
        labels = labels[beat_start:beat_end]
        if self.max_len is not None and feats.shape[0] > self.max_len:
            feats = feats[: self.max_len]
            beat_ids = beat_ids[: self.max_len]
        return feats, beat_ids, labels, beat_end - beat_start

    def __getitem__(self, idx):
        sample = self.samples[idx]
        group = self._load_group(sample["base"])
        feats = group["note_feats"]
        beat_ids = group["beat_ids"]
        labels = group["labels"]
        num_beats = group["num_beats"]

        if sample["beat_start"] is not None:
            feats, beat_ids, labels, num_beats = self._slice_by_beats(
                feats, beat_ids, labels, sample["beat_start"], sample["beat_end"]
            )
        else:
            if self.max_len is not None and feats.shape[0] > self.max_len:
                feats = feats[: self.max_len]
                beat_ids = beat_ids[: self.max_len]

        feats = self._apply_position_mode(feats)
        feats = self._normalize_features(feats)
        if self.add_beat_pos:
            denom = max(num_beats - 1, 1)
            beat_pos = np.where(beat_ids >= 0, beat_ids, 0).astype(np.float32) / float(denom)
            feats = np.concatenate([feats, beat_pos[:, None]], axis=1)

        return {
            "note_feats": torch.tensor(feats, dtype=torch.float32),
            "beat_ids": torch.tensor(beat_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.float32),
            "num_beats": int(num_beats),
            "length": int(feats.shape[0]),
            "performer_id": self._get_performer_id(sample["base"]),
        }


def parse_int_list(text: Optional[str], default: List[int]) -> List[int]:
    if not text:
        return list(default)
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_float_list(text: Optional[str], count: int) -> List[float]:
    if not text:
        vals = list(range(1, count + 1))
    else:
        vals = [float(x.strip()) for x in text.split(",") if x.strip()]
    if len(vals) != count:
        raise ValueError(f"Expected {count} weights, got {len(vals)}")
    s = sum(vals)
    if s <= 0:
        raise ValueError("level_weights must sum to > 0")
    return [v / s for v in vals]


def piece_id_from_base(base: str, cfg) -> str:
    regex = cfg.get("data", {}).get("piece_id_regex")
    if regex:
        m = re.search(regex, base)
        if not m and "\\\\" in regex:
            m = re.search(regex.encode("utf-8").decode("unicode_escape"), base)
        if m:
            return m.group(1) if m.groups() else m.group(0)
    delim = cfg.get("data", {}).get("piece_id_delim")
    if delim and delim in base:
        return base.split(delim)[0]
    return base


def create_dataloaders(cfg, levels, level_weights, split_file=None, aux_split_file=None, aux_mode=None, aux_targets=None):
    dataset = BeatSalienceDataset(
        data_dir=cfg["data"]["data_dir"],
        levels=levels,
        level_weights=level_weights,
        file_ext=cfg["data"]["file_ext"],
        max_len=cfg["data"]["max_len"],
        beat_sequence_length=cfg["data"].get("beat_sequence_length"),
        beat_stride=cfg["data"].get("beat_stride"),
        drop_short=cfg["data"].get("drop_short", True),
        position_mode=cfg["data"].get("position_mode", "absolute"),
        use_base_features_only=cfg["data"].get("use_base_features_only", False),
        add_beat_pos=cfg["data"].get("add_beat_pos", False),
        max_samples=cfg["data"].get("max_samples"),
        value_ranges=cfg["data"].get("value_ranges"),
        performer_id_regex=cfg["data"].get("performer_id_regex"),
    )

    aux_summary = {"mode": aux_mode, "selected_targets": [], "excluded_ids": 0, "restricted_pieces": []}
    if aux_split_file and aux_mode:
        aux_data = load_aux_split(aux_split_file)
        aux_filter = build_aux_filters(aux_data, aux_mode=aux_mode, aux_targets=aux_targets)
        filtered = []
        if aux_mode == "heldout_pianists":
            excluded = set(aux_filter["excluded_ids"])
            for g in dataset.groups:
                rid = recording_id_from_path(Path(g["base"]))
                if rid not in excluded:
                    filtered.append(g)
            aux_summary = {
                "mode": aux_mode,
                "selected_targets": aux_filter["selected_targets"],
                "excluded_ids": len(excluded),
                "restricted_pieces": [],
            }
        else:
            keep_map = aux_filter["keep_map"]
            for g in dataset.groups:
                piece = piece_id_from_base(g["base"], cfg)
                rid = recording_id_from_path(Path(g["base"]))
                if piece in keep_map:
                    if rid in keep_map[piece]:
                        filtered.append(g)
                else:
                    filtered.append(g)
            aux_summary = {
                "mode": aux_mode,
                "selected_targets": aux_filter["selected_targets"],
                "excluded_ids": 0,
                "restricted_pieces": sorted(keep_map.keys()),
            }
        dataset.groups = filtered
        dataset.samples = dataset._build_samples()

    group_piece_ids = [piece_id_from_base(g["base"], cfg) for g in dataset.groups]
    unique_pieces = sorted(set(group_piece_ids))
    group_index_by_base = {g["base"]: i for i, g in enumerate(dataset.groups)}

    def sample_piece_id(sample):
        return piece_id_from_base(sample["base"], cfg)

    if split_file:
        split_meta = load_piece_split(split_file)
        known = set(unique_pieces)
        requested = split_meta["train"] | split_meta["val"] | split_meta["test"]
        missing = sorted(requested - known)
        if missing:
            raise ValueError(f"Split file references pieces not found in dataset: {missing}")
        train_idx = [i for i, s in enumerate(dataset.samples) if sample_piece_id(s) in split_meta["train"]]
        val_idx = [i for i, s in enumerate(dataset.samples) if sample_piece_id(s) in split_meta["val"]]
        test_idx = [i for i, s in enumerate(dataset.samples) if sample_piece_id(s) in split_meta["test"]]
    else:
        rng = random.Random(cfg["training"]["seed"])
        pieces = unique_pieces[:]
        rng.shuffle(pieces)
        train_count = max(1, min(int(cfg["data"]["train_split"] * len(pieces)), len(pieces) - 1))
        train_pieces = set(pieces[:train_count])
        val_pieces = set(pieces[train_count:])
        train_idx = [i for i, s in enumerate(dataset.samples) if sample_piece_id(s) in train_pieces]
        val_idx = [i for i, s in enumerate(dataset.samples) if sample_piece_id(s) in val_pieces]
        test_idx = []

    collate_fn = lambda batch: collate_beat(batch, pad_to=cfg["data"]["max_len"])
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=cfg["data"]["batch_size"], shuffle=True, num_workers=cfg["data"]["num_workers"], collate_fn=collate_fn, pin_memory=True)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=cfg["data"]["batch_size"], shuffle=False, num_workers=cfg["data"]["num_workers"], collate_fn=collate_fn, pin_memory=True)
    test_loader = None
    if test_idx:
        test_loader = DataLoader(Subset(dataset, test_idx), batch_size=cfg["data"]["batch_size"], shuffle=False, num_workers=cfg["data"]["num_workers"], collate_fn=collate_fn, pin_memory=True)

    split_summary = {
        "all_pieces": unique_pieces,
        "train_pieces": sorted({sample_piece_id(dataset.samples[i]) for i in train_idx}),
        "val_pieces": sorted({sample_piece_id(dataset.samples[i]) for i in val_idx}),
        "test_pieces": sorted({sample_piece_id(dataset.samples[i]) for i in test_idx}),
        "aux_filter": aux_summary,
        "aggregation": {
            "levels": levels,
            "level_weights": level_weights,
        },
    }
    return train_loader, val_loader, test_loader, dataset.feature_dim, dataset, split_summary


def main():
    parser = argparse.ArgumentParser(description="Train a single beat-salience model from weighted multi-level labels")
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--levels", default="1,2,3,4,5", help="Comma-separated levels to aggregate")
    parser.add_argument("--level_weights", default=None, help="Comma-separated weights aligned with --levels")
    parser.add_argument("--sanity_batch", action="store_true")
    parser.add_argument("--bias_only", action="store_true")
    parser.add_argument("--freeze_base", action="store_true")
    parser.add_argument("--split_file", default=None)
    parser.add_argument("--aux_split_file", default=None)
    parser.add_argument("--aux_mode", default=None, choices=["heldout_pianists", "same_piece_80"])
    parser.add_argument("--aux_targets", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = args.device or cfg["training"].get("device", "auto")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    print(f"Using device: {device}")

    levels = parse_int_list(args.levels, [1, 2, 3, 4, 5])
    level_weights = parse_float_list(args.level_weights, len(levels))
    set_seed(cfg["training"]["seed"])

    aux_targets = [x.strip() for x in args.aux_targets.split(",")] if args.aux_targets else None
    train_loader, val_loader, test_loader, input_dim, dataset, split_summary = create_dataloaders(
        cfg,
        levels=levels,
        level_weights=level_weights,
        split_file=args.split_file,
        aux_split_file=args.aux_split_file,
        aux_mode=args.aux_mode,
        aux_targets=aux_targets,
    )

    if args.sanity_batch:
        print_batch_sanity(next(iter(train_loader)))
        return

    if cfg.get("model", {}).get("performer_cond"):
        cfg["model"]["performer_vocab_size"] = int(dataset.num_performers) + 1
    model = build_model(cfg, input_dim=input_dim).to(device)

    trainable_params = model.parameters()
    weight_decay = cfg["training"]["weight_decay"]
    if args.bias_only:
        trainable_params = set_bias_only(model)
        weight_decay = 0.0
    if args.freeze_base or cfg["training"].get("freeze_base", False):
        for name, param in model.named_parameters():
            param.requires_grad = name.startswith("performer_")
        trainable_params = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW(trainable_params, lr=cfg["training"]["lr"], weight_decay=weight_decay)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"{cfg['trainer']['experiment_name']}_salience_{ts}"
    save_dir = Path(cfg["trainer"]["save_dir"]) / "salience" / exp_name
    save_dir.mkdir(parents=True, exist_ok=True)
    with (save_dir / "split_summary.yaml").open("w") as f:
        yaml.safe_dump(split_summary, f, sort_keys=False)

    best_val = float("inf")
    for epoch in range(1, cfg["training"]["epochs"] + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, cfg["training"].get("grad_clip"))
        val_loss = evaluate(model, val_loader, device)
        print(f"Epoch {epoch}/{cfg['training']['epochs']} | train_loss {train_loss:.4f} | val_loss {val_loss:.4f}")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), save_dir / "best.pt")
        torch.save(model.state_dict(), save_dir / "last.pt")

    if test_loader is not None:
        best_model = build_model(cfg, input_dim=input_dim).to(device)
        best_model.load_state_dict(torch.load(save_dir / "best.pt", map_location=device))
        best_test = evaluate(best_model, test_loader, device)
        print(f"Test | best_loss {best_test:.4f}")
        with (save_dir / "test_metrics.yaml").open("w") as f:
            yaml.safe_dump({"best_test_loss": float(best_test)}, f, sort_keys=False)


if __name__ == "__main__":
    main()
