import glob
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.signal import find_peaks


class BeatBoundaryDataset(Dataset):
    """
    Dataset for beat-level boundary prediction (note-level input).

    Optional sliding-window sampling (sequence_length/stride) mirrors the
    main model dataset: fixed-length note windows with an extra last window
    aligned to the end of the piece.
    If beat_sequence_length is set, windows are built by beat index instead
    of note index (beat_start/beat_end), and notes are filtered to those beats.

    Expects each sample file to contain:
      - note_feats: [notes, feature_dim] float array
      - beat_ids: [notes] int array (0-based beat index)
      - boundary_probs: [beats] float array in [0, 1]
      - num_beats: int (optional, inferred if missing)
    Supported formats: .npz (with keys above) or .pt (dict with same keys).
    Files missing required keys will be skipped.
    """

    def __init__(
        self,
        data_dir: str,
        file_ext: str = "npz",
        max_len: Optional[int] = None,
        sequence_length: Optional[int] = None,
        stride: Optional[int] = None,
        beat_sequence_length: Optional[int] = None,
        beat_stride: Optional[int] = None,
        drop_short: bool = True,
        position_mode: str = "absolute",
        use_base_features_only: bool = False,
        label_mode: str = "ratio",
        dist_min_dist: int = 6,
        dist_height: float = 0.15,
        dist_prominence: float = 0.05,
        dist_tau: float = 4.0,
        add_beat_pos: bool = False,
        max_samples: Optional[int] = None,
        value_ranges: Optional[Dict[str, List[float]]] = None,
        label_binarize_threshold: Optional[float] = None,
        performer_id_regex: Optional[str] = None,
    ):
        super().__init__()
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
        self.label_mode = label_mode
        self.dist_min_dist = int(dist_min_dist)
        self.dist_height = float(dist_height)
        self.dist_prominence = float(dist_prominence)
        self.dist_tau = float(dist_tau)
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
        if self.label_mode not in {"ratio", "dist", "dual"}:
            raise ValueError("label_mode must be one of: ratio, dist, dual")

        pattern = str(self.data_dir / f"*.{self.file_ext}")
        all_files: List[Path] = sorted(Path(p) for p in glob.glob(pattern))
        if not all_files:
            raise FileNotFoundError(f"No *.{self.file_ext} files found in {self.data_dir}")

        self.files = [p for p in all_files if self._has_required_keys(p)]
        if not self.files:
            raise FileNotFoundError(
                f"No valid *.{self.file_ext} files with required keys in {self.data_dir}"
            )
        self.performer_map = self._build_performer_map()
        self.num_performers = len(self.performer_map)
        self._sampling_meta_cache: Dict[Path, Dict[str, np.ndarray]] = {}

        # Peek to infer feature dimension
        first = self._load_file(self.files[0])
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

    def _has_required_keys(self, path: Path) -> bool:
        required = {"note_feats", "beat_ids", "boundary_probs"}
        try:
            if self.file_ext == "npz":
                files = set(np.load(path, mmap_mode="r").files)
                return required.issubset(files)
            if self.file_ext == "pt":
                data = torch.load(path, map_location="cpu")
                return required.issubset(set(data.keys()))
        except Exception:
            return False
        return False

    def _extract_performer_tag(self, path: Path) -> Optional[str]:
        stem = path.stem
        regex = self.performer_id_regex or r"(pid[^_]+)"
        try:
            m = re.search(regex, stem)
        except re.error:
            return None
        if not m:
            return None
        return m.group(1) if m.groups() else m.group(0)

    def _build_performer_map(self) -> Dict[str, int]:
        tags = []
        for path in self.files:
            tag = self._extract_performer_tag(path)
            if tag:
                tags.append(tag)
        if not tags:
            return {}
        uniq = sorted(set(tags))
        return {tag: idx + 1 for idx, tag in enumerate(uniq)}

    def _get_performer_id(self, path: Path) -> int:
        if not self.performer_map:
            return 0
        tag = self._extract_performer_tag(path)
        if not tag:
            return 0
        return int(self.performer_map.get(tag, 0))

    def _get_note_len(self, path: Path) -> int:
        if self.file_ext == "npz":
            with np.load(path, mmap_mode="r") as data:
                return int(data["note_feats"].shape[0])
        if self.file_ext == "pt":
            data = torch.load(path, map_location="cpu")
            note_feats = data["note_feats"]
            if isinstance(note_feats, torch.Tensor):
                return int(note_feats.shape[0])
            return int(np.asarray(note_feats).shape[0])
        raise ValueError(f"Unsupported file_ext: {self.file_ext}")

    def _get_num_beats(self, path: Path) -> int:
        if self.file_ext == "npz":
            with np.load(path, mmap_mode="r") as data:
                if "num_beats" in data:
                    return int(data["num_beats"])
                if "boundary_probs" in data:
                    return int(data["boundary_probs"].shape[0])
                if "beat_ids" in data:
                    beat_ids = data["beat_ids"]
                    return int(np.max(beat_ids) + 1) if beat_ids.size > 0 else 0
        if self.file_ext == "pt":
            data = torch.load(path, map_location="cpu")
            if "num_beats" in data:
                return int(data["num_beats"])
            if "boundary_probs" in data:
                return int(np.asarray(data["boundary_probs"]).shape[0])
            if "beat_ids" in data:
                beat_ids = np.asarray(data["beat_ids"])
                return int(np.max(beat_ids) + 1) if beat_ids.size > 0 else 0
        raise ValueError(f"Unsupported file_ext: {self.file_ext}")

    def _build_samples(self) -> List[Dict[str, Any]]:
        if self.beat_sequence_length is not None:
            return self._build_samples_by_beat()
        if self.sequence_length is None:
            return [
                {"path": p, "start": None, "end": None, "beat_start": None, "beat_end": None}
                for p in self.files
            ]

        if self.stride is None:
            self.stride = self.sequence_length
        if self.sequence_length <= 0 or self.stride <= 0:
            raise ValueError("sequence_length and stride must be > 0")

        samples: List[Dict[str, Any]] = []
        for path in self.files:
            num_tokens = self._get_note_len(path)
            if num_tokens < self.sequence_length:
                if not self.drop_short:
                    samples.append(
                        {
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
                        "path": path,
                        "start": start,
                        "end": start + self.sequence_length,
                        "beat_start": None,
                        "beat_end": None,
                    }
                )

        if not samples:
            raise FileNotFoundError(
                "No samples built. Check sequence_length/stride or set drop_short=False."
            )
        return samples

    def _build_samples_by_beat(self) -> List[Dict[str, Any]]:
        if self.beat_stride is None:
            self.beat_stride = self.beat_sequence_length
        if self.beat_sequence_length is None or self.beat_sequence_length <= 0 or self.beat_stride <= 0:
            raise ValueError("beat_sequence_length and beat_stride must be > 0")

        samples: List[Dict[str, Any]] = []
        for path in self.files:
            num_beats = self._get_num_beats(path)
            if num_beats <= 0:
                continue
            if num_beats < self.beat_sequence_length:
                if not self.drop_short:
                    samples.append(
                        {
                            "path": path,
                            "start": None,
                            "end": None,
                            "beat_start": 0,
                            "beat_end": num_beats,
                        }
                    )
                continue

            start_indices = list(
                range(0, num_beats - self.beat_sequence_length + 1, self.beat_stride)
            )
            last_start = num_beats - self.beat_sequence_length
            if not start_indices or start_indices[-1] != last_start:
                start_indices.append(last_start)

            for start in start_indices:
                samples.append(
                    {
                        "path": path,
                        "start": None,
                        "end": None,
                        "beat_start": start,
                        "beat_end": start + self.beat_sequence_length,
                    }
                )

        if not samples:
            raise FileNotFoundError(
                "No beat samples built. Check beat_sequence_length/beat_stride or set drop_short=False."
            )
        return samples

    def _load_sampling_meta(self, path: Path) -> Dict[str, np.ndarray]:
        cached = self._sampling_meta_cache.get(path)
        if cached is not None:
            return cached

        if self.file_ext == "npz":
            with np.load(path, mmap_mode="r") as data:
                beat_ids = np.asarray(data["beat_ids"])
                boundary = np.asarray(data["boundary_probs"], dtype=np.float32)
        elif self.file_ext == "pt":
            data = torch.load(path, map_location="cpu")
            beat_ids = data["beat_ids"].numpy() if isinstance(data["beat_ids"], torch.Tensor) else np.asarray(data["beat_ids"])
            raw_boundary = data["boundary_probs"].numpy() if isinstance(data["boundary_probs"], torch.Tensor) else data["boundary_probs"]
            boundary = np.asarray(raw_boundary, dtype=np.float32)
        else:
            raise ValueError(f"Unsupported file_ext: {self.file_ext}")

        out = {"beat_ids": beat_ids, "boundary_probs": boundary}
        self._sampling_meta_cache[path] = out
        return out

    def sample_boundary_stats(self, idx: int, threshold: float = 0.0) -> Dict[str, int]:
        sample = self.samples[idx]
        meta = self._load_sampling_meta(sample["path"])
        beat_ids = meta["beat_ids"]
        labels_ratio = meta["boundary_probs"]

        beat_start = sample.get("beat_start")
        if beat_start is not None:
            beat_start = max(int(beat_start), 0)
            beat_end = min(int(sample["beat_end"]), int(labels_ratio.shape[0]))
            window = labels_ratio[beat_start:beat_end]
        elif sample["start"] is not None:
            beat_ids_win = beat_ids[sample["start"] : sample["end"]]
            valid = beat_ids_win >= 0
            if np.any(valid):
                bmin = int(np.min(beat_ids_win[valid]))
                bmax = int(np.max(beat_ids_win[valid]))
                window = labels_ratio[bmin : bmax + 1]
            else:
                window = labels_ratio[:0]
        else:
            window = labels_ratio

        pos_count = int((window > float(threshold)).sum())
        return {
            "num_beats": int(window.shape[0]),
            "pos_count": pos_count,
            "has_boundary": int(pos_count > 0),
        }

    def _load_file(self, path: Path) -> Dict[str, Any]:
        if self.file_ext == "npz":
            data = np.load(path)
            note_feats = data["note_feats"]
            beat_ids = data["beat_ids"]
            boundary = data["boundary_probs"]
            num_beats = int(data["num_beats"]) if "num_beats" in data else None
        elif self.file_ext == "pt":
            data = torch.load(path, map_location="cpu")
            note_feats = data["note_feats"].numpy() if isinstance(data["note_feats"], torch.Tensor) else data[
                "note_feats"]
            beat_ids = data["beat_ids"].numpy() if isinstance(data["beat_ids"], torch.Tensor) else data["beat_ids"]
            boundary = data["boundary_probs"].numpy() if isinstance(data["boundary_probs"], torch.Tensor) else data[
                "boundary_probs"]
            num_beats = int(data["num_beats"]) if "num_beats" in data else None
        else:
            raise ValueError(f"Unsupported file_ext: {self.file_ext}")

        if num_beats is None:
            num_beats = int(np.max(beat_ids) + 1) if len(beat_ids) > 0 else 0
        if self.use_base_features_only:
            if note_feats.shape[1] < len(self.feature_names):
                raise ValueError("note_feats has fewer columns than base features.")
            note_feats = note_feats[:, : len(self.feature_names)]
        return {"note_feats": note_feats, "beat_ids": beat_ids, "boundary_probs": boundary, "num_beats": num_beats}

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

    def _apply_position_mode(self, feats: np.ndarray) -> np.ndarray:
        if feats.size == 0 or feats.shape[1] < 3:
            return feats
        if self.position_mode == "absolute":
            return feats
        feats = feats.copy()
        if self.position_mode == "zero":
            feats[:, 2] = 0.0
            return feats
        # window-relative position in [0, 1]
        pos = feats[:, 2]
        pmin = float(pos.min())
        pmax = float(pos.max())
        denom = max(pmax - pmin, 1e-6)
        feats[:, 2] = (pos - pmin) / denom
        return feats

    def _normalize_features(self, feats: np.ndarray) -> np.ndarray:
        feats = feats.copy()
        for i, name in enumerate(self.feature_names):
            min_v, max_v = self.value_ranges[name]
            denom = max(max_v - min_v, 1e-6)
            vals = 2.0 * (feats[:, i] - min_v) / denom - 1.0
            feats[:, i] = np.clip(vals, -1.0, 1.0)
        return feats

    def _ratio_to_locs(self, boundary_ratio: np.ndarray) -> np.ndarray:
        x = np.asarray(boundary_ratio, dtype=float)
        if x.size == 0:
            return np.array([], dtype=int)
        peaks, _ = find_peaks(
            x,
            distance=max(self.dist_min_dist, 1),
            height=self.dist_height,
            prominence=self.dist_prominence,
        )
        return peaks.astype(int)

    def _distance_to_nearest_boundary(self, length: int, locs: np.ndarray) -> np.ndarray:
        if length <= 0:
            return np.zeros(0, dtype=np.float32)
        d = np.full(length, length, dtype=np.float32)
        if locs.size == 0:
            return d
        locs = locs[(locs >= 0) & (locs < length)]
        if locs.size == 0:
            return d
        d[locs] = 0.0
        for i in range(1, length):
            d[i] = min(d[i], d[i - 1] + 1.0)
        for i in range(length - 2, -1, -1):
            d[i] = min(d[i], d[i + 1] + 1.0)
        return d

    def _distance_target(self, boundary_ratio: np.ndarray) -> np.ndarray:
        locs = self._ratio_to_locs(boundary_ratio)
        d = self._distance_to_nearest_boundary(len(boundary_ratio), locs)
        tau = max(self.dist_tau, 1e-6)
        return np.exp(-d / tau).astype(np.float32)

    def __len__(self) -> int:
        return len(self.samples)

    def _window_and_rebase(
        self,
        feats: np.ndarray,
        beat_ids: np.ndarray,
        labels: np.ndarray,
        start: int,
        end: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int, int]:
        feats = feats[start:end]
        beat_ids = beat_ids[start:end]

        if self.max_len is not None and feats.shape[0] > self.max_len:
            feats = feats[: self.max_len]
            beat_ids = beat_ids[: self.max_len]

        if beat_ids.size == 0:
            return feats, beat_ids, labels[:0], 0, 0, -1

        valid_mask = beat_ids >= 0
        if not np.any(valid_mask):
            return feats, beat_ids, labels[:0], 0, 0, -1

        bmin = int(np.min(beat_ids[valid_mask]))
        bmax = int(np.max(beat_ids[valid_mask]))
        beat_ids = beat_ids.copy()
        beat_ids[valid_mask] -= bmin
        num_beats = bmax - bmin + 1
        labels = labels[bmin : bmax + 1] if num_beats > 0 else labels[:0]
        return feats, beat_ids, labels, num_beats, bmin, bmax

    def _slice_by_beats(
        self,
        feats: np.ndarray,
        beat_ids: np.ndarray,
        labels: np.ndarray,
        beat_start: int,
        beat_end: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int, int]:
        if beat_end <= beat_start:
            return feats[:0], beat_ids[:0], labels[:0], 0, 0, 0
        beat_end = min(int(beat_end), int(labels.shape[0]))
        beat_start = max(int(beat_start), 0)
        if beat_end <= beat_start:
            return feats[:0], beat_ids[:0], labels[:0], 0, beat_start, beat_end

        mask = (beat_ids >= beat_start) & (beat_ids < beat_end)
        feats = feats[mask]
        beat_ids = beat_ids[mask] - beat_start
        labels = labels[beat_start:beat_end]
        num_beats = beat_end - beat_start

        if self.max_len is not None and feats.shape[0] > self.max_len:
            feats = feats[: self.max_len]
            beat_ids = beat_ids[: self.max_len]

        return feats, beat_ids, labels, num_beats, beat_start, beat_end

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        data = self._load_file(sample["path"])
        feats = data["note_feats"]
        beat_ids = data["beat_ids"]
        labels_ratio = data["boundary_probs"]
        num_beats = data["num_beats"]
        performer_id = self._get_performer_id(sample["path"])

        labels_dist = None
        if self.label_mode in {"dist", "dual"}:
            labels_dist = self._distance_target(labels_ratio)
        elif self.label_mode != "ratio":
            raise ValueError(f"Unsupported label_mode: {self.label_mode}. Use 'ratio', 'dist', or 'dual'.")

        beat_start = sample.get("beat_start")
        if beat_start is not None:
            feats, beat_ids, labels_ratio, num_beats, beat_start, beat_end = self._slice_by_beats(
                feats, beat_ids, labels_ratio, beat_start, sample["beat_end"]
            )
            if labels_dist is not None:
                labels_dist = labels_dist[beat_start:beat_end]
        elif sample["start"] is not None:
            feats, beat_ids, labels_ratio, num_beats, bmin, bmax = self._window_and_rebase(
                feats, beat_ids, labels_ratio, sample["start"], sample["end"]
            )
            if labels_dist is not None:
                labels_dist = labels_dist[bmin : bmax + 1] if num_beats > 0 else labels_dist[:0]
        else:
            # Optional truncate
            if self.max_len is not None and feats.shape[0] > self.max_len:
                feats = feats[: self.max_len]
                beat_ids = beat_ids[: self.max_len]
                num_beats = int(np.max(beat_ids) + 1) if len(beat_ids) > 0 else 0

            if labels_ratio.shape[0] > num_beats:
                labels_ratio = labels_ratio[:num_beats]
            if labels_dist is not None and labels_dist.shape[0] > num_beats:
                labels_dist = labels_dist[:num_beats]

        if self.label_binarize_threshold is not None and self.label_mode == "ratio":
            labels_ratio = (labels_ratio > self.label_binarize_threshold).astype(np.float32)

        feats = self._apply_position_mode(feats)
        feats = self._normalize_features(feats)

        if self.add_beat_pos:
            denom = max(num_beats - 1, 1)
            beat_pos = np.where(beat_ids >= 0, beat_ids, 0).astype(np.float32) / float(denom)
            feats = np.concatenate([feats, beat_pos[:, None]], axis=1)

        feats_t = torch.tensor(feats, dtype=torch.float32)
        beat_ids_t = torch.tensor(beat_ids, dtype=torch.long)
        if self.label_mode == "ratio":
            labels_t = torch.tensor(labels_ratio, dtype=torch.float32)
        else:
            labels_t = torch.tensor(labels_dist, dtype=torch.float32)
        length = feats_t.shape[0]

        out = {
            "note_feats": feats_t,
            "beat_ids": beat_ids_t,
            "labels": labels_t,
            "num_beats": num_beats,
            "length": length,
            "performer_id": performer_id,
        }
        if self.label_mode == "dual":
            out["labels_prob"] = torch.tensor(labels_ratio, dtype=torch.float32)
        return out


def collate_beat(batch: List[Dict[str, Any]], pad_to: Optional[int] = None) -> Dict[str, torch.Tensor]:
    """Pad note sequences to max length and pad labels to max beats."""
    lengths = [b["length"] for b in batch]
    max_len = pad_to if pad_to is not None else max(lengths)
    feat_dim = batch[0]["note_feats"].shape[-1]
    max_beats = max(b["num_beats"] for b in batch) if batch else 0

    note_feats = torch.zeros(len(batch), max_len, feat_dim, dtype=torch.float32)
    beat_ids = torch.full((len(batch), max_len), -1, dtype=torch.long)
    labels = torch.zeros(len(batch), max_beats, dtype=torch.float32)
    has_prob = any("labels_prob" in b for b in batch)
    labels_prob = torch.zeros(len(batch), max_beats, dtype=torch.float32) if has_prob else None
    has_performer = any("performer_id" in b for b in batch)
    performer_ids = torch.zeros(len(batch), dtype=torch.long) if has_performer else None
    attn_mask = torch.zeros(len(batch), max_len, dtype=torch.bool)

    for i, item in enumerate(batch):
        l = min(item["length"], max_len)
        note_feats[i, :l] = item["note_feats"][:l]
        beat_ids[i, :l] = item["beat_ids"][:l]
        if item["labels"].numel() > 0:
            labels[i, : item["labels"].shape[0]] = item["labels"]
        if has_prob and "labels_prob" in item and item["labels_prob"].numel() > 0:
            labels_prob[i, : item["labels_prob"].shape[0]] = item["labels_prob"]
        attn_mask[i, :l] = True
        if has_performer:
            performer_ids[i] = int(item.get("performer_id", 0))

    out = {
        "note_feats": note_feats,
        "beat_ids": beat_ids,
        "labels": labels,
        "attn_mask": attn_mask,
        "lengths": torch.tensor(lengths),
        "num_beats": torch.tensor([b["num_beats"] for b in batch], dtype=torch.long),
        "max_beats": torch.tensor(max_beats, dtype=torch.long),
    }
    if has_prob:
        out["labels_prob"] = labels_prob
    if has_performer:
        out["performer_ids"] = performer_ids
    return out
