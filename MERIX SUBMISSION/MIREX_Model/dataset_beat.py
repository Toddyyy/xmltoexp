import glob
import os
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


class BeatBoundaryDataset(Dataset):
    """
    Dataset for beat-level boundary prediction.

    Expects each sample file to contain:
      - score_feats: [beats, feature_dim] float array
      - boundary_probs: [beats] float array in [0, 1]
    Supported formats: .npz (with keys above) or .pt (dict with same keys).
    """

    def __init__(self, data_dir: str, file_ext: str = "npz", max_len: Optional[int] = None):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.file_ext = file_ext
        self.max_len = max_len

        pattern = str(self.data_dir / f"*.{self.file_ext}")
        self.files: List[Path] = sorted(Path(p) for p in glob.glob(pattern))
        if not self.files:
            raise FileNotFoundError(f"No *.{self.file_ext} files found in {self.data_dir}")

        # Peek to infer feature dimension
        first = self._load_file(self.files[0])
        self.feature_dim = first["score_feats"].shape[-1]

    def _load_file(self, path: Path) -> Dict[str, Any]:
        if self.file_ext == "npz":
            data = np.load(path)
            score_feats = data["score_feats"]
            boundary = data["boundary_probs"]
        elif self.file_ext == "pt":
            data = torch.load(path, map_location="cpu")
            score_feats = data["score_feats"].numpy() if isinstance(data["score_feats"], torch.Tensor) else data[
                "score_feats"]
            boundary = data["boundary_probs"].numpy() if isinstance(data["boundary_probs"], torch.Tensor) else data[
                "boundary_probs"]
        else:
            raise ValueError(f"Unsupported file_ext: {self.file_ext}")

        return {"score_feats": score_feats, "boundary_probs": boundary}

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        data = self._load_file(self.files[idx])
        feats = data["score_feats"]
        labels = data["boundary_probs"]

        # Optional truncate
        if self.max_len is not None and feats.shape[0] > self.max_len:
            feats = feats[: self.max_len]
            labels = labels[: self.max_len]

        feats_t = torch.tensor(feats, dtype=torch.float32)
        labels_t = torch.tensor(labels, dtype=torch.float32)
        length = feats_t.shape[0]

        return {"score_feats": feats_t, "labels": labels_t, "length": length}


def collate_beat(batch: List[Dict[str, Any]], pad_to: Optional[int] = None) -> Dict[str, torch.Tensor]:
    """Pad batch to max length (or pad_to if provided)."""
    lengths = [b["length"] for b in batch]
    max_len = pad_to if pad_to is not None else max(lengths)
    feat_dim = batch[0]["score_feats"].shape[-1]

    score_feats = torch.zeros(len(batch), max_len, feat_dim, dtype=torch.float32)
    labels = torch.zeros(len(batch), max_len, dtype=torch.float32)
    attn_mask = torch.zeros(len(batch), max_len, dtype=torch.bool)

    for i, item in enumerate(batch):
        l = min(item["length"], max_len)
        score_feats[i, :l] = item["score_feats"][:l]
        labels[i, :l] = item["labels"][:l]
        attn_mask[i, :l] = True

    return {"score_feats": score_feats, "labels": labels, "attn_mask": attn_mask, "lengths": torch.tensor(lengths)}
