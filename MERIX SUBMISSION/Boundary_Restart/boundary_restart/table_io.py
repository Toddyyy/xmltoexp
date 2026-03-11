from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import yaml

PIECE_RE = re.compile(r"^(M\d+-\d+)")
PERFORMER_RE = re.compile(r"(pid[^_]+)")
LEVEL_RE = re.compile(r"_L(\d+)$")

META_COLUMNS = {
    "source_path",
    "sample_id",
    "piece_id",
    "performer_id",
    "level",
    "split",
    "beat_idx",
    "num_beats",
    "boundary_prob",
    "boundary_peak",
}


def is_meta_column(column: str) -> bool:
    return (
        column in META_COLUMNS
        or column.startswith("boundary_")
        or column.startswith("target_")
        or column.startswith("salience_")
        or column.startswith("stage_")
    )


def extract_piece_id(name: str) -> str:
    match = PIECE_RE.search(name)
    return match.group(1) if match else name


def extract_performer_id(name: str) -> str:
    match = PERFORMER_RE.search(name)
    return match.group(1) if match else "unknown"


def extract_level(name: str) -> int | None:
    match = LEVEL_RE.search(name)
    return int(match.group(1)) if match else None


def list_npz_files(npz_dir: Path, level: int | None = None, max_files: int | None = None) -> list[Path]:
    files = sorted(npz_dir.glob("*.npz"))
    if level is not None:
        suffix = f"_L{int(level)}.npz"
        files = [path for path in files if path.name.endswith(suffix)]
    if max_files is not None:
        files = files[: int(max_files)]
    return files


def load_piece_split(path: Path) -> dict[str, set[str]]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return {
        "train": set(data.get("train", [])),
        "val": set(data.get("val", [])),
        "test": set(data.get("test", [])),
    }


def assign_split(piece_id: str, split_cfg: dict[str, set[str]] | None) -> str:
    if split_cfg is None:
        return "all"
    for split_name in ("train", "val", "test"):
        if piece_id in split_cfg[split_name]:
            return split_name
    return "unused"


def load_table(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if not is_meta_column(col)]


def samples_from_table(
    df: pd.DataFrame,
    feature_cols: Iterable[str],
    split: str,
    target_col: str = "boundary_peak",
    score_col: str = "boundary_prob",
) -> list[dict]:
    subset = df[df["split"] == split].copy()
    subset = subset.sort_values(["sample_id", "beat_idx"])
    samples = []
    cols = list(feature_cols)
    for sample_id, group in subset.groupby("sample_id", sort=False):
        group = group.sort_values("beat_idx")
        samples.append(
            {
                "sample_id": sample_id,
                "piece_id": group["piece_id"].iloc[0],
                "performer_id": group["performer_id"].iloc[0],
                "level": int(group["level"].iloc[0]),
                "features": group[cols].to_numpy(dtype=np.float32),
                "labels": group[target_col].to_numpy(dtype=np.float32),
                "scores_ref": group[score_col].to_numpy(dtype=np.float32),
                "beat_idx": group["beat_idx"].to_numpy(dtype=np.int32),
            }
        )
    return samples
