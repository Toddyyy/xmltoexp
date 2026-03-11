from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml


def load_config(path: str | Path) -> dict:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    cfg["_config_path"] = str(config_path)
    return cfg


def config_dir(cfg: dict) -> Path:
    return Path(cfg["_config_path"]).resolve().parent


def resolve_path(cfg: dict, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    cwd_candidate = path.resolve()
    if cwd_candidate.exists():
        return cwd_candidate
    return (config_dir(cfg) / path).resolve()


def threshold_grid(cfg: dict) -> np.ndarray:
    grid = cfg.get("evaluation", {}).get("threshold_grid", {})
    start = float(grid.get("start", 0.05))
    stop = float(grid.get("stop", 0.95))
    steps = int(grid.get("steps", 37))
    return np.linspace(start, stop, steps, dtype=np.float32)
