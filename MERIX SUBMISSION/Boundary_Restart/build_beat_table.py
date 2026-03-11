#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from boundary_restart.config import load_config, resolve_path
from boundary_restart.features import (
    BASE_NOTE_FEATURES,
    PeakConfig,
    build_beat_feature_frame,
    build_grouped_salience_frame,
    build_weighted_salience_frame,
)
from boundary_restart.table_io import feature_columns, list_npz_files, load_piece_split


def main():
    parser = argparse.ArgumentParser(description="Build beat-level feature table from boundary npz files.")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--out_path", default=None, help="Optional override for table output path")
    parser.add_argument("--max_files", type=int, default=None, help="Optional file cap for smoke runs")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_cfg = cfg.get("data", {})
    npz_dir = resolve_path(cfg, data_cfg["npz_dir"])
    split_file = resolve_path(cfg, data_cfg["split_file"]) if data_cfg.get("split_file") else None
    out_path = resolve_path(cfg, args.out_path or data_cfg["beat_table_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    split_cfg = load_piece_split(split_file) if split_file is not None else None
    task_type = str(data_cfg.get("task_type", "single_level"))
    feature_level = int(data_cfg.get("feature_level", data_cfg.get("level", 1)))
    files = list_npz_files(
        npz_dir=npz_dir,
        level=feature_level if task_type in {"weighted_salience", "grouped_salience"} else data_cfg.get("level"),
        max_files=args.max_files,
    )
    if not files:
        raise FileNotFoundError(f"No npz files found in {npz_dir}")

    peak_cfg = PeakConfig(
        distance=int(data_cfg.get("peak_distance", 6)),
        height=float(data_cfg.get("peak_height", 0.15)),
        prominence=float(data_cfg.get("peak_prominence", 0.05)),
    )
    long_note_threshold = float(data_cfg.get("long_note_threshold", 1.0))
    beat_unit_fallback = float(data_cfg.get("beat_unit_fallback", 1.0))
    symmetry_window = int(data_cfg.get("symmetry_window", 4))
    deviation_window = int(data_cfg.get("deviation_window", 8))
    measure_cycle = int(data_cfg.get("measure_cycle", 3))
    xml_score_dir = resolve_path(cfg, data_cfg["xml_score_dir"]) if data_cfg.get("xml_score_dir") else None
    xml_expand_repeats = bool(data_cfg.get("xml_expand_repeats", True))

    if task_type == "weighted_salience":
        levels = [int(v) for v in data_cfg.get("target_levels", [1, 2, 3, 4, 5, 6])]
        raw_weights = np.asarray(data_cfg.get("target_weights", [1.0] * len(levels)), dtype=np.float32)
        if raw_weights.size != len(levels):
            raise ValueError("target_levels and target_weights must have the same length")
        frames = [
            build_weighted_salience_frame(
                npz_path=path,
                levels=levels,
                level_weights=raw_weights.tolist(),
                peak_cfg=peak_cfg,
                split_cfg=split_cfg,
                long_note_threshold=long_note_threshold,
                beat_unit_fallback=beat_unit_fallback,
                symmetry_window=symmetry_window,
                deviation_window=deviation_window,
                measure_cycle=measure_cycle,
                xml_score_dir=xml_score_dir,
                xml_expand_repeats=xml_expand_repeats,
            )
            for path in files
        ]
    elif task_type == "grouped_salience":
        target_groups = [[int(level) for level in group] for group in data_cfg["target_groups"]]
        raw_weights = np.asarray(data_cfg.get("group_weights", [1.0] * len(target_groups)), dtype=np.float32)
        if raw_weights.size != len(target_groups):
            raise ValueError("target_groups and group_weights must have the same length")
        group_merge = str(data_cfg.get("group_merge", "max"))
        frames = [
            build_grouped_salience_frame(
                npz_path=path,
                target_groups=target_groups,
                group_weights=raw_weights.tolist(),
                group_merge=group_merge,
                peak_cfg=peak_cfg,
                split_cfg=split_cfg,
                long_note_threshold=long_note_threshold,
                beat_unit_fallback=beat_unit_fallback,
                symmetry_window=symmetry_window,
                deviation_window=deviation_window,
                measure_cycle=measure_cycle,
                xml_score_dir=xml_score_dir,
                xml_expand_repeats=xml_expand_repeats,
            )
            for path in files
        ]
    else:
        frames = [
            build_beat_feature_frame(
                npz_path=path,
                peak_cfg=peak_cfg,
                split_cfg=split_cfg,
                long_note_threshold=long_note_threshold,
                beat_unit_fallback=beat_unit_fallback,
                symmetry_window=symmetry_window,
                deviation_window=deviation_window,
                measure_cycle=measure_cycle,
                xml_score_dir=xml_score_dir,
                xml_expand_repeats=xml_expand_repeats,
            )
            for path in files
        ]
    table = pd.concat(frames, ignore_index=True)
    table.to_csv(out_path, index=False, compression="gzip")

    meta_path = out_path.with_suffix("").with_suffix(".meta.json")
    target_column = str(data_cfg.get("target_column", "boundary_peak"))
    meta = {
        "source_dir": str(npz_dir),
        "task_type": task_type,
        "rows": int(len(table)),
        "files": int(len(files)),
        "splits": table["split"].value_counts().to_dict(),
        "target_column": target_column,
        "target_mean": float(table[target_column].mean()),
        "feature_columns": feature_columns(table),
        "base_note_features": BASE_NOTE_FEATURES,
        "peak_config": {
            "distance": peak_cfg.distance,
            "height": peak_cfg.height,
            "prominence": peak_cfg.prominence,
        },
        "long_note_threshold": long_note_threshold,
        "symmetry_window": symmetry_window,
        "deviation_window": deviation_window,
        "measure_cycle": measure_cycle,
        "xml_score_dir": None if xml_score_dir is None else str(xml_score_dir),
        "xml_expand_repeats": xml_expand_repeats,
    }
    if task_type == "weighted_salience":
        meta["feature_level"] = feature_level
        meta["target_levels"] = [int(v) for v in data_cfg.get("target_levels", [1, 2, 3, 4, 5, 6])]
        meta["target_weights"] = [float(v) for v in data_cfg.get("target_weights", [])]
    if task_type == "grouped_salience":
        meta["feature_level"] = feature_level
        meta["target_groups"] = [[int(level) for level in group] for group in data_cfg.get("target_groups", [])]
        meta["group_weights"] = [float(v) for v in data_cfg.get("group_weights", [])]
        meta["group_merge"] = str(data_cfg.get("group_merge", "max"))
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Wrote beat table to {out_path}")
    print(f"Rows: {len(table)} | Files: {len(files)} | Target mean: {table[target_column].mean():.6f}")
    print(f"Metadata: {meta_path}")


if __name__ == "__main__":
    main()
