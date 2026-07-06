from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
BUILD_SCRIPT = ROOT / "MERIX SUBMISSION" / "MIREX_Model" / "build_mazurka_beat_npz_performer_levels.py"
BEAT_TIME_DIR = ROOT / "MazurkaBL-master" / "beat_time"
OUT_DIR = ROOT / "MERIX SUBMISSION" / "MIREX_Model" / "mazurkabl_raw_non_nested_level_overlap"
STR_VEC = [3, 2, 2, 2, 2, 2]
LEVELS = [1, 2, 3, 4, 5, 6]

spec = importlib.util.spec_from_file_location("mazurka_level_builder_raw_overlap", BUILD_SCRIPT)
builder = importlib.util.module_from_spec(spec)
sys.modules["mazurka_level_builder_raw_overlap"] = builder
assert spec.loader is not None
spec.loader.exec_module(builder)


def overlap_metrics(a: np.ndarray, b: np.ndarray) -> dict:
    a = a.astype(bool)
    b = b.astype(bool)
    inter = int(np.count_nonzero(a & b))
    union = int(np.count_nonzero(a | b))
    a_count = int(np.count_nonzero(a))
    b_count = int(np.count_nonzero(b))
    return {
        "a_count": a_count,
        "b_count": b_count,
        "intersection": inter,
        "union": union,
        "jaccard": inter / union if union else np.nan,
        "a_covered_by_b": inter / a_count if a_count else np.nan,
        "b_covered_by_a": inter / b_count if b_count else np.nan,
        "overlap_min_count": inter / min(a_count, b_count) if min(a_count, b_count) else np.nan,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pair_rows = []
    level_rows = []
    sample_count = 0
    for file_path in sorted(BEAT_TIME_DIR.glob("*beat_time.csv")):
        piece = file_path.name.replace("beat_time.csv", "")
        df, performer_cols = builder.load_beat_time(file_path)
        curves = builder.compute_tempo_curves(df, performer_cols, smooth_window=3, clip_max=600)
        for performer, curve in curves.items():
            raw, level_sets = builder.group_analysis_hierarchy(curve, STR_VEC, enforce_nested=False)
            sample_count += 1
            masks = {level: raw[level - 1].astype(bool) for level in LEVELS}
            for level in LEVELS:
                level_rows.append(
                    {
                        "piece": piece,
                        "performer": performer,
                        "level": level,
                        "num_beats": len(curve),
                        "event_count": int(np.count_nonzero(masks[level])),
                        "event_rate": float(np.mean(masks[level])) if len(curve) else np.nan,
                    }
                )
            for i, a_level in enumerate(LEVELS):
                for b_level in LEVELS[i + 1 :]:
                    row = {
                        "piece": piece,
                        "performer": performer,
                        "level_a": a_level,
                        "level_b": b_level,
                    }
                    row.update(overlap_metrics(masks[a_level], masks[b_level]))
                    pair_rows.append(row)

    level_df = pd.DataFrame(level_rows)
    pair_df = pd.DataFrame(pair_rows)
    level_df.to_csv(OUT_DIR / "raw_non_nested_level_event_counts_by_performer.csv", index=False)
    pair_df.to_csv(OUT_DIR / "raw_non_nested_pairwise_level_overlap_by_performer.csv", index=False)

    level_summary = (
        level_df.groupby("level")
        .agg(
            samples=("performer", "size"),
            mean_event_count=("event_count", "mean"),
            median_event_count=("event_count", "median"),
            mean_event_rate=("event_rate", "mean"),
        )
        .reset_index()
    )
    level_summary.to_csv(OUT_DIR / "raw_non_nested_level_event_count_summary.csv", index=False)

    pair_summary = (
        pair_df.groupby(["level_a", "level_b"])
        .agg(
            samples=("performer", "size"),
            mean_a_count=("a_count", "mean"),
            mean_b_count=("b_count", "mean"),
            mean_intersection=("intersection", "mean"),
            mean_jaccard=("jaccard", "mean"),
            median_jaccard=("jaccard", "median"),
            mean_a_covered_by_b=("a_covered_by_b", "mean"),
            mean_b_covered_by_a=("b_covered_by_a", "mean"),
            mean_overlap_min_count=("overlap_min_count", "mean"),
        )
        .reset_index()
    )
    pair_summary.to_csv(OUT_DIR / "raw_non_nested_pairwise_level_overlap_summary.csv", index=False)

    matrices = {}
    for metric in ["mean_jaccard", "mean_overlap_min_count", "mean_b_covered_by_a"]:
        mat = pd.DataFrame(np.eye(len(LEVELS)), index=LEVELS, columns=LEVELS, dtype=float)
        for _, row in pair_summary.iterrows():
            a = int(row["level_a"])
            b = int(row["level_b"])
            mat.loc[a, b] = row[metric]
            mat.loc[b, a] = row[metric]
        mat.to_csv(OUT_DIR / f"{metric}_matrix.csv")
        matrices[metric] = mat

    metadata = {
        "source": str(BEAT_TIME_DIR),
        "str_vec": STR_VEC,
        "enforce_nested": False,
        "sample_count": sample_count,
        "metrics": {
            "jaccard": "intersection / union",
            "a_covered_by_b": "fraction of level_a events also present in level_b at exact same beat",
            "b_covered_by_a": "fraction of level_b events also present in level_a at exact same beat",
            "overlap_min_count": "intersection / min(count_a, count_b)",
        },
    }
    (OUT_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("Level event count summary:")
    print(level_summary.round(4).to_string(index=False))
    print("\nPairwise overlap summary:")
    print(pair_summary.round(4).to_string(index=False))
    print(f"\nWrote {OUT_DIR}")


if __name__ == "__main__":
    main()
