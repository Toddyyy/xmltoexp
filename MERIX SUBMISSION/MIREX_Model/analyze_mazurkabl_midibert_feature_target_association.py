from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression


ROOT = Path(__file__).resolve().parents[2]
MIREX = ROOT / "MERIX SUBMISSION" / "MIREX_Model"
SOURCE_SCRIPT = MIREX / "run_mazurkabl_l2plus_sqrtmass_branchwise_cnn.py"
OUT_DIR = MIREX / "mazurkabl_midibert_target_association"

EVENT_MIN = 0.01
HIDDEN_DIM = 768
SCALAR_NAMES = [
    "log1p_note_count",
    "log1p_onset_count",
    "log1p_sustain_count",
    "pitch_span_over_88",
    "rest_flag",
    "sustain_only_flag",
    "overlap_weight_sum",
]
BRANCHES = [
    ("onset_mean", 0, HIDDEN_DIM),
    ("sustain_mean", HIDDEN_DIM, HIDDEN_DIM * 2),
    ("all_mean", HIDDEN_DIM * 2, HIDDEN_DIM * 3),
    ("highest_note", HIDDEN_DIM * 3, HIDDEN_DIM * 4),
    ("lowest_note", HIDDEN_DIM * 4, HIDDEN_DIM * 5),
    ("duration_weighted", HIDDEN_DIM * 5, HIDDEN_DIM * 6),
    ("scalars", HIDDEN_DIM * 6, HIDDEN_DIM * 6 + 7),
]


def load_source():
    os.environ["MAZURKA_EVENT_MIN"] = str(EVENT_MIN)
    spec = importlib.util.spec_from_file_location("mazurkabl_corr_source", SOURCE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {SOURCE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["mazurkabl_corr_source"] = module
    spec.loader.exec_module(module)
    return module


def pearson_by_column(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x_centered = x - x.mean(axis=0, keepdims=True)
    y_centered = y - y.mean()
    num = x_centered.T @ y_centered
    den = np.sqrt(np.sum(x_centered * x_centered, axis=0) * np.sum(y_centered * y_centered))
    out = np.zeros(x.shape[1], dtype=np.float64)
    valid = den > 1e-12
    out[valid] = num[valid] / den[valid]
    return out


def feature_name(idx: int) -> tuple[str, str]:
    for branch, start, end in BRANCHES:
        if start <= idx < end:
            local = idx - start
            if branch == "scalars":
                return branch, SCALAR_NAMES[local]
            return branch, f"{branch}_{local:03d}"
    raise ValueError(idx)


def summarize_group(name: str, indices: np.ndarray, pearson: np.ndarray, mi: np.ndarray) -> dict:
    abs_corr = np.abs(pearson[indices])
    mi_values = mi[indices]
    return {
        "group": name,
        "dims": int(len(indices)),
        "mean_abs_pearson": float(np.mean(abs_corr)),
        "median_abs_pearson": float(np.median(abs_corr)),
        "p95_abs_pearson": float(np.percentile(abs_corr, 95)),
        "max_abs_pearson": float(np.max(abs_corr)),
        "dims_abs_pearson_ge_0_02": int(np.count_nonzero(abs_corr >= 0.02)),
        "dims_abs_pearson_ge_0_05": int(np.count_nonzero(abs_corr >= 0.05)),
        "dims_abs_pearson_ge_0_10": int(np.count_nonzero(abs_corr >= 0.10)),
        "mean_mi": float(np.mean(mi_values)),
        "median_mi": float(np.median(mi_values)),
        "p95_mi": float(np.percentile(mi_values, 95)),
        "max_mi": float(np.max(mi_values)),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    src = load_source()
    cfg = src.bw.runner.load_config()
    pieces, labels, _components, _argmax = src.load_sqrtmass_l2plus_labels()
    base_features, feature_cols = src.bw.runner.load_piece_features(pieces, cfg)
    rich_features = src.bw.runner.load_rich_features(pieces)

    missing = sorted(set(pieces) - set(rich_features))
    if missing:
        raise RuntimeError(f"Missing rich features: {missing}")
    bad = [
        (p, len(base_features[p]), len(labels[p]), len(rich_features[p]))
        for p in pieces
        if len(base_features[p]) != len(labels[p]) or len(labels[p]) != len(rich_features[p])
    ]
    if bad:
        raise RuntimeError(f"Length mismatch: {bad[:10]}")

    pieces = sorted(pieces)
    x_rich = np.concatenate([rich_features[p] for p in pieces], axis=0).astype(np.float32)
    x_base = np.concatenate([base_features[p] for p in pieces], axis=0).astype(np.float32)
    y = np.concatenate([labels[p] for p in pieces], axis=0).astype(np.float32)
    y_binary = (y >= EVENT_MIN).astype(np.float32)

    rich_pearson = pearson_by_column(x_rich, y)
    rich_point_biserial = pearson_by_column(x_rich, y_binary)
    base_pearson = pearson_by_column(x_base, y)
    base_point_biserial = pearson_by_column(x_base, y_binary)

    rng = np.random.default_rng(42)
    mi_n = min(12000, len(y))
    mi_idx = np.sort(rng.choice(len(y), size=mi_n, replace=False))
    rich_mi = mutual_info_regression(
        x_rich[mi_idx],
        y[mi_idx],
        discrete_features=False,
        n_neighbors=3,
        random_state=42,
    )
    base_mi = mutual_info_regression(
        x_base[mi_idx],
        y[mi_idx],
        discrete_features=False,
        n_neighbors=3,
        random_state=42,
    )

    rows = []
    for idx in range(x_rich.shape[1]):
        group, name = feature_name(idx)
        rows.append(
            {
                "feature_index": idx,
                "group": group,
                "feature": name,
                "pearson_target": float(rich_pearson[idx]),
                "abs_pearson_target": float(abs(rich_pearson[idx])),
                "point_biserial_true_event": float(rich_point_biserial[idx]),
                "abs_point_biserial_true_event": float(abs(rich_point_biserial[idx])),
                "mutual_info_target_sampled": float(rich_mi[idx]),
            }
        )
    rich_df = pd.DataFrame(rows).sort_values("abs_pearson_target", ascending=False)
    rich_df.to_csv(OUT_DIR / "rich_dim_association.csv", index=False)
    rich_df.head(50).to_csv(OUT_DIR / "top50_rich_dims_by_abs_pearson.csv", index=False)

    base_rows = []
    for idx, name in enumerate(feature_cols):
        base_rows.append(
            {
                "feature_index": idx,
                "feature": name,
                "pearson_target": float(base_pearson[idx]),
                "abs_pearson_target": float(abs(base_pearson[idx])),
                "point_biserial_true_event": float(base_point_biserial[idx]),
                "abs_point_biserial_true_event": float(abs(base_point_biserial[idx])),
                "mutual_info_target_sampled": float(base_mi[idx]),
            }
        )
    base_df = pd.DataFrame(base_rows).sort_values("abs_pearson_target", ascending=False)
    base_df.to_csv(OUT_DIR / "handcrafted_dim_association.csv", index=False)
    base_df.head(50).to_csv(OUT_DIR / "top50_handcrafted_dims_by_abs_pearson.csv", index=False)

    summaries = []
    for name, start, end in BRANCHES:
        summaries.append(summarize_group(name, np.arange(start, end), rich_pearson, rich_mi))
    summaries.append(summarize_group("all_rich", np.arange(x_rich.shape[1]), rich_pearson, rich_mi))
    summaries.append(
        {
            "group": "handcrafted",
            "dims": int(x_base.shape[1]),
            "mean_abs_pearson": float(np.mean(np.abs(base_pearson))),
            "median_abs_pearson": float(np.median(np.abs(base_pearson))),
            "p95_abs_pearson": float(np.percentile(np.abs(base_pearson), 95)),
            "max_abs_pearson": float(np.max(np.abs(base_pearson))),
            "dims_abs_pearson_ge_0_02": int(np.count_nonzero(np.abs(base_pearson) >= 0.02)),
            "dims_abs_pearson_ge_0_05": int(np.count_nonzero(np.abs(base_pearson) >= 0.05)),
            "dims_abs_pearson_ge_0_10": int(np.count_nonzero(np.abs(base_pearson) >= 0.10)),
            "mean_mi": float(np.mean(base_mi)),
            "median_mi": float(np.median(base_mi)),
            "p95_mi": float(np.percentile(base_mi, 95)),
            "max_mi": float(np.max(base_mi)),
        }
    )
    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(OUT_DIR / "group_association_summary.csv", index=False)

    metadata = {
        "pieces": pieces,
        "num_beats": int(len(y)),
        "target_min": float(y.min()),
        "target_max": float(y.max()),
        "target_mean": float(y.mean()),
        "event_min": float(EVENT_MIN),
        "true_events": int(np.count_nonzero(y_binary)),
        "rich_dim": int(x_rich.shape[1]),
        "handcrafted_dim": int(x_base.shape[1]),
        "mi_sample_size": int(mi_n),
        "mi_random_seed": 42,
    }
    (OUT_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("Metadata:")
    print(json.dumps(metadata, indent=2))
    print("\nGroup summary:")
    print(summary_df.round(6).to_string(index=False))
    print("\nTop rich dims by abs Pearson:")
    print(rich_df.head(20).round(6).to_string(index=False))
    print("\nTop handcrafted dims by abs Pearson:")
    print(base_df.head(20).round(6).to_string(index=False))
    print(f"\nWrote {OUT_DIR}")


if __name__ == "__main__":
    main()
