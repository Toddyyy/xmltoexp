from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
MIREX = ROOT / "MERIX SUBMISSION" / "MIREX_Model"
SOURCE_SCRIPT = MIREX / "run_atepp30_l2plus_sqrtmass_handcrafted_plus_branchwise.py"
OUT_DIR = MIREX / "atepp30_l2plus_sqrtmass_branchwise_rich_only"


def load_source():
    spec = importlib.util.spec_from_file_location("atepp30_hp_branchwise_source", SOURCE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {SOURCE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["atepp30_hp_branchwise_source"] = module
    spec.loader.exec_module(module)
    return module


src = load_source()
bw = src.bw
runner = src.runner
base = src.base


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = runner.load_config()
    pieces, labels, components, argmax = src.load_sqrtmass_labels()
    base_features, feature_cols = src.load_piece_features(pieces, cfg)
    rich_features = src.load_rich_features(pieces)
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

    stats = src.label_stats(labels, components)
    stats.to_csv(OUT_DIR / "label_stats.csv", index=False)
    argmax.to_csv(OUT_DIR / "argmax_contribution_by_piece.csv", index=False)
    argmax_summary = []
    for level in src.SQRT_MASS_WEIGHTS:
        argmax_summary.append(
            {
                "level": f"L{level}",
                "unique_max_true_events": int(argmax[f"L{level}_unique_max"].sum()),
                "tied_max_includes_level": int(argmax[f"L{level}_tied_includes"].sum()),
            }
        )
    pd.DataFrame(argmax_summary).to_csv(OUT_DIR / "argmax_contribution_summary.csv", index=False)

    pieces = sorted(pieces)
    folds = base.make_folds(pieces, n_folds=5, seed=42)
    device = runner.resolve_device()

    original_event_min = base.EVENT_MIN
    original_min_distance = base.MIN_DISTANCE
    original_expected_count = base.expected_count_from_train_density
    original_extract_top_density = base.extract_top_density
    base.EVENT_MIN = float(src.EVENT_MIN)
    base.MIN_DISTANCE = int(src.DENSITY_MIN_DISTANCE)
    base.expected_count_from_train_density = src.fixed_two_bar_density

    def top_density_min_distance(scores, expected_count, min_distance=None):
        return original_extract_top_density(scores, expected_count, min_distance=max(int(src.DENSITY_MIN_DISTANCE), 1))

    base.extract_top_density = top_density_min_distance

    print(
        f"dataset={src.DATASET_NAME}; setting=branchwise_rich_only; device={device}; pieces={len(pieces)}; "
        f"weights={src.SQRT_MASS_WEIGHTS}; event_min={src.EVENT_MIN}; "
        f"density=1 per {src.DENSITY_BEATS:g} beats; min_distance={src.DENSITY_MIN_DISTANCE}; "
        f"true_events={int(stats['true_events_ge_eval_threshold'].sum())}; "
        f"target_sum={float(stats['target_sum'].sum()):.4f}; "
        f"base_dim={base_features[pieces[0]].shape[1]}; rich_dim={rich_features[pieces[0]].shape[1]}"
    )

    try:
        fold_df, aggregate = bw.run_setting(
            "branchwise_rich_only",
            cfg,
            pieces,
            labels,
            base_features,
            rich_features,
            folds,
            device,
        )
    finally:
        base.EVENT_MIN = original_event_min
        base.MIN_DISTANCE = original_min_distance
        base.expected_count_from_train_density = original_expected_count
        base.extract_top_density = original_extract_top_density

    fold_df.to_csv(OUT_DIR / "fold_summary.csv", index=False)
    agg = pd.DataFrame([aggregate])
    agg.to_csv(OUT_DIR / "aggregate_totals.csv", index=False)
    (OUT_DIR / "metadata.json").write_text(
        json.dumps(
            {
                "dataset_name": src.DATASET_NAME,
                "beat_table": str(src.BEAT_TABLE),
                "label_dir": str(src.LABEL_DIR),
                "manifest": str(src.MANIFEST),
                "rich_dir": str(src.RICH_DIR),
                "weights": src.SQRT_MASS_WEIGHTS,
                "event_min": src.EVENT_MIN,
                "density_beats": src.DENSITY_BEATS,
                "density_min_distance": src.DENSITY_MIN_DISTANCE,
                "setting": "branchwise_rich_only",
                "branch_dim": bw.BRANCH_DIM,
                "scalar_dim": bw.SCALAR_DIM,
                "beat_emb_dim": bw.BEAT_EMB_DIM,
                "config_path": str(runner.CONFIG_PATH),
                "pieces": pieces,
                "folds": folds,
                "feature_columns": feature_cols,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("\nAggregate:")
    print(agg.round(4).to_string(index=False))
    print(f"\nWrote {OUT_DIR}")


if __name__ == "__main__":
    main()
