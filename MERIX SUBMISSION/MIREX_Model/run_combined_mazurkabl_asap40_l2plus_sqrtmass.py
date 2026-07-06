from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
MIREX = ROOT / "MERIX SUBMISSION" / "MIREX_Model"
MAZURKA_SCRIPT = MIREX / "run_mazurkabl_l2plus_sqrtmass_branchwise_cnn.py"
ASAP_SCRIPT = MIREX / "run_asap30_l2plus_sqrtmass_baseline_branchwise.py"
OUT_DIR = MIREX / "combined_mazurkabl_asap40_l2plus_sqrtmass_baseline_branchwise"

EVENT_MIN = 0.01
DENSITY_BEATS = 6.0
DENSITY_MIN_DISTANCE = 1


os.environ["MAZURKA_EVENT_MIN"] = str(EVENT_MIN)
os.environ["MAZURKA_DENSITY_MODE"] = "fixed_2bars"
os.environ["MAZURKA_DENSITY_BEATS"] = str(DENSITY_BEATS)
os.environ["MAZURKA_DENSITY_MIN_DISTANCE"] = str(DENSITY_MIN_DISTANCE)
os.environ["ASAP_TOP_N"] = "40"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


mz = load_module("combined_mz_runner", MAZURKA_SCRIPT)
asap = load_module("combined_asap40_runner", ASAP_SCRIPT)
bw = mz.bw
runner = bw.runner
base = runner.base


def prefix_dict(prefix: str, values: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {f"{prefix}__{piece}": array for piece, array in values.items()}


def prefix_components(prefix: str, components: dict[str, dict[int, np.ndarray]]) -> dict[str, dict[int, np.ndarray]]:
    return {f"{prefix}__{piece}": value for piece, value in components.items()}


def subset_features(
    features: dict[str, np.ndarray],
    columns: list[str],
    common_columns: list[str],
) -> dict[str, np.ndarray]:
    index = [columns.index(col) for col in common_columns]
    return {piece: values[:, index].astype(np.float32) for piece, values in features.items()}


def label_stats(dataset: str, labels: dict[str, np.ndarray], pieces: list[str]) -> pd.DataFrame:
    rows = []
    for piece in pieces:
        target = labels[piece]
        rows.append(
            {
                "dataset": dataset,
                "piece": piece,
                "num_beats": int(len(target)),
                "target_sum": float(target.sum()),
                "true_events_ge_eval_threshold": int(np.count_nonzero(target >= EVENT_MIN)),
                "target_max": float(target.max()) if len(target) else 0.0,
            }
        )
    return pd.DataFrame(rows)


def fixed_density(_labels, _train_pieces, num_beats: int) -> int:
    return max(1, int(round(float(num_beats) / max(float(DENSITY_BEATS), 1e-9))))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = runner.load_config()

    mz_pieces_raw, mz_labels_raw, mz_components_raw, mz_argmax = mz.load_sqrtmass_l2plus_labels()
    mz_base_raw, mz_cols = runner.load_piece_features(mz_pieces_raw, cfg)
    mz_rich_raw = runner.load_rich_features(mz_pieces_raw)

    asap_pieces_raw, asap_labels_raw, asap_components_raw, asap_argmax = asap.load_sqrtmass_labels()
    asap_base_raw, asap_cols = asap.load_piece_features(asap_pieces_raw, cfg)
    asap_rich_raw = asap.load_rich_features(asap_pieces_raw)

    common_cols = [col for col in mz_cols if col in set(asap_cols)]
    if not common_cols:
        raise RuntimeError("No common handcrafted feature columns between MazurkaBL and ASAP40")

    mz_base_common = subset_features(mz_base_raw, mz_cols, common_cols)
    asap_base_common = subset_features(asap_base_raw, asap_cols, common_cols)

    mz_labels = prefix_dict("mazurka", mz_labels_raw)
    asap_labels = prefix_dict("asap40", asap_labels_raw)
    labels = {**mz_labels, **asap_labels}
    base_features = {
        **prefix_dict("mazurka", mz_base_common),
        **prefix_dict("asap40", asap_base_common),
    }
    rich_features = {
        **prefix_dict("mazurka", mz_rich_raw),
        **prefix_dict("asap40", asap_rich_raw),
    }
    pieces = sorted(labels)

    missing_rich = sorted(set(pieces) - set(rich_features))
    if missing_rich:
        raise RuntimeError(f"Missing rich features: {missing_rich}")
    bad = [
        (piece, len(base_features[piece]), len(labels[piece]), len(rich_features[piece]))
        for piece in pieces
        if len(base_features[piece]) != len(labels[piece]) or len(labels[piece]) != len(rich_features[piece])
    ]
    if bad:
        raise RuntimeError(f"Length mismatch: {bad[:20]}")

    stats = pd.concat(
        [
            label_stats("mazurka", mz_labels, sorted(mz_labels)),
            label_stats("asap40", asap_labels, sorted(asap_labels)),
        ],
        ignore_index=True,
    )
    stats.to_csv(OUT_DIR / "label_stats.csv", index=False)

    folds = base.make_folds(pieces, n_folds=5, seed=42)
    device = runner.resolve_device()

    original_event_min = base.EVENT_MIN
    original_min_distance = base.MIN_DISTANCE
    original_expected_count = base.expected_count_from_train_density
    original_extract_top_density = base.extract_top_density
    base.EVENT_MIN = float(EVENT_MIN)
    base.MIN_DISTANCE = int(DENSITY_MIN_DISTANCE)
    base.expected_count_from_train_density = fixed_density

    def top_density_min_distance(scores, expected_count, min_distance=None):
        return original_extract_top_density(scores, expected_count, min_distance=max(int(DENSITY_MIN_DISTANCE), 1))

    base.extract_top_density = top_density_min_distance

    print(
        f"dataset=mazurkabl+asap40; pieces={len(pieces)}; "
        f"mazurka_pieces={len(mz_labels)}; asap40_pieces={len(asap_labels)}; "
        f"event_min={EVENT_MIN}; density=1 per {DENSITY_BEATS:g} beats; min_distance={DENSITY_MIN_DISTANCE}; "
        f"true_events={int(stats['true_events_ge_eval_threshold'].sum())}; "
        f"base_dim={base_features[pieces[0]].shape[1]}; rich_dim={rich_features[pieces[0]].shape[1]}; "
        f"device={device}"
    )
    print(
        f"common handcrafted columns={len(common_cols)}; "
        f"mazurka original columns={len(mz_cols)}; asap40 columns={len(asap_cols)}"
    )

    fold_frames = []
    aggregates = []

    def write_partial() -> None:
        if fold_frames:
            pd.concat(fold_frames, ignore_index=True).to_csv(OUT_DIR / "fold_summary.csv", index=False)
        if aggregates:
            pd.DataFrame(aggregates).to_csv(OUT_DIR / "aggregate_totals.csv", index=False)

    try:
        baseline_df, baseline_agg = runner.run_setting(
            "baseline_cnn",
            cfg,
            pieces,
            labels,
            base_features,
            rich_features,
            folds,
            device,
        )
        fold_frames.append(baseline_df)
        aggregates.append(baseline_agg)
        write_partial()

        branch_df, branch_agg = bw.run_setting(
            "handcrafted_plus_branchwise",
            cfg,
            pieces,
            labels,
            base_features,
            rich_features,
            folds,
            device,
        )
        fold_frames.append(branch_df)
        aggregates.append(branch_agg)
        write_partial()
    finally:
        base.EVENT_MIN = original_event_min
        base.MIN_DISTANCE = original_min_distance
        base.expected_count_from_train_density = original_expected_count
        base.extract_top_density = original_extract_top_density

    agg = pd.DataFrame(aggregates)
    (OUT_DIR / "metadata.json").write_text(
        json.dumps(
            {
                "dataset": "mazurkabl+asap40",
                "event_min": EVENT_MIN,
                "density_beats": DENSITY_BEATS,
                "density_min_distance": DENSITY_MIN_DISTANCE,
                "mazurka_piece_count": len(mz_labels),
                "asap40_piece_count": len(asap_labels),
                "piece_count": len(pieces),
                "common_feature_columns": common_cols,
                "mazurka_original_feature_columns": mz_cols,
                "asap40_feature_columns": asap_cols,
                "folds": folds,
                "settings": ["baseline_cnn", "handcrafted_plus_branchwise"],
                "mazurka_argmax_summary": mz_argmax.to_dict(orient="records"),
                "asap40_argmax_summary": asap_argmax.to_dict(orient="records"),
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
