from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = ROOT / "src" / "experiments"
RESULTS = ROOT / "results"
COMBINED_SCRIPT = EXPERIMENTS / "run_combined_mazurkabl_asap40_l2plus_sqrtmass.py"
RUN_NAME = os.environ.get("COMBINED_L23_L456_RUN_NAME", "combined_mazurkabl_asap40_l23_l456_targets")
RESULT_NAME_MAP = {
    "combined_mazurkabl_asap40_l23_l456_targets": "combined_l23_l456_density6",
    "combined_mazurkabl_asap40_l23_l456_targets_density12": "combined_l23_l456_density12",
    "combined_mazurkabl_asap40_l23_l456_targets_density24": "combined_l23_l456_density24",
}
OUT_DIR = RESULTS / RESULT_NAME_MAP.get(RUN_NAME, RUN_NAME)

EVENT_MIN = 0.01
DENSITY_BEATS = float(os.environ.get("COMBINED_L456_DENSITY_BEATS", "6.0"))
DENSITY_MIN_DISTANCE = 1
WEIGHTS = {
    2: 0.205,
    3: 0.284,
    4: 0.408,
    5: 0.613,
    6: 1.000,
}
TARGET_GROUPS = {
    "L23": [2, 3],
    "L456": [4, 5, 6],
}


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


cmb = load_module("combined_l23_l456_source", COMBINED_SCRIPT)
bw = cmb.bw
runner = cmb.runner
base = cmb.base


def build_group_labels(
    components: dict[str, dict[int, np.ndarray]],
    levels: list[int],
) -> dict[str, np.ndarray]:
    labels = {}
    for piece, per_level in components.items():
        stack = np.stack([float(WEIGHTS[level]) * per_level[level] for level in levels], axis=0)
        labels[piece] = np.max(stack, axis=0).astype(np.float32)
    return labels


def label_stats(dataset: str, target_name: str, labels: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for piece, target in labels.items():
        rows.append(
            {
                "dataset": dataset,
                "target_name": target_name,
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


def patch_eval_globals():
    original = {
        "event_min": base.EVENT_MIN,
        "min_distance": base.MIN_DISTANCE,
        "expected_count": base.expected_count_from_train_density,
        "extract_top_density": base.extract_top_density,
    }
    base.EVENT_MIN = float(EVENT_MIN)
    base.MIN_DISTANCE = int(DENSITY_MIN_DISTANCE)
    base.expected_count_from_train_density = fixed_density

    def top_density_min_distance(scores, expected_count, min_distance=None):
        return original["extract_top_density"](
            scores,
            expected_count,
            min_distance=max(int(DENSITY_MIN_DISTANCE), 1),
        )

    base.extract_top_density = top_density_min_distance
    return original


def restore_eval_globals(original: dict) -> None:
    base.EVENT_MIN = original["event_min"]
    base.MIN_DISTANCE = original["min_distance"]
    base.expected_count_from_train_density = original["expected_count"]
    base.extract_top_density = original["extract_top_density"]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = runner.load_config()

    mz_pieces_raw, _mz_labels_raw, mz_components_raw, _mz_argmax = cmb.mz.load_sqrtmass_l2plus_labels()
    mz_base_raw, mz_cols = runner.load_piece_features(mz_pieces_raw, cfg)
    mz_rich_raw = runner.load_rich_features(mz_pieces_raw)

    asap_pieces_raw, _asap_labels_raw, asap_components_raw, _asap_argmax = cmb.asap.load_sqrtmass_labels()
    asap_base_raw, asap_cols = cmb.asap.load_piece_features(asap_pieces_raw, cfg)
    asap_rich_raw = cmb.asap.load_rich_features(asap_pieces_raw)

    common_cols = [col for col in mz_cols if col in set(asap_cols)]
    mz_base_common = cmb.subset_features(mz_base_raw, mz_cols, common_cols)
    asap_base_common = cmb.subset_features(asap_base_raw, asap_cols, common_cols)
    base_features = {
        **cmb.prefix_dict("mazurka", mz_base_common),
        **cmb.prefix_dict("asap40", asap_base_common),
    }
    rich_features = {
        **cmb.prefix_dict("mazurka", mz_rich_raw),
        **cmb.prefix_dict("asap40", asap_rich_raw),
    }
    pieces = sorted(base_features)
    folds = base.make_folds(pieces, n_folds=5, seed=42)
    device = runner.resolve_device()

    fold_frames = []
    aggregates = []
    stats_frames = []

    original = patch_eval_globals()
    try:
        for target_name, levels in TARGET_GROUPS.items():
            mz_group = build_group_labels(mz_components_raw, levels)
            asap_group = build_group_labels(asap_components_raw, levels)
            labels = {
                **cmb.prefix_dict("mazurka", mz_group),
                **cmb.prefix_dict("asap40", asap_group),
            }
            bad = [
                (piece, len(base_features[piece]), len(labels[piece]), len(rich_features[piece]))
                for piece in pieces
                if len(base_features[piece]) != len(labels[piece]) or len(labels[piece]) != len(rich_features[piece])
            ]
            if bad:
                raise RuntimeError(f"{target_name}: length mismatch {bad[:20]}")

            stats = pd.concat(
                [
                    label_stats("mazurka", target_name, cmb.prefix_dict("mazurka", mz_group)),
                    label_stats("asap40", target_name, cmb.prefix_dict("asap40", asap_group)),
                ],
                ignore_index=True,
            )
            stats_frames.append(stats)
            print(
                f"target={target_name}; levels={levels}; pieces={len(pieces)}; "
                f"true_events={int(stats['true_events_ge_eval_threshold'].sum())}; "
                f"target_sum={float(stats['target_sum'].sum()):.4f}; "
                f"base_dim={base_features[pieces[0]].shape[1]}; rich_dim={rich_features[pieces[0]].shape[1]}; device={device}",
                flush=True,
            )

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
            baseline_df.insert(0, "target_name", target_name)
            baseline_agg = {"target_name": target_name, **baseline_agg}
            fold_frames.append(baseline_df)
            aggregates.append(baseline_agg)
            pd.concat(fold_frames, ignore_index=True).to_csv(OUT_DIR / "fold_summary.csv", index=False)
            pd.DataFrame(aggregates).to_csv(OUT_DIR / "aggregate_totals.csv", index=False)

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
            branch_df.insert(0, "target_name", target_name)
            branch_agg = {"target_name": target_name, **branch_agg}
            fold_frames.append(branch_df)
            aggregates.append(branch_agg)
            pd.concat(fold_frames, ignore_index=True).to_csv(OUT_DIR / "fold_summary.csv", index=False)
            pd.DataFrame(aggregates).to_csv(OUT_DIR / "aggregate_totals.csv", index=False)
    finally:
        restore_eval_globals(original)

    pd.concat(stats_frames, ignore_index=True).to_csv(OUT_DIR / "label_stats.csv", index=False)
    agg = pd.DataFrame(aggregates)
    (OUT_DIR / "metadata.json").write_text(
        json.dumps(
            {
                "dataset": "mazurkabl+asap40",
                "target_groups": TARGET_GROUPS,
                "weights": WEIGHTS,
                "event_min": EVENT_MIN,
                "density_beats": DENSITY_BEATS,
                "density_min_distance": DENSITY_MIN_DISTANCE,
                "piece_count": len(pieces),
                "mazurka_piece_count": len(mz_pieces_raw),
                "asap40_piece_count": len(asap_pieces_raw),
                "common_feature_columns": common_cols,
                "settings": ["baseline_cnn", "handcrafted_plus_branchwise"],
                "folds": folds,
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
