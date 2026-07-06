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
BRANCHWISE_SCRIPT = MIREX / "run_mazurkabl_l2plus_branchwise_bottleneck_cnn.py"
EVENT_MIN_OVERRIDE = float(os.environ.get("MAZURKA_EVENT_MIN", "0.05"))
DENSITY_MODE = os.environ.get("MAZURKA_DENSITY_MODE", "train_density")
DENSITY_BEATS = float(os.environ.get("MAZURKA_DENSITY_BEATS", "6.0"))
DENSITY_MIN_DISTANCE = int(os.environ.get("MAZURKA_DENSITY_MIN_DISTANCE", "1"))
OUT_DIR = MIREX / (
    "mazurkabl_l2plus_sqrtmass_branchwise_cnn"
    if abs(EVENT_MIN_OVERRIDE - 0.05) < 1e-9 and DENSITY_MODE == "train_density"
    else (
        f"mazurkabl_l2plus_sqrtmass_branchwise_cnn_"
        f"event{str(EVENT_MIN_OVERRIDE).replace('.', 'p')}_density{DENSITY_MODE}"
        f"_mindist{DENSITY_MIN_DISTANCE}"
    )
)

SQRT_MASS_WEIGHTS = {
    2: 0.205,
    3: 0.284,
    4: 0.408,
    5: 0.613,
    6: 1.000,
}


def load_branchwise_runner():
    spec = importlib.util.spec_from_file_location("mazurka_branchwise_sqrtmass", BRANCHWISE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {BRANCHWISE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["mazurka_branchwise_sqrtmass"] = module
    spec.loader.exec_module(module)
    return module


bw = load_branchwise_runner()


def load_sqrtmass_l2plus_labels():
    base = bw.runner.base
    pieces = sorted({base.piece_id_from_npz(path) for path in base.LABEL_DIR.glob("*_L2.npz")})
    labels = {}
    components = {}
    argmax_rows = []
    for piece in pieces:
        weighted = []
        components[piece] = {}
        levels = []
        for level, weight in SQRT_MASS_WEIGHTS.items():
            freq = base.load_level_frequency(piece, level)
            components[piece][level] = freq
            weighted.append(float(weight) * freq)
            levels.append(level)
        matrix = np.stack(weighted, axis=0).astype(np.float32)
        target = np.max(matrix, axis=0).astype(np.float32)
        labels[piece] = target

        true = target >= EVENT_MIN_OVERRIDE
        is_max = np.isclose(matrix, target[None, :], atol=1e-8) & true[None, :]
        ties = is_max.sum(axis=0)
        row = {"piece": piece, "true_events": int(true.sum()), "tied_true_events": int(np.count_nonzero(ties > 1))}
        for idx, level in enumerate(levels):
            row[f"L{level}_unique_max"] = int(np.count_nonzero(true & (ties == 1) & is_max[idx]))
            row[f"L{level}_tied_includes"] = int(np.count_nonzero(true & (ties > 1) & is_max[idx]))
        argmax_rows.append(row)
    return pieces, labels, components, pd.DataFrame(argmax_rows)


def label_stats(labels: dict[str, np.ndarray], components: dict[str, dict[int, np.ndarray]]) -> pd.DataFrame:
    rows = []
    for piece, target in labels.items():
        row = {
            "piece": piece,
            "num_beats": int(len(target)),
            "target_sum": float(target.sum()),
            "true_events_ge_eval_threshold": int(np.count_nonzero(target >= EVENT_MIN_OVERRIDE)),
            "target_max": float(target.max()),
        }
        for level, weight in SQRT_MASS_WEIGHTS.items():
            row[f"L{level}_support"] = int(np.count_nonzero(components[piece][level] > 0))
            row[f"L{level}_weighted_ge_0p05"] = int(np.count_nonzero(float(weight) * components[piece][level] >= bw.runner.base.EVENT_MIN))
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = bw.runner.load_config()
    pieces, labels, components, argmax = load_sqrtmass_l2plus_labels()
    base_features, feature_cols = bw.runner.load_piece_features(pieces, cfg)
    rich_features = bw.runner.load_rich_features(pieces)
    missing = sorted(set(pieces) - set(rich_features))
    if missing:
        raise RuntimeError(f"Missing rich features: {missing}")
    bad = [
        (p, len(base_features[p]), len(labels[p]), len(rich_features[p]))
        for p in pieces
        if len(base_features[p]) != len(labels[p]) or len(labels[p]) != len(rich_features[p])
    ]
    if bad:
        raise RuntimeError(f"Length mismatch: {bad}")

    stats = label_stats(labels, components)
    stats.to_csv(OUT_DIR / "label_stats.csv", index=False)
    argmax.to_csv(OUT_DIR / "argmax_contribution_by_piece.csv", index=False)
    argmax_summary = []
    for level in SQRT_MASS_WEIGHTS:
        argmax_summary.append(
            {
                "level": f"L{level}",
                "unique_max_true_events": int(argmax[f"L{level}_unique_max"].sum()),
                "tied_max_includes_level": int(argmax[f"L{level}_tied_includes"].sum()),
            }
        )
    pd.DataFrame(argmax_summary).to_csv(OUT_DIR / "argmax_contribution_summary.csv", index=False)

    pieces = sorted(pieces)
    folds = bw.runner.base.make_folds(pieces, n_folds=5, seed=42)
    device = bw.runner.resolve_device()
    original_event_min = bw.runner.base.EVENT_MIN
    bw.runner.base.EVENT_MIN = float(EVENT_MIN_OVERRIDE)
    original_expected_count = bw.runner.base.expected_count_from_train_density
    original_extract_top_density = bw.runner.base.extract_top_density
    if DENSITY_MODE == "fixed_2bars":
        def fixed_two_bar_density(_labels, _train_pieces, num_beats):
            return max(1, int(round(float(num_beats) / max(float(DENSITY_BEATS), 1e-9))))

        bw.runner.base.expected_count_from_train_density = fixed_two_bar_density
        def fixed_min_distance_top_density(scores, expected_count, min_distance=None):
            return original_extract_top_density(
                scores,
                expected_count,
                min_distance=max(int(DENSITY_MIN_DISTANCE), 1),
            )

        bw.runner.base.extract_top_density = fixed_min_distance_top_density
    elif DENSITY_MODE != "train_density":
        raise ValueError(f"Unsupported MAZURKA_DENSITY_MODE={DENSITY_MODE}")

    print(
        f"device={device}; weights={SQRT_MASS_WEIGHTS}; "
        f"event_min={EVENT_MIN_OVERRIDE}; "
        f"density_mode={DENSITY_MODE}; density_beats={DENSITY_BEATS}; density_min_distance={DENSITY_MIN_DISTANCE}; "
        f"true_events={int(stats['true_events_ge_eval_threshold'].sum())}; "
        f"target_sum={float(stats['target_sum'].sum()):.4f}"
    )

    fold_frames = []
    aggregates = []
    baseline_df, baseline_agg = bw.runner.run_setting(
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
    for setting in ["random_branchwise", "branchwise_rich_only", "handcrafted_plus_branchwise"]:
        fold_df, aggregate = bw.run_setting(setting, cfg, pieces, labels, base_features, rich_features, folds, device)
        fold_frames.append(fold_df)
        aggregates.append(aggregate)

    pd.concat(fold_frames, ignore_index=True).to_csv(OUT_DIR / "fold_summary.csv", index=False)
    agg = pd.DataFrame(aggregates)
    agg.to_csv(OUT_DIR / "aggregate_totals.csv", index=False)
    (OUT_DIR / "metadata.json").write_text(
        json.dumps(
            {
                "weights": SQRT_MASS_WEIGHTS,
                "event_min": bw.runner.base.EVENT_MIN,
                "original_event_min": original_event_min,
                "density_mode": DENSITY_MODE,
                "density_beats": DENSITY_BEATS,
                "density_min_distance": DENSITY_MIN_DISTANCE,
                "original_expected_count_fn": getattr(original_expected_count, "__name__", str(original_expected_count)),
                "original_extract_top_density_fn": getattr(original_extract_top_density, "__name__", str(original_extract_top_density)),
                "rich_dir": str(bw.runner.RICH_DIR),
                "branch_dim": bw.BRANCH_DIM,
                "scalar_dim": bw.SCALAR_DIM,
                "branchwise_concat_dim": 6 * bw.BRANCH_DIM + bw.SCALAR_DIM,
                "beat_emb_dim": bw.BEAT_EMB_DIM,
                "pieces": pieces,
                "folds": folds,
                "feature_columns": feature_cols,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("\nArgmax contribution:")
    print(pd.DataFrame(argmax_summary).to_string(index=False))
    print("\nAggregate:")
    print(agg.round(4).to_string(index=False))
    print(f"\nWrote {OUT_DIR}")


if __name__ == "__main__":
    main()
