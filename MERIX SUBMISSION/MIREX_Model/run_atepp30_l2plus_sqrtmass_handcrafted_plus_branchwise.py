from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
MIREX = ROOT / "MERIX SUBMISSION" / "MIREX_Model"
METER_AUTO = ROOT / "MERIX SUBMISSION" / "MIREX_Model_meter_auto"
BRANCHWISE_SCRIPT = MIREX / "run_mazurkabl_l2plus_branchwise_bottleneck_cnn.py"

DATASET_NAME = "atepp30"
BEAT_TABLE = METER_AUTO / "outputs" / "atepp30_beat_table.csv.gz"
LABEL_DIR = METER_AUTO / "beat_data_atepp30_performer_levels"
MANIFEST = METER_AUTO / "outputs" / "atepp30_manifest.csv"
RICH_DIR = MIREX / "atepp30_midibert_rich_beat_features"
OUT_DIR = MIREX / "atepp30_l2plus_sqrtmass_handcrafted_plus_branchwise"

EVENT_MIN = 0.01
DENSITY_BEATS = 6.0
DENSITY_MIN_DISTANCE = 1
SQRT_MASS_WEIGHTS = {
    2: 0.205,
    3: 0.284,
    4: 0.408,
    5: 0.613,
    6: 1.000,
}


def load_branchwise():
    spec = importlib.util.spec_from_file_location("atepp30_branchwise_runner", BRANCHWISE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {BRANCHWISE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["atepp30_branchwise_runner"] = module
    spec.loader.exec_module(module)
    return module


bw = load_branchwise()
runner = bw.runner
base = runner.base


def piece_id_from_npz(path: Path) -> str:
    match = re.match(r"(.+)_\d+_L[1-6]\.npz$", path.name)
    if not match:
        raise ValueError(f"Cannot parse ATEPP piece id from {path.name}")
    return match.group(1)


def load_level_frequency(piece: str, level: int) -> np.ndarray:
    files = sorted(LABEL_DIR.glob(f"{piece}_*_L{level}.npz"))
    if not files:
        raise FileNotFoundError(f"No label files for {piece} L{level}")
    arrays = [np.load(path, allow_pickle=True)["boundary_probs"].astype(np.float32) for path in files]
    lengths = {len(a) for a in arrays}
    if len(lengths) != 1:
        raise RuntimeError(f"{piece} L{level}: inconsistent label lengths {sorted(lengths)}")
    return np.mean(np.stack(arrays, axis=0), axis=0).astype(np.float32)


def load_sqrtmass_labels() -> tuple[list[str], dict[str, np.ndarray], dict[str, dict[int, np.ndarray]], pd.DataFrame]:
    pieces = sorted({piece_id_from_npz(path) for path in LABEL_DIR.glob("*_L2.npz")})
    labels: dict[str, np.ndarray] = {}
    components: dict[str, dict[int, np.ndarray]] = {}
    argmax_rows = []
    for piece in pieces:
        weighted = []
        levels = []
        components[piece] = {}
        for level, weight in SQRT_MASS_WEIGHTS.items():
            freq = load_level_frequency(piece, level)
            components[piece][level] = freq
            weighted.append(float(weight) * freq)
            levels.append(level)
        matrix = np.stack(weighted, axis=0).astype(np.float32)
        target = np.max(matrix, axis=0).astype(np.float32)
        labels[piece] = target
        true = target >= EVENT_MIN
        is_max = np.isclose(matrix, target[None, :], atol=1e-8) & true[None, :]
        ties = is_max.sum(axis=0)
        row = {
            "piece": piece,
            "true_events": int(true.sum()),
            "tied_true_events": int(np.count_nonzero(ties > 1)),
        }
        for idx, level in enumerate(levels):
            row[f"L{level}_unique_max"] = int(np.count_nonzero(true & (ties == 1) & is_max[idx]))
            row[f"L{level}_tied_includes"] = int(np.count_nonzero(true & (ties > 1) & is_max[idx]))
        argmax_rows.append(row)
    return pieces, labels, components, pd.DataFrame(argmax_rows)


def load_piece_features(pieces: list[str], cfg: dict) -> tuple[dict[str, np.ndarray], list[str]]:
    cfg = dict(cfg)
    cfg["data"] = dict(cfg.get("data", {}))
    cfg["data"]["beat_table_path"] = str(BEAT_TABLE)
    return runner.load_piece_features(pieces, cfg)


def load_rich_features(pieces: list[str]) -> dict[str, np.ndarray]:
    out = {}
    for piece in pieces:
        path = RICH_DIR / f"{piece}_midibert_rich_beat_features.npz"
        if not path.exists():
            continue
        with np.load(path) as data:
            out[piece] = np.asarray(data["rich_beat_features"], dtype=np.float32)
    return out


def label_stats(labels: dict[str, np.ndarray], components: dict[str, dict[int, np.ndarray]]) -> pd.DataFrame:
    rows = []
    manifest_df = pd.read_csv(MANIFEST) if MANIFEST.exists() else pd.DataFrame()
    manifest_by_piece = {str(row.piece_id): row for row in manifest_df.itertuples(index=False)} if not manifest_df.empty else {}
    for piece, target in labels.items():
        mrow = manifest_by_piece.get(piece)
        row = {
            "piece": piece,
            "num_beats": int(len(target)),
            "target_sum": float(target.sum()),
            "true_events_ge_eval_threshold": int(np.count_nonzero(target >= EVENT_MIN)),
            "target_max": float(target.max()),
            "performance_midis": int(getattr(mrow, "performance_midis", 0)) if mrow is not None else None,
            "usable_tempo_curves": int(getattr(mrow, "usable_tempo_curves", 0)) if mrow is not None else None,
        }
        for level, weight in SQRT_MASS_WEIGHTS.items():
            row[f"L{level}_support"] = int(np.count_nonzero(components[piece][level] > 0))
            row[f"L{level}_weighted_ge_event_min"] = int(np.count_nonzero(float(weight) * components[piece][level] >= EVENT_MIN))
        rows.append(row)
    return pd.DataFrame(rows)


def fixed_two_bar_density(_labels, _train_pieces, num_beats: int) -> int:
    return max(1, int(round(float(num_beats) / max(float(DENSITY_BEATS), 1e-9))))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = runner.load_config()
    pieces, labels, components, argmax = load_sqrtmass_labels()
    base_features, feature_cols = load_piece_features(pieces, cfg)
    rich_features = load_rich_features(pieces)
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
    folds = base.make_folds(pieces, n_folds=5, seed=42)
    device = runner.resolve_device()

    original_event_min = base.EVENT_MIN
    original_min_distance = base.MIN_DISTANCE
    original_expected_count = base.expected_count_from_train_density
    original_extract_top_density = base.extract_top_density
    base.EVENT_MIN = float(EVENT_MIN)
    base.MIN_DISTANCE = int(DENSITY_MIN_DISTANCE)
    base.expected_count_from_train_density = fixed_two_bar_density

    def top_density_min_distance(scores, expected_count, min_distance=None):
        return original_extract_top_density(scores, expected_count, min_distance=max(int(DENSITY_MIN_DISTANCE), 1))

    base.extract_top_density = top_density_min_distance

    print(
        f"dataset={DATASET_NAME}; setting=handcrafted_plus_branchwise; device={device}; pieces={len(pieces)}; "
        f"weights={SQRT_MASS_WEIGHTS}; event_min={EVENT_MIN}; "
        f"density=1 per {DENSITY_BEATS:g} beats; min_distance={DENSITY_MIN_DISTANCE}; "
        f"true_events={int(stats['true_events_ge_eval_threshold'].sum())}; "
        f"target_sum={float(stats['target_sum'].sum()):.4f}; "
        f"base_dim={base_features[pieces[0]].shape[1]}; rich_dim={rich_features[pieces[0]].shape[1]}"
    )

    try:
        fold_df, aggregate = bw.run_setting(
            "handcrafted_plus_branchwise",
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
                "dataset_name": DATASET_NAME,
                "beat_table": str(BEAT_TABLE),
                "label_dir": str(LABEL_DIR),
                "manifest": str(MANIFEST),
                "rich_dir": str(RICH_DIR),
                "weights": SQRT_MASS_WEIGHTS,
                "event_min": EVENT_MIN,
                "density_beats": DENSITY_BEATS,
                "density_min_distance": DENSITY_MIN_DISTANCE,
                "setting": "handcrafted_plus_branchwise",
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
    print("\nArgmax contribution:")
    print(pd.DataFrame(argmax_summary).to_string(index=False))
    print("\nAggregate:")
    print(agg.round(4).to_string(index=False))
    print(f"\nWrote {OUT_DIR}")


if __name__ == "__main__":
    main()
