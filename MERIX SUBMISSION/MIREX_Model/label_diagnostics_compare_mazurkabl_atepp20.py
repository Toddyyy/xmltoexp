from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "MERIX SUBMISSION" / "MIREX_Model" / "label_diagnostics_mazurkabl_vs_atepp20"
MAZ_SCRIPT = ROOT / "MERIX SUBMISSION" / "MIREX_Model" / "run_mazurkabl_l2plus_weighted_target_experiment.py"
ATEPP_SCRIPT = ROOT / "MERIX SUBMISSION" / "MIREX_Model" / "run_atepp20_l2plus_weighted_target_experiment.py"
LEVELS = (2, 3, 4, 5, 6)
WEIGHTS = {2: 0.28, 3: 0.46, 4: 0.64, 5: 0.82, 6: 1.0}
CUTOFFS = (0.001, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2)
OVERLAP_TOL = 1


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def nearest_overlap(a: np.ndarray, b: np.ndarray, tol: int = OVERLAP_TOL) -> float:
    a = np.asarray(a, dtype=np.int32)
    b = np.asarray(b, dtype=np.int32)
    if a.size == 0:
        return np.nan
    if b.size == 0:
        return 0.0
    count = 0
    b_sorted = np.sort(b)
    for x in a:
        pos = np.searchsorted(b_sorted, x)
        ok = False
        for j in (pos - 1, pos, pos + 1):
            if 0 <= j < len(b_sorted) and abs(int(b_sorted[j]) - int(x)) <= tol:
                ok = True
                break
        count += int(ok)
    return count / len(a)


def spacing(events: np.ndarray) -> np.ndarray:
    events = np.asarray(events, dtype=np.int32)
    if len(events) < 2:
        return np.zeros(0, dtype=np.int32)
    return np.diff(np.sort(events)).astype(np.int32)


def diagnostics_for_dataset(name: str, labels: dict[str, np.ndarray], components: dict[str, dict[int, np.ndarray]]) -> dict[str, pd.DataFrame]:
    value_rows = []
    cutoff_rows = []
    level_rows = []
    piece_rows = []
    overlap_rows = []
    spacing_rows = []

    all_targets = np.concatenate([labels[p] for p in labels])
    positive = all_targets[all_targets > 0]
    quantiles = [0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0]
    value_rows.append(
        {
            "dataset": name,
            "beats": int(len(all_targets)),
            "positive_beats": int(np.count_nonzero(all_targets > 0)),
            "positive_ratio": float(np.mean(all_targets > 0)),
            "positive_sum": float(np.sum(all_targets[all_targets > 0])),
            **{
                f"positive_q{str(q).replace('.', 'p')}": float(np.quantile(positive, q)) if positive.size else np.nan
                for q in quantiles
            },
        }
    )

    for cutoff in CUTOFFS:
        events_total = 0
        target_weight_total = 0.0
        for piece, target in labels.items():
            mask = target >= cutoff
            events_total += int(np.count_nonzero(mask))
            target_weight_total += float(np.sum(target[mask]))
        cutoff_rows.append(
            {
                "dataset": name,
                "cutoff": float(cutoff),
                "true_events": events_total,
                "events_per_100_beats": float(100.0 * events_total / max(len(all_targets), 1)),
                "target_weight_sum": target_weight_total,
            }
        )

    for piece, target in labels.items():
        piece_rows.append(
            {
                "dataset": name,
                "piece": piece,
                "num_beats": int(len(target)),
                "positive_beats": int(np.count_nonzero(target > 0)),
                "target_sum": float(np.sum(target)),
                **{f"events_ge_{str(c).replace('.', 'p')}": int(np.count_nonzero(target >= c)) for c in CUTOFFS},
                **{f"events_per_100_ge_{str(c).replace('.', 'p')}": float(100.0 * np.count_nonzero(target >= c) / max(len(target), 1)) for c in CUTOFFS},
            }
        )

        for cutoff in CUTOFFS:
            sp = spacing(np.flatnonzero(target >= cutoff))
            for val in sp.tolist():
                spacing_rows.append({"dataset": name, "piece": piece, "cutoff": float(cutoff), "spacing": int(val)})

    for level in LEVELS:
        level_values = []
        level_positive = 0
        level_events_by_piece = []
        for piece, comp in components.items():
            arr = np.asarray(comp[level], dtype=np.float32)
            pos = arr[arr > 0]
            if pos.size:
                level_values.append(pos)
            level_positive += int(np.count_nonzero(arr > 0))
            level_events_by_piece.append(int(np.count_nonzero(arr > 0)))
        vals = np.concatenate(level_values) if level_values else np.zeros(0, dtype=np.float32)
        level_rows.append(
            {
                "dataset": name,
                "level": level,
                "weight": WEIGHTS[level],
                "positive_events": level_positive,
                "events_per_piece_mean": float(np.mean(level_events_by_piece)) if level_events_by_piece else 0.0,
                "consensus_q25": float(np.quantile(vals, 0.25)) if vals.size else np.nan,
                "consensus_q50": float(np.quantile(vals, 0.50)) if vals.size else np.nan,
                "consensus_q75": float(np.quantile(vals, 0.75)) if vals.size else np.nan,
                "consensus_q90": float(np.quantile(vals, 0.90)) if vals.size else np.nan,
                "consensus_ge_0p25": int(np.count_nonzero(vals >= 0.25)) if vals.size else 0,
                "consensus_ge_0p50": int(np.count_nonzero(vals >= 0.50)) if vals.size else 0,
                "consensus_ge_0p75": int(np.count_nonzero(vals >= 0.75)) if vals.size else 0,
            }
        )

    for src in LEVELS:
        for dst in LEVELS:
            overlaps = []
            for piece, comp in components.items():
                a = np.flatnonzero(np.asarray(comp[piece_level := src], dtype=np.float32) > 0)
                b = np.flatnonzero(np.asarray(comp[dst], dtype=np.float32) > 0)
                overlaps.append(nearest_overlap(a, b, tol=OVERLAP_TOL))
            overlaps = [x for x in overlaps if np.isfinite(x)]
            overlap_rows.append(
                {
                    "dataset": name,
                    "source_level": src,
                    "target_level": dst,
                    "overlap_rate_mean": float(np.mean(overlaps)) if overlaps else np.nan,
                    "overlap_rate_median": float(np.median(overlaps)) if overlaps else np.nan,
                    "tolerance_beats": OVERLAP_TOL,
                }
            )

    return {
        "target_value_summary": pd.DataFrame(value_rows),
        "cutoff_event_counts": pd.DataFrame(cutoff_rows),
        "consensus_by_level": pd.DataFrame(level_rows),
        "piece_density": pd.DataFrame(piece_rows),
        "spacing": pd.DataFrame(spacing_rows),
        "level_overlap": pd.DataFrame(overlap_rows),
    }


def plot_cutoff_counts(frame: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for dataset, group in frame.groupby("dataset"):
        group = group.sort_values("cutoff")
        ax.plot(group["cutoff"], group["true_events"], marker="o", linewidth=2, label=dataset)
    ax.set_xscale("log")
    ax.set_xlabel("target cutoff")
    ax.set_ylabel("true event count")
    ax.set_title("True event count vs target cutoff")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "cutoff_event_count_compare.png", dpi=180)
    plt.close(fig)


def plot_piece_density(frame: pd.DataFrame, out_dir: Path) -> None:
    col = "events_per_100_ge_0p05"
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    data = [group[col].to_numpy(dtype=float) for _, group in frame.groupby("dataset")]
    labels = [dataset for dataset, _ in frame.groupby("dataset")]
    ax.boxplot(data, labels=labels, showmeans=True)
    ax.set_ylabel("events per 100 beats (target >= 0.05)")
    ax.set_title("Piece-level boundary density")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "piece_density_boxplot_ge_0p05.png", dpi=180)
    plt.close(fig)


def plot_overlap_heatmaps(frame: pd.DataFrame, out_dir: Path) -> None:
    for dataset, group in frame.groupby("dataset"):
        pivot = group.pivot(index="source_level", columns="target_level", values="overlap_rate_mean").loc[list(LEVELS), list(LEVELS)]
        fig, ax = plt.subplots(figsize=(5.5, 4.8))
        im = ax.imshow(pivot.to_numpy(dtype=float), vmin=0, vmax=1, cmap="viridis")
        ax.set_xticks(range(len(LEVELS)), LEVELS)
        ax.set_yticks(range(len(LEVELS)), LEVELS)
        ax.set_xlabel("target level")
        ax.set_ylabel("source level")
        ax.set_title(f"{dataset} level overlap within +/-{OVERLAP_TOL} beat")
        for i in range(len(LEVELS)):
            for j in range(len(LEVELS)):
                ax.text(j, i, f"{pivot.iloc[i, j]:.2f}", ha="center", va="center", color="white" if pivot.iloc[i, j] < 0.55 else "black", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(out_dir / f"{dataset}_level_overlap_heatmap.png", dpi=180)
        plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    maz = load_module(MAZ_SCRIPT, "diag_mazurkabl_quick")
    atepp = load_module(ATEPP_SCRIPT, "diag_atepp20_quick")
    datasets = []
    for name, module in [("MazurkaBL44", maz), ("ATEPP20", atepp)]:
        pieces, labels, components = module.load_l2plus_weighted_labels()
        datasets.append((name, pieces, labels, components))
        print(name, "pieces", len(pieces), "beats", sum(len(labels[p]) for p in pieces))

    collected: dict[str, list[pd.DataFrame]] = {}
    for name, _, labels, components in datasets:
        result = diagnostics_for_dataset(name, labels, components)
        for key, value in result.items():
            collected.setdefault(key, []).append(value)

    outputs = {}
    for key, frames in collected.items():
        merged = pd.concat(frames, ignore_index=True)
        path = OUT_DIR / f"{key}.csv"
        merged.to_csv(path, index=False)
        outputs[key] = merged

    plot_cutoff_counts(outputs["cutoff_event_counts"], OUT_DIR)
    plot_piece_density(outputs["piece_density"], OUT_DIR)
    plot_overlap_heatmaps(outputs["level_overlap"], OUT_DIR)

    print("\nTarget value summary:")
    print(outputs["target_value_summary"].to_string(index=False))
    print("\nCutoff event counts:")
    print(outputs["cutoff_event_counts"].to_string(index=False))
    print(f"\nWrote {OUT_DIR}")


if __name__ == "__main__":
    main()
