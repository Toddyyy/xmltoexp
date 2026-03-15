#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


LEVELS = ("low", "mid", "high")


def load_level_frame(piece_root: Path, level: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    run_dir = piece_root / f"tcn_{level}_boundary_union_recall_cpu"
    val_df = pd.read_csv(run_dir / "val_predictions.csv.gz").sort_values("beat_idx").reset_index(drop=True)
    event_df = pd.read_csv(run_dir / "predicted_events.csv.gz").sort_values("beat_idx").reset_index(drop=True)
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    return val_df, event_df, summary


def compute_alignment_stats(val_df: pd.DataFrame, event_df: pd.DataFrame, summary: dict) -> dict:
    score = val_df["detector_score"]
    freq = val_df["frequency_target"]
    union = val_df["union_target"]
    return {
        "pearson_score_vs_frequency": float(score.corr(freq, method="pearson")),
        "spearman_score_vs_frequency": float(score.corr(freq, method="spearman")),
        "pearson_score_vs_union": float(score.corr(union, method="pearson")),
        "spearman_score_vs_union": float(score.corr(union, method="spearman")),
        "pred_event_count": int(len(event_df)),
        "matched_event_count": int(event_df["matched_union"].fillna(False).sum()),
        "true_union_event_count": int(summary["union_metrics"]["true_union_events"]),
        "threshold": float(summary["union_metrics"]["threshold"]),
        "union_precision": float(summary["union_metrics"]["union_precision"]),
        "weighted_recall": float(summary["union_metrics"]["weighted_recall"]),
        "consensus_recall": float(summary["union_metrics"]["consensus_recall"]),
    }


def plot_level(ax, level: str, val_df: pd.DataFrame, event_df: pd.DataFrame, stats: dict) -> None:
    beat_idx = val_df["beat_idx"].to_numpy()
    ax.plot(beat_idx, val_df["detector_score"].to_numpy(), color="#004c6d", linewidth=1.5, label="detector score")
    ax.bar(
        beat_idx,
        val_df["frequency_target"].to_numpy(),
        width=0.9,
        color="#ffa600",
        alpha=0.35,
        label="true union frequency",
        zorder=1,
    )

    true_beats = val_df.loc[val_df["union_target"] > 0.5, "beat_idx"].to_numpy()
    if true_beats.size:
        ax.vlines(true_beats, ymin=0.0, ymax=1.0, color="black", alpha=0.08, linewidth=0.7, zorder=0)

    if not event_df.empty:
        matched = event_df[event_df["matched_union"].fillna(False)]
        unmatched = event_df[~event_df["matched_union"].fillna(False)]
        if not matched.empty:
            ax.scatter(
                matched["beat_idx"],
                matched["detector_score"],
                marker="o",
                s=18,
                color="#2f9e44",
                label="predicted event (matched)",
                zorder=3,
            )
        if not unmatched.empty:
            ax.scatter(
                unmatched["beat_idx"],
                unmatched["detector_score"],
                marker="x",
                s=22,
                color="#d9480f",
                label="predicted event (unmatched)",
                zorder=3,
            )

    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel(level)
    ax.grid(axis="y", alpha=0.2, linewidth=0.6)
    ax.set_title(
        (
            f"{level} | pred={stats['pred_event_count']} match={stats['matched_event_count']}/{stats['true_union_event_count']} | "
            f"thr={stats['threshold']:.2f}\n"
            f"pearson={stats['pearson_score_vs_frequency']:.3f} spearman={stats['spearman_score_vs_frequency']:.3f} | "
            f"P={stats['union_precision']:.3f} WR={stats['weighted_recall']:.3f} CR={stats['consensus_recall']:.3f}"
        ),
        fontsize=9,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot direct level detector scores against true union frequencies.")
    parser.add_argument("--piece-root", required=True, help="Directory containing tcn_low/mid/high_boundary_union_recall_cpu runs.")
    parser.add_argument("--output-prefix", default=None, help="Output prefix for png/pdf/json summary.")
    args = parser.parse_args()

    piece_root = Path(args.piece_root).resolve()
    output_prefix = Path(args.output_prefix).resolve() if args.output_prefix else piece_root / "tcn_direct_levels_alignment"
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    stats_summary = {}
    fig, axes = plt.subplots(len(LEVELS), 1, figsize=(18, 10), sharex=True, constrained_layout=True)

    for ax, level in zip(axes, LEVELS):
        val_df, event_df, summary = load_level_frame(piece_root, level)
        stats = compute_alignment_stats(val_df, event_df, summary)
        stats_summary[level] = stats
        plot_level(ax, level, val_df, event_df, stats)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    axes[-1].set_xlabel("beat index")
    fig.suptitle(f"TCN direct level predictions vs true union frequency: {piece_root.name}", fontsize=14, y=1.02)

    png_path = output_prefix.with_suffix(".png")
    pdf_path = output_prefix.with_suffix(".pdf")
    json_path = output_prefix.with_suffix(".json")
    fig.savefig(png_path, dpi=180, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    json_path.write_text(json.dumps(stats_summary, indent=2), encoding="utf-8")
    print(str(png_path))
    print(str(pdf_path))
    print(str(json_path))


if __name__ == "__main__":
    main()
