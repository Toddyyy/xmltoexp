#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
VELOCITY_ROOT = ROOT / "MERIX SUBMISSION" / "Velocity"
ATEPP_EVAL_DIR = VELOCITY_ROOT / "atepp_op110_i_eval"

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(ATEPP_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(ATEPP_EVAL_DIR))

import plot_clean_outer_merge56_reconstruction as recon_plot  # noqa: E402
import run_atepp_op110_i_eval as atepp_eval  # noqa: E402


DEFAULT_EVAL_ROOT = VELOCITY_ROOT / "atepp_pure34_top3_eval" / "results"
DEFAULT_BETA_METADATA = (
    SCRIPT_DIR
    / "clean_outer_reconstruction_merge56_seed44_per_performer_zscore_predcount"
    / "reconstruction_metadata.json"
)

LEVEL_FILE_STEMS = {
    "L1+": "L1plus",
    "L2+": "L2plus",
    "L3+": "L3plus",
    "L4+": "L4plus",
    "L5+6": "L5plus6",
}
PIECE_METRIC_CHOICES = ("union_precision", "weighted_recall", "f1_union_weighted")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reconstruct a selected pure-3/4 ATEPP movement from predicted hierarchical phrase boundaries."
    )
    parser.add_argument("--eval-root", type=Path, default=DEFAULT_EVAL_ROOT)
    parser.add_argument("--piece-slug", default=None)
    parser.add_argument("--piece-metric", choices=PIECE_METRIC_CHOICES, default="union_precision")
    parser.add_argument("--beta-metadata", type=Path, default=DEFAULT_BETA_METADATA)
    parser.add_argument("--train-floor", type=float, default=0.05)
    parser.add_argument("--beat-unit", type=float, default=1.0)
    parser.add_argument("--smooth-window", type=int, default=3)
    parser.add_argument("--bpm-max", type=float, default=600.0)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def load_piece_metrics(eval_root: Path) -> pd.DataFrame:
    metrics_path = eval_root / "mean_metrics_by_piece.csv"
    df = pd.read_csv(metrics_path)
    df["f1_union_weighted"] = (
        2.0
        * df["union_precision"]
        * df["weighted_recall"]
        / (df["union_precision"] + df["weighted_recall"]).replace(0.0, np.nan)
    ).fillna(0.0)
    return df


def select_piece(piece_metrics: pd.DataFrame, piece_slug: str | None, metric: str) -> pd.Series:
    if piece_slug:
        matches = piece_metrics[piece_metrics["piece_slug"] == piece_slug]
        if matches.empty:
            raise ValueError(f"Unknown piece slug: {piece_slug}")
        return matches.iloc[0]
    ranked = piece_metrics.sort_values([metric, "piece_slug"], ascending=[False, True]).reset_index(drop=True)
    return ranked.iloc[0]


def normalize_per_performer_zscore(
    tempo_arrays: dict[str, np.ndarray],
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, float]]:
    normalized: dict[str, np.ndarray] = {}
    performer_means = []
    performer_stds = []
    for performer_id, curve in sorted(tempo_arrays.items()):
        curve_mean = float(np.nanmean(curve))
        curve_std = float(np.nanstd(curve))
        if not np.isfinite(curve_std) or curve_std < 1e-8:
            curve_std = 1.0
        performer_means.append(curve_mean)
        performer_stds.append(curve_std)
        normalized[performer_id] = ((curve - curve_mean) / curve_std).astype(np.float32)
    mean_curve = np.nanmean(np.stack(list(normalized.values()), axis=0), axis=0).astype(np.float32)
    raw_piece_mean = float(np.nanmean(np.stack(list(tempo_arrays.values()), axis=0)))
    return mean_curve, normalized, {
        "raw_piece_mean_tempo": raw_piece_mean,
        "mean_performer_mean_tempo": float(np.mean(performer_means)) if performer_means else raw_piece_mean,
        "mean_performer_std_tempo": float(np.mean(performer_stds)) if performer_stds else 1.0,
        "num_performers": float(len(normalized)),
    }


def load_level_truth_and_events(
    piece_root: Path,
    train_floor: float,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray], dict[str, pd.DataFrame]]:
    true_level_sets: dict[str, np.ndarray] = {}
    pred_level_sets: dict[str, np.ndarray] = {}
    pred_strengths: dict[str, np.ndarray] = {}
    truth_frames: dict[str, pd.DataFrame] = {}
    for level in recon_plot.LEVEL_SPECS:
        stem = LEVEL_FILE_STEMS[level]
        truth = pd.read_csv(piece_root / f"{stem}_truth.csv").sort_values("beat_idx").reset_index(drop=True)
        events = pd.read_csv(piece_root / f"{stem}_events.csv").sort_values(["event_rank", "beat_idx"]).reset_index(drop=True)
        truth_frames[level] = truth
        true_level_sets[level] = truth.loc[truth["frequency_target"] >= float(train_floor), "beat_idx"].to_numpy(dtype=int)
        pred_level_sets[level] = events["beat_idx"].to_numpy(dtype=int)
        pred_strengths[level] = events["detector_score"].to_numpy(dtype=float)
    return true_level_sets, pred_level_sets, pred_strengths, truth_frames


def save_reconstruction_outputs(
    output_dir: Path,
    piece_slug: str,
    mean_tempo: np.ndarray,
    tempo_arrays: dict[str, np.ndarray],
    true_level_sets: dict[str, np.ndarray],
    pred_level_sets: dict[str, np.ndarray],
    pred_strengths: dict[str, np.ndarray],
    truth_frames: dict[str, pd.DataFrame],
    beta: np.ndarray,
    train_floor: float,
) -> tuple[dict[str, float], dict[str, float]]:
    true_recon, true_metrics = recon_plot.apply_params(mean_tempo, true_level_sets, beta)
    pred_recon, pred_metrics = recon_plot.apply_params(mean_tempo, pred_level_sets, beta, strengths_by_level=pred_strengths)

    x = np.arange(len(mean_tempo))
    fig = plt.figure(figsize=(15, 12), constrained_layout=True)
    gs = fig.add_gridspec(6, 1, height_ratios=[2.4, 1, 1, 1, 1, 1], hspace=0.18)

    ax0 = fig.add_subplot(gs[0])
    for curve in tempo_arrays.values():
        ax0.plot(x, curve, color="0.8", linewidth=0.7, alpha=0.22)
    ax0.plot(x, mean_tempo, color="black", linewidth=2.0, label="Mean performer tempo curve")
    ax0.plot(
        x,
        true_recon,
        color="#1f77b4",
        linewidth=1.8,
        linestyle="--",
        label=f"True-boundary reconstruction (rmse={true_metrics['rmse']:.2f}, corr={true_metrics['corr']:.3f})",
    )
    ax0.plot(
        x,
        pred_recon,
        color="#d62728",
        linewidth=1.8,
        label=f"Predicted-boundary reconstruction (rmse={pred_metrics['rmse']:.2f}, corr={pred_metrics['corr']:.3f})",
    )
    ax0.set_ylabel("Tempo z-score")
    ax0.set_title(f"{piece_slug}: ATEPP pure 3/4 predicted tempo reconstruction")
    ax0.grid(alpha=0.25)
    ax0.legend(frameon=False, fontsize=9, loc="upper right")

    level_rows: list[dict[str, object]] = []
    for row_idx, level in enumerate(recon_plot.LEVEL_SPECS, start=1):
        ax = fig.add_subplot(gs[row_idx], sharex=ax0)
        truth = truth_frames[level]
        ax.plot(
            truth["beat_idx"],
            truth["frequency_target"],
            color=recon_plot.LEVEL_SPECS[level]["color"],
            linewidth=1.3,
            label=f"True {level} frequency",
        )
        ax.scatter(
            pred_level_sets[level],
            pred_strengths[level],
            color=recon_plot.LEVEL_SPECS[level]["color"],
            edgecolors="black",
            linewidths=0.3,
            s=26,
            alpha=0.9,
            label=f"Predicted events ({len(pred_level_sets[level])})",
            zorder=3,
        )
        ax.axhline(float(train_floor), color="0.6", linestyle="--", linewidth=0.8)
        ax.set_ylim(-0.02, 1.05)
        ax.set_ylabel(level)
        ax.grid(alpha=0.2)
        ax.legend(frameon=False, fontsize=8, loc="upper right")
        level_rows.append(
            {
                "piece_id": piece_slug,
                "level": level,
                "predicted_count": int(len(pred_level_sets[level])),
                "selected_beats": pred_level_sets[level].tolist(),
            }
        )
    fig.axes[-1].set_xlabel("Beat index")
    reconstruction_png = output_dir / f"{piece_slug}_atepp_per_performer_zscore_reconstruction.png"
    reconstruction_pdf = output_dir / f"{piece_slug}_atepp_per_performer_zscore_reconstruction.pdf"
    fig.savefig(reconstruction_png, dpi=180, bbox_inches="tight")
    fig.savefig(reconstruction_pdf, bbox_inches="tight")
    plt.close(fig)

    fig_all, ax_all = plt.subplots(figsize=(15, 5.2))
    for curve in tempo_arrays.values():
        ax_all.plot(x, curve, color="0.7", linewidth=0.8, alpha=0.25)
    ax_all.plot(
        x,
        pred_recon,
        color="#d62728",
        linewidth=1.8,
        label=f"Pred reconstruction (rmse={pred_metrics['rmse']:.2f}, corr={pred_metrics['corr']:.3f})",
    )
    ax_all.set_title(f"{piece_slug}: Pred reconstruction vs ALL performer tempo curves")
    ax_all.set_xlabel("Beat index")
    ax_all.set_ylabel("Tempo z-score")
    ax_all.grid(alpha=0.22)
    ax_all.legend(frameon=False, loc="upper right")
    all_true_png = output_dir / f"{piece_slug}_atepp_per_performer_zscore_pred_vs_all_true_tempo.png"
    all_true_pdf = output_dir / f"{piece_slug}_atepp_per_performer_zscore_pred_vs_all_true_tempo.pdf"
    fig_all.savefig(all_true_png, dpi=180, bbox_inches="tight")
    fig_all.savefig(all_true_pdf, bbox_inches="tight")
    plt.close(fig_all)

    pd.DataFrame(level_rows).to_csv(
        output_dir / f"{piece_slug}_selected_level_breakpoints.csv",
        index=False,
    )
    return true_metrics, pred_metrics


def main() -> None:
    args = parse_args()
    eval_root = args.eval_root.resolve()
    beta_metadata_path = args.beta_metadata.resolve()

    piece_metrics = load_piece_metrics(eval_root)
    selected_piece = select_piece(piece_metrics, args.piece_slug, args.piece_metric)
    piece_slug = str(selected_piece["piece_slug"])
    piece_root = eval_root / piece_slug

    beta_metadata = json.loads(beta_metadata_path.read_text(encoding="utf-8"))
    beta = np.array(beta_metadata["beta"], dtype=float)

    manifest = json.loads((piece_root / "manifest.json").read_text(encoding="utf-8"))
    piece_dir = Path(manifest["selected_piece_dir"])
    num_beats = int(manifest["num_beats"])
    tempo_arrays, failed_matches = atepp_eval.load_tempo_arrays(
        piece_dir=piece_dir,
        num_beats=num_beats,
        beat_unit=float(args.beat_unit),
        smooth_window=int(args.smooth_window),
        bpm_max=float(args.bpm_max),
    )
    mean_tempo, normalized_tempo_arrays, tempo_stats = normalize_per_performer_zscore(tempo_arrays)
    true_level_sets, pred_level_sets, pred_strengths, truth_frames = load_level_truth_and_events(
        piece_root=piece_root,
        train_floor=float(args.train_floor),
    )

    default_output = SCRIPT_DIR / f"atepp_pure34_best_{args.piece_metric}_per_performer_zscore_seed44"
    output_dir = (args.output_dir or default_output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    true_metrics, pred_metrics = save_reconstruction_outputs(
        output_dir=output_dir,
        piece_slug=piece_slug,
        mean_tempo=mean_tempo,
        tempo_arrays=normalized_tempo_arrays,
        true_level_sets=true_level_sets,
        pred_level_sets=pred_level_sets,
        pred_strengths=pred_strengths,
        truth_frames=truth_frames,
        beta=beta,
        train_floor=float(args.train_floor),
    )

    summary = pd.DataFrame(
        [
            {
                "piece_slug": piece_slug,
                "piece_metric": args.piece_metric,
                "piece_metric_value": float(selected_piece[args.piece_metric]),
                "mean_union_precision": float(selected_piece["union_precision"]),
                "mean_weighted_recall": float(selected_piece["weighted_recall"]),
                "mean_f1_union_weighted": float(selected_piece["f1_union_weighted"]),
                "num_beats": num_beats,
                "num_performers": int(len(normalized_tempo_arrays)),
                "true_rmse": true_metrics["rmse"],
                "true_corr": true_metrics["corr"],
                "pred_rmse": pred_metrics["rmse"],
                "pred_corr": pred_metrics["corr"],
                **tempo_stats,
            }
        ]
    )
    summary.to_csv(output_dir / "reconstruction_summary.csv", index=False)

    output_metadata = {
        "piece_metric": args.piece_metric,
        "selected_piece": selected_piece.to_dict(),
        "piece_dir": str(piece_dir),
        "num_beats": num_beats,
        "failed_match_files": failed_matches,
        "train_floor": float(args.train_floor),
        "beat_unit": float(args.beat_unit),
        "smooth_window": int(args.smooth_window),
        "bpm_max": float(args.bpm_max),
        "beta_metadata_path": str(beta_metadata_path),
        "beta_standardization": beta_metadata.get("standardization"),
        "true_metrics": true_metrics,
        "pred_metrics": pred_metrics,
        "tempo_stats": tempo_stats,
    }
    (output_dir / "reconstruction_metadata.json").write_text(json.dumps(output_metadata, indent=2), encoding="utf-8")
    print(str(output_dir))


if __name__ == "__main__":
    main()
