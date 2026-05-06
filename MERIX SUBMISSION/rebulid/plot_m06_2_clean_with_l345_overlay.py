#!/usr/bin/env python3

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import plot_clean_outer_merge56_reconstruction as recon_plot


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = recon_plot.BOUNDARY_RESTART_DIR / recon_plot.DEFAULT_CONFIG
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "merge56_per_performer_zscore_m06_2_top_panel_no_true"
DEFAULT_SELECTED_DIR = SCRIPT_DIR / "clean_outer_reconstruction_merge56_seed44_per_performer_zscore_predcount"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Overlay L3/L4/L5-only reconstruction on top of the existing "
            "M06-2 clean predicted tempo reconstruction figure."
        )
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--piece-id", default="M06-2")
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--normalization", choices=["robust_relative", "per_performer_zscore"], default="per_performer_zscore")
    parser.add_argument("--selection-mode", choices=["predicted_events", "train_avg_topk"], default="predicted_events")
    parser.add_argument("--high-level-mode", choices=["l345", "l45"], default="l345")
    parser.add_argument("--high-level-refit", action="store_true")
    parser.add_argument("--train-floor", type=float, default=0.05)
    parser.add_argument("--beat-start", type=int, default=0)
    parser.add_argument("--beat-end", type=int, default=100)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--selected-breakpoints-dir", default=str(DEFAULT_SELECTED_DIR))
    parser.add_argument("--hide-performer-median", action="store_true")
    parser.add_argument("--hide-legend", action="store_true")
    parser.add_argument("--hide-title", action="store_true")
    parser.add_argument("--x-label", default="Beat index")
    parser.add_argument("--y-label", default=None)
    return parser.parse_args()


def _zero_level(num_beats: int) -> tuple[np.ndarray, np.ndarray]:
    return np.asarray([0, num_beats - 1], dtype=int), np.asarray([0.0, 0.0], dtype=float)


def _build_design_matrix_subset(
    num_beats: int,
    level_sets: dict[str, np.ndarray],
    subset_levels: tuple[str, ...],
    strengths_by_level: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    cols = [np.ones(num_beats, dtype=float)]
    for level in subset_levels:
        strengths = None if strengths_by_level is None else strengths_by_level.get(level)
        cols.append(recon_plot.build_level_basis(num_beats, level_sets[level], strengths=strengths))
    return np.stack(cols, axis=1)


def _fit_beta_subset(
    normalized_tempo_by_piece: dict[str, dict[str, np.ndarray]],
    target_freq_by_piece: dict[str, dict[str, pd.DataFrame]],
    train_pieces: list[str],
    threshold: float,
    subset_levels: tuple[str, ...],
) -> np.ndarray:
    x_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []
    for piece_id in train_pieces:
        level_sets: dict[str, np.ndarray] = {}
        for level in subset_levels:
            frame = target_freq_by_piece[level][piece_id]
            level_sets[level] = frame.loc[frame["frequency_target"] >= threshold, "beat_idx"].to_numpy(dtype=int)
        piece_curves = normalized_tempo_by_piece[piece_id]
        x_piece = _build_design_matrix_subset(
            num_beats=len(next(iter(piece_curves.values()))),
            level_sets=level_sets,
            subset_levels=subset_levels,
            strengths_by_level=None,
        )
        for curve in piece_curves.values():
            x_list.append(x_piece)
            y_list.append(np.asarray(curve, dtype=float))
    x_all = np.concatenate(x_list, axis=0)
    y_all = np.concatenate(y_list, axis=0)
    beta, *_ = np.linalg.lstsq(x_all, y_all, rcond=None)
    return beta


def _apply_params_subset(
    mean_tempo: np.ndarray,
    level_sets: dict[str, np.ndarray],
    beta: np.ndarray,
    subset_levels: tuple[str, ...],
    strengths_by_level: dict[str, np.ndarray] | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    x = _build_design_matrix_subset(
        num_beats=len(mean_tempo),
        level_sets=level_sets,
        subset_levels=subset_levels,
        strengths_by_level=strengths_by_level,
    )
    y_hat = x @ beta
    rmse = float(np.sqrt(np.mean((y_hat - mean_tempo) ** 2)))
    corr = float(np.corrcoef(mean_tempo, y_hat)[0, 1])
    return y_hat, {"rmse": rmse, "corr": corr}


def _plot_performer_density(
    ax: plt.Axes,
    x: np.ndarray,
    performer_curves: np.ndarray,
    show_median: bool = True,
) -> None:
    q00 = np.nanquantile(performer_curves, 0.00, axis=0)
    q05 = np.nanquantile(performer_curves, 0.05, axis=0)
    q10 = np.nanquantile(performer_curves, 0.10, axis=0)
    q20 = np.nanquantile(performer_curves, 0.20, axis=0)
    q30 = np.nanquantile(performer_curves, 0.30, axis=0)
    q40 = np.nanquantile(performer_curves, 0.40, axis=0)
    q50 = np.nanquantile(performer_curves, 0.50, axis=0)
    q60 = np.nanquantile(performer_curves, 0.60, axis=0)
    q70 = np.nanquantile(performer_curves, 0.70, axis=0)
    q80 = np.nanquantile(performer_curves, 0.80, axis=0)
    q90 = np.nanquantile(performer_curves, 0.90, axis=0)
    q95 = np.nanquantile(performer_curves, 0.95, axis=0)
    q100 = np.nanquantile(performer_curves, 1.00, axis=0)

    # Layered quantile envelopes: darker center indicates higher performer density.
    ax.fill_between(x, q05, q95, color="0.55", alpha=0.10, linewidth=0.0, label="Performer density (5-95%)")
    ax.fill_between(x, q10, q90, color="0.50", alpha=0.13, linewidth=0.0, label="Performer density (10-90%)")
    ax.fill_between(x, q20, q80, color="0.45", alpha=0.16, linewidth=0.0, label="Performer density (20-80%)")
    ax.fill_between(x, q30, q70, color="0.40", alpha=0.20, linewidth=0.0, label="Performer density (30-70%)")
    ax.fill_between(x, q40, q60, color="0.32", alpha=0.26, linewidth=0.0, label="Performer density (40-60%)")

    ax.plot(x, q100, color="0.58", linewidth=1.0, linestyle="--", label="Performer max")
    ax.plot(x, q00, color="0.58", linewidth=1.0, linestyle="--", label="Performer min")
    if show_median:
        ax.plot(x, q50, color="0.22", linewidth=1.2, linestyle="-", label="Performer median")


def _load_selected_breakpoint_csv(
    selected_dir: Path,
    piece_id: str,
    level_order: list[str],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    csv_path = selected_dir / f"{piece_id}_selected_level_breakpoints_seed44.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing selected breakpoints CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    df = df[df["piece_id"] == piece_id].copy()
    if df.empty:
        raise ValueError(f"No rows for piece {piece_id} in {csv_path}")
    beat_sets: dict[str, np.ndarray] = {}
    strengths: dict[str, np.ndarray] = {}
    for level in level_order:
        row = df[df["level"] == level]
        if row.empty:
            beat_sets[level] = np.asarray([], dtype=int)
            strengths[level] = np.asarray([], dtype=float)
            continue
        beats = ast.literal_eval(str(row.iloc[0]["selected_beats"]))
        beats_np = np.asarray(beats, dtype=int)
        beat_sets[level] = beats_np
        strengths[level] = np.ones(len(beats_np), dtype=float)
    return beat_sets, strengths


def main() -> None:
    args = parse_args()
    cfg = recon_plot.load_config(args.config)
    piece_id = str(args.piece_id)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_breakpoints_dir = Path(args.selected_breakpoints_dir).resolve()

    target_freq_by_piece, mean_tempo_by_piece, tempo_arrays_by_piece, _ = recon_plot.build_target_frequencies(
        cfg,
        [piece_id],
        str(args.normalization),
    )
    if piece_id not in mean_tempo_by_piece:
        raise ValueError(f"Piece {piece_id} is not available.")

    all_pieces = sorted(mean_tempo_by_piece.keys())
    train_pieces = [pid for pid in all_pieces if pid != piece_id]
    beta = recon_plot.fit_beta(
        normalized_tempo_by_piece=tempo_arrays_by_piece,
        target_freq_by_piece=target_freq_by_piece,
        train_pieces=train_pieces,
        threshold=float(args.train_floor),
    )

    avg_counts = recon_plot.compute_training_average_counts(
        target_freq_by_piece=target_freq_by_piece,
        train_pieces=train_pieces,
        threshold=float(args.train_floor),
    )

    report_root = (
        recon_plot.REPO_ROOT
        / "MERIX SUBMISSION"
        / "Boundary_Restart"
        / "reports"
        / "clean_outer_test"
    )

    mean_tempo = mean_tempo_by_piece[piece_id]
    num_beats = len(mean_tempo)
    pred_level_sets: dict[str, np.ndarray] = {}
    pred_strengths: dict[str, np.ndarray] = {}
    used_fallback_csv = False

    try:
        for level_label, spec in recon_plot.LEVEL_SPECS.items():
            if str(args.selection_mode) == "predicted_events":
                pred_events = recon_plot.load_outer_events(
                    report_root, spec["target"], int(args.seed), piece_id
                )
                beats = pred_events["beat_idx"].to_numpy(dtype=int)
                if "detector_score" in pred_events.columns:
                    strengths = pred_events["detector_score"].to_numpy(dtype=float)
                else:
                    strengths = np.ones(len(beats), dtype=float)
            else:
                outer_scores = recon_plot.load_outer_scores(
                    report_root, spec["target"], int(args.seed), piece_id
                )
                beats, strengths = recon_plot.select_topk_spaced_peaks(
                    outer_scores["detector_score"].to_numpy(dtype=float),
                    k=int(avg_counts[level_label]),
                    distance=int(cfg.get("data", {}).get("peak_distance", 6)),
                )
            pred_level_sets[level_label] = beats
            pred_strengths[level_label] = strengths
    except FileNotFoundError:
        pred_level_sets, pred_strengths = _load_selected_breakpoint_csv(
            selected_breakpoints_dir,
            piece_id,
            list(recon_plot.LEVEL_SPECS.keys()),
        )
        used_fallback_csv = True

    pred_all_recon, pred_all_metrics = recon_plot.apply_params(
        mean_tempo, pred_level_sets, beta, strengths_by_level=pred_strengths
    )

    high_level_mode = str(args.high_level_mode)
    high_level_refit = bool(args.high_level_refit)
    high_sets = dict(pred_level_sets)
    high_strengths = dict(pred_strengths)
    zero_levels = ("L1+", "L2+") if high_level_mode == "l345" else ("L1+", "L2+", "L3+")
    for low_level in zero_levels:
        z_beats, z_scores = _zero_level(num_beats)
        high_sets[low_level] = z_beats
        high_strengths[low_level] = z_scores

    if high_level_mode == "l345":
        subset_levels = ("L3+", "L4+", "L5+6")
    else:
        subset_levels = ("L4+", "L5+6")

    if high_level_refit:
        beta_high = _fit_beta_subset(
            normalized_tempo_by_piece=tempo_arrays_by_piece,
            target_freq_by_piece=target_freq_by_piece,
            train_pieces=train_pieces,
            threshold=float(args.train_floor),
            subset_levels=subset_levels,
        )
        high_recon, high_metrics = _apply_params_subset(
            mean_tempo=mean_tempo,
            level_sets=high_sets,
            beta=beta_high,
            subset_levels=subset_levels,
            strengths_by_level=high_strengths,
        )
    else:
        high_recon, high_metrics = recon_plot.apply_params(
            mean_tempo, high_sets, beta, strengths_by_level=high_strengths
        )

    x = np.arange(num_beats, dtype=int)
    x_start = max(0, int(args.beat_start))
    x_end = min(num_beats - 1, int(args.beat_end))
    if x_end <= x_start:
        raise ValueError("beat-end must be greater than beat-start.")

    mode_label = {
        "robust_relative": "robust relative tempo",
        "per_performer_zscore": "per-performer z-score tempo",
    }[str(args.normalization)]
    y_label = str(args.y_label) if args.y_label is not None else mode_label.title()
    high_label = "L3/L4/L5 only" if high_level_mode == "l345" else "L4/L5 only"

    fig, ax = plt.subplots(figsize=(11.5, 4.8))
    performer_curves = np.vstack([curve for curve in tempo_arrays_by_piece[piece_id].values()])
    _plot_performer_density(ax, x, performer_curves, show_median=not bool(args.hide_performer_median))
    ax.plot(x, mean_tempo, color="black", linewidth=2.0, label="Formal tempo curve (mean)")
    ax.plot(
        x,
        pred_all_recon,
        color="#d62728",
        linewidth=2.0,
        label=f"Predicted reconstruction (all levels, corr={pred_all_metrics['corr']:.3f})",
    )
    ax.plot(
        x,
        high_recon,
        color="#ff7f0e",
        linewidth=2.0,
        linestyle="--",
        label=f"Predicted reconstruction ({high_label}, corr={high_metrics['corr']:.3f})",
    )
    ax.set_xlim(x_start, x_end)
    ax.set_xlabel(str(args.x_label))
    ax.set_ylabel(y_label)
    if not bool(args.hide_title):
        ax.set_title(f"{piece_id}: predicted tempo reconstruction (beat {x_start}-{x_end})")
    ax.grid(alpha=0.25)
    if not bool(args.hide_legend):
        handles, labels = ax.get_legend_handles_labels()
        keep_labels = {
            "Performer min",
            "Performer max",
            "Performer median",
            "Formal tempo curve (mean)",
            f"Predicted reconstruction (all levels, corr={pred_all_metrics['corr']:.3f})",
            f"Predicted reconstruction ({high_label}, corr={high_metrics['corr']:.3f})",
        }
        filtered = [(h, l) for h, l in zip(handles, labels) if l in keep_labels]
        if filtered:
            ax.legend(
                [h for h, _ in filtered],
                [l for _, l in filtered],
                frameon=False,
                fontsize=9,
                loc="best",
            )
    fig.tight_layout()

    stem = f"{piece_id}_predicted_tempo_reconstruction_{x_start}_{x_end}_clean_with_{high_level_mode}_overlay"
    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    metrics_path = output_dir / f"{stem}_metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "piece_id": piece_id,
                "seed": int(args.seed),
                "normalization": str(args.normalization),
                "selection_mode": str(args.selection_mode),
                "beat_range": [x_start, x_end],
                "train_floor": float(args.train_floor),
                "all_levels": pred_all_metrics,
                "high_level_mode": high_level_mode,
                "high_level_refit": high_level_refit,
                "high_level_only": high_metrics,
                "performer_count": int(performer_curves.shape[0]),
                "used_fallback_selected_breakpoints_csv": used_fallback_csv,
                "selected_breakpoints_dir": str(selected_breakpoints_dir),
                "output_pdf": str(pdf_path),
                "output_png": str(png_path),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(str(pdf_path))


if __name__ == "__main__":
    main()
