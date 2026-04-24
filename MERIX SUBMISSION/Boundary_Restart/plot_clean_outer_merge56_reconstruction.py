from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from scipy.signal import find_peaks


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
MIREX_MODEL_DIR = REPO_ROOT / "MERIX SUBMISSION" / "MIREX_Model"
BOUNDARY_RESTART_DIR = REPO_ROOT / "MERIX SUBMISSION" / "Boundary_Restart"

if str(MIREX_MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MIREX_MODEL_DIR))
if str(BOUNDARY_RESTART_DIR) not in sys.path:
    sys.path.insert(0, str(BOUNDARY_RESTART_DIR))

from build_mazurka_beat_npz_performer_levels import compute_tempo_curves, load_beat_time  # noqa: E402
from boundary_restart.config import load_config, resolve_path  # noqa: E402
from boundary_restart.cumulative_targets import (  # noqa: E402
    build_piece_frequency_for_raw_levels,
    build_topdown_cumulative_frequency,
    cumulative_components_for_target,
)
from boundary_restart.features import PeakConfig  # noqa: E402


DEFAULT_CONFIG = "configs/salience_grouped3_hi8_score_only_xml_curated.yaml"
OUTER_PIECES = ("M06-1", "M06-2", "M06-3")
LEVEL_SPECS = {
    "L1+": {"target": "level1plus_boundary", "component_order": ("level56", "level4", "level3", "level2", "level1"), "color": "#0a6cff"},
    "L2+": {"target": "level2plus_boundary", "component_order": ("level56", "level4", "level3", "level2"), "color": "#00a35c"},
    "L3+": {"target": "level3plus_boundary", "component_order": ("level56", "level4", "level3"), "color": "#ff8a00"},
    "L4+": {"target": "level4plus_boundary", "component_order": ("level56", "level4"), "color": "#7b1fa2"},
    "L5+6": {"target": "level56_boundary", "component_order": ("level56",), "color": "#c2185b"},
}
COMPONENT_RAW_LEVELS = {
    "level1": (1,),
    "level2": (2,),
    "level3": (3,),
    "level4": (4,),
    "level56": (5, 6),
}
TRAIN_FLOOR = 0.05
MERGE_TOLERANCE = 2
COMPONENT_WEIGHTS = {
    "level56": 1.0,
    "level4": 0.64,
    "level3": 0.46,
    "level2": 0.28,
    "level1": 0.16,
}
SEED = 44
NORMALIZATION_CHOICES = ("robust_relative", "per_performer_zscore")
SELECTION_CHOICES = ("predicted_events", "train_avg_topk")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reconstruct tempo curves from clean outer merged56 predictions.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--train-floor", type=float, default=TRAIN_FLOOR)
    parser.add_argument("--outer-piece", nargs="+", default=list(OUTER_PIECES))
    parser.add_argument("--normalization", choices=NORMALIZATION_CHOICES, default="robust_relative")
    parser.add_argument("--selection-mode", choices=SELECTION_CHOICES, default="predicted_events")
    parser.add_argument("--name-prefix", default=None)
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory relative to Boundary_Restart.",
    )
    return parser.parse_args()


def load_piece_robust_relative_tempos(
    beat_time_dir: Path,
    piece_id: str,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, float]]:
    beat_time_path = beat_time_dir / f"{piece_id}beat_time.csv"
    df_bt, performer_cols = load_beat_time(beat_time_path)
    tempo_arrays = compute_tempo_curves(
        df_bt,
        performer_cols,
        smooth_window=3,
        bpm_range=(0, 5000),
        clip_max=600,
    )
    normalized = {}
    performer_medians = []
    performer_iqrs = []
    for name, curve in tempo_arrays.items():
        curve_median = float(np.nanmedian(curve))
        performer_medians.append(curve_median)
        if np.isfinite(curve_median) and abs(curve_median) > 1e-8:
            relative = (curve / curve_median) - 1.0
        else:
            relative = curve - curve_median
        q25, q75 = np.nanpercentile(relative, [25.0, 75.0])
        curve_iqr = float(q75 - q25)
        if not np.isfinite(curve_iqr) or curve_iqr < 1e-8:
            curve_iqr = 1.0
        performer_iqrs.append(curve_iqr)
        normalized[name] = (relative / curve_iqr).astype(np.float32)
    mean_tempo = np.nanmean(np.vstack([normalized[name] for name in normalized]), axis=0)
    raw_piece_mean = float(np.nanmean(np.vstack([tempo_arrays[name] for name in tempo_arrays])))
    mean_performer_median = float(np.mean(performer_medians)) if performer_medians else raw_piece_mean
    mean_performer_iqr = float(np.mean(performer_iqrs)) if performer_iqrs else 1.0
    return mean_tempo, normalized, {
        "raw_piece_mean_tempo": raw_piece_mean,
        "mean_performer_median_tempo": mean_performer_median,
        "mean_performer_relative_iqr": mean_performer_iqr,
        "num_performers": float(len(performer_medians)),
    }


def load_piece_performer_zscore_tempos(
    beat_time_dir: Path,
    piece_id: str,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, float]]:
    beat_time_path = beat_time_dir / f"{piece_id}beat_time.csv"
    df_bt, performer_cols = load_beat_time(beat_time_path)
    tempo_arrays = compute_tempo_curves(
        df_bt,
        performer_cols,
        smooth_window=3,
        bpm_range=(0, 5000),
        clip_max=600,
    )
    normalized = {}
    performer_means = []
    performer_stds = []
    for name, curve in tempo_arrays.items():
        curve_mean = float(np.nanmean(curve))
        curve_std = float(np.nanstd(curve))
        if not np.isfinite(curve_std) or curve_std < 1e-8:
            curve_std = 1.0
        performer_means.append(curve_mean)
        performer_stds.append(curve_std)
        normalized[name] = ((curve - curve_mean) / curve_std).astype(np.float32)
    mean_tempo = np.nanmean(np.vstack([normalized[name] for name in normalized]), axis=0)
    raw_piece_mean = float(np.nanmean(np.vstack([tempo_arrays[name] for name in tempo_arrays])))
    return mean_tempo, normalized, {
        "raw_piece_mean_tempo": raw_piece_mean,
        "mean_performer_mean_tempo": float(np.mean(performer_means)) if performer_means else raw_piece_mean,
        "mean_performer_std_tempo": float(np.mean(performer_stds)) if performer_stds else 1.0,
        "num_performers": float(len(performer_means)),
    }


def build_level_basis(num_beats: int, boundaries: np.ndarray, strengths: np.ndarray | None = None) -> np.ndarray:
    b = np.asarray(boundaries, dtype=int)
    b = b[(b >= 0) & (b < num_beats)]
    b = np.unique(b)
    if b.size == 0:
        b = np.array([0, num_beats - 1], dtype=int)
    if b[0] != 0:
        b = np.insert(b, 0, 0)
    if b[-1] != num_beats - 1:
        b = np.append(b, num_beats - 1)
    b.sort()

    if strengths is None or len(strengths) == 0:
        s = np.ones(len(b), dtype=float)
    else:
        s = np.zeros(len(b), dtype=float)
        orig_b = np.asarray(boundaries, dtype=int)
        orig_s = np.asarray(strengths, dtype=float)
        for idx, beat in enumerate(b):
            if orig_b.size == 0:
                s[idx] = 1.0
            else:
                nearest = int(np.argmin(np.abs(orig_b - beat)))
                s[idx] = orig_s[nearest]

    basis = np.zeros(num_beats, dtype=float)
    for idx in range(len(b) - 1):
        start = int(b[idx])
        end = int(b[idx + 1])
        if end <= start:
            continue
        amp = 0.5 * (s[idx] + s[idx + 1])
        t = np.arange(start, end + 1)
        u = (t - start) / (end - start)
        basis[t] = amp * (-4.0 * u * (1.0 - u))
    return basis


def build_design_matrix(
    num_beats: int,
    level_sets: dict[str, np.ndarray],
    strengths_by_level: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    cols = [np.ones(num_beats, dtype=float)]
    for level in LEVEL_SPECS:
        strengths = None if strengths_by_level is None else strengths_by_level.get(level)
        cols.append(build_level_basis(num_beats, level_sets[level], strengths=strengths))
    return np.stack(cols, axis=1)


def apply_params(
    mean_tempo: np.ndarray,
    level_sets: dict[str, np.ndarray],
    beta: np.ndarray,
    strengths_by_level: dict[str, np.ndarray] | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    X = build_design_matrix(len(mean_tempo), level_sets, strengths_by_level)
    y_hat = X @ beta
    rmse = float(np.sqrt(np.mean((y_hat - mean_tempo) ** 2)))
    corr = float(np.corrcoef(mean_tempo, y_hat)[0, 1])
    return y_hat, {"rmse": rmse, "corr": corr}


def load_outer_scores(report_root: Path, detector_target: str, seed: int, piece_id: str) -> pd.DataFrame:
    path = report_root / f"weighted_topdown_merge56_{detector_target}_seed{seed}" / "outer_predictions.csv.gz"
    df = pd.read_csv(path)
    df = df[df["piece_id"] == piece_id].copy().sort_values("beat_idx").reset_index(drop=True)
    return df


def load_outer_events(report_root: Path, detector_target: str, seed: int, piece_id: str) -> pd.DataFrame:
    path = report_root / f"weighted_topdown_merge56_{detector_target}_seed{seed}" / "predicted_events.csv.gz"
    df = pd.read_csv(path)
    df = df[df["piece_id"] == piece_id].copy()
    if "event_rank" in df.columns:
        df = df.sort_values(["event_rank", "beat_idx"]).reset_index(drop=True)
    else:
        df = df.sort_values("beat_idx").reset_index(drop=True)
    return df


def select_topk_spaced_peaks(scores: np.ndarray, k: int, distance: int) -> tuple[np.ndarray, np.ndarray]:
    if scores.size == 0 or k <= 0:
        return np.array([], dtype=int), np.array([], dtype=float)

    candidate_idx, _ = find_peaks(scores, distance=max(1, int(distance)))
    candidate_idx = candidate_idx[(candidate_idx > 0) & (candidate_idx < len(scores) - 1)]
    if candidate_idx.size == 0:
        candidate_idx = np.arange(1, len(scores) - 1, dtype=int)

    order = sorted(candidate_idx.tolist(), key=lambda idx: float(scores[idx]), reverse=True)
    selected: list[int] = []
    for idx in order:
        if all(abs(idx - prev) >= int(distance) for prev in selected):
            selected.append(int(idx))
        if len(selected) >= int(k):
            break

    if len(selected) < int(k):
        fallback = np.argsort(scores)[::-1].tolist()
        for idx in fallback:
            idx = int(idx)
            if idx <= 0 or idx >= len(scores) - 1:
                continue
            if any(abs(idx - prev) < int(distance) for prev in selected):
                continue
            selected.append(idx)
            if len(selected) >= int(k):
                break

    selected = sorted(set(selected))
    strengths = np.asarray([float(scores[idx]) for idx in selected], dtype=float)
    return np.asarray(selected, dtype=int), strengths


def fit_beta(
    normalized_tempo_by_piece: dict[str, dict[str, np.ndarray]],
    target_freq_by_piece: dict[str, dict[str, pd.DataFrame]],
    train_pieces: list[str],
    threshold: float,
) -> np.ndarray:
    X_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []
    for piece_id in train_pieces:
        level_sets = {}
        for level, spec in LEVEL_SPECS.items():
            frame = target_freq_by_piece[level][piece_id]
            level_sets[level] = frame.loc[frame["frequency_target"] >= threshold, "beat_idx"].to_numpy(dtype=int)
        piece_curves = normalized_tempo_by_piece[piece_id]
        X_piece = build_design_matrix(len(next(iter(piece_curves.values()))), level_sets)
        for curve in piece_curves.values():
            X_list.append(X_piece)
            y_list.append(np.asarray(curve, dtype=float))
    X_all = np.concatenate(X_list, axis=0)
    y_all = np.concatenate(y_list, axis=0)
    beta, *_ = np.linalg.lstsq(X_all, y_all, rcond=None)
    return beta


def compute_training_average_counts(
    target_freq_by_piece: dict[str, dict[str, pd.DataFrame]],
    train_pieces: list[str],
    threshold: float,
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for level in LEVEL_SPECS:
        level_counts = []
        for piece_id in train_pieces:
            frame = target_freq_by_piece[level][piece_id]
            level_counts.append(int((frame["frequency_target"] >= threshold).sum()))
        counts[level] = max(1, int(round(float(np.mean(level_counts))))) if level_counts else 1
    return counts


def build_target_frequencies(
    cfg: dict,
    outer_pieces: list[str],
    normalization: str,
) -> tuple[
    dict[str, dict[str, pd.DataFrame]],
    dict[str, np.ndarray],
    dict[str, dict[str, np.ndarray]],
    dict[str, dict[str, float]],
]:
    data_cfg = cfg.get("data", {})
    table_path = resolve_path(cfg, data_cfg["beat_table_path"])
    peak_cfg = PeakConfig(
        distance=int(data_cfg.get("peak_distance", 6)),
        height=float(data_cfg.get("peak_height", 0.15)),
        prominence=float(data_cfg.get("peak_prominence", 0.05)),
    )
    beat_unit_fallback = float(data_cfg.get("beat_unit_fallback", 1.0))
    beat_time_dir = REPO_ROOT / "MazurkaBL-master" / "beat_time"

    df = pd.read_csv(table_path, usecols=["piece_id", "sample_id", "source_path", "beat_idx"]).drop_duplicates()
    pieces = sorted(df["piece_id"].unique())

    mean_tempo_by_piece: dict[str, np.ndarray] = {}
    tempo_arrays_by_piece: dict[str, dict[str, np.ndarray]] = {}
    tempo_stats_by_piece: dict[str, dict[str, float]] = {}
    loader = {
        "robust_relative": load_piece_robust_relative_tempos,
        "per_performer_zscore": load_piece_performer_zscore_tempos,
    }[normalization]
    for piece_id in pieces:
        mean_tempo, tempo_arrays, tempo_stats = loader(beat_time_dir, piece_id)
        mean_tempo_by_piece[piece_id] = mean_tempo
        tempo_arrays_by_piece[piece_id] = tempo_arrays
        tempo_stats_by_piece[piece_id] = tempo_stats

    component_map: dict[str, pd.DataFrame] = {}
    for component_name, raw_levels in COMPONENT_RAW_LEVELS.items():
        component_map[component_name] = build_piece_frequency_for_raw_levels(
            df.copy(),
            raw_levels=raw_levels,
            peak_cfg=peak_cfg,
            beat_unit_fallback=beat_unit_fallback,
        )

    base_piece = df[["piece_id", "beat_idx"]].drop_duplicates().sort_values(["piece_id", "beat_idx"]).reset_index(drop=True)
    target_freq_by_piece: dict[str, dict[str, pd.DataFrame]] = {level: {} for level in LEVEL_SPECS}
    for level, spec in LEVEL_SPECS.items():
        merged = build_topdown_cumulative_frequency(
            base_piece=base_piece,
            component_map=component_map,
            component_order=spec["component_order"],
            tolerance=MERGE_TOLERANCE,
            component_weights=COMPONENT_WEIGHTS,
        )
        for piece_id, group in merged.groupby("piece_id", sort=False):
            target_freq_by_piece[level][piece_id] = group[["beat_idx", "frequency_target"]].reset_index(drop=True)

    return target_freq_by_piece, mean_tempo_by_piece, tempo_arrays_by_piece, tempo_stats_by_piece


def main() -> None:
    args = parse_args()
    mode = str(args.normalization)
    selection_mode = str(args.selection_mode)
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (SCRIPT_DIR / config_path).resolve()
    cfg = load_config(config_path)

    outer_pieces = list(args.outer_piece)
    default_output = f"outputs/local_runs/merge56_{mode}_{selection_mode}_seed44"
    output_dir = Path(args.output_dir or default_output)
    if not output_dir.is_absolute():
        output_dir = (SCRIPT_DIR / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    target_freq_by_piece, mean_tempo_by_piece, tempo_arrays_by_piece, tempo_stats_by_piece = build_target_frequencies(
        cfg,
        outer_pieces,
        mode,
    )
    all_pieces = sorted(mean_tempo_by_piece.keys())
    train_pieces = [piece_id for piece_id in all_pieces if piece_id not in outer_pieces]

    beta = fit_beta(tempo_arrays_by_piece, target_freq_by_piece, train_pieces, threshold=float(args.train_floor))
    avg_counts = compute_training_average_counts(target_freq_by_piece, train_pieces, threshold=float(args.train_floor))

    report_root = REPO_ROOT / "MERIX SUBMISSION" / "Boundary_Restart" / "reports" / "clean_outer_test"
    x_limit = max(len(mean_tempo_by_piece[p]) for p in outer_pieces)

    summary_rows: list[dict[str, object]] = []
    mode_label = {
        "robust_relative": "robust relative tempo",
        "per_performer_zscore": "per-performer z-score tempo",
    }[mode]
    mode_stem = {
        "robust_relative": "robust_relative",
        "per_performer_zscore": "per_performer_zscore",
    }[mode]
    selection_stem = {
        "predicted_events": "predcount",
        "train_avg_topk": "topkavg",
    }[selection_mode]
    name_prefix = str(args.name_prefix or f"merge56_{mode_stem}_{selection_stem}_seed44")
    pdf_path = output_dir / f"{name_prefix}_reconstruction.pdf"
    all_true_pdf_path = output_dir / f"{name_prefix}_pred_vs_all_true_tempo.pdf"
    with PdfPages(pdf_path) as pdf, PdfPages(all_true_pdf_path) as all_true_pdf:
        for piece_id in outer_pieces:
            mean_tempo = mean_tempo_by_piece[piece_id]
            tempo_arrays = tempo_arrays_by_piece[piece_id]

            true_level_sets: dict[str, np.ndarray] = {}
            pred_level_sets: dict[str, np.ndarray] = {}
            pred_strengths: dict[str, np.ndarray] = {}
            level_detail_rows: list[dict[str, object]] = []

            for level, spec in LEVEL_SPECS.items():
                frame = target_freq_by_piece[level][piece_id]
                true_level_sets[level] = frame.loc[frame["frequency_target"] >= float(args.train_floor), "beat_idx"].to_numpy(dtype=int)

                if selection_mode == "predicted_events":
                    pred_events = load_outer_events(report_root, spec["target"], args.seed, piece_id)
                    selected_beats = pred_events["beat_idx"].to_numpy(dtype=int)
                    selected_scores = pred_events["detector_score"].to_numpy(dtype=float)
                else:
                    outer_scores = load_outer_scores(report_root, spec["target"], args.seed, piece_id)
                    selected_beats, selected_scores = select_topk_spaced_peaks(
                        outer_scores["detector_score"].to_numpy(dtype=float),
                        k=avg_counts[level],
                        distance=int(cfg.get("data", {}).get("peak_distance", 6)),
                    )
                pred_level_sets[level] = selected_beats
                pred_strengths[level] = selected_scores
                level_detail_rows.append(
                    {
                        "piece_id": piece_id,
                        "level": level,
                        "selection_mode": selection_mode,
                        "target_count": int(avg_counts[level]) if selection_mode == "train_avg_topk" else int(len(selected_beats)),
                        "predicted_count": int(len(selected_beats)),
                        "selected_count": int(len(selected_beats)),
                        "selected_beats": selected_beats.tolist(),
                    }
                )

            true_recon, true_metrics = apply_params(mean_tempo, true_level_sets, beta)
            pred_recon, pred_metrics = apply_params(mean_tempo, pred_level_sets, beta, strengths_by_level=pred_strengths)

            x = np.arange(len(mean_tempo))
            fig = plt.figure(figsize=(15, 12))
            gs = fig.add_gridspec(6, 1, height_ratios=[2.4, 1, 1, 1, 1, 1], hspace=0.18)

            ax0 = fig.add_subplot(gs[0])
            for curve in tempo_arrays.values():
                ax0.plot(x, curve, color="0.8", linewidth=0.7, alpha=0.22)
            ax0.plot(x, mean_tempo, color="black", linewidth=2.0, label="Formal tempo curve (mean)")
            ax0.plot(
                x,
                true_recon,
                color="#1f77b4",
                linewidth=1.8,
                linestyle="--",
                label=f"Reconstruction from true cumulative boundaries (rmse={true_metrics['rmse']:.2f}, corr={true_metrics['corr']:.3f})",
            )
            ax0.plot(
                x,
                pred_recon,
                color="#d62728",
                linewidth=1.8,
                label=f"Reconstruction from clean outer predicted events (rmse={pred_metrics['rmse']:.2f}, corr={pred_metrics['corr']:.3f})",
            )
            ax0.set_ylabel(mode_label.title())
            ax0.set_title(f"{piece_id}: merged L5+6 clean outer {mode_label} reconstruction (seed {args.seed})")
            ax0.grid(alpha=0.25)
            ax0.legend(frameon=False, fontsize=9, loc="upper right")

            for row_idx, level in enumerate(LEVEL_SPECS, start=1):
                ax = fig.add_subplot(gs[row_idx], sharex=ax0)
                true_frame = target_freq_by_piece[level][piece_id]
                ax.plot(
                    true_frame["beat_idx"],
                    true_frame["frequency_target"],
                    color=LEVEL_SPECS[level]["color"],
                    linewidth=1.3,
                    label=f"True {level} frequency",
                )
                ax.scatter(
                    pred_level_sets[level],
                    pred_strengths[level],
                    color=LEVEL_SPECS[level]["color"],
                    edgecolors="black",
                    linewidths=0.3,
                    s=26,
                    alpha=0.9,
                    label=f"Predicted events ({len(pred_level_sets[level])})",
                    zorder=3,
                )
                ax.axhline(float(args.train_floor), color="0.6", linestyle="--", linewidth=0.8)
                ax.set_ylim(-0.02, 1.05)
                ax.set_ylabel(level)
                ax.grid(alpha=0.2)
                ax.legend(frameon=False, fontsize=8, loc="upper right")
            fig.axes[-1].set_xlabel("Beat index")
            fig.tight_layout()

            png_path = output_dir / f"{piece_id}_{name_prefix}_reconstruction.png"
            fig.savefig(png_path, dpi=180, bbox_inches="tight")
            pdf.savefig(fig, bbox_inches="tight")
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
            ax_all.set_title(f"{piece_id}: Pred reconstruction vs ALL {mode_label} true tempo curves")
            ax_all.set_xlabel("Beat index")
            ax_all.set_ylabel(mode_label.title())
            ax_all.grid(alpha=0.22)
            ax_all.legend(frameon=False, loc="upper right")
            all_true_png_path = output_dir / f"{piece_id}_{name_prefix}_pred_vs_all_true_tempo.png"
            all_true_single_pdf_path = output_dir / f"{piece_id}_{name_prefix}_pred_vs_all_true_tempo.pdf"
            fig_all.savefig(all_true_png_path, dpi=180, bbox_inches="tight")
            fig_all.savefig(all_true_single_pdf_path, bbox_inches="tight")
            all_true_pdf.savefig(fig_all, bbox_inches="tight")
            plt.close(fig_all)

            summary_rows.append(
                {
                    "piece_id": piece_id,
                    "true_rmse": true_metrics["rmse"],
                    "true_corr": true_metrics["corr"],
                    "pred_rmse": pred_metrics["rmse"],
                    "pred_corr": pred_metrics["corr"],
                    "raw_piece_mean_tempo": tempo_stats_by_piece[piece_id]["raw_piece_mean_tempo"],
                    **tempo_stats_by_piece[piece_id],
                }
            )

            pd.DataFrame(level_detail_rows).to_csv(
                output_dir / f"{piece_id}_selected_level_breakpoints_seed{args.seed}.csv",
                index=False,
            )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_dir / "reconstruction_summary.csv", index=False)

    metadata = {
        "seed": int(args.seed),
        "outer_pieces": outer_pieces,
        "train_piece_count": len(train_pieces),
        "train_floor": float(args.train_floor),
        "merge_tolerance": MERGE_TOLERANCE,
        "component_weights": COMPONENT_WEIGHTS,
        "standardization": {
            "robust_relative": "per-performer relative tempo ((tempo / median) - 1) with robust IQR scaling",
            "per_performer_zscore": "per-performer z-score ((tempo - mean) / std)",
        }[mode],
        "prediction_selection": {
            "predicted_events": "use clean outer predicted_events directly",
            "train_avg_topk": "use top-K peaks from clean outer scores, where K is the mean training boundary count",
        }[selection_mode],
        "selection_mode": selection_mode,
        "avg_train_counts": avg_counts,
        "name_prefix": name_prefix,
        "beta": beta.tolist(),
        "summary": summary_rows,
        "pdf_path": str(pdf_path),
        "pred_vs_all_true_pdf_path": str(all_true_pdf_path),
    }
    (output_dir / "reconstruction_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(str(output_dir))


if __name__ == "__main__":
    main()
