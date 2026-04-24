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
from boundary_restart.cumulative_targets import build_piece_frequency_for_raw_levels, build_topdown_cumulative_frequency  # noqa: E402
from boundary_restart.features import PeakConfig  # noqa: E402


DEFAULT_CONFIG = "configs/salience_grouped3_hi8_score_only_xml_curated.yaml"
OUTER_PIECES = ("M06-1", "M06-2", "M06-3")
TRAIN_FLOOR = 0.05
MERGE_TOLERANCE = 2
SEED = 44
LEVEL_SPECS = {
    "L1+": {
        "target": "level1plus_split56_boundary",
        "component_order": ("level6", "level5", "level4", "level3", "level2", "level1"),
        "color": "#0a6cff",
    },
    "L2+": {
        "target": "level2plus_split56_boundary",
        "component_order": ("level6", "level5", "level4", "level3", "level2"),
        "color": "#00a35c",
    },
    "L3+": {
        "target": "level3plus_split56_boundary",
        "component_order": ("level6", "level5", "level4", "level3"),
        "color": "#ff8a00",
    },
    "L4+": {
        "target": "level4plus_split56_boundary",
        "component_order": ("level6", "level5", "level4"),
        "color": "#7b1fa2",
    },
    "L5+": {
        "target": "level5plus_split56_boundary",
        "component_order": ("level6", "level5"),
        "color": "#c2185b",
    },
    "L6": {
        "target": "level6_boundary",
        "component_order": ("level6",),
        "color": "#6d4c41",
    },
}
COMPONENT_RAW_LEVELS = {
    "level1": (1,),
    "level2": (2,),
    "level3": (3,),
    "level4": (4,),
    "level5": (5,),
    "level6": (6,),
}
COMPONENT_WEIGHTS = {
    "level6": 1.0,
    "level5": 0.82,
    "level4": 0.64,
    "level3": 0.46,
    "level2": 0.28,
    "level1": 0.16,
}
PREDICTION_ROOT = REPO_ROOT / "MERIX SUBMISSION" / "Boundary_Restart" / "outputs" / "local_runs" / "clean_outer_test"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reconstruct tempo curves from split56 clean outer predictions.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--train-floor", type=float, default=TRAIN_FLOOR)
    parser.add_argument("--outer-piece", nargs="+", default=list(OUTER_PIECES))
    parser.add_argument(
        "--output-dir",
        default="outputs/local_runs/clean_outer_reconstruction_split56_seed44",
        help="Output directory relative to Boundary_Restart.",
    )
    return parser.parse_args()


def load_piece_mean_tempo(beat_time_dir: Path, piece_id: str) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    beat_time_path = beat_time_dir / f"{piece_id}beat_time.csv"
    df_bt, performer_cols = load_beat_time(beat_time_path)
    tempo_arrays = compute_tempo_curves(
        df_bt,
        performer_cols,
        smooth_window=3,
        bpm_range=(0, 5000),
        clip_max=600,
    )
    mean_tempo = np.nanmean(np.vstack([tempo_arrays[name] for name in tempo_arrays]), axis=0)
    return mean_tempo, tempo_arrays


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
            nearest = int(np.argmin(np.abs(orig_b - beat))) if orig_b.size else 0
            s[idx] = 1.0 if orig_b.size == 0 else orig_s[nearest]

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


def load_outer_scores(detector_target: str, seed: int, piece_id: str) -> pd.DataFrame:
    path = PREDICTION_ROOT / f"weighted_topdown_{detector_target}_seed{seed}" / "outer_predictions.csv.gz"
    df = pd.read_csv(path)
    df = df[df["piece_id"] == piece_id].copy().sort_values("beat_idx").reset_index(drop=True)
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
    mean_tempo_by_piece: dict[str, np.ndarray],
    target_freq_by_piece: dict[str, dict[str, pd.DataFrame]],
    train_pieces: list[str],
    threshold: float,
) -> np.ndarray:
    X_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []
    for piece_id in train_pieces:
        y = mean_tempo_by_piece[piece_id]
        level_sets = {}
        for level, frame_map in target_freq_by_piece.items():
            frame = frame_map[piece_id]
            level_sets[level] = frame.loc[frame["frequency_target"] >= threshold, "beat_idx"].to_numpy(dtype=int)
        X_list.append(build_design_matrix(len(y), level_sets))
        y_list.append(y)
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
        counts[level] = max(1, int(round(float(np.mean(level_counts)))))
    return counts


def build_target_frequencies(
    cfg: dict,
) -> tuple[dict[str, dict[str, pd.DataFrame]], dict[str, np.ndarray], dict[str, dict[str, np.ndarray]]]:
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
    for piece_id in pieces:
        mean_tempo, tempo_arrays = load_piece_mean_tempo(beat_time_dir, piece_id)
        mean_tempo_by_piece[piece_id] = mean_tempo
        tempo_arrays_by_piece[piece_id] = tempo_arrays

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

    return target_freq_by_piece, mean_tempo_by_piece, tempo_arrays_by_piece


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (SCRIPT_DIR / config_path).resolve()
    cfg = load_config(config_path)

    outer_pieces = list(args.outer_piece)
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = (SCRIPT_DIR / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    target_freq_by_piece, mean_tempo_by_piece, tempo_arrays_by_piece = build_target_frequencies(cfg)
    all_pieces = sorted(mean_tempo_by_piece.keys())
    train_pieces = [piece_id for piece_id in all_pieces if piece_id not in outer_pieces]

    beta = fit_beta(mean_tempo_by_piece, target_freq_by_piece, train_pieces, threshold=float(args.train_floor))
    avg_counts = compute_training_average_counts(target_freq_by_piece, train_pieces, threshold=float(args.train_floor))
    peak_distance = int(cfg.get("data", {}).get("peak_distance", 6))

    summary_rows: list[dict[str, object]] = []
    pdf_path = output_dir / f"clean_outer_split56_reconstruction_seed{args.seed}.pdf"
    all_true_pdf_path = output_dir / f"clean_outer_split56_pred_vs_all_true_tempo_seed{args.seed}.pdf"

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

                outer_scores = load_outer_scores(spec["target"], args.seed, piece_id)
                if len(outer_scores) != len(mean_tempo):
                    outer_scores = outer_scores.iloc[: min(len(outer_scores), len(mean_tempo))].copy()

                selected_beats, selected_scores = select_topk_spaced_peaks(
                    outer_scores["detector_score"].to_numpy(dtype=float),
                    k=avg_counts[level],
                    distance=peak_distance,
                )
                pred_level_sets[level] = selected_beats
                pred_strengths[level] = selected_scores
                level_detail_rows.append(
                    {
                        "piece_id": piece_id,
                        "level": level,
                        "avg_train_count": avg_counts[level],
                        "selected_count": int(len(selected_beats)),
                        "selected_beats": selected_beats.tolist(),
                    }
                )

            true_recon, true_metrics = apply_params(mean_tempo, true_level_sets, beta)
            pred_recon, pred_metrics = apply_params(mean_tempo, pred_level_sets, beta, strengths_by_level=pred_strengths)

            x = np.arange(len(mean_tempo))
            fig = plt.figure(figsize=(15, 13.2))
            gs = fig.add_gridspec(len(LEVEL_SPECS) + 1, 1, height_ratios=[2.4, 1, 1, 1, 1, 1, 1], hspace=0.18)

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
                label=f"Reconstruction from clean outer predictions with train-avg K (rmse={pred_metrics['rmse']:.2f}, corr={pred_metrics['corr']:.3f})",
            )
            ax0.set_ylabel("Tempo (BPM)")
            ax0.set_title(f"{piece_id}: split L5/L6 clean outer reconstruction (seed {args.seed})")
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
                    label=f"Predicted top-{avg_counts[level]} peaks",
                    zorder=3,
                )
                ax.axhline(float(args.train_floor), color="0.6", linestyle="--", linewidth=0.8)
                ax.set_ylim(-0.02, 1.05)
                ax.set_ylabel(level)
                ax.grid(alpha=0.2)
                ax.legend(frameon=False, fontsize=8, loc="upper right")
            fig.axes[-1].set_xlabel("Beat index")
            fig.tight_layout()

            png_path = output_dir / f"{piece_id}_clean_outer_split56_reconstruction_seed{args.seed}.png"
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
            ax_all.set_title(f"{piece_id}: Pred reconstruction vs ALL true tempo curves")
            ax_all.set_xlabel("Beat index")
            ax_all.set_ylabel("Tempo (BPM)")
            ax_all.grid(alpha=0.22)
            ax_all.legend(frameon=False, loc="upper right")
            all_true_png_path = output_dir / f"{piece_id}_clean_outer_split56_pred_vs_all_true_tempo_seed{args.seed}.png"
            all_true_single_pdf_path = output_dir / f"{piece_id}_clean_outer_split56_pred_vs_all_true_tempo_seed{args.seed}.pdf"
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
        "avg_train_counts": avg_counts,
        "beta": beta.tolist(),
        "summary": summary_rows,
        "pdf_path": str(pdf_path),
        "pred_vs_all_true_pdf_path": str(all_true_pdf_path),
    }
    (output_dir / "reconstruction_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(str(output_dir))


if __name__ == "__main__":
    main()
