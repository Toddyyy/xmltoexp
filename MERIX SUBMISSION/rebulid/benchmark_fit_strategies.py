from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.linear_model import RidgeCV


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
COMPONENT_WEIGHTS = {
    "level56": 1.0,
    "level4": 0.64,
    "level3": 0.46,
    "level2": 0.28,
    "level1": 0.16,
}
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
RIDGE_ALPHAS = np.logspace(-4, 4, 17)


@dataclass(frozen=True)
class MethodSpec:
    name: str
    display_name: str
    color: str
    builder_kind: str
    regression: str


METHODS = [
    MethodSpec("segment_ols", "Segment Parabola OLS", "#1f77b4", "segment", "ols"),
    MethodSpec("segment_ridge", "Segment Parabola Ridge", "#ff7f0e", "segment", "ridge"),
    MethodSpec("gaussian_ridge", "Gaussian Local Ridge", "#2ca02c", "gaussian", "ridge"),
    MethodSpec("triangular_ridge", "Triangular Local Ridge", "#d62728", "triangular", "ridge"),
    MethodSpec("asym_gaussian_ridge", "Asymmetric Gaussian Ridge", "#9467bd", "asym_gaussian", "ridge"),
]

GAUSSIAN_SIGMA = {"L1+": 4.0, "L2+": 5.0, "L3+": 6.0, "L4+": 8.0, "L5+6": 10.0}
TRIANGULAR_WIDTH = {"L1+": 6.0, "L2+": 7.0, "L3+": 8.0, "L4+": 10.0, "L5+6": 12.0}
ASYM_PRE_SIGMA = {"L1+": 3.0, "L2+": 4.0, "L3+": 5.0, "L4+": 6.0, "L5+6": 7.0}
ASYM_POST_SIGMA = {"L1+": 6.0, "L2+": 7.0, "L3+": 8.0, "L4+": 10.0, "L5+6": 12.0}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark multiple reconstruction fit strategies.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--outer-piece", nargs="+", default=list(OUTER_PIECES))
    parser.add_argument("--train-floor", type=float, default=TRAIN_FLOOR)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--output-dir", default="MERIX SUBMISSION/rebulid/merge56_fit_benchmark_per_piece_zscore_predcount_seed44")
    return parser.parse_args()


def load_piece_per_piece_zscore_tempos(beat_time_dir: Path, piece_id: str) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, float]]:
    beat_time_path = beat_time_dir / f"{piece_id}beat_time.csv"
    df_bt, performer_cols = load_beat_time(beat_time_path)
    tempo_arrays = compute_tempo_curves(df_bt, performer_cols, smooth_window=3, bpm_range=(0, 5000), clip_max=600)
    mat = np.vstack([tempo_arrays[name] for name in tempo_arrays])
    piece_mean = float(np.nanmean(mat))
    piece_std = float(np.nanstd(mat))
    if not np.isfinite(piece_std) or piece_std < 1e-8:
        piece_std = 1.0
    normalized = {name: ((curve - piece_mean) / piece_std).astype(np.float32) for name, curve in tempo_arrays.items()}
    mean_curve = np.nanmean(np.vstack([normalized[name] for name in normalized]), axis=0)
    return mean_curve, normalized, {"raw_piece_mean_tempo": piece_mean, "raw_piece_std_tempo": piece_std, "num_performers": float(len(normalized))}


def build_target_frequencies(cfg: dict) -> tuple[dict[str, dict[str, pd.DataFrame]], dict[str, np.ndarray], dict[str, dict[str, np.ndarray]], dict[str, dict[str, float]]]:
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

    mean_tempo_by_piece = {}
    tempo_arrays_by_piece = {}
    tempo_stats_by_piece = {}
    for piece_id in pieces:
        mean_tempo, tempo_arrays, tempo_stats = load_piece_per_piece_zscore_tempos(beat_time_dir, piece_id)
        mean_tempo_by_piece[piece_id] = mean_tempo
        tempo_arrays_by_piece[piece_id] = tempo_arrays
        tempo_stats_by_piece[piece_id] = tempo_stats

    component_map = {}
    for component_name, raw_levels in COMPONENT_RAW_LEVELS.items():
        component_map[component_name] = build_piece_frequency_for_raw_levels(
            df.copy(),
            raw_levels=raw_levels,
            peak_cfg=peak_cfg,
            beat_unit_fallback=beat_unit_fallback,
        )

    base_piece = df[["piece_id", "beat_idx"]].drop_duplicates().sort_values(["piece_id", "beat_idx"]).reset_index(drop=True)
    target_freq_by_piece = {level: {} for level in LEVEL_SPECS}
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


def load_outer_scores(report_root: Path, detector_target: str, seed: int, piece_id: str) -> pd.DataFrame:
    path = report_root / f"weighted_topdown_merge56_{detector_target}_seed{seed}" / "outer_predictions.csv.gz"
    df = pd.read_csv(path)
    return df[df["piece_id"] == piece_id].copy().sort_values("beat_idx").reset_index(drop=True)


def load_outer_events(report_root: Path, detector_target: str, seed: int, piece_id: str) -> pd.DataFrame:
    path = report_root / f"weighted_topdown_merge56_{detector_target}_seed{seed}" / "predicted_events.csv.gz"
    df = pd.read_csv(path)
    df = df[df["piece_id"] == piece_id].copy()
    if "event_rank" in df.columns:
        return df.sort_values(["event_rank", "beat_idx"]).reset_index(drop=True)
    return df.sort_values("beat_idx").reset_index(drop=True)


def compute_training_average_counts(target_freq_by_piece: dict[str, dict[str, pd.DataFrame]], train_pieces: list[str], threshold: float) -> dict[str, int]:
    counts = {}
    for level in LEVEL_SPECS:
        vals = [int((target_freq_by_piece[level][pid]["frequency_target"] >= threshold).sum()) for pid in train_pieces]
        counts[level] = max(1, int(round(float(np.mean(vals))))) if vals else 1
    return counts


def select_topk_spaced_peaks(scores: np.ndarray, k: int, distance: int) -> tuple[np.ndarray, np.ndarray]:
    from scipy.signal import find_peaks
    if scores.size == 0 or k <= 0:
        return np.array([], dtype=int), np.array([], dtype=float)
    candidate_idx, _ = find_peaks(scores, distance=max(1, int(distance)))
    candidate_idx = candidate_idx[(candidate_idx > 0) & (candidate_idx < len(scores) - 1)]
    if candidate_idx.size == 0:
        candidate_idx = np.arange(1, len(scores) - 1, dtype=int)
    order = sorted(candidate_idx.tolist(), key=lambda idx: float(scores[idx]), reverse=True)
    selected = []
    for idx in order:
        if all(abs(idx - prev) >= int(distance) for prev in selected):
            selected.append(int(idx))
        if len(selected) >= int(k):
            break
    selected = sorted(set(selected))
    return np.asarray(selected, dtype=int), np.asarray([float(scores[idx]) for idx in selected], dtype=float)


def build_segment_basis(num_beats: int, boundaries: np.ndarray, strengths: np.ndarray | None = None) -> np.ndarray:
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
            s[idx] = orig_s[nearest] if orig_b.size else 1.0

    basis = np.zeros(num_beats, dtype=float)
    for idx in range(len(b) - 1):
        start, end = int(b[idx]), int(b[idx + 1])
        if end <= start:
            continue
        amp = 0.5 * (s[idx] + s[idx + 1])
        t = np.arange(start, end + 1)
        u = (t - start) / (end - start)
        basis[t] = amp * (-4.0 * u * (1.0 - u))
    return basis


def build_gaussian_basis(num_beats: int, boundaries: np.ndarray, strengths: np.ndarray | None, sigma: float) -> np.ndarray:
    x = np.arange(num_beats, dtype=float)
    basis = np.zeros(num_beats, dtype=float)
    b = np.asarray(boundaries, dtype=int)
    s = np.ones(len(b), dtype=float) if strengths is None or len(strengths) == 0 else np.asarray(strengths, dtype=float)
    for beat, amp in zip(b, s):
        basis += amp * np.exp(-0.5 * ((x - float(beat)) / float(sigma)) ** 2)
    return basis


def build_triangular_basis(num_beats: int, boundaries: np.ndarray, strengths: np.ndarray | None, width: float) -> np.ndarray:
    x = np.arange(num_beats, dtype=float)
    basis = np.zeros(num_beats, dtype=float)
    b = np.asarray(boundaries, dtype=int)
    s = np.ones(len(b), dtype=float) if strengths is None or len(strengths) == 0 else np.asarray(strengths, dtype=float)
    for beat, amp in zip(b, s):
        basis += amp * np.maximum(0.0, 1.0 - np.abs(x - float(beat)) / float(width))
    return basis


def build_asym_gaussian_basis(num_beats: int, boundaries: np.ndarray, strengths: np.ndarray | None, sigma_pre: float, sigma_post: float) -> tuple[np.ndarray, np.ndarray]:
    x = np.arange(num_beats, dtype=float)
    pre = np.zeros(num_beats, dtype=float)
    post = np.zeros(num_beats, dtype=float)
    b = np.asarray(boundaries, dtype=int)
    s = np.ones(len(b), dtype=float) if strengths is None or len(strengths) == 0 else np.asarray(strengths, dtype=float)
    for beat, amp in zip(b, s):
        dx = x - float(beat)
        pre += amp * np.where(dx <= 0.0, np.exp(-0.5 * (dx / float(sigma_pre)) ** 2), 0.0)
        post += amp * np.where(dx >= 0.0, np.exp(-0.5 * (dx / float(sigma_post)) ** 2), 0.0)
    return pre, post


def build_design_matrix(num_beats: int, level_sets: dict[str, np.ndarray], strengths_by_level: dict[str, np.ndarray] | None, method: MethodSpec) -> np.ndarray:
    cols = [np.ones(num_beats, dtype=float)]
    if method.builder_kind == "segment":
        for level in LEVEL_SPECS:
            strengths = None if strengths_by_level is None else strengths_by_level.get(level)
            cols.append(build_segment_basis(num_beats, level_sets[level], strengths))
    elif method.builder_kind == "gaussian":
        for level in LEVEL_SPECS:
            strengths = None if strengths_by_level is None else strengths_by_level.get(level)
            cols.append(build_gaussian_basis(num_beats, level_sets[level], strengths, GAUSSIAN_SIGMA[level]))
    elif method.builder_kind == "triangular":
        for level in LEVEL_SPECS:
            strengths = None if strengths_by_level is None else strengths_by_level.get(level)
            cols.append(build_triangular_basis(num_beats, level_sets[level], strengths, TRIANGULAR_WIDTH[level]))
    elif method.builder_kind == "asym_gaussian":
        for level in LEVEL_SPECS:
            strengths = None if strengths_by_level is None else strengths_by_level.get(level)
            pre, post = build_asym_gaussian_basis(num_beats, level_sets[level], strengths, ASYM_PRE_SIGMA[level], ASYM_POST_SIGMA[level])
            cols.extend([pre, post])
    else:
        raise ValueError(method.builder_kind)
    return np.stack(cols, axis=1)


def fit_method(method: MethodSpec, y_by_piece: dict[str, dict[str, np.ndarray]], target_freq_by_piece: dict[str, dict[str, pd.DataFrame]], train_pieces: list[str], threshold: float) -> tuple[np.ndarray, float | None]:
    x_list = []
    y_list = []
    for piece_id in train_pieces:
        level_sets = {level: target_freq_by_piece[level][piece_id].loc[target_freq_by_piece[level][piece_id]["frequency_target"] >= threshold, "beat_idx"].to_numpy(dtype=int) for level in LEVEL_SPECS}
        piece_curves = y_by_piece[piece_id]
        x_piece = build_design_matrix(len(next(iter(piece_curves.values()))), level_sets, None, method)
        for curve in piece_curves.values():
            x_list.append(x_piece)
            y_list.append(np.asarray(curve, dtype=float))
    X = np.concatenate(x_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    if method.regression == "ols":
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        return beta, None
    model = RidgeCV(alphas=RIDGE_ALPHAS, fit_intercept=False)
    model.fit(X, y)
    return model.coef_.astype(float), float(model.alpha_)


def apply_method(mean_curve: np.ndarray, level_sets: dict[str, np.ndarray], strengths_by_level: dict[str, np.ndarray] | None, method: MethodSpec, beta: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    X = build_design_matrix(len(mean_curve), level_sets, strengths_by_level, method)
    y_hat = X @ beta
    rmse = float(np.sqrt(np.mean((y_hat - mean_curve) ** 2)))
    corr = float(np.corrcoef(mean_curve, y_hat)[0, 1])
    return y_hat, {"rmse": rmse, "corr": corr}


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (BOUNDARY_RESTART_DIR / config_path).resolve()
    cfg = load_config(config_path)

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    target_freq_by_piece, mean_tempo_by_piece, tempo_arrays_by_piece, tempo_stats_by_piece = build_target_frequencies(cfg)
    outer_pieces = list(args.outer_piece)
    train_pieces = [p for p in sorted(mean_tempo_by_piece) if p not in outer_pieces]
    avg_counts = compute_training_average_counts(target_freq_by_piece, train_pieces, float(args.train_floor))
    peak_distance = int(cfg.get("data", {}).get("peak_distance", 6))
    report_root = REPO_ROOT / "MERIX SUBMISSION" / "Boundary_Restart" / "reports" / "clean_outer_test"

    trained = {}
    for method in METHODS:
        beta, alpha = fit_method(method, tempo_arrays_by_piece, target_freq_by_piece, train_pieces, float(args.train_floor))
        trained[method.name] = {"beta": beta, "alpha": alpha}

    summary_rows = []
    detail_rows = []
    pdf_overlay = output_dir / "merge56_fit_strategy_overlay_seed44.pdf"
    pdf_best = output_dir / "merge56_fit_strategy_best_pred_vs_all_seed44.pdf"
    with PdfPages(pdf_overlay) as overlay_pdf, PdfPages(pdf_best) as best_pdf:
        for piece_id in outer_pieces:
            mean_curve = mean_tempo_by_piece[piece_id]
            tempo_arrays = tempo_arrays_by_piece[piece_id]
            true_level_sets = {
                level: target_freq_by_piece[level][piece_id].loc[target_freq_by_piece[level][piece_id]["frequency_target"] >= float(args.train_floor), "beat_idx"].to_numpy(dtype=int)
                for level in LEVEL_SPECS
            }
            pred_level_sets = {}
            pred_strengths = {}
            for level, spec in LEVEL_SPECS.items():
                pred_events = load_outer_events(report_root, spec["target"], args.seed, piece_id)
                pred_level_sets[level] = pred_events["beat_idx"].to_numpy(dtype=int)
                pred_strengths[level] = pred_events["detector_score"].to_numpy(dtype=float)

                outer_scores = load_outer_scores(report_root, spec["target"], args.seed, piece_id)
                topk_beats, topk_scores = select_topk_spaced_peaks(
                    outer_scores["detector_score"].to_numpy(dtype=float),
                    k=avg_counts[level],
                    distance=peak_distance,
                )
                detail_rows.append(
                    {
                        "piece_id": piece_id,
                        "level": level,
                        "predicted_event_count": int(len(pred_level_sets[level])),
                        "train_avg_k": int(avg_counts[level]),
                        "topk_selected_count": int(len(topk_beats)),
                    }
                )

            piece_results = []
            for method in METHODS:
                true_recon, true_metrics = apply_method(mean_curve, true_level_sets, None, method, trained[method.name]["beta"])
                pred_recon, pred_metrics = apply_method(mean_curve, pred_level_sets, pred_strengths, method, trained[method.name]["beta"])
                piece_results.append((method, pred_recon, pred_metrics))
                summary_rows.append(
                    {
                        "method": method.name,
                        "display_name": method.display_name,
                        "piece_id": piece_id,
                        "true_rmse": true_metrics["rmse"],
                        "true_corr": true_metrics["corr"],
                        "pred_rmse": pred_metrics["rmse"],
                        "pred_corr": pred_metrics["corr"],
                        "ridge_alpha": trained[method.name]["alpha"],
                    }
                )

            fig, ax = plt.subplots(figsize=(15, 6))
            x = np.arange(len(mean_curve))
            ax.plot(x, mean_curve, color="black", linewidth=2.2, label="True mean z-score tempo")
            for method, pred_recon, pred_metrics in piece_results:
                ax.plot(
                    x,
                    pred_recon,
                    color=method.color,
                    linewidth=1.5,
                    label=f"{method.display_name} (corr={pred_metrics['corr']:.3f})",
                )
            ax.set_title(f"{piece_id}: fit-strategy comparison")
            ax.set_xlabel("Beat index")
            ax.set_ylabel("Per-piece z-score tempo")
            ax.grid(alpha=0.25)
            ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper right")
            fig.tight_layout()
            fig.savefig(output_dir / f"{piece_id}_fit_strategy_overlay.png", dpi=180, bbox_inches="tight")
            overlay_pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

            best_method, best_curve, best_metrics = max(piece_results, key=lambda item: item[2]["corr"])
            fig_best, ax_best = plt.subplots(figsize=(15, 5.2))
            for curve in tempo_arrays.values():
                ax_best.plot(x, curve, color="0.7", linewidth=0.8, alpha=0.25)
            ax_best.plot(x, best_curve, color=best_method.color, linewidth=1.8, label=f"{best_method.display_name} (corr={best_metrics['corr']:.3f})")
            ax_best.set_title(f"{piece_id}: best fitted curve vs all true performer curves")
            ax_best.set_xlabel("Beat index")
            ax_best.set_ylabel("Per-piece z-score tempo")
            ax_best.grid(alpha=0.22)
            ax_best.legend(frameon=False, loc="upper right")
            fig_best.tight_layout()
            fig_best.savefig(output_dir / f"{piece_id}_best_fit_vs_all_true.png", dpi=180, bbox_inches="tight")
            best_pdf.savefig(fig_best, bbox_inches="tight")
            plt.close(fig_best)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_dir / "fit_strategy_summary.csv", index=False)
    mean_df = (
        summary_df.groupby(["method", "display_name"], as_index=False)[["pred_rmse", "pred_corr"]]
        .mean()
        .sort_values("pred_corr", ascending=False)
    )
    mean_df.to_csv(output_dir / "fit_strategy_mean_summary.csv", index=False)
    pd.DataFrame(detail_rows).to_csv(output_dir / "selection_count_detail.csv", index=False)
    metadata = {
        "normalization": "per-piece z-score tempo",
        "selection_mode": "predicted events for fitting benchmark; train-average top-K counts recorded only for reference",
        "methods": [method.__dict__ for method in METHODS],
        "ridge_alphas": RIDGE_ALPHAS.tolist(),
        "avg_train_counts": avg_counts,
        "outer_pieces": outer_pieces,
        "seed": int(args.seed),
    }
    (output_dir / "fit_strategy_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(str(output_dir))


if __name__ == "__main__":
    main()
