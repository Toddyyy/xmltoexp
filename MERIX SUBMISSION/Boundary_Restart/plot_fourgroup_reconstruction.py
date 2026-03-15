from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT / "MERIX SUBMISSION" / "MIREX_Model") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "MERIX SUBMISSION" / "MIREX_Model"))
if str(REPO_ROOT / "MERIX SUBMISSION" / "Boundary_Restart") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "MERIX SUBMISSION" / "Boundary_Restart"))

from build_mazurka_beat_npz_performer_levels import compute_tempo_curves, load_beat_time  # noqa: E402
from boundary_restart.config import load_config, resolve_path  # noqa: E402
from boundary_restart.features import PeakConfig, boundary_probs_to_binary, load_boundary_npz, replace_level_suffix  # noqa: E402


DEFAULT_CONFIG = "configs/salience_grouped3_hi8_score_only_xml_curated.yaml"
DEFAULT_PIECES = ("M06-2", "M17-1", "M30-1")
THRESHOLD = 0.3
GROUPS = {
    "L1": (1,),
    "L2": (2,),
    "L34": (3, 4),
    "L56": (5, 6),
}
GROUP_COLORS = {
    "L1": "#4C78A8",
    "L2": "#72B7B2",
    "L34": "#F58518",
    "L56": "#E45756",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot four-group reconstruction vs formal tempo curve.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--piece_id", nargs="+", default=list(DEFAULT_PIECES))
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
        for i, bi in enumerate(b):
            if orig_b.size == 0:
                s[i] = 1.0
            else:
                j = int(np.argmin(np.abs(orig_b - bi)))
                s[i] = orig_s[j]

    basis = np.zeros(num_beats, dtype=float)
    for i in range(len(b) - 1):
        start = int(b[i])
        end = int(b[i + 1])
        if end <= start:
            continue
        amp = 0.5 * (s[i] + s[i + 1])
        t = np.arange(start, end + 1)
        u = (t - start) / (end - start)
        basis[t] = amp * (-4.0 * u * (1.0 - u))
    return basis


def build_design_matrix(num_beats: int, group_sets: dict[str, np.ndarray], strengths_by_group: dict[str, np.ndarray] | None = None) -> np.ndarray:
    cols = [np.ones(num_beats, dtype=float)]
    for group in GROUPS:
        strengths = None if strengths_by_group is None else strengths_by_group.get(group)
        cols.append(build_level_basis(num_beats, group_sets[group], strengths=strengths))
    return np.stack(cols, axis=1)


def apply_params(mean_tempo: np.ndarray, group_sets: dict[str, np.ndarray], beta: np.ndarray, strengths_by_group: dict[str, np.ndarray] | None = None) -> tuple[np.ndarray, dict[str, float]]:
    X = build_design_matrix(len(mean_tempo), group_sets, strengths_by_group)
    y_hat = X @ beta
    rmse = float(np.sqrt(np.mean((y_hat - mean_tempo) ** 2)))
    corr = float(np.corrcoef(mean_tempo, y_hat)[0, 1])
    return y_hat, {"rmse": rmse, "corr": corr}


def aggregate_piece_group_frequencies(
    piece_df: pd.DataFrame,
    peak_cfg: PeakConfig,
    beat_unit_fallback: float,
) -> pd.DataFrame:
    num_beats = int(piece_df["num_beats"].iloc[0])
    counts = {group: np.zeros(num_beats, dtype=np.float32) for group in GROUPS}
    performer_count = 0

    for source_path, group in piece_df.groupby("source_path", sort=False):
        performer_count += 1
        beat_idx = group["beat_idx"].to_numpy(dtype=np.int32)
        loaded_level_binary: dict[int, np.ndarray] = {}
        for raw_level in {1, 2, 3, 4, 5, 6}:
            level_path = replace_level_suffix(Path(str(source_path)), level=raw_level)
            loaded = load_boundary_npz(level_path, beat_unit_fallback=beat_unit_fallback)
            loaded_level_binary[raw_level] = boundary_probs_to_binary(
                np.asarray(loaded["boundary_probs"], dtype=np.float32),
                peak_cfg,
            ).astype(np.float32)
        for group_name, raw_levels in GROUPS.items():
            binary = None
            for raw_level in raw_levels:
                arr = loaded_level_binary[raw_level]
                binary = arr if binary is None else np.maximum(binary, arr)
            counts[group_name][beat_idx] += binary[beat_idx]

    out = pd.DataFrame({"beat_idx": np.arange(num_beats, dtype=np.int32)})
    for group_name in GROUPS:
        out[group_name] = counts[group_name] / max(performer_count, 1)
    out["performer_count"] = performer_count
    return out


def fit_beta(
    mean_tempo_by_piece: dict[str, np.ndarray],
    group_freq_by_piece: dict[str, pd.DataFrame],
    heldout_piece: str,
) -> np.ndarray:
    X_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []
    for piece_id, y in mean_tempo_by_piece.items():
        if piece_id == heldout_piece:
            continue
        agg = group_freq_by_piece[piece_id]
        if len(agg) != len(y):
            n = min(len(agg), len(y))
            agg = agg.iloc[:n].copy()
            y = y[:n]
        group_sets = {
            group: agg.loc[agg[group] >= THRESHOLD, "beat_idx"].to_numpy(dtype=int)
            for group in GROUPS
        }
        X_list.append(build_design_matrix(len(y), group_sets))
        y_list.append(y)
    X_all = np.concatenate(X_list, axis=0)
    y_all = np.concatenate(y_list, axis=0)
    beta, *_ = np.linalg.lstsq(X_all, y_all, rcond=None)
    return beta


def load_predicted_events(base_dir: Path) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    group_sets: dict[str, np.ndarray] = {}
    strengths: dict[str, np.ndarray] = {}
    target_name = {
        "L1": "level1_boundary",
        "L2": "level2_boundary",
        "L34": "level34_boundary",
        "L56": "level56_boundary",
    }
    for group_name, detector_target in target_name.items():
        pred_path = base_dir / f"tcn_{detector_target}_union_recall_cpu" / "predicted_events.csv.gz"
        pred_df = pd.read_csv(pred_path)
        group_sets[group_name] = pred_df["beat_idx"].to_numpy(dtype=int)
        strengths[group_name] = pred_df["detector_score"].to_numpy(dtype=float)
    return group_sets, strengths


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (SCRIPT_DIR / config_path).resolve()
    cfg = load_config(config_path)
    data_cfg = cfg.get("data", {})
    beat_time_dir = REPO_ROOT / "MazurkaBL-master" / "beat_time"
    table_path = resolve_path(cfg, data_cfg["beat_table_path"])
    peak_cfg = PeakConfig(
        distance=int(data_cfg.get("peak_distance", 6)),
        height=float(data_cfg.get("peak_height", 0.15)),
        prominence=float(data_cfg.get("peak_prominence", 0.05)),
    )
    beat_unit_fallback = float(data_cfg.get("beat_unit_fallback", 1.0))

    df = pd.read_csv(table_path, usecols=["piece_id", "source_path", "beat_idx", "num_beats"]).drop_duplicates()
    pieces = sorted(df["piece_id"].unique())

    mean_tempo_by_piece: dict[str, np.ndarray] = {}
    tempo_arrays_by_piece: dict[str, dict[str, np.ndarray]] = {}
    group_freq_by_piece: dict[str, pd.DataFrame] = {}
    for piece_id in pieces:
        mean_tempo, tempo_arrays = load_piece_mean_tempo(beat_time_dir, piece_id)
        mean_tempo_by_piece[piece_id] = mean_tempo
        tempo_arrays_by_piece[piece_id] = tempo_arrays
        group_freq_by_piece[piece_id] = aggregate_piece_group_frequencies(
            df[df["piece_id"] == piece_id].copy(),
            peak_cfg=peak_cfg,
            beat_unit_fallback=beat_unit_fallback,
        )

    for piece_id in args.piece_id:
        beta = fit_beta(mean_tempo_by_piece, group_freq_by_piece, heldout_piece=piece_id)
        mean_tempo = mean_tempo_by_piece[piece_id]
        tempo_arrays = tempo_arrays_by_piece[piece_id]
        true_freq = group_freq_by_piece[piece_id]

        true_group_sets = {
            group: true_freq.loc[true_freq[group] >= THRESHOLD, "beat_idx"].to_numpy(dtype=int)
            for group in GROUPS
        }
        true_recon, true_metrics = apply_params(mean_tempo, true_group_sets, beta)

        pred_base_dir = REPO_ROOT / "MERIX SUBMISSION" / "Boundary_Restart" / "outputs" / "local_runs" / f"{piece_id}_4groups"
        pred_group_sets, pred_strengths = load_predicted_events(pred_base_dir)
        pred_recon, pred_metrics = apply_params(mean_tempo, pred_group_sets, beta, strengths_by_group=pred_strengths)

        x = np.arange(len(mean_tempo))
        fig = plt.figure(figsize=(15, 11))
        gs = fig.add_gridspec(5, 1, height_ratios=[2.3, 1, 1, 1, 1], hspace=0.18)

        ax0 = fig.add_subplot(gs[0])
        for curve in tempo_arrays.values():
            ax0.plot(x, curve, color="0.8", linewidth=0.7, alpha=0.25)
        ax0.plot(x, mean_tempo, color="black", linewidth=2.0, label="Formal tempo curve (mean)")
        ax0.plot(
            x,
            true_recon,
            color="#1f77b4",
            linewidth=1.8,
            linestyle="--",
            label=f"Reconstruction from true L1/L2/L3+4/L5+6 union (rmse={true_metrics['rmse']:.2f}, corr={true_metrics['corr']:.3f})",
        )
        ax0.plot(
            x,
            pred_recon,
            color="#d62728",
            linewidth=1.8,
            label=f"Reconstruction from TCN direct four groups (rmse={pred_metrics['rmse']:.2f}, corr={pred_metrics['corr']:.3f})",
        )
        ax0.set_ylabel("Tempo (BPM)")
        ax0.set_title(f"{piece_id}: four-group boundary reconstruction vs formal tempo curve")
        ax0.grid(alpha=0.25)
        ax0.legend(frameon=False, fontsize=9, loc="upper right")

        for row_idx, group_name in enumerate(GROUPS, start=1):
            ax = fig.add_subplot(gs[row_idx], sharex=ax0)
            color = GROUP_COLORS[group_name]
            ax.plot(
                true_freq["beat_idx"],
                true_freq[group_name],
                color=color,
                linewidth=1.3,
                label=f"True {group_name} union frequency",
            )
            pred_beats = pred_group_sets[group_name]
            pred_scores = pred_strengths[group_name]
            ax.scatter(
                pred_beats,
                pred_scores,
                color=color,
                edgecolors="black",
                linewidths=0.3,
                s=28,
                alpha=0.9,
                label=f"Predicted {group_name} event score",
                zorder=3,
            )
            ax.axhline(THRESHOLD, color="0.6", linestyle="--", linewidth=0.8)
            ax.set_ylim(-0.02, 1.05)
            ax.set_ylabel(group_name)
            ax.grid(alpha=0.2)
            ax.legend(frameon=False, fontsize=8, loc="upper right")

        fig.axes[-1].set_xlabel("Beat index")
        fig.tight_layout()

        stem = f"{piece_id}_fourgroup_reconstruction_vs_tempo"
        png_path = pred_base_dir / f"{stem}.png"
        pdf_path = pred_base_dir / f"{stem}.pdf"
        fig.savefig(png_path, dpi=180, bbox_inches="tight")
        fig.savefig(pdf_path, bbox_inches="tight")
        plt.close(fig)

        metrics = {
            "piece_id": piece_id,
            "threshold": THRESHOLD,
            "beta": beta.tolist(),
            "true_reconstruction": true_metrics,
            "pred_reconstruction": pred_metrics,
            "pred_event_counts": {group: int(len(pred_group_sets[group])) for group in GROUPS},
            "true_union_counts_over_threshold": {
                group: int(np.sum(true_freq[group].to_numpy() >= THRESHOLD)) for group in GROUPS
            },
        }
        json_path = pred_base_dir / f"{stem}.json"
        json_path.write_text(json.dumps(metrics, indent=2))
        print(f"Saved figure to {png_path}")
        print(f"Saved figure to {pdf_path}")
        print(f"Saved metrics to {json_path}")
        print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
