#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

import plot_clean_outer_merge56_reconstruction as recon_plot


SCRIPT_DIR = Path(__file__).resolve().parent
SOURCE_DIR = SCRIPT_DIR / "clean_outer_reconstruction_merge56_seed44_per_performer_zscore_predcount"
OUTPUT_DIR = SCRIPT_DIR / "merge56_reconstruction_controls_seed44"


def build_barline_map(cfg: dict, pieces: list[str]) -> dict[str, np.ndarray]:
    table_path = recon_plot.resolve_path(cfg, cfg["data"]["beat_table_path"])
    df = pd.read_csv(table_path, usecols=["piece_id", "beat_idx", "xml_measure_start"])
    df = df[df["piece_id"].isin(pieces)].copy()
    rows = []
    for piece_id, group in df.groupby(["piece_id", "beat_idx"], sort=False):
        rows.append(
            {
                "piece_id": piece_id[0],
                "beat_idx": int(piece_id[1]),
                "xml_measure_start": float(group["xml_measure_start"].iloc[0]),
            }
        )
    beat_df = pd.DataFrame(rows)
    out: dict[str, np.ndarray] = {}
    for piece_id, group in beat_df.groupby("piece_id", sort=False):
        beats = group.loc[group["xml_measure_start"] > 0.5, "beat_idx"].to_numpy(dtype=int)
        out[str(piece_id)] = np.unique(beats.astype(int))
    return out


def fit_barline_beta(
    tempo_arrays_by_piece: dict[str, dict[str, np.ndarray]],
    barline_map: dict[str, np.ndarray],
    train_pieces: list[str],
) -> np.ndarray:
    x_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []
    for piece_id in train_pieces:
        beats = barline_map[piece_id]
        basis = recon_plot.build_level_basis(
            num_beats=len(next(iter(tempo_arrays_by_piece[piece_id].values()))),
            boundaries=beats,
            strengths=None,
        )
        x_piece = np.stack([np.ones_like(basis, dtype=float), basis.astype(float)], axis=1)
        for curve in tempo_arrays_by_piece[piece_id].values():
            x_list.append(x_piece)
            y_list.append(np.asarray(curve, dtype=float))
    x_all = np.concatenate(x_list, axis=0)
    y_all = np.concatenate(y_list, axis=0)
    beta, *_ = np.linalg.lstsq(x_all, y_all, rcond=None)
    return beta


def apply_barline_beta(mean_curve: np.ndarray, beats: np.ndarray, beta: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    basis = recon_plot.build_level_basis(len(mean_curve), beats, strengths=None)
    x = np.stack([np.ones_like(basis, dtype=float), basis.astype(float)], axis=1)
    y_hat = x @ beta
    rmse = float(np.sqrt(np.mean((y_hat - mean_curve) ** 2)))
    corr = float(np.corrcoef(mean_curve, y_hat)[0, 1])
    return y_hat, {"rmse": rmse, "corr": corr}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    metadata = json.loads((SOURCE_DIR / "reconstruction_metadata.json").read_text(encoding="utf-8"))
    beta = np.asarray(metadata["beta"], dtype=float)
    outer_pieces = list(metadata["outer_pieces"])
    cfg = recon_plot.load_config(recon_plot.BOUNDARY_RESTART_DIR / recon_plot.DEFAULT_CONFIG)

    target_freq_by_piece, mean_tempo_by_piece, tempo_arrays_by_piece, tempo_stats_by_piece = recon_plot.build_target_frequencies(
        cfg,
        outer_pieces,
        "per_performer_zscore",
    )
    train_pieces = sorted(piece_id for piece_id in mean_tempo_by_piece if piece_id not in outer_pieces)
    all_pieces = sorted(mean_tempo_by_piece.keys())
    barline_map = build_barline_map(cfg, all_pieces)
    barline_beta = fit_barline_beta(tempo_arrays_by_piece, barline_map, train_pieces)

    report_root = recon_plot.REPO_ROOT / "MERIX SUBMISSION" / "Boundary_Restart" / "reports" / "clean_outer_test"
    summary_rows: list[dict[str, object]] = []
    with PdfPages(OUTPUT_DIR / "reconstruction_controls_overlay.pdf") as pdf:
        for piece_id in outer_pieces:
            mean_curve = mean_tempo_by_piece[piece_id]
            tempo_arrays = tempo_arrays_by_piece[piece_id]

            true_level_sets = {}
            pred_level_sets = {}
            pred_strengths = {}
            equal_strengths = {}
            for level, spec in recon_plot.LEVEL_SPECS.items():
                truth = target_freq_by_piece[level][piece_id]
                true_level_sets[level] = truth.loc[
                    truth["frequency_target"] >= float(metadata["train_floor"]),
                    "beat_idx",
                ].to_numpy(dtype=int)
                pred_events = recon_plot.load_outer_events(report_root, spec["target"], int(metadata["seed"]), piece_id)
                pred_level_sets[level] = pred_events["beat_idx"].to_numpy(dtype=int)
                pred_strengths[level] = pred_events["detector_score"].to_numpy(dtype=float)
                equal_strengths[level] = np.ones(len(pred_level_sets[level]), dtype=float)

            true_recon, true_metrics = recon_plot.apply_params(mean_curve, true_level_sets, beta)
            pred_recon, pred_metrics = recon_plot.apply_params(mean_curve, pred_level_sets, beta, strengths_by_level=pred_strengths)
            equal_recon, equal_metrics = recon_plot.apply_params(mean_curve, pred_level_sets, beta, strengths_by_level=equal_strengths)
            barline_recon, barline_metrics = apply_barline_beta(mean_curve, barline_map[piece_id], barline_beta)

            x = np.arange(len(mean_curve))
            fig, ax = plt.subplots(figsize=(15, 5.5))
            for curve in tempo_arrays.values():
                ax.plot(x, curve, color="0.82", linewidth=0.7, alpha=0.2)
            ax.plot(x, mean_curve, color="black", linewidth=2.0, label="Mean performer tempo curve")
            ax.plot(x, true_recon, color="#1f77b4", linewidth=1.8, linestyle="--", label=f"True cumulative upper bound (corr={true_metrics['corr']:.3f})")
            ax.plot(x, pred_recon, color="#d62728", linewidth=1.8, label=f"Predicted events + detector strength (corr={pred_metrics['corr']:.3f})")
            ax.plot(x, equal_recon, color="#2ca02c", linewidth=1.6, label=f"Predicted events + equal strength (corr={equal_metrics['corr']:.3f})")
            ax.plot(x, barline_recon, color="#9467bd", linewidth=1.6, label=f"Barline-only control (corr={barline_metrics['corr']:.3f})")
            ax.set_title(f"{piece_id}: tempo reconstruction controls")
            ax.set_xlabel("Beat index")
            ax.set_ylabel("Tempo z-score")
            ax.grid(alpha=0.22)
            ax.legend(frameon=False, loc="upper right")
            fig.tight_layout()
            fig.savefig(OUTPUT_DIR / f"{piece_id}_reconstruction_controls.png", dpi=180, bbox_inches="tight")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

            summary_rows.extend(
                [
                    {"piece_id": piece_id, "control": "true_cumulative_upper_bound", **true_metrics},
                    {"piece_id": piece_id, "control": "predicted_detector_strength", **pred_metrics},
                    {"piece_id": piece_id, "control": "predicted_equal_strength", **equal_metrics},
                    {"piece_id": piece_id, "control": "barline_only", **barline_metrics},
                ]
            )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUTPUT_DIR / "reconstruction_control_summary.csv", index=False)
    mean_df = summary_df.groupby("control", sort=False)[["rmse", "corr"]].mean().reset_index()
    mean_df.to_csv(OUTPUT_DIR / "reconstruction_control_mean.csv", index=False)
    (OUTPUT_DIR / "reconstruction_control_metadata.json").write_text(
        json.dumps(
            {
                "source_metadata": metadata,
                "barline_beta": barline_beta.tolist(),
                "output_dir": str(OUTPUT_DIR),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(str(OUTPUT_DIR))


if __name__ == "__main__":
    main()
