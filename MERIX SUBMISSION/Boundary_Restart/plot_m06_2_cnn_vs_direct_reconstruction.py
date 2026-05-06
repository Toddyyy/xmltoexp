#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from boundary_restart.config import load_config, resolve_path
from boundary_restart.features import PeakConfig
from boundary_restart.models import build_sequence_model
from boundary_restart.table_io import load_table
from plot_clean_outer_merge56_reconstruction import (
    COMPONENT_WEIGHTS,
    LEVEL_SPECS,
    MERGE_TOLERANCE,
    build_target_frequencies,
    apply_params,
    fit_beta,
)
from train_piece_union_protocol import (
    PieceUnionDataset,
    build_piece_union_frame,
    build_predicted_event_frame,
    collate_piece_union,
    predict_detector,
    resolve_device,
)


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "salience_grouped3_hi8_score_only_xml_curated.yaml"
LEVEL_TARGETS = {label: spec["target"] for label, spec in LEVEL_SPECS.items()}
TARGET_TO_LABEL = {target: label for label, target in LEVEL_TARGETS.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reconstruct M06-2 tempo curve using seed42 CNN and direct baselines "
            "(all_boundary/periodic/downbeat), with 0-100 beat display."
        )
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--piece-id", default="M06-2")
    parser.add_argument("--normalization", choices=["robust_relative", "per_performer_zscore"], default="per_performer_zscore")
    parser.add_argument("--train-floor", type=float, default=0.05)
    parser.add_argument("--beat-start", type=int, default=0)
    parser.add_argument("--beat-end", type=int, default=100)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "rebulid" / "cnn_seed42_m06_2_0_100_reconstruction"),
    )
    return parser.parse_args()


def _safe_level_events(events: pd.DataFrame, piece_id: str, num_beats: int) -> tuple[np.ndarray, np.ndarray]:
    if events.empty or "piece_id" not in events.columns:
        return np.asarray([0, num_beats - 1], dtype=int), np.asarray([0.0, 0.0], dtype=float)
    piece_events = events[events["piece_id"] == piece_id].copy()
    if piece_events.empty:
        return np.asarray([0, num_beats - 1], dtype=int), np.asarray([0.0, 0.0], dtype=float)
    piece_events = piece_events.sort_values("beat_idx")
    piece_events = piece_events[(piece_events["beat_idx"] >= 0) & (piece_events["beat_idx"] < num_beats)]
    if piece_events.empty:
        return np.asarray([0, num_beats - 1], dtype=int), np.asarray([0.0, 0.0], dtype=float)
    grouped = (
        piece_events.groupby("beat_idx", as_index=False)["detector_score"]
        .max()
        .sort_values("beat_idx")
    )
    return grouped["beat_idx"].to_numpy(dtype=int), grouped["detector_score"].to_numpy(dtype=float)


def _build_piece_sample(piece_df: pd.DataFrame, feature_cols: list[str]) -> dict:
    piece_df = piece_df.sort_values("beat_idx").reset_index(drop=True)
    return {
        "sample_id": piece_df["piece_sample_id"].iloc[0],
        "piece_id": piece_df["piece_id"].iloc[0],
        "beat_idx": piece_df["beat_idx"].to_numpy(dtype=np.int32),
        "features": piece_df[feature_cols].to_numpy(dtype=np.float32),
        "union_target": piece_df["union_target"].to_numpy(dtype=np.float32),
        "frequency_target": piece_df["frequency_target"].to_numpy(dtype=np.float32),
        "train_union_target": piece_df["union_target"].to_numpy(dtype=np.float32),
        "train_frequency_target": piece_df["frequency_target"].to_numpy(dtype=np.float32),
        "performer_count": piece_df["performer_count"].to_numpy(dtype=np.int32),
        "train_loss_factor": np.ones(len(piece_df), dtype=np.float32),
    }


def _load_cnn_runtime(cfg: dict, report_dir: Path, target_mode: str, device: torch.device) -> dict:
    summary_path = report_dir / target_mode / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    outer_summary = summary["outer_test_summary"]
    candidate_slug = str(summary["best_candidate"]["candidate_slug"])
    ckpt_path = report_dir / target_mode / "outer_test" / candidate_slug / "detector_best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing CNN checkpoint: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = build_sequence_model(
        str(ckpt["model_type"]),
        input_dim=len(ckpt["feature_columns"]),
        cfg=cfg,
        output_dim=1,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return {
        "model": model,
        "feature_columns": list(ckpt["feature_columns"]),
        "mean": np.asarray(ckpt["mean"], dtype=np.float32),
        "std": np.asarray(ckpt["std"], dtype=np.float32),
        "threshold": float(outer_summary["union_metrics"]["threshold"]),
    }


def _infer_cnn_events_for_piece(
    cfg: dict,
    beat_table: pd.DataFrame,
    runtime: dict,
    target_mode: str,
    piece_id: str,
    eval_cfg: dict,
    data_cfg: dict,
    device: torch.device,
) -> pd.DataFrame:
    peak_cfg = PeakConfig(
        distance=int(data_cfg.get("peak_distance", 6)),
        height=float(data_cfg.get("peak_height", 0.15)),
        prominence=float(data_cfg.get("peak_prominence", 0.05)),
    )
    feature_cols = list(runtime["feature_columns"])
    local_df = beat_table.copy()
    if "protocol_split" not in local_df.columns:
        local_df["protocol_split"] = "val"
    if "num_beats" not in local_df.columns:
        local_df["num_beats"] = (
            local_df.groupby("piece_id")["beat_idx"].transform("max").astype(int) + 1
        )
    for col in feature_cols:
        if col not in local_df.columns:
            local_df[col] = 0.0

    piece_level = build_piece_union_frame(
        local_df,
        feature_cols=feature_cols,
        target_mode=target_mode,
        peak_cfg=peak_cfg,
        beat_unit_fallback=float(data_cfg.get("beat_unit_fallback", 1.0)),
        cumulative_merge_tolerance=int(MERGE_TOLERANCE),
        cumulative_component_weights=dict(COMPONENT_WEIGHTS),
    )
    piece_level = piece_level[piece_level["piece_id"] == piece_id].copy()
    if piece_level.empty:
        raise ValueError(f"No piece rows found for {piece_id} under target {target_mode}")
    piece_level["protocol_split"] = "val"

    sample = _build_piece_sample(piece_level, feature_cols)
    dataset = PieceUnionDataset([sample], mean=runtime["mean"], std=runtime["std"])
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_piece_union)
    pred_df = predict_detector(runtime["model"], loader, device=device)
    events = build_predicted_event_frame(
        pred_df=pred_df,
        threshold=float(runtime["threshold"]),
        min_distance=int(eval_cfg.get("min_distance", 6)),
        prominence=float(eval_cfg.get("prominence", 0.0)),
        tolerance=int(eval_cfg.get("event_tolerance", 1)),
    )
    return events


def _zero_level(num_beats: int) -> tuple[np.ndarray, np.ndarray]:
    return np.asarray([0, num_beats - 1], dtype=int), np.asarray([0.0, 0.0], dtype=float)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    data_cfg = cfg.get("data", {})
    eval_cfg = cfg.get("evaluation", {})
    piece_id = str(args.piece_id)
    seed = int(args.seed)
    device = resolve_device(args.device)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    target_freq_by_piece, mean_tempo_by_piece, tempo_arrays_by_piece, _ = build_target_frequencies(
        cfg,
        [piece_id],
        str(args.normalization),
    )
    if piece_id not in mean_tempo_by_piece:
        raise ValueError(f"Piece {piece_id} is not present in target frequencies.")
    all_pieces = sorted(mean_tempo_by_piece.keys())
    train_pieces = [pid for pid in all_pieces if pid != piece_id]
    beta = fit_beta(tempo_arrays_by_piece, target_freq_by_piece, train_pieces, threshold=float(args.train_floor))
    mean_tempo = mean_tempo_by_piece[piece_id]
    num_beats = len(mean_tempo)

    beat_table_path = resolve_path(cfg, data_cfg["beat_table_path"])
    beat_table = load_table(beat_table_path)

    cnn_report_dir = ROOT / "reports" / f"paper_outer_baselines_weighted_topdown_all_seed{seed}_cnn1d"
    cnn_level_sets: dict[str, np.ndarray] = {}
    cnn_level_strengths: dict[str, np.ndarray] = {}
    cnn_event_rows: list[dict[str, object]] = []
    for level_label, target_mode in LEVEL_TARGETS.items():
        runtime = _load_cnn_runtime(cfg, cnn_report_dir, target_mode=target_mode, device=device)
        events = _infer_cnn_events_for_piece(
            cfg=cfg,
            beat_table=beat_table,
            runtime=runtime,
            target_mode=target_mode,
            piece_id=piece_id,
            eval_cfg=eval_cfg,
            data_cfg=data_cfg,
            device=device,
        )
        beats, scores = _safe_level_events(events, piece_id=piece_id, num_beats=num_beats)
        cnn_level_sets[level_label] = beats
        cnn_level_strengths[level_label] = scores
        cnn_event_rows.append(
            {
                "method": "cnn_seed42_all_levels",
                "level": level_label,
                "event_count": int(len(beats)),
                "beats": beats.tolist(),
            }
        )

    direct_root = ROOT / "reports" / f"paper_outer_missing_baselines_seed{seed}"
    direct_methods = {
        "all_boundary_direct": direct_root / "all_boundary",
        "periodic_k_direct": direct_root / "periodic",
        "downbeat_only_direct": direct_root / "downbeat",
    }
    direct_level_sets: dict[str, dict[str, np.ndarray]] = {}
    direct_level_strengths: dict[str, dict[str, np.ndarray]] = {}
    for method_name, method_dir in direct_methods.items():
        level_sets: dict[str, np.ndarray] = {}
        level_strengths: dict[str, np.ndarray] = {}
        for level_label, target_mode in LEVEL_TARGETS.items():
            events_path = method_dir / target_mode / "predicted_events.csv.gz"
            events = pd.read_csv(events_path)
            beats, scores = _safe_level_events(events, piece_id=piece_id, num_beats=num_beats)
            level_sets[level_label] = beats
            level_strengths[level_label] = scores
            cnn_event_rows.append(
                {
                    "method": method_name,
                    "level": level_label,
                    "event_count": int(len(beats)),
                    "beats": beats.tolist(),
                }
            )
        direct_level_sets[method_name] = level_sets
        direct_level_strengths[method_name] = level_strengths

    cnn_l345_sets = dict(cnn_level_sets)
    cnn_l345_strengths = dict(cnn_level_strengths)
    for low_level in ("L1+", "L2+"):
        zb, zs = _zero_level(num_beats)
        cnn_l345_sets[low_level] = zb
        cnn_l345_strengths[low_level] = zs

    recon_variants: dict[str, tuple[dict[str, np.ndarray], dict[str, np.ndarray]]] = {
        "CNN (all levels)": (cnn_level_sets, cnn_level_strengths),
        "CNN (L3/L4/L5 only)": (cnn_l345_sets, cnn_l345_strengths),
        "all_boundary_direct": (
            direct_level_sets["all_boundary_direct"],
            direct_level_strengths["all_boundary_direct"],
        ),
        "periodic_k_direct": (
            direct_level_sets["periodic_k_direct"],
            direct_level_strengths["periodic_k_direct"],
        ),
        "downbeat_only_direct": (
            direct_level_sets["downbeat_only_direct"],
            direct_level_strengths["downbeat_only_direct"],
        ),
    }

    metrics_rows: list[dict[str, object]] = []
    recon_curves: dict[str, np.ndarray] = {}
    for label, (level_sets, level_strengths) in recon_variants.items():
        recon, m = apply_params(mean_tempo, level_sets, beta, strengths_by_level=level_strengths)
        rmse = float(m["rmse"])
        corr = float(m["corr"])
        if not np.isfinite(rmse):
            rmse = 0.0
        if not np.isfinite(corr):
            corr = 0.0
        recon_curves[label] = recon
        metrics_rows.append(
            {
                "method": label,
                "rmse": rmse,
                "corr": corr,
            }
        )

    x = np.arange(num_beats, dtype=int)
    x_start = max(0, int(args.beat_start))
    x_end = min(num_beats - 1, int(args.beat_end))
    if x_end <= x_start:
        raise ValueError("beat-end must be greater than beat-start.")

    colors = {
        "CNN (all levels)": "#d62728",
        "CNN (L3/L4/L5 only)": "#ff7f0e",
        "all_boundary_direct": "#1f77b4",
        "periodic_k_direct": "#2ca02c",
        "downbeat_only_direct": "#9467bd",
    }

    fig, ax = plt.subplots(figsize=(12.8, 5.2))
    ax.plot(x, mean_tempo, color="black", linewidth=2.0, label="Formal tempo curve")
    for method, curve in recon_curves.items():
        m = next(item for item in metrics_rows if item["method"] == method)
        ax.plot(
            x,
            curve,
            color=colors.get(method, None),
            linewidth=1.7,
            label=f"{method} (rmse={m['rmse']:.2f}, corr={m['corr']:.3f})",
        )
    ax.set_xlim(x_start, x_end)
    ax.set_xlabel("Beat index")
    ax.set_ylabel(str(args.normalization))
    ax.set_title(f"{piece_id}: tempo-curve reconstruction (beat {x_start}-{x_end})")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8, loc="best")
    fig.tight_layout()

    stem = f"{piece_id}_cnn_seed{seed}_vs_direct_reconstruction_{x_start}_{x_end}"
    fig.savefig(output_dir / f"{stem}.png", dpi=220, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)

    pd.DataFrame(metrics_rows).to_csv(output_dir / f"{stem}_metrics.csv", index=False)
    pd.DataFrame(cnn_event_rows).to_csv(output_dir / f"{stem}_events_by_level.csv", index=False)
    (output_dir / f"{stem}_metadata.json").write_text(
        json.dumps(
            {
                "piece_id": piece_id,
                "seed": seed,
                "normalization": str(args.normalization),
                "beat_range": [x_start, x_end],
                "train_floor": float(args.train_floor),
                "merge_tolerance": int(MERGE_TOLERANCE),
                "component_weights": COMPONENT_WEIGHTS,
                "cnn_report_dir": str(cnn_report_dir.resolve()),
                "direct_report_dir": str(direct_root.resolve()),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(str(output_dir))


if __name__ == "__main__":
    main()
