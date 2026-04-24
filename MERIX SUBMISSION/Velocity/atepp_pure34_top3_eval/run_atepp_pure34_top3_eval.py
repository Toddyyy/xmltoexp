#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
EVAL_ROOT = ROOT / "MERIX SUBMISSION" / "Velocity" / "atepp_op110_i_eval"
if str(EVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(EVAL_ROOT))

import run_atepp_op110_i_eval as eval_one  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the merged56 boundary evaluation on the top-N pure 3/4 ATEPP pieces."
    )
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--beat_unit", type=float, default=1.0)
    parser.add_argument("--smooth_window", type=int, default=3)
    parser.add_argument("--bpm_max", type=float, default=600.0)
    parser.add_argument("--cumulative_merge_tolerance", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--top_n", type=int, default=3)
    parser.add_argument(
        "--output_dir",
        default=str(Path(__file__).resolve().parent / "results"),
    )
    return parser.parse_args()


def slugify_piece(piece_dir: Path) -> str:
    relative = piece_dir.relative_to(eval_one.ATEPP_ROOT)
    joined = "__".join(relative.parts)
    slug = re.sub(r"[^A-Za-z0-9]+", "_", joined).strip("_").lower()
    return slug


def find_top_pure_three_four_pieces(top_n: int) -> list[dict]:
    rows: list[dict] = []
    for piece_dir, score_path in eval_one.iter_score_dirs(eval_one.ATEPP_ROOT):
        time_sigs = eval_one.extract_time_signatures(score_path)
        if not time_sigs or time_sigs != eval_one.PURE_THREE_FOUR:
            continue
        midi_count = eval_one.count_performance_midis(piece_dir)
        if midi_count <= 0:
            continue
        rows.append(
            {
                "piece_dir": piece_dir,
                "score_path": score_path,
                "midi_count": midi_count,
                "time_signatures": sorted(time_sigs),
                "piece_slug": slugify_piece(piece_dir),
                "piece_relpath": str(piece_dir.relative_to(eval_one.ATEPP_ROOT)),
            }
        )
    rows.sort(key=lambda row: (-row["midi_count"], row["piece_relpath"]))
    return rows[: int(top_n)]


def raw_boundary_means(per_performance: pd.DataFrame) -> dict[str, float]:
    out: dict[str, float] = {}
    for level in range(1, 7):
        level_df = per_performance[per_performance["level"] == level]
        curves = [
            df["boundary"].to_numpy(dtype=np.float32)
            for _, df in level_df.groupby("performer_id", sort=True)
        ]
        out[f"L{level}"] = float(np.mean(np.stack(curves, axis=0).sum(axis=1))) if curves else 0.0
    return out


def evaluate_piece(selected: dict, args: argparse.Namespace, root_output: Path) -> tuple[pd.DataFrame, dict]:
    piece_dir = selected["piece_dir"]
    score_path = selected["score_path"]
    piece_slug = selected["piece_slug"]
    piece_output = root_output / piece_slug
    piece_output.mkdir(parents=True, exist_ok=True)

    alignment_info = eval_one.align_piece(piece_dir)

    cfg = eval_one.load_config(str(eval_one.DEFAULT_CONFIG))
    data_cfg = cfg.get("data", {})
    peak_cfg = eval_one.PeakConfig(
        distance=int(data_cfg.get("peak_distance", 6)),
        height=float(data_cfg.get("peak_height", 0.15)),
        prominence=float(data_cfg.get("peak_prominence", 0.05)),
    )
    piece_frame = eval_one.build_feature_frame_for_score(
        score_path=score_path,
        piece_id=piece_slug,
        peak_cfg=peak_cfg,
        beat_unit=float(args.beat_unit),
        measure_cycle=int(data_cfg.get("measure_cycle", 3)),
        symmetry_window=int(data_cfg.get("symmetry_window", 4)),
        deviation_window=int(data_cfg.get("deviation_window", 8)),
        expand_repeats=False,
    )
    num_beats = int(len(piece_frame))

    tempo_arrays, failed_matches = eval_one.load_tempo_arrays(
        piece_dir=piece_dir,
        num_beats=num_beats,
        beat_unit=float(args.beat_unit),
        smooth_window=int(args.smooth_window),
        bpm_max=float(args.bpm_max),
    )
    component_frames, component_arrays, per_performance = eval_one.build_component_truth_frames(
        tempo_arrays=tempo_arrays,
        piece_id=piece_slug,
        num_beats=num_beats,
    )
    truth_frames = eval_one.build_cumulative_truth_frames(
        component_frames=component_frames,
        piece_id=piece_slug,
        num_beats=num_beats,
        tolerance=int(args.cumulative_merge_tolerance),
    )

    per_performance.to_csv(piece_output / "per_performance_raw_boundaries.csv.gz", index=False, compression="gzip")
    for component_name, frame in component_frames.items():
        frame.to_csv(piece_output / f"{component_name}_piece_frequency.csv", index=False)
    for label, frame in truth_frames.items():
        frame.to_csv(piece_output / f"{label.replace('+', 'plus')}_truth.csv", index=False)

    eval_one.save_truth_plots(
        output_dir=piece_output,
        piece_id=piece_slug,
        tempo_arrays=tempo_arrays,
        truth_frames=truth_frames,
    )
    metrics_df = eval_one.predict_and_evaluate(
        score_path=score_path,
        piece_id=piece_slug,
        truth_frames=truth_frames,
        seed=int(args.seed),
        beat_unit=float(args.beat_unit),
        device_name=str(args.device),
        output_dir=piece_output,
    )
        metrics_df.insert(1, "piece_slug", piece_slug)
        metrics_df.insert(2, "piece_relpath", selected["piece_relpath"])
    metrics_df.to_csv(piece_output / "evaluation_summary.csv", index=False)

    manifest = {
        "selected_piece_dir": str(piece_dir),
        "selected_score_path": str(score_path),
        "selection": {
            "time_signatures": selected["time_signatures"],
            "performance_midis": int(selected["midi_count"]),
        },
        "alignment": alignment_info,
        "num_beats": int(num_beats),
        "usable_tempo_curves": int(len(tempo_arrays)),
        "failed_match_files": failed_matches,
        "seed": int(args.seed),
        "beat_unit": float(args.beat_unit),
        "cumulative_merge_tolerance": int(args.cumulative_merge_tolerance),
        "component_weights": eval_one.COMPONENT_WEIGHTS,
        "raw_level_mean_boundary_count": raw_boundary_means(per_performance),
    }
    (piece_output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return metrics_df, manifest


def main() -> None:
    args = parse_args()
    root_output = Path(args.output_dir).resolve()
    root_output.mkdir(parents=True, exist_ok=True)

    selected_pieces = find_top_pure_three_four_pieces(top_n=int(args.top_n))
    if not selected_pieces:
        raise RuntimeError("No pure 3/4 ATEPP pieces with performance MIDIs were found.")

    all_metrics = []
    manifest_rows = []
    for selected in selected_pieces:
        metrics_df, manifest = evaluate_piece(selected, args, root_output)
        all_metrics.append(metrics_df)
        manifest_rows.append(
            {
                "piece_slug": selected["piece_slug"],
                "piece_relpath": selected["piece_relpath"],
                "midi_count": int(selected["midi_count"]),
                "num_beats": int(manifest["num_beats"]),
                "usable_tempo_curves": int(manifest["usable_tempo_curves"]),
            }
        )

    all_metrics_df = pd.concat(all_metrics, ignore_index=True)
    all_metrics_df.to_csv(root_output / "all_piece_evaluation_summary.csv", index=False)

    piece_manifest_df = pd.DataFrame(manifest_rows)
    piece_manifest_df.to_csv(root_output / "piece_manifest.csv", index=False)

    mean_by_level = (
        all_metrics_df.groupby("level_label", sort=False)[
            [
                "union_precision",
                "frequency_weighted_precision",
                "consensus_precision",
                "union_recall",
                "weighted_recall",
                "consensus_recall",
                "predicted_event_count",
                "true_union_events",
            ]
        ]
        .mean()
        .reset_index()
    )
    mean_by_level.to_csv(root_output / "mean_metrics_by_level.csv", index=False)

    mean_by_piece = (
        all_metrics_df.groupby(["piece_slug", "piece_relpath"], sort=False)[
            ["union_precision", "weighted_recall", "predicted_event_count", "true_union_events"]
        ]
        .mean()
        .reset_index()
    )
    mean_by_piece.to_csv(root_output / "mean_metrics_by_piece.csv", index=False)

    print(piece_manifest_df.to_string(index=False))
    print()
    print(all_metrics_df.to_string(index=False))
    print()
    print(mean_by_level.to_string(index=False))


if __name__ == "__main__":
    main()
