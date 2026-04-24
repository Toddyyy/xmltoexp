#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
VELOCITY_ROOT = ROOT.parent / "Velocity"
ATEPP_EVAL_DIR = VELOCITY_ROOT / "atepp_op110_i_eval"

if str(ATEPP_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(ATEPP_EVAL_DIR))

from boundary_restart.config import load_config  # noqa: E402
from boundary_restart.features import PeakConfig  # noqa: E402
from boundary_restart.lbdm import compute_lbdm_beat_salience  # noqa: E402
from boundary_restart.metrics import evaluate_union_frequency_sequences  # noqa: E402
from predict_new_scores_merge56_seed44 import build_feature_frame_for_score, score_to_npz_arrays  # noqa: E402


DEFAULT_CONFIG = ROOT / "configs" / "salience_grouped3_hi8_score_only_xml_curated.yaml"
DEFAULT_ATEPP_RESULTS_ROOT = VELOCITY_ROOT / "atepp_pure34_top3_eval" / "results"
LEVEL_FILE_STEMS = {
    "level1plus_boundary": "L1plus",
    "level2plus_boundary": "L2plus",
    "level3plus_boundary": "L3plus",
    "level4plus_boundary": "L4plus",
    "level56_boundary": "L5plus6",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate frozen score-only baselines on ATEPP truth sets.")
    parser.add_argument("--baseline_root", required=True)
    parser.add_argument("--model", choices=["logreg", "mlp", "lbdm"], required=True)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--atepp_results_root", default=str(DEFAULT_ATEPP_RESULTS_ROOT))
    parser.add_argument("--piece_slug", nargs="*", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--beat_unit", type=float, default=1.0)
    parser.add_argument("--no_expand_repeats", action="store_true")
    return parser.parse_args()


def load_truth_frame(piece_dir: Path, target_mode: str) -> pd.DataFrame:
    stem = LEVEL_FILE_STEMS[target_mode]
    return pd.read_csv(piece_dir / f"{stem}_truth.csv").sort_values("beat_idx").reset_index(drop=True)


def load_runtime(baseline_root: Path, model_name: str, target_mode: str) -> dict:
    model_dir = baseline_root / model_name / target_mode
    payload = pickle.loads((model_dir / "model.pkl").read_bytes())
    summary = json.loads((model_dir / "summary.json").read_text(encoding="utf-8"))
    return {
        "model": payload["model"],
        "scaler": payload["scaler"],
        "feature_columns": payload["feature_columns"],
        "threshold": float(summary["fixed_threshold"]),
        "summary": summary,
    }


def score_piece(frame: pd.DataFrame, runtime: dict) -> np.ndarray:
    if runtime["summary"]["model"] == "lbdm":
        if "lbdm_score" not in frame.columns:
            raise KeyError("piece frame is missing lbdm_score")
        return frame["lbdm_score"].to_numpy(dtype=np.float32)
    work = frame.copy()
    for col in runtime["feature_columns"]:
        if col not in work.columns:
            work[col] = 0.0
    x = work[runtime["feature_columns"]].to_numpy(dtype=np.float32)
    x = runtime["scaler"].transform(x)
    return runtime["model"].predict_proba(x)[:, 1].astype(np.float32)


def main() -> None:
    args = parse_args()
    baseline_root = Path(args.baseline_root).resolve()
    atepp_root = Path(args.atepp_results_root).resolve()
    cfg = load_config(args.config)
    data_cfg = cfg.get("data", {})
    eval_cfg = cfg.get("evaluation", {})
    peak_cfg = PeakConfig(
        distance=int(data_cfg.get("peak_distance", 6)),
        height=float(data_cfg.get("peak_height", 0.15)),
        prominence=float(data_cfg.get("peak_prominence", 0.05)),
    )

    output_dir = Path(
        args.output_dir
        or (baseline_root / f"{args.model}_atepp_transfer")
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    piece_dirs = [path for path in sorted(atepp_root.iterdir()) if path.is_dir() and (path / "manifest.json").exists()]
    if args.piece_slug:
        wanted = set(args.piece_slug)
        piece_dirs = [path for path in piece_dirs if path.name in wanted]

    runtimes = {
        target_mode: load_runtime(baseline_root, args.model, target_mode)
        for target_mode in LEVEL_FILE_STEMS
    }

    rows: list[dict[str, object]] = []
    for piece_dir in piece_dirs:
        manifest = json.loads((piece_dir / "manifest.json").read_text(encoding="utf-8"))
        piece_slug = str(piece_dir.name)
        score_path = Path(manifest["selected_score_path"])
        piece_frame = build_feature_frame_for_score(
            score_path=score_path,
            piece_id=piece_slug,
            peak_cfg=peak_cfg,
            beat_unit=float(args.beat_unit),
            measure_cycle=int(data_cfg.get("measure_cycle", 3)),
            symmetry_window=int(data_cfg.get("symmetry_window", 4)),
            deviation_window=int(data_cfg.get("deviation_window", 8)),
            expand_repeats=not bool(args.no_expand_repeats),
        )
        if args.model == "lbdm":
            arrays = score_to_npz_arrays(
                score_path=score_path,
                beat_unit=float(args.beat_unit),
                expand_repeats=not bool(args.no_expand_repeats),
            )
            piece_frame["lbdm_score"] = compute_lbdm_beat_salience(
                note_feats=np.asarray(arrays["note_feats"], dtype=np.float32),
                beat_ids=np.asarray(arrays["beat_ids"], dtype=np.int32),
                num_beats=int(arrays["num_beats"]),
            )
        piece_out = output_dir / piece_slug
        piece_out.mkdir(parents=True, exist_ok=True)
        piece_frame.to_csv(piece_out / "score_beat_features.csv.gz", index=False, compression="gzip")

        all_pred = []
        for target_mode in LEVEL_FILE_STEMS:
            truth = load_truth_frame(piece_dir, target_mode)
            runtime = runtimes[target_mode]
            scores = score_piece(piece_frame, runtime)
            pred_df = pd.DataFrame(
                {
                    "sample_id": piece_slug,
                    "piece_id": piece_slug,
                    "beat_idx": piece_frame["beat_idx"].to_numpy(dtype=np.int32),
                    "performer_count": int(manifest["usable_tempo_curves"]),
                    "union_target": truth["union_target"].to_numpy(dtype=np.float32),
                    "frequency_target": truth["frequency_target"].to_numpy(dtype=np.float32),
                    "detector_score": scores,
                }
            )
            pred_df.to_csv(piece_out / f"{target_mode}_predictions.csv.gz", index=False, compression="gzip")
            all_pred.append(pred_df.assign(target_mode=target_mode))
            metrics = evaluate_union_frequency_sequences(
                sequence_scores={piece_slug: scores},
                sequence_union_labels={piece_slug: truth["union_target"].to_numpy(dtype=np.float32)},
                sequence_frequency_targets={piece_slug: truth["frequency_target"].to_numpy(dtype=np.float32)},
                threshold=float(runtime["threshold"]),
                tolerance=int(eval_cfg.get("event_tolerance", 1)),
                min_distance=int(eval_cfg.get("min_distance", 6)),
                consensus_threshold=0.5,
                prominence=float(eval_cfg.get("prominence", 0.0)),
            )
            rows.append(
                {
                    "piece_slug": piece_slug,
                    "target_mode": target_mode,
                    "threshold": float(runtime["threshold"]),
                    "predicted_event_count": int(metrics.pred_events),
                    "true_union_events": int(metrics.true_union_events),
                    "true_consensus_events": int(metrics.true_consensus_events),
                    "union_precision": float(metrics.union_precision),
                    "frequency_weighted_precision": float(metrics.frequency_weighted_precision),
                    "consensus_precision": float(metrics.consensus_precision),
                    "union_recall": float(metrics.union_recall),
                    "weighted_recall": float(metrics.weighted_recall),
                    "consensus_recall": float(metrics.consensus_recall),
                    "mean_offset": None if metrics.mean_offset is None else float(metrics.mean_offset),
                }
            )
        if all_pred:
            pd.concat(all_pred, ignore_index=True).to_csv(piece_out / "all_predictions.csv.gz", index=False, compression="gzip")

    result_df = pd.DataFrame(rows)
    result_df.to_csv(output_dir / "all_piece_evaluation_summary.csv", index=False)
    mean_df = (
        result_df.groupby("target_mode", sort=False)[
            [
                "union_precision",
                "frequency_weighted_precision",
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
    mean_df.to_csv(output_dir / "mean_metrics_by_level.csv", index=False)
    print(str(output_dir))


if __name__ == "__main__":
    main()
