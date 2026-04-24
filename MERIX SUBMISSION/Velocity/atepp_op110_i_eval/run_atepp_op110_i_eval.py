#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
MERIX_ROOT = ROOT / "MERIX SUBMISSION"
VELOCITY_ROOT = MERIX_ROOT / "Velocity"
BOUNDARY_ROOT = MERIX_ROOT / "Boundary_Restart"
MIREX_ROOT = MERIX_ROOT / "MIREX_Model"
ATEPP_ROOT = ROOT / "ATEPP-1.2" / "ATEPP-1.2"
ALIGN_PROG = ROOT / "AlignmentTool" / "Programs"

sys.path.insert(0, str(VELOCITY_ROOT))
sys.path.insert(0, str(BOUNDARY_ROOT))
sys.path.insert(0, str(MIREX_ROOT))

from build_mazurka_velocity_npz_performer_levels import group_analysis_hierarchy  # noqa: E402
from boundary_restart.config import load_config  # noqa: E402
from boundary_restart.cumulative_targets import (  # noqa: E402
    build_topdown_cumulative_frequency,
    cumulative_components_for_target,
)
from boundary_restart.metrics import evaluate_union_frequency_sequences  # noqa: E402
from predict_new_scores_merge56_seed44 import (  # noqa: E402
    DEFAULT_CLEAN_ROOT,
    DEFAULT_CONFIG,
    LEVEL_SPECS,
    build_feature_frame_for_score,
    load_level_runtime,
    run_inference_for_level,
)
from train_piece_union_protocol import resolve_device  # noqa: E402
from boundary_restart.features import PeakConfig  # noqa: E402

PURE_THREE_FOUR = {(3, 4)}
STR_VEC = [3, 2, 2, 2, 2, 2]
COMPONENT_WEIGHTS = {
    "level56": 1.0,
    "level4": 0.64,
    "level3": 0.46,
    "level2": 0.28,
    "level1": 0.16,
}
LEVEL_COMPONENTS = {
    "L1+": ("level56", "level4", "level3", "level2", "level1"),
    "L2+": ("level56", "level4", "level3", "level2"),
    "L3+": ("level56", "level4", "level3"),
    "L4+": ("level56", "level4"),
    "L5+6": ("level56",),
}
LEVEL_TARGETS = {
    "L1+": "level1plus_boundary",
    "L2+": "level2plus_boundary",
    "L3+": "level3plus_boundary",
    "L4+": "level4plus_boundary",
    "L5+6": "level56_boundary",
}
LEVEL_COLORS = {
    "L1+": "#1f77b4",
    "L2+": "#ff7f0e",
    "L3+": "#2ca02c",
    "L4+": "#d62728",
    "L5+6": "#9467bd",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the final merged56 score model on the pure 3/4 ATEPP piece with the most performance MIDIs."
    )
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--beat_unit", type=float, default=1.0)
    parser.add_argument("--smooth_window", type=int, default=3)
    parser.add_argument("--bpm_max", type=float, default=600.0)
    parser.add_argument("--cumulative_merge_tolerance", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output_dir", default=str(Path(__file__).resolve().parent / "results"))
    return parser.parse_args()


def iter_score_dirs(root: Path) -> list[tuple[Path, Path]]:
    candidates: dict[Path, Path] = {}
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        lower = path.name.lower()
        if lower.endswith(".musicxml") or lower.endswith(".xml") or lower.endswith(".mxl"):
            candidates.setdefault(path.parent, path)
    return sorted((piece_dir, score_path) for piece_dir, score_path in candidates.items())


def read_musicxml_text(score_path: Path) -> str:
    suffix = score_path.suffix.lower()
    if suffix != ".mxl":
        return score_path.read_text(encoding="utf-8", errors="replace")
    with zipfile.ZipFile(score_path) as zf:
        container_text = zf.read("META-INF/container.xml")
        root = ET.fromstring(container_text)
        ns = {"c": "urn:oasis:names:tc:opendocument:xmlns:container"}
        rootfile = root.find(".//c:rootfile", ns)
        if rootfile is None:
            raise ValueError(f"Missing rootfile in {score_path}")
        full_path = rootfile.attrib["full-path"]
        return zf.read(full_path).decode("utf-8", errors="replace")


def extract_time_signatures(score_path: Path) -> set[tuple[int, int]]:
    try:
        xml_text = read_musicxml_text(score_path)
        root = ET.fromstring(xml_text)
    except Exception:
        return set()
    signatures: set[tuple[int, int]] = set()
    for time_el in root.findall(".//time"):
        beats = time_el.findtext("beats")
        beat_type = time_el.findtext("beat-type")
        if beats is None or beat_type is None:
            continue
        try:
            signatures.add((int(beats), int(beat_type)))
        except ValueError:
            continue
    return signatures


def count_performance_midis(piece_dir: Path) -> int:
    count = 0
    for path in piece_dir.glob("*.mid"):
        if path.stem.isdigit():
            count += 1
    return count


def find_best_pure_three_four_piece(atepp_root: Path) -> dict:
    best: dict | None = None
    for piece_dir, score_path in iter_score_dirs(atepp_root):
        time_sigs = extract_time_signatures(score_path)
        if not time_sigs or time_sigs != PURE_THREE_FOUR:
            continue
        midi_count = count_performance_midis(piece_dir)
        if midi_count <= 0:
            continue
        record = {
            "piece_dir": piece_dir,
            "score_path": score_path,
            "midi_count": midi_count,
            "time_signatures": sorted(time_sigs),
        }
        if best is None or record["midi_count"] > best["midi_count"]:
            best = record
    if best is None:
        raise RuntimeError("No pure 3/4 ATEPP piece with performance MIDIs was found.")
    return best


def run_checked(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def align_piece(piece_dir: Path) -> dict:
    required = [
        "midi2pianoroll",
        "MusicXMLToFmt3x",
        "MusicXMLToHMM",
        "ScorePerfmMatcher",
        "ErrorDetection",
        "RealignmentMOHMM",
    ]
    for name in required:
        prog = ALIGN_PROG / name
        if not prog.exists():
            raise FileNotFoundError(f"Missing alignment program: {prog}")

    score_xml = piece_dir / "musicxml_cleaned.musicxml"
    if not score_xml.exists():
        raise FileNotFoundError(f"Missing score XML: {score_xml}")

    score_fmt3x = piece_dir / "score_fmt3x.txt"
    score_hmm = piece_dir / "score_hmm.txt"
    if not score_fmt3x.exists():
        run_checked([str(ALIGN_PROG / "MusicXMLToFmt3x"), str(score_xml), str(score_fmt3x)])
    if not score_hmm.exists():
        run_checked([str(ALIGN_PROG / "MusicXMLToHMM"), str(score_xml), str(score_hmm)])

    midi_paths = sorted(path for path in piece_dir.glob("*.mid") if path.stem.isdigit())
    created = 0
    existing = 0
    for midi_path in midi_paths:
        base = midi_path.with_suffix("")
        spr = Path(f"{base}_spr.txt")
        pre = Path(f"{base}_pre_match.txt")
        err = Path(f"{base}_err_match.txt")
        match = Path(f"{base}_match.txt")
        if match.exists():
            existing += 1
            continue
        run_checked([str(ALIGN_PROG / "midi2pianoroll"), "0", str(base)])
        run_checked([str(ALIGN_PROG / "ScorePerfmMatcher"), str(score_hmm), str(spr), str(pre), "1.0"])
        run_checked([str(ALIGN_PROG / "ErrorDetection"), str(score_fmt3x), str(score_hmm), str(pre), str(err), "0"])
        run_checked([str(ALIGN_PROG / "RealignmentMOHMM"), str(score_fmt3x), str(score_hmm), str(err), str(match), "0.3"])
        created += 1

    match_paths = sorted(path for path in piece_dir.glob("*_match.txt") if path.stem.replace("_match", "").isdigit())
    return {
        "score_xml": str(score_xml),
        "score_fmt3x": str(score_fmt3x),
        "score_hmm": str(score_hmm),
        "performance_midis": len(midi_paths),
        "existing_matches": existing,
        "created_matches": created,
        "total_matches": len(match_paths),
    }


def read_tpqn(fmt3x_path: Path, default: float = 24.0) -> float:
    if not fmt3x_path.exists():
        return default
    text = fmt3x_path.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"TPQN:\s*(\d+)", text)
    return float(match.group(1)) if match else default


def tempo_curve_from_match(
    match_path: Path,
    fmt3x_path: Path,
    num_beats: int,
    beat_unit: float,
    smooth_window: int,
    bpm_max: float,
) -> np.ndarray | None:
    tpqn = read_tpqn(fmt3x_path)
    rows = []
    for line in match_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip() or line.startswith("//") or line.startswith("Missing"):
            continue
        parts = re.split(r"\s+", line.strip())
        if len(parts) < 12:
            continue
        rows.append(
            {
                "onset": float(parts[1]),
                "score_time": float(parts[8]),
                "score_note": parts[9],
                "err_idx": parts[10],
            }
        )
    if not rows:
        return None
    df = pd.DataFrame(rows)
    df = df[(df["score_note"] != "*") & (df["err_idx"] == "0")]
    if df.empty:
        return None

    score_quarter = df["score_time"].astype(float) / float(tpqn)
    beat_idx = np.floor(score_quarter / max(float(beat_unit), 1e-9)).astype(int)
    df["beat_idx"] = beat_idx
    df = df[(df["beat_idx"] >= 0) & (df["beat_idx"] < int(num_beats))]
    if df.empty:
        return None

    beat_time = df.groupby("beat_idx")["onset"].median().sort_index()
    full_idx = pd.Index(range(int(num_beats)), name="beat_idx")
    beat_time = beat_time.reindex(full_idx).interpolate("linear", limit_direction="both")
    dt = beat_time.diff()
    tempo = (60.0 * float(beat_unit)) / dt
    tempo = tempo[(tempo > 0.0) & (tempo < float(bpm_max))]
    tempo = tempo.rolling(window=int(smooth_window), center=True, min_periods=1).mean()
    tempo = tempo.clip(upper=float(bpm_max))
    tempo = tempo.reindex(full_idx).interpolate("linear", limit_direction="both")
    return tempo.to_numpy(dtype=np.float32)


def load_tempo_arrays(
    piece_dir: Path,
    num_beats: int,
    beat_unit: float,
    smooth_window: int,
    bpm_max: float,
) -> tuple[dict[str, np.ndarray], list[str]]:
    fmt3x_path = piece_dir / "score_fmt3x.txt"
    match_paths = sorted(path for path in piece_dir.glob("*_match.txt") if path.stem.replace("_match", "").isdigit())
    tempo_arrays: dict[str, np.ndarray] = {}
    failed: list[str] = []
    for match_path in match_paths:
        curve = tempo_curve_from_match(
            match_path=match_path,
            fmt3x_path=fmt3x_path,
            num_beats=int(num_beats),
            beat_unit=float(beat_unit),
            smooth_window=int(smooth_window),
            bpm_max=float(bpm_max),
        )
        if curve is None:
            failed.append(match_path.name)
            continue
        tempo_arrays[match_path.stem.replace("_match", "")] = curve
    if not tempo_arrays:
        raise RuntimeError(f"No usable tempo curves were produced for {piece_dir}")
    return tempo_arrays, failed


def build_component_truth_frames(
    tempo_arrays: dict[str, np.ndarray],
    piece_id: str,
    num_beats: int,
) -> tuple[dict[str, pd.DataFrame], dict[str, np.ndarray], pd.DataFrame]:
    raw_masks = {level: [] for level in range(1, 7)}
    union56_masks: list[np.ndarray] = []
    per_performance_rows = []

    for performer_id, curve in sorted(tempo_arrays.items()):
        results_raw, _ = group_analysis_hierarchy(curve, str_vec=STR_VEC, enforce_nested=False)
        for level in range(1, 7):
            mask = results_raw[level - 1].astype(np.float32)
            raw_masks[level].append(mask)
            per_performance_rows.append(
                pd.DataFrame(
                    {
                        "piece_id": piece_id,
                        "performer_id": performer_id,
                        "level": level,
                        "beat_idx": np.arange(num_beats, dtype=np.int32),
                        "boundary": mask,
                    }
                )
            )
        union56_masks.append(np.maximum(results_raw[4], results_raw[5]).astype(np.float32))

    component_arrays = {
        "level1": np.mean(np.stack(raw_masks[1], axis=0), axis=0).astype(np.float32),
        "level2": np.mean(np.stack(raw_masks[2], axis=0), axis=0).astype(np.float32),
        "level3": np.mean(np.stack(raw_masks[3], axis=0), axis=0).astype(np.float32),
        "level4": np.mean(np.stack(raw_masks[4], axis=0), axis=0).astype(np.float32),
        "level56": np.mean(np.stack(union56_masks, axis=0), axis=0).astype(np.float32),
    }
    component_frames = {
        name: pd.DataFrame(
            {
                "piece_id": piece_id,
                "beat_idx": np.arange(num_beats, dtype=np.int32),
                "frequency_target": values,
            }
        )
        for name, values in component_arrays.items()
    }
    per_performance = pd.concat(per_performance_rows, ignore_index=True)
    return component_frames, component_arrays, per_performance


def build_cumulative_truth_frames(
    component_frames: dict[str, pd.DataFrame],
    piece_id: str,
    num_beats: int,
    tolerance: int,
) -> dict[str, pd.DataFrame]:
    base_piece = pd.DataFrame(
        {
            "piece_id": piece_id,
            "beat_idx": np.arange(num_beats, dtype=np.int32),
        }
    )
    truth_frames: dict[str, pd.DataFrame] = {}
    for label, component_order in LEVEL_COMPONENTS.items():
        merged = build_topdown_cumulative_frequency(
            base_piece=base_piece,
            component_map=component_frames,
            component_order=component_order,
            tolerance=int(tolerance),
            component_weights=COMPONENT_WEIGHTS,
        )
        merged["union_target"] = (merged["frequency_target"] > 0.0).astype(np.float32)
        truth_frames[label] = merged
    return truth_frames


def save_truth_plots(
    output_dir: Path,
    piece_id: str,
    tempo_arrays: dict[str, np.ndarray],
    truth_frames: dict[str, pd.DataFrame],
) -> None:
    mean_tempo = np.mean(np.stack(list(tempo_arrays.values()), axis=0), axis=0)
    x = np.arange(mean_tempo.shape[0], dtype=np.int32)
    plt.figure(figsize=(18, 5))
    plt.plot(x, mean_tempo, color="black", linewidth=1.2, label="mean tempo")
    for label, frame in truth_frames.items():
        beats = frame.loc[frame["union_target"] > 0.5, "beat_idx"].to_numpy(dtype=np.int32)
        if beats.size == 0:
            continue
        plt.scatter(beats, mean_tempo[beats], s=18, color=LEVEL_COLORS[label], label=label, alpha=0.8)
    plt.title(f"{piece_id}: mean tempo with performance-derived cumulative boundaries")
    plt.xlabel("Beat index")
    plt.ylabel("BPM")
    plt.grid(alpha=0.3)
    plt.legend(ncol=5, fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "mean_tempo_truth_boundaries.png", dpi=200)
    plt.close()


def predict_and_evaluate(
    score_path: Path,
    piece_id: str,
    truth_frames: dict[str, pd.DataFrame],
    seed: int,
    beat_unit: float,
    device_name: str,
    output_dir: Path,
) -> pd.DataFrame:
    cfg = load_config(str(DEFAULT_CONFIG))
    data_cfg = cfg.get("data", {})
    eval_cfg = cfg.get("evaluation", {})
    peak_cfg = PeakConfig(
        distance=int(data_cfg.get("peak_distance", 6)),
        height=float(data_cfg.get("peak_height", 0.15)),
        prominence=float(data_cfg.get("peak_prominence", 0.05)),
    )
    piece_frame = build_feature_frame_for_score(
        score_path=score_path,
        piece_id=piece_id,
        peak_cfg=peak_cfg,
        beat_unit=float(beat_unit),
        measure_cycle=int(data_cfg.get("measure_cycle", 3)),
        symmetry_window=int(data_cfg.get("symmetry_window", 4)),
        deviation_window=int(data_cfg.get("deviation_window", 8)),
        expand_repeats=False,
    )
    piece_frame.to_csv(output_dir / "score_beat_features.csv.gz", index=False, compression="gzip")

    device = resolve_device(device_name)
    rows = []
    all_pred_frames = []
    all_event_frames = []
    for label, detector_target in LEVEL_SPECS.items():
        runtime = load_level_runtime(
            cfg=cfg,
            clean_root=DEFAULT_CLEAN_ROOT,
            detector_target=detector_target,
            seed=int(seed),
            device=device,
        )
        pred_df, events = run_inference_for_level(
            frame=piece_frame,
            runtime=runtime,
            device=device,
            batch_size=1,
            eval_cfg=eval_cfg,
        )
        pred_df.insert(0, "level_label", label)
        events.insert(0, "level_label", label)
        pred_df.to_csv(output_dir / f"{label.replace('+', 'plus')}_predictions.csv.gz", index=False, compression="gzip")
        events.to_csv(output_dir / f"{label.replace('+', 'plus')}_events.csv", index=False)
        all_pred_frames.append(pred_df)
        all_event_frames.append(events)

        truth = truth_frames[label].sort_values("beat_idx").reset_index(drop=True)
        metrics = evaluate_union_frequency_sequences(
            sequence_scores={piece_id: pred_df.sort_values("beat_idx")["detector_score"].to_numpy(dtype=np.float32)},
            sequence_union_labels={piece_id: truth["union_target"].to_numpy(dtype=np.float32)},
            sequence_frequency_targets={piece_id: truth["frequency_target"].to_numpy(dtype=np.float32)},
            threshold=float(runtime["threshold"]),
            tolerance=int(eval_cfg.get("event_tolerance", 1)),
            min_distance=int(eval_cfg.get("min_distance", 6)),
            consensus_threshold=0.5,
            prominence=float(eval_cfg.get("prominence", 0.0)),
        )
        rows.append(
            {
                "piece_id": piece_id,
                "level_label": label,
                "detector_target": detector_target,
                "threshold": float(runtime["threshold"]),
                "frozen_epochs": int(runtime["frozen_epochs"]),
                "predicted_event_count": int(len(events)),
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

    if all_pred_frames:
        pd.concat(all_pred_frames, ignore_index=True).to_csv(
            output_dir / "all_predictions.csv.gz", index=False, compression="gzip"
        )
    if all_event_frames:
        pd.concat(all_event_frames, ignore_index=True).to_csv(output_dir / "all_events.csv", index=False)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    selected = find_best_pure_three_four_piece(ATEPP_ROOT)
    piece_dir = selected["piece_dir"]
    score_path = selected["score_path"]
    piece_id = "beethoven_op110_i"

    alignment_info = align_piece(piece_dir)

    cfg = load_config(str(DEFAULT_CONFIG))
    data_cfg = cfg.get("data", {})
    peak_cfg = PeakConfig(
        distance=int(data_cfg.get("peak_distance", 6)),
        height=float(data_cfg.get("peak_height", 0.15)),
        prominence=float(data_cfg.get("peak_prominence", 0.05)),
    )
    piece_frame = build_feature_frame_for_score(
        score_path=score_path,
        piece_id=piece_id,
        peak_cfg=peak_cfg,
        beat_unit=float(args.beat_unit),
        measure_cycle=int(data_cfg.get("measure_cycle", 3)),
        symmetry_window=int(data_cfg.get("symmetry_window", 4)),
        deviation_window=int(data_cfg.get("deviation_window", 8)),
        expand_repeats=False,
    )
    num_beats = int(len(piece_frame))

    tempo_arrays, failed_matches = load_tempo_arrays(
        piece_dir=piece_dir,
        num_beats=num_beats,
        beat_unit=float(args.beat_unit),
        smooth_window=int(args.smooth_window),
        bpm_max=float(args.bpm_max),
    )
    component_frames, component_arrays, per_performance = build_component_truth_frames(
        tempo_arrays=tempo_arrays,
        piece_id=piece_id,
        num_beats=num_beats,
    )
    truth_frames = build_cumulative_truth_frames(
        component_frames=component_frames,
        piece_id=piece_id,
        num_beats=num_beats,
        tolerance=int(args.cumulative_merge_tolerance),
    )

    per_performance.to_csv(output_dir / "per_performance_raw_boundaries.csv.gz", index=False, compression="gzip")
    for component_name, frame in component_frames.items():
        frame.to_csv(output_dir / f"{component_name}_piece_frequency.csv", index=False)
    for label, frame in truth_frames.items():
        frame.to_csv(output_dir / f"{label.replace('+', 'plus')}_truth.csv", index=False)

    save_truth_plots(
        output_dir=output_dir,
        piece_id=piece_id,
        tempo_arrays=tempo_arrays,
        truth_frames=truth_frames,
    )
    metrics_df = predict_and_evaluate(
        score_path=score_path,
        piece_id=piece_id,
        truth_frames=truth_frames,
        seed=int(args.seed),
        beat_unit=float(args.beat_unit),
        device_name=str(args.device),
        output_dir=output_dir,
    )
    metrics_df.to_csv(output_dir / "evaluation_summary.csv", index=False)

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
        "component_weights": COMPONENT_WEIGHTS,
        "raw_level_mean_boundary_count": {
            f"L{level}": float(np.mean(np.stack(masks, axis=0).sum(axis=1)))
            for level, masks in {
                1: [df["boundary"].to_numpy(dtype=np.float32) for _, df in per_performance[per_performance["level"] == 1].groupby("performer_id")],
                2: [df["boundary"].to_numpy(dtype=np.float32) for _, df in per_performance[per_performance["level"] == 2].groupby("performer_id")],
                3: [df["boundary"].to_numpy(dtype=np.float32) for _, df in per_performance[per_performance["level"] == 3].groupby("performer_id")],
                4: [df["boundary"].to_numpy(dtype=np.float32) for _, df in per_performance[per_performance["level"] == 4].groupby("performer_id")],
                5: [df["boundary"].to_numpy(dtype=np.float32) for _, df in per_performance[per_performance["level"] == 5].groupby("performer_id")],
                6: [df["boundary"].to_numpy(dtype=np.float32) for _, df in per_performance[per_performance["level"] == 6].groupby("performer_id")],
            }.items()
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(json.dumps(manifest, indent=2))
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
