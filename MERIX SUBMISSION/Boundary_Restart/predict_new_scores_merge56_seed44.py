#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import tempfile
from pathlib import Path

import music21
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from boundary_restart.config import load_config, resolve_path
from boundary_restart.derived_features import add_highlevel_derived_features
from boundary_restart.features import PeakConfig, build_beat_feature_frame
from boundary_restart.models import build_sequence_model
from boundary_restart.xml_score_features import extract_xml_beat_features_from_path
from train_piece_union_protocol import (
    PieceUnionDataset,
    build_predicted_event_frame,
    collate_piece_union,
    predict_detector,
    resolve_device,
)

ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "configs" / "salience_grouped3_hi8_score_only_xml_curated.yaml"
DEFAULT_CLEAN_ROOT = ROOT / "reports" / "clean_outer_test"
DEFAULT_TARGET_SCORES = [
    (
        "beethoven_pathetique_ii",
        ROOT.parent.parent / "ATEPP-1.2" / "ATEPP-1.2" / "Ludwig_van_Beethoven" / 'Piano_Sonata_No._8_in_C_Minor,_Op._13_"Pathétique"' / "II._Adagio_cantabile" / "musicxml_cleaned.musicxml",
    ),
    (
        "mozart_k283_i",
        ROOT.parent.parent / "ATEPP-1.2" / "ATEPP-1.2" / "Wolfgang_Amadeus_Mozart" / "Piano_Sonata_No._5_in_G_Major,_K._283" / "I._Allegro" / "Sonata_No._5_1st_Movement_K._283.mxl",
    ),
    (
        "mozart_k331_i",
        ROOT.parent.parent / "ATEPP-1.2" / "ATEPP-1.2" / "Wolfgang_Amadeus_Mozart" / "Piano_Sonata_No._11_in_A_Major,_K._331" / "1._Tema_(Andante_grazioso)_con_variazioni" / "Sonata_No._11_1st_Movement_K._331.mxl",
    ),
]
LEVEL_SPECS = {
    "L1+": "level1plus_boundary",
    "L2+": "level2plus_boundary",
    "L3+": "level3plus_boundary",
    "L4+": "level4plus_boundary",
    "L5+6": "level56_boundary",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict hierarchical phrase boundaries for new scores with merged56 seed44 models.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output_dir", default=str(ROOT / "reports" / "new_score_predictions_merge56_seed44"))
    parser.add_argument("--score", nargs="*", default=None, help="Optional score paths. If omitted, the three default target scores are used.")
    parser.add_argument("--piece_id", nargs="*", default=None, help="Optional piece ids matching --score.")
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--beat_unit", type=float, default=1.0)
    parser.add_argument("--no_expand_repeats", action="store_true")
    return parser.parse_args()


def absolute_offset(el, score, part) -> float:
    try:
        return float(el.getOffsetInHierarchy(score))
    except Exception:
        try:
            return float(el.getOffsetInHierarchy(part))
        except Exception:
            return float(part.flatten().elementOffset(el))


def element_flags(el) -> tuple[float, float]:
    articulations = list(getattr(el, "articulations", []) or [])
    accent = 0.0
    staccato = 0.0
    for articulation in articulations:
        classes = {cls.lower() for cls in getattr(articulation, "classes", [])}
        if any("accent" in cls for cls in classes):
            accent = 1.0
        if any("staccato" in cls for cls in classes):
            staccato = 1.0
    return accent, staccato


def score_to_npz_arrays(score_path: Path, beat_unit: float, expand_repeats: bool) -> dict[str, np.ndarray | float | int]:
    score = music21.converter.parse(str(score_path))
    if expand_repeats:
        try:
            score = score.expandRepeats()
        except Exception:
            pass
    parts = list(score.parts) if len(score.parts) > 0 else [score]
    note_rows: list[list[float]] = []
    highest_time = 0.0
    eps = 1e-9
    for part_idx, part in enumerate(parts):
        for el in part.recurse():
            if isinstance(el, music21.note.Note):
                duration = float(el.duration.quarterLength or 0.0)
                if duration <= 0:
                    continue
                pos = absolute_offset(el, score, part)
                highest_time = max(highest_time, pos + duration)
                accent, staccato = element_flags(el)
                note_rows.append(
                    [float(el.pitch.midi), duration, pos, float(part_idx), accent, staccato]
                )
            elif isinstance(el, music21.chord.Chord):
                duration = float(el.duration.quarterLength or 0.0)
                if duration <= 0:
                    continue
                pos = absolute_offset(el, score, part)
                highest_time = max(highest_time, pos + duration)
                accent, staccato = element_flags(el)
                for chord_note in el.notes:
                    note_rows.append(
                        [float(chord_note.pitch.midi), duration, pos, float(part_idx), accent, staccato]
                    )
    if not note_rows:
        raise ValueError(f"No pitched note events found in {score_path}")
    note_feats = np.asarray(note_rows, dtype=np.float32)
    beat_ids = np.floor(note_feats[:, 2] / max(float(beat_unit), eps)).astype(np.int32)
    num_beats = max(int(math.ceil(max(highest_time - eps, 0.0) / max(float(beat_unit), eps))), 1)
    boundary_probs = np.zeros(num_beats, dtype=np.float32)
    return {
        "note_feats": note_feats,
        "beat_ids": beat_ids,
        "boundary_probs": boundary_probs,
        "num_beats": int(num_beats),
        "beat_unit": float(beat_unit),
    }


def build_measure_map(score_path: Path, num_beats: int, beat_unit: float, expand_repeats: bool) -> pd.DataFrame:
    score = music21.converter.parse(str(score_path))
    if expand_repeats:
        try:
            score = score.expandRepeats()
        except Exception:
            pass
    primary_part = score.parts[0] if len(score.parts) > 0 else score
    measure_number = np.zeros(num_beats, dtype=np.int32)
    beat_in_measure = np.zeros(num_beats, dtype=np.int32)
    for measure in primary_part.getElementsByClass(music21.stream.Measure):
        measure_offset = absolute_offset(measure, score, primary_part)
        measure_len = float(getattr(measure.barDuration, "quarterLength", 0.0) or measure.duration.quarterLength or 0.0)
        if measure_len <= 0:
            continue
        start = int(math.floor(measure_offset / beat_unit))
        end = min(num_beats, int(math.ceil((measure_offset + measure_len) / beat_unit)))
        local_beats = max(end - start, 1)
        try:
            number = int(getattr(measure, "number", 0) or 0)
        except Exception:
            number = 0
        for beat_idx in range(start, end):
            measure_number[beat_idx] = number
            beat_in_measure[beat_idx] = beat_idx - start + 1
    return pd.DataFrame(
        {
            "beat_idx": np.arange(num_beats, dtype=np.int32),
            "measure_number": measure_number,
            "beat_in_measure": beat_in_measure,
        }
    )


def build_feature_frame_for_score(
    score_path: Path,
    piece_id: str,
    peak_cfg: PeakConfig,
    beat_unit: float,
    measure_cycle: int,
    symmetry_window: int,
    deviation_window: int,
    expand_repeats: bool,
) -> pd.DataFrame:
    arrays = score_to_npz_arrays(score_path=score_path, beat_unit=beat_unit, expand_repeats=expand_repeats)
    with tempfile.TemporaryDirectory(prefix="boundary_score_npz_") as tmpdir:
        tmp_path = Path(tmpdir) / f"{piece_id}.npz"
        np.savez(
            tmp_path,
            note_feats=arrays["note_feats"],
            beat_ids=arrays["beat_ids"],
            boundary_probs=arrays["boundary_probs"],
            num_beats=arrays["num_beats"],
            beat_unit=arrays["beat_unit"],
        )
        frame = build_beat_feature_frame(
            npz_path=tmp_path,
            peak_cfg=peak_cfg,
            split_cfg=None,
            long_note_threshold=1.0,
            beat_unit_fallback=beat_unit,
            symmetry_window=symmetry_window,
            deviation_window=deviation_window,
            measure_cycle=measure_cycle,
            xml_score_dir=None,
        )
    xml_frame = extract_xml_beat_features_from_path(
        xml_path=score_path,
        num_beats=int(arrays["num_beats"]),
        beat_unit=beat_unit,
        expand_repeats=expand_repeats,
    )
    beat_map = build_measure_map(
        score_path=score_path,
        num_beats=int(arrays["num_beats"]),
        beat_unit=beat_unit,
        expand_repeats=expand_repeats,
    )
    frame = frame.merge(xml_frame, on="beat_idx", how="left", validate="one_to_one")
    frame = frame.merge(beat_map, on="beat_idx", how="left", validate="one_to_one")
    frame["piece_id"] = piece_id
    frame["sample_id"] = piece_id
    frame["performer_id"] = "score_only"
    frame["protocol_split"] = "val"
    frame["split"] = "val"
    frame["source_score_path"] = str(score_path.resolve())
    frame = add_highlevel_derived_features(frame)
    return frame


def checkpoint_path_from_nested_dir(nested_dir: Path) -> Path:
    matches = sorted((nested_dir / "outer_test").rglob("detector_best.pt"))
    if not matches:
        raise FileNotFoundError(f"No outer_test detector_best.pt found under {nested_dir}")
    return matches[0]


def load_level_runtime(cfg: dict, clean_root: Path, detector_target: str, seed: int, device: torch.device) -> dict:
    summary_path = clean_root / f"weighted_topdown_merge56_{detector_target}_seed{seed}" / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing clean outer summary: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    checkpoint_path = checkpoint_path_from_nested_dir(Path(summary["nested_report_dir"]))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = build_sequence_model(
        str(checkpoint["model_type"]),
        input_dim=len(checkpoint["feature_columns"]),
        cfg=cfg,
        output_dim=1,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return {
        "detector_target": detector_target,
        "model": model,
        "feature_columns": list(checkpoint["feature_columns"]),
        "mean": np.asarray(checkpoint["mean"], dtype=np.float32),
        "std": np.asarray(checkpoint["std"], dtype=np.float32),
        "threshold": float(summary["frozen_threshold"]),
        "frozen_epochs": int(summary["frozen_epochs"]),
        "checkpoint_path": str(checkpoint_path.resolve()),
        "clean_summary_path": str(summary_path.resolve()),
    }


def make_prediction_sample(frame: pd.DataFrame, feature_columns: list[str]) -> dict:
    sample_frame = frame.copy()
    for col in feature_columns:
        if col not in sample_frame.columns:
            sample_frame[col] = 0.0
    sample_frame = sample_frame.sort_values("beat_idx").reset_index(drop=True)
    num_rows = len(sample_frame)
    return {
        "sample_id": sample_frame["sample_id"].iloc[0],
        "piece_id": sample_frame["piece_id"].iloc[0],
        "beat_idx": sample_frame["beat_idx"].to_numpy(dtype=np.int32),
        "features": sample_frame[feature_columns].to_numpy(dtype=np.float32),
        "union_target": np.zeros(num_rows, dtype=np.float32),
        "frequency_target": np.zeros(num_rows, dtype=np.float32),
        "train_union_target": np.zeros(num_rows, dtype=np.float32),
        "train_frequency_target": np.zeros(num_rows, dtype=np.float32),
        "performer_count": np.ones(num_rows, dtype=np.int32),
        "train_loss_factor": np.ones(num_rows, dtype=np.float32),
    }


def run_inference_for_level(
    frame: pd.DataFrame,
    runtime: dict,
    device: torch.device,
    batch_size: int,
    eval_cfg: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sample = make_prediction_sample(frame, runtime["feature_columns"])
    dataset = PieceUnionDataset([sample], mean=runtime["mean"], std=runtime["std"])
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_piece_union)
    pred_df = predict_detector(runtime["model"], loader, device=device)
    events = build_predicted_event_frame(
        pred_df=pred_df,
        threshold=float(runtime["threshold"]),
        min_distance=int(eval_cfg.get("min_distance", 6)),
        prominence=float(eval_cfg.get("prominence", 0.0)),
        tolerance=int(eval_cfg.get("event_tolerance", 1)),
    )
    beat_meta = frame[["beat_idx", "measure_number", "beat_in_measure"]].drop_duplicates("beat_idx")
    pred_df = pred_df.merge(beat_meta, on="beat_idx", how="left")
    if "beat_idx" not in events.columns:
        events = pd.DataFrame(
            columns=[
                "sample_id",
                "piece_id",
                "event_rank",
                "beat_idx",
                "detector_score",
                "threshold",
                "union_target_at_beat",
                "frequency_target_at_beat",
                "performer_count",
                "matched_union",
                "match_offset",
                "matched_true_beat_idx",
                "measure_number",
                "beat_in_measure",
            ]
        )
    else:
        events = events.merge(beat_meta, on="beat_idx", how="left")
    return pred_df, events


def resolve_targets(args: argparse.Namespace) -> list[tuple[str, Path]]:
    if not args.score:
        return [(piece_id, Path(path).resolve()) for piece_id, path in DEFAULT_TARGET_SCORES]
    score_paths = [Path(path).resolve() for path in args.score]
    if args.piece_id and len(args.piece_id) != len(score_paths):
        raise ValueError("--piece_id must match the number of --score paths")
    piece_ids = list(args.piece_id) if args.piece_id else [path.stem for path in score_paths]
    return list(zip(piece_ids, score_paths))


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    eval_cfg = cfg.get("evaluation", {})
    data_cfg = cfg.get("data", {})
    peak_cfg = PeakConfig(
        distance=int(data_cfg.get("peak_distance", 6)),
        height=float(data_cfg.get("peak_height", 0.15)),
        prominence=float(data_cfg.get("peak_prominence", 0.05)),
    )
    device = resolve_device(args.device)
    clean_root = DEFAULT_CLEAN_ROOT
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    runtimes = {
        label: load_level_runtime(cfg, clean_root=clean_root, detector_target=detector_target, seed=int(args.seed), device=device)
        for label, detector_target in LEVEL_SPECS.items()
    }
    targets = resolve_targets(args)
    expand_repeats = not args.no_expand_repeats
    measure_cycle = int(data_cfg.get("measure_cycle", 3))
    symmetry_window = int(data_cfg.get("symmetry_window", 4))
    deviation_window = int(data_cfg.get("deviation_window", 8))

    all_event_frames: list[pd.DataFrame] = []
    all_pred_frames: list[pd.DataFrame] = []
    summary_rows: list[dict] = []
    runtime_manifest = {
        "seed": int(args.seed),
        "config": str(Path(args.config).resolve()),
        "device": str(device),
        "levels": {},
    }

    for label, runtime in runtimes.items():
        runtime_manifest["levels"][label] = {
            "detector_target": runtime["detector_target"],
            "threshold": float(runtime["threshold"]),
            "frozen_epochs": int(runtime["frozen_epochs"]),
            "checkpoint_path": runtime["checkpoint_path"],
            "clean_summary_path": runtime["clean_summary_path"],
        }

    for piece_id, score_path in targets:
        if not score_path.exists():
            raise FileNotFoundError(f"Missing score: {score_path}")
        piece_frame = build_feature_frame_for_score(
            score_path=score_path,
            piece_id=piece_id,
            peak_cfg=peak_cfg,
            beat_unit=float(args.beat_unit),
            measure_cycle=measure_cycle,
            symmetry_window=symmetry_window,
            deviation_window=deviation_window,
            expand_repeats=expand_repeats,
        )
        piece_dir = output_dir / piece_id
        piece_dir.mkdir(parents=True, exist_ok=True)
        piece_frame.to_csv(piece_dir / "beat_features.csv.gz", index=False, compression="gzip")

        for label, runtime in runtimes.items():
            pred_df, events = run_inference_for_level(
                frame=piece_frame,
                runtime=runtime,
                device=device,
                batch_size=int(args.batch_size),
                eval_cfg=eval_cfg,
            )
            pred_df.insert(0, "level_label", label)
            pred_df.insert(1, "source_score_path", str(score_path))
            events.insert(0, "level_label", label)
            events.insert(1, "source_score_path", str(score_path))
            pred_df.to_csv(piece_dir / f"{label.replace('+', 'plus').replace('/', '_')}_predictions.csv.gz", index=False, compression="gzip")
            events.to_csv(piece_dir / f"{label.replace('+', 'plus').replace('/', '_')}_events.csv", index=False)
            all_pred_frames.append(pred_df)
            all_event_frames.append(events)
            summary_rows.append(
                {
                    "piece_id": piece_id,
                    "score_path": str(score_path),
                    "level_label": label,
                    "detector_target": runtime["detector_target"],
                    "threshold": float(runtime["threshold"]),
                    "predicted_event_count": int(len(events)),
                    "max_detector_score": float(pred_df["detector_score"].max()),
                    "mean_detector_score": float(pred_df["detector_score"].mean()),
                }
            )

    if all_pred_frames:
        pd.concat(all_pred_frames, ignore_index=True).to_csv(output_dir / "all_predictions.csv.gz", index=False, compression="gzip")
    if all_event_frames:
        pd.concat(all_event_frames, ignore_index=True).to_csv(output_dir / "all_events.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(output_dir / "summary.csv", index=False)
    (output_dir / "runtime_manifest.json").write_text(json.dumps(runtime_manifest, indent=2), encoding="utf-8")

    print(f"Wrote predictions to {output_dir}")
    for row in summary_rows:
        print(
            f"{row['piece_id']} | {row['level_label']} | threshold={row['threshold']:.3f} | "
            f"pred_events={row['predicted_event_count']}"
        )


if __name__ == "__main__":
    main()
