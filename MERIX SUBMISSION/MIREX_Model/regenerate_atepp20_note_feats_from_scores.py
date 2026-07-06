from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
BOUNDARY_ROOT = ROOT / "MERIX SUBMISSION" / "Boundary_Restart"
MIREX_ROOT = ROOT / "MERIX SUBMISSION" / "MIREX_Model"
METER_AUTO_ROOT = ROOT / "MERIX SUBMISSION" / "MIREX_Model_meter_auto"

for path in (BOUNDARY_ROOT, MIREX_ROOT, METER_AUTO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from build_atepp_auto_meter_training_table import read_meter_segments  # noqa: E402
from predict_new_scores_merge56_seed44 import score_to_npz_arrays  # noqa: E402


MANIFEST_PATH = METER_AUTO_ROOT / "outputs" / "atepp_top20_mixed_by_segment_manifest.csv"
BEAT_TABLE_PATH = METER_AUTO_ROOT / "outputs" / "atepp_top20_mixed_by_segment_nonan_beat_table.csv.gz"
OUT_DIR = MIREX_ROOT / "atepp20_regenerated_note_feats"


def _score_to_mixed_meter_arrays(
    score_path: Path,
    *,
    expand_repeats: bool,
) -> tuple[np.ndarray, np.ndarray, int, list[dict]]:
    """Mirror build_mixed_meter_feature_frame's per-segment beat grid.

    The ATEPP mixed-meter table was not made on one global quarter-note grid.
    Each meter segment was tokenized with its own beat_unit, sliced in that
    local grid, and then concatenated into global beat_idx. We do the same here
    so regenerated note_feats/beat_ids line up with the existing labels.
    """
    segments = read_meter_segments(
        score_path,
        expand_repeats=expand_repeats,
        hierarchy_depth=6,
        hierarchy_first_group_mode="numerator",
    )
    num_beats = int(sum(int(segment.num_beats) for segment in segments))
    all_feats: list[np.ndarray] = []
    all_beat_ids: list[np.ndarray] = []
    for segment in segments:
        arrays = score_to_npz_arrays(
            score_path=score_path,
            beat_unit=float(segment.beat_unit),
            expand_repeats=expand_repeats,
        )
        segment_feats = np.asarray(arrays["note_feats"], dtype=np.float32)
        segment_beat_ids = np.asarray(arrays["beat_ids"], dtype=np.int32)
        local_start = int(segment.local_start_beat)
        local_end = local_start + int(segment.num_beats)
        keep = (segment_beat_ids >= local_start) & (segment_beat_ids < local_end)
        if not np.any(keep):
            continue
        global_ids = (
            int(segment.global_start_beat)
            + segment_beat_ids[keep].astype(np.int32)
            - local_start
        )
        all_feats.append(segment_feats[keep])
        all_beat_ids.append(global_ids.astype(np.int32))
    segment_rows = [
        {
            "index": int(segment.index),
            "signature": segment.signature,
            "beat_unit": float(segment.beat_unit),
            "start_quarter": float(segment.start_quarter),
            "end_quarter": float(segment.end_quarter),
            "local_start_beat": int(segment.local_start_beat),
            "global_start_beat": int(segment.global_start_beat),
            "num_beats": int(segment.num_beats),
        }
        for segment in segments
    ]
    if not all_feats:
        return (
            np.zeros((0, 6), dtype=np.float32),
            np.zeros((0,), dtype=np.int32),
            num_beats,
            segment_rows,
        )
    return (
        np.concatenate(all_feats, axis=0).astype(np.float32),
        np.concatenate(all_beat_ids, axis=0).astype(np.int32),
        num_beats,
        segment_rows,
    )


def _expected_note_counts() -> dict[str, float]:
    if not BEAT_TABLE_PATH.exists():
        return {}
    usecols = ["piece_id", "sample_id", "level", "note_count"]
    frame = pd.read_csv(BEAT_TABLE_PATH, usecols=usecols)
    per_sample = (
        frame.groupby(["piece_id", "sample_id", "level"], sort=False)["note_count"]
        .sum()
        .reset_index()
    )
    return (
        per_sample.groupby("piece_id")["note_count"]
        .median()
        .astype(float)
        .to_dict()
    )


def _single_signature(text: str) -> tuple[int, int]:
    parts = str(text).strip().split()
    if len(parts) != 1 or "/" not in parts[0]:
        return 0, 0
    num, den = parts[0].split("/", 1)
    return int(num), int(den)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(MANIFEST_PATH)
    expected_counts = _expected_note_counts()
    rows: list[dict] = []
    expand_repeats = True

    for row in manifest.to_dict("records"):
        piece_id = str(row["piece_id"])
        score_path = Path(str(row["score_path"]))
        mixed = bool(row.get("mixed_meter_by_segment", False))
        expected_beats = int(row["num_beats"])
        if mixed:
            note_feats, beat_ids, num_beats, segments = _score_to_mixed_meter_arrays(
                score_path,
                expand_repeats=expand_repeats,
            )
            beat_unit = np.nan
            ts_num, ts_den = 0, 0
        else:
            beat_unit = float(row["beat_unit"])
            ts_num, ts_den = _single_signature(str(row["time_signature"]))
            single = score_to_npz_arrays(
                score_path=score_path,
                beat_unit=beat_unit,
                expand_repeats=expand_repeats,
            )
            note_feats = np.asarray(single["note_feats"], dtype=np.float32)
            beat_ids = np.asarray(single["beat_ids"], dtype=np.int32)
            num_beats = int(single["num_beats"])
            segments = []

        valid = (beat_ids >= 0) & (beat_ids < expected_beats)
        dropped = int(np.size(beat_ids) - int(valid.sum()))
        if dropped:
            note_feats = note_feats[valid]
            beat_ids = beat_ids[valid]
        num_beats = expected_beats
        order = np.lexsort((note_feats[:, 0], note_feats[:, 2], beat_ids))
        note_feats = note_feats[order]
        beat_ids = beat_ids[order]

        out_path = OUT_DIR / f"{piece_id}_note_feats.npz"
        np.savez_compressed(
            out_path,
            piece_id=piece_id,
            score_path=str(score_path),
            note_feats=note_feats.astype(np.float32),
            beat_ids=beat_ids.astype(np.int32),
            num_beats=int(num_beats),
            beat_unit=float(beat_unit) if not mixed else np.nan,
            time_signature_numerator=int(ts_num),
            time_signature_denominator=int(ts_den),
            mixed_meter_by_segment=bool(mixed),
            segment_count=int(len(segments)),
            segment_index=np.asarray([r["index"] for r in segments], dtype=np.int32),
            segment_numerator=np.asarray([str(r["signature"]).split("/")[0] for r in segments], dtype=np.int32),
            segment_denominator=np.asarray([str(r["signature"]).split("/")[1] for r in segments], dtype=np.int32),
            segment_beat_unit=np.asarray([r["beat_unit"] for r in segments], dtype=np.float32),
            segment_local_start_beat=np.asarray([r["local_start_beat"] for r in segments], dtype=np.int32),
            segment_global_start_beat=np.asarray([r["global_start_beat"] for r in segments], dtype=np.int32),
            segment_num_beats=np.asarray([r["num_beats"] for r in segments], dtype=np.int32),
        )

        expected_notes = expected_counts.get(piece_id, np.nan)
        rows.append(
            {
                "piece_id": piece_id,
                "score_path": str(score_path),
                "mixed_meter_by_segment": bool(mixed),
                "num_beats": int(num_beats),
                "manifest_num_beats": int(expected_beats),
                "beat_count_match": int(num_beats) == int(expected_beats),
                "note_count": int(note_feats.shape[0]),
                "beat_table_note_count_median": expected_notes,
                "note_count_delta_vs_beat_table": (
                    float(note_feats.shape[0]) - float(expected_notes)
                    if not pd.isna(expected_notes)
                    else np.nan
                ),
                "dropped_out_of_grid_notes": dropped,
                "output_npz": str(out_path),
            }
        )
        print(
            f"{piece_id}: notes={note_feats.shape[0]} beats={num_beats} "
            f"mixed={mixed} dropped={dropped}"
        )

    summary = pd.DataFrame(rows)
    summary.to_csv(OUT_DIR / "regenerated_note_feats_summary.csv", index=False)
    print(f"wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
