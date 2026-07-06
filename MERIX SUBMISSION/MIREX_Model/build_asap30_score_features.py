from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
ASAP_ROOT = ROOT / "asap-dataset-master"
MIREX = ROOT / "MERIX SUBMISSION" / "MIREX_Model"
BR = ROOT / "MERIX SUBMISSION" / "Boundary_Restart"
TOP_N = int(os.environ.get("ASAP_TOP_N", "30"))
DATASET_NAME = f"asap{TOP_N}"
LABEL_OUT = MIREX / f"{DATASET_NAME}_tempo_boundary_labels"
MANIFEST = LABEL_OUT / f"asap_top{TOP_N}_manifest.csv"
SUMMARY = LABEL_OUT / f"asap_top{TOP_N}_boundary_summary.csv"
BEAT_TABLE = LABEL_OUT / f"asap_top{TOP_N}_beat_table.csv.gz"
NOTE_DIR = MIREX / f"{DATASET_NAME}_score_note_feats"

for path in (MIREX, BR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from boundary_restart.xml_score_features import extract_xml_beat_features_from_path  # noqa: E402
from tokenizer_beat import build_note_features, extract_score_tokens  # noqa: E402


def load_asap_label_builder():
    path = MIREX / "build_asap30_tempo_boundary_labels.py"
    spec = importlib.util.spec_from_file_location(f"{DATASET_NAME}_label_builder_for_features", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[f"{DATASET_NAME}_label_builder_for_features"] = module
    spec.loader.exec_module(module)
    return module


label_builder = load_asap_label_builder()


def parse_meter_label(label: str) -> tuple[int, int]:
    text = str(label).strip()
    if "/" not in text:
        return 4, 4
    num, den = text.split("/", 1)
    return int(num), int(den)


def beat_unit_from_meter(meter_label: str, beats_per_measure: int) -> float:
    num, den = parse_meter_label(meter_label)
    measure_quarter = float(num) * 4.0 / float(den)
    return measure_quarter / max(int(beats_per_measure), 1)


def build_note_npz(piece_id: str, xml_path: Path, num_beats: int, beat_unit: float, meter_label: str, beats_per_measure: int) -> dict:
    tokens, _metadata = extract_score_tokens(xml_path, expand_repeats=False)
    note_feats, beat_ids, raw_num_beats = build_note_features(tokens, beat_unit=beat_unit)
    valid = (beat_ids >= 0) & (beat_ids < int(num_beats))
    dropped = int(np.size(beat_ids) - int(valid.sum()))
    note_feats = note_feats[valid].astype(np.float32)
    beat_ids = beat_ids[valid].astype(np.int32)
    order = np.lexsort((note_feats[:, 0], note_feats[:, 2], beat_ids)) if len(note_feats) else np.array([], dtype=int)
    note_feats = note_feats[order]
    beat_ids = beat_ids[order]

    num, den = parse_meter_label(meter_label)
    out_path = NOTE_DIR / f"{piece_id}_note_feats.npz"
    np.savez_compressed(
        out_path,
        piece_id=piece_id,
        score_path=str(xml_path),
        note_feats=note_feats.astype(np.float32),
        beat_ids=beat_ids.astype(np.int32),
        num_beats=int(num_beats),
        raw_num_beats=int(raw_num_beats),
        beat_unit=float(beat_unit),
        beats_per_measure=int(beats_per_measure),
        meter_label=str(meter_label),
        time_signature_numerator=int(num),
        time_signature_denominator=int(den),
        mixed_meter_by_segment=bool(True),
        segment_count=int(1),
        segment_index=np.asarray([1], dtype=np.int32),
        segment_numerator=np.asarray([int(beats_per_measure)], dtype=np.int32),
        segment_denominator=np.asarray([int(den)], dtype=np.int32),
        segment_beat_unit=np.asarray([float(beat_unit)], dtype=np.float32),
        segment_local_start_beat=np.asarray([0], dtype=np.int32),
        segment_global_start_beat=np.asarray([0], dtype=np.int32),
        segment_num_beats=np.asarray([int(num_beats)], dtype=np.int32),
    )
    return {
        "piece_id": piece_id,
        "score_path": str(xml_path),
        "num_beats": int(num_beats),
        "raw_num_beats": int(raw_num_beats),
        "beat_unit": float(beat_unit),
        "beats_per_measure": int(beats_per_measure),
        "meter_label": str(meter_label),
        "note_count": int(note_feats.shape[0]),
        "dropped_out_of_grid_notes": dropped,
        "output_npz": str(out_path),
    }


def main() -> None:
    NOTE_DIR.mkdir(parents=True, exist_ok=True)
    if not MANIFEST.exists() or not SUMMARY.exists():
        raise FileNotFoundError(f"Run ASAP_TOP_N={TOP_N} build_asap30_tempo_boundary_labels.py first.")

    manifest = pd.read_csv(MANIFEST)
    summary = pd.read_csv(SUMMARY)
    summary_by_piece = {str(row.piece_id): row for row in summary.itertuples(index=False)}

    rows: list[pd.DataFrame] = []
    note_rows: list[dict] = []
    for row in manifest.itertuples(index=False):
        piece_id = str(row.piece_id)
        srow = summary_by_piece[piece_id]
        num_beats = int(srow.num_beats)
        beats_per_measure = int(srow.beats_per_measure)
        meter_label = str(srow.meter_label)
        beat_unit = beat_unit_from_meter(meter_label, beats_per_measure)
        xml_path = Path(str(row.xml_score))
        if not xml_path.is_absolute():
            xml_path = ASAP_ROOT / xml_path

        feat = extract_xml_beat_features_from_path(
            xml_path=xml_path,
            num_beats=num_beats,
            beat_unit=beat_unit,
            expand_repeats=False,
        )
        feat.insert(0, "source_path", str(xml_path))
        feat.insert(1, "sample_id", piece_id)
        feat.insert(2, "piece_id", piece_id)
        feat.insert(3, "performer_id", "score")
        feat.insert(4, "level", 0)
        feat["split"] = "all"
        feat["num_beats"] = int(num_beats)
        feat["boundary_prob"] = 0.0
        feat["boundary_peak"] = 0.0
        rows.append(feat)
        note_rows.append(build_note_npz(piece_id, xml_path, num_beats, beat_unit, meter_label, beats_per_measure))
        print(f"{piece_id}: beats={num_beats} beat_unit={beat_unit:g} meter={meter_label} xml_features={feat.shape[1]}")

    table = pd.concat(rows, ignore_index=True)
    table.to_csv(BEAT_TABLE, index=False, compression="gzip")
    pd.DataFrame(note_rows).to_csv(NOTE_DIR / f"{DATASET_NAME}_score_note_feats_summary.csv", index=False)
    (NOTE_DIR / "metadata.json").write_text(
        json.dumps(
            {
                "beat_table": str(BEAT_TABLE),
                "manifest": str(MANIFEST),
                "summary": str(SUMMARY),
                "beat_unit_rule": "time_signature_measure_quarter_length / beats_per_measure",
                "expand_repeats": False,
                "note_dir": str(NOTE_DIR),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {BEAT_TABLE}")
    print(f"wrote {NOTE_DIR}")


if __name__ == "__main__":
    main()
