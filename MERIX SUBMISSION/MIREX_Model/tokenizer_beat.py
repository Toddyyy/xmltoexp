#!/usr/bin/env python3
"""
tokenizer_beat.py

Beat-level tokenizer for MusicXML (score-only) using music21.

What it does:
 1) Parse MusicXML, read note onset (absolute in score), duration, pitch, part, articulations, tie.
 2) Expand chords into multiple pitch events.
 3) Sort by absolute time for deterministic ordering.
 4) Build note-level numeric features + beat_ids.
 5) Save .npz (note_feats, beat_ids, num_beats). Optionally save .json with tokens + metadata.

Key fix vs the buggy version:
- Use *absolute* onset time across the whole score:
    offset = el.getOffsetInHierarchy(score)
  instead of el.offset (which can reset inside each measure).

Beat semantics:
- beat_unit is in quarterLength units:
    1.0 => quarter note grid
    0.5 => eighth note grid (recommended for 3/8 if you want "beat=8th")

Outputs:
  - .npz: note_feats [N, 6], beat_ids [N], num_beats (int)
  - optional .json: metadata + score_tokens list

Example:
  python tokenizer_beat.py --xml_path path/to/score.musicxml \
    --output_path out/note_data.npz --json_out out/tokens.json --beat_unit 0.5
"""

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import music21
import numpy as np


def extract_score_tokens(xml_path: Path, expand_repeats: bool = False) -> Tuple[List[Dict], Dict]:
    """
    Parse MusicXML and return sorted score tokens + metadata.

    Each token corresponds to a single pitch event (notes + chord tones expanded).
    position is ABSOLUTE onset in quarterLength across the entire score.
    """
    score = music21.converter.parse(str(xml_path))
    if expand_repeats:
        try:
            score = score.expandRepeats()
        except Exception:
            pass
    tokens: List[Dict] = []

    for part_idx, part in enumerate(score.parts):
        part_id = str(part.id) if part.id is not None else f"P{part_idx + 1}"

        for el in part.recurse().notes:  # includes Chord; excludes Rest
            # ✅ critical: absolute onset in score (quarterLength)
            try:
                offset = float(el.getOffsetInHierarchy(score))
            except Exception:
                # fallback: best-effort (still better than measure-local el.offset)
                flat_part = part.flatten()
                offset = float(flat_part.elementOffset(el))

            duration = float(el.quarterLength)
            tie = str(el.tie.type) if getattr(el, "tie", None) else None
            measure = el.getContextByClass(music21.stream.Measure)
            measure_number = int(measure.number) if measure is not None and measure.number is not None else None
            measure_length = None
            measure_progress = None
            if measure is not None:
                try:
                    measure_length = float(measure.barDuration.quarterLength)
                except Exception:
                    measure_length = None
                try:
                    measure_offset = float(el.getOffsetBySite(measure))
                except Exception:
                    measure_offset = float(el.offset)
                if measure_length and measure_length > 0:
                    measure_progress = max(0.0, min(measure_offset / measure_length, 1.0))

            articulations = [type(a).__name__.lower() for a in el.articulations]
            is_accent = any("accent" in a for a in articulations)
            is_staccato = any("staccat" in a for a in articulations)

            # Expand chord to per-pitch tokens
            if el.isChord:
                pitches = el.pitches
            else:
                pitches = [el.pitch]

            for pitch in pitches:
                token = {
                    "position": offset,
                    "duration": duration,
                    "pitch_name": pitch.name,  # e.g., C#, Eb (music21 may use '-' for flat)
                    "octave": pitch.octave,
                    "accidental": pitch.accidental.name if pitch.accidental else "natural",
                    "pitch_midi": int(pitch.midi),
                    "part_id": part_id,
                    "part_idx": part_idx,
                    "measure_number": measure_number,
                    "measure_progress": measure_progress,
                    "tie": tie,
                    "is_accent": is_accent,
                    "is_staccato": is_staccato,
                }
                tokens.append(token)

    # sort by onset, then by part_id to ensure deterministic order
    tokens.sort(key=lambda t: (t["position"], t["part_id"], t["pitch_midi"]))

    metadata = {
        "score_path": str(xml_path),
        "num_tokens": len(tokens),
    }
    return tokens, metadata


def build_note_features(tokens: List[Dict], beat_unit: float) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Build note-level features and beat_ids.

    Features (6-dim):
      pitch_midi, duration, position, part_idx, is_accent, is_staccato

    beat_ids:
      Quantize position onto a beat grid of size beat_unit (quarterLength).
      We use floor((pos + eps)/beat_unit) for stability (avoids round() issues).

    num_beats:
      Compute from max end time to include final note sustain:
        ceil(max(position + duration)/beat_unit)
      (More robust than max(beat_id)+1; also accounts for last note duration.)
    """
    if beat_unit <= 0:
        raise ValueError("beat_unit must be > 0")

    note_feats: List[List[float]] = []
    beat_ids: List[int] = []
    eps = 1e-9

    max_end = 0.0
    for t in tokens:
        pos = float(t["position"])
        dur = float(t["duration"])

        beat_idx = int(math.floor((pos + eps) / beat_unit))
        beat_ids.append(beat_idx)

        note_feats.append([
            float(t["pitch_midi"]),
            dur,
            pos,
            float(t["part_idx"]),
            1.0 if t["is_accent"] else 0.0,
            1.0 if t["is_staccato"] else 0.0,
        ])

        max_end = max(max_end, pos + dur)

    note_feats_arr = np.array(note_feats, dtype=np.float32)
    beat_ids_arr = np.array(beat_ids, dtype=np.int32)

    num_beats = int(math.ceil((max_end + eps) / beat_unit)) if len(tokens) > 0 else 0
    return note_feats_arr, beat_ids_arr, num_beats


def main():
    parser = argparse.ArgumentParser(description="Tokenize MusicXML into note-level features + beat ids.")
    parser.add_argument("--xml_path", required=True, help="Path to MusicXML file")
    parser.add_argument("--output_path", required=True, help="Path to save .npz output")
    parser.add_argument("--json_out", default=None, help="Optional path to save score tokens JSON")
    parser.add_argument("--beat_unit", type=float, default=1.0, help="QuarterLength per beat (default 1.0)")
    parser.add_argument("--expand_repeats", action="store_true", help="Expand repeats before tokenizing")
    args = parser.parse_args()

    xml_path = Path(args.xml_path)
    if not xml_path.exists():
        raise FileNotFoundError(f"XML not found: {xml_path}")

    tokens, metadata = extract_score_tokens(xml_path, expand_repeats=args.expand_repeats)
    if not tokens:
        raise ValueError(f"No notes found in {xml_path} (rests are ignored).")

    note_feats, beat_ids, num_beats = build_note_features(tokens, beat_unit=args.beat_unit)

    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, note_feats=note_feats, beat_ids=beat_ids, num_beats=num_beats, beat_unit=float(args.beat_unit))

    print(f"Saved note features to {out_path}")
    print(f"notes={note_feats.shape[0]} | feat_dim={note_feats.shape[1]}")
    print(f"beat_unit={args.beat_unit} quarterLength | num_beats={num_beats} | beat_id_range=[{int(beat_ids.min())},{int(beat_ids.max())}]")

    if args.json_out:
        json_path = Path(args.json_out)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        metadata["num_beats"] = num_beats
        metadata["beat_unit"] = float(args.beat_unit)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({"metadata": metadata, "score_tokens": tokens}, f, ensure_ascii=False, indent=2)
        print(f"Saved tokens JSON to {json_path}")


if __name__ == "__main__":
    main()
