#!/usr/bin/env python3
"""
check_beat_alignment.py

Quick sanity check for beat-index alignment between:
  - note-level npz (note_feats + beat_ids + num_beats)
  - boundary_prob_by_beat.csv (beat_index, boundary_probability)

It reports:
  - basic stats (beat range, missing beats, etc.)
  - shift sweep: try shifting labels by [-max_shift, +max_shift]
    and report how labels align with beats that actually contain notes.
"""

import argparse
import csv
from pathlib import Path
from typing import Tuple

import numpy as np


def load_boundary_csv(path: Path) -> Tuple[np.ndarray, int, int, int]:
    indices = []
    values = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            indices.append(int(row["beat_index"]))
            values.append(float(row["boundary_probability"]))

    if not indices:
        raise ValueError(f"No rows found in {path}")

    min_idx = min(indices)
    max_idx = max(indices)
    arr = np.zeros(max_idx + 1, dtype=np.float32)
    for idx, val in zip(indices, values):
        if 0 <= idx < arr.size:
            arr[idx] = val

    return arr, min_idx, max_idx, len(indices)


def shift_labels(labels: np.ndarray, shift: int, num_beats: int) -> np.ndarray:
    out = np.zeros(num_beats, dtype=np.float32)
    if shift >= 0:
        if shift < num_beats:
            out[shift:] = labels[: num_beats - shift]
    else:
        k = -shift
        if k < num_beats:
            out[: num_beats - k] = labels[k:]
    return out


def main():
    parser = argparse.ArgumentParser(description="Check beat alignment between note npz and boundary csv.")
    parser.add_argument("--note_npz", required=True, help="Path to note-level npz (note_feats, beat_ids, num_beats)")
    parser.add_argument("--boundary_csv", required=True, help="Path to boundary_prob_by_beat.csv")
    parser.add_argument("--max_shift", type=int, default=8, help="Sweep shifts in [-max_shift, max_shift]")
    args = parser.parse_args()

    note_path = Path(args.note_npz)
    boundary_path = Path(args.boundary_csv)

    data = np.load(note_path)
    if "note_feats" not in data or "beat_ids" not in data:
        raise KeyError("note_npz must contain note_feats and beat_ids")

    note_feats = data["note_feats"]
    beat_ids = data["beat_ids"].astype(np.int64)
    num_beats = int(data["num_beats"]) if "num_beats" in data else int(beat_ids.max() + 1)
    beat_unit = float(data["beat_unit"]) if "beat_unit" in data else None

    boundary, min_idx, max_idx, row_count = load_boundary_csv(boundary_path)
    labels = np.zeros(num_beats, dtype=np.float32)
    n = min(num_beats, boundary.size)
    labels[:n] = boundary[:n]

    valid_mask = (beat_ids >= 0) & (beat_ids < num_beats)
    valid_beat_ids = beat_ids[valid_mask]

    counts = np.zeros(num_beats, dtype=np.int64)
    np.add.at(counts, valid_beat_ids, 1)

    note_mask = counts > 0
    empty_mask = ~note_mask

    print("=== Basic Stats ===")
    print(f"note_npz: {note_path}")
    print(f"boundary_csv: {boundary_path}")
    print(f"num_beats: {num_beats}")
    print(f"beat_unit: {beat_unit if beat_unit is not None else 'N/A'}")
    print(f"beat_ids range: [{int(valid_beat_ids.min())}, {int(valid_beat_ids.max())}]")
    print(f"beats with notes: {note_mask.sum()} | empty beats: {empty_mask.sum()}")
    print(f"boundary index range: [{min_idx}, {max_idx}] | rows: {row_count}")
    print(f"boundary length used: {n} (padded to num_beats)")

    print("\n=== Shift Sweep ===")
    print("shift\tmean_note\tmean_empty\tnonzero_on_note\tcorr_note_count")
    best = {"shift": 0, "diff": -1e9}
    best_corr = {"shift": 0, "corr": -1e9}
    best_nz = {"shift": 0, "frac": -1e9}

    for shift in range(-args.max_shift, args.max_shift + 1):
        shifted = shift_labels(labels, shift, num_beats)

        mean_note = shifted[note_mask].mean() if note_mask.any() else 0.0
        mean_empty = shifted[empty_mask].mean() if empty_mask.any() else 0.0
        diff = mean_note - mean_empty

        nonzero = shifted > 0
        nz_total = nonzero.sum()
        nz_on_note = (nonzero & note_mask).sum()
        nz_frac = (nz_on_note / nz_total) if nz_total else 0.0

        corr = 0.0
        if counts.std() > 0 and shifted.std() > 0:
            corr = float(np.corrcoef(counts, shifted)[0, 1])

        print(f"{shift:+d}\t{mean_note:.6f}\t{mean_empty:.6f}\t{nz_frac:.3f}\t\t{corr:.4f}")

        if diff > best["diff"]:
            best = {"shift": shift, "diff": diff}
        if corr > best_corr["corr"]:
            best_corr = {"shift": shift, "corr": corr}
        if nz_frac > best_nz["frac"]:
            best_nz = {"shift": shift, "frac": nz_frac}

    print("\n=== Suggested Shifts (heuristics) ===")
    print(f"best mean_note - mean_empty: shift {best['shift']:+d} (diff {best['diff']:.6f})")
    print(f"best corr(label, note_count): shift {best_corr['shift']:+d} (corr {best_corr['corr']:.4f})")
    print(f"best nonzero_on_note fraction: shift {best_nz['shift']:+d} (frac {best_nz['frac']:.3f})")


if __name__ == "__main__":
    main()
