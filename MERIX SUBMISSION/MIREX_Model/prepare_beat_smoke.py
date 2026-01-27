"""
Prepare a tiny note-level dataset from existing smoke_data for sanity check.

We reuse smoke_data/sample.json to build a single-file note dataset with:
  - note_feats: [notes, feature_dim]
  - beat_ids: [notes]
  - boundary_probs: [beats] random probs
  - num_beats: int
Saved to beat_data_smoke/sample.npz
"""

import json
from pathlib import Path
import numpy as np


def main():
    root = Path(__file__).parent
    src = root / "smoke_data" / "sample.json"
    out_dir = root / "beat_data_smoke"
    out_dir.mkdir(parents=True, exist_ok=True)

    data = json.load(open(src))
    full_tokens = data["full_tokens"]

    # Build simple note-level features:
    # [pitch_midi, duration, position, part_idx, is_accent, is_staccato]
    positions = [t["score_note_token"]["position"] for t in full_tokens]
    durations = [t["score_note_token"]["duration"] for t in full_tokens]
    pitches = [t["performance_note_token"]["pitch"] for t in full_tokens]
    part_ids = [t["score_note_token"]["part_id"] for t in full_tokens]
    is_accent = [1.0 if t["score_note_token"].get("is_accent") else 0.0 for t in full_tokens]
    is_staccato = [1.0 if t["score_note_token"].get("is_staccato") else 0.0 for t in full_tokens]

    # Map part_id to index
    part_vocab = {pid: idx for idx, pid in enumerate(sorted(set(part_ids)))}
    part_idx = [float(part_vocab[p]) for p in part_ids]

    note_feats = np.array(
        list(zip(pitches, durations, positions, part_idx, is_accent, is_staccato)),
        dtype=np.float32,
    )

    beat_unit = 1.0
    beat_ids = np.array([int(round(p / beat_unit)) for p in positions], dtype=np.int32)
    num_beats = int(beat_ids.max() + 1) if beat_ids.size > 0 else 0
    boundary_probs = np.random.rand(num_beats).astype(np.float32)  # random supervision

    out_path = out_dir / "sample.npz"
    np.savez(out_path, note_feats=note_feats, beat_ids=beat_ids, boundary_probs=boundary_probs, num_beats=num_beats)
    print(f"Wrote smoke note data to {out_path} with shape {note_feats.shape}, beats={num_beats}")


if __name__ == "__main__":
    main()
