"""
Prepare a tiny beat-level dataset from existing smoke_data for sanity check.

We reuse smoke_data/sample.json to build a single-file beat dataset with:
  - score_feats: simple per-beat features [beats, feature_dim]
  - boundary_probs: random probs [beats]
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

    # Build a very simple beat-level representation: here we just take positions (beats) and durations
    positions = [t["score_note_token"]["position"] for t in full_tokens]
    durations = [t["score_note_token"]["duration"] for t in full_tokens]
    pitches = [t["performance_note_token"]["pitch"] for t in full_tokens]

    # To keep it trivial, we sort by position and group unique beat positions
    # Feature vector: [avg_pitch/128, avg_duration, beat_index_norm]
    beats = sorted(set(positions))
    feats = []
    for i, beat in enumerate(beats):
        idxs = [j for j, p in enumerate(positions) if p == beat]
        avg_pitch = np.mean([pitches[j] for j in idxs]) / 128.0
        avg_dur = np.mean([durations[j] for j in idxs])
        beat_norm = i / max(len(beats) - 1, 1)
        feats.append([avg_pitch, avg_dur, beat_norm])

    feats = np.array(feats, dtype=np.float32)  # [beats, 3]
    boundary_probs = np.random.rand(len(beats)).astype(np.float32)  # random supervision

    out_path = out_dir / "sample.npz"
    np.savez(out_path, score_feats=feats, boundary_probs=boundary_probs)
    print(f"Wrote smoke beat data to {out_path} with shape {feats.shape}")


if __name__ == "__main__":
    main()
