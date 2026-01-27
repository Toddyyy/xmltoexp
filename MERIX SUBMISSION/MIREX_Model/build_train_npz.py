import csv
from pathlib import Path

import numpy as np


def main():
    note = np.load("/Users/toddywang/Documents/VsCodeProjects/xmltoexp/out/mephisto_note_data.npz")
    note_feats = note["note_feats"]
    beat_ids = note["beat_ids"].astype(np.int32)
    num_beats = int(beat_ids.max()) + 1 if beat_ids.size else 0

    boundary = np.zeros(num_beats, dtype=np.float32)
    with open("/Users/toddywang/Documents/VsCodeProjects/xmltoexp/boundary_prob_by_beat.csv", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            idx = int(row["beat_index"])
            prob = float(row["boundary_probability"])
            if 0 <= idx < num_beats:
                boundary[idx] = prob

    out_dir = Path("/Users/toddywang/Documents/VsCodeProjects/xmltoexp/MERIX SUBMISSION/MIREX_Model/beat_data_full")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "mephisto_train.npz"
    np.savez(out_path, note_feats=note_feats, beat_ids=beat_ids, boundary_probs=boundary, num_beats=num_beats)
    print("saved", out_path, "notes", note_feats.shape[0], "beats", num_beats)


if __name__ == "__main__":
    main()
