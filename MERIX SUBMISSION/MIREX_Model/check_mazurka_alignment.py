import argparse
import csv
import re
from pathlib import Path

import numpy as np


def extract_mazurka_id(name: str) -> str | None:
    m = re.search(r"(?i)M(\d+)-(\d+)", name)
    if not m:
        return None
    opus = int(m.group(1))
    num = int(m.group(2))
    return f"M{opus:02d}-{num}"


def count_csv_rows(path: Path) -> int:
    with path.open(newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)
    if not rows:
        return 0
    return max(0, len(rows) - 1)


def load_npz_info(path: Path) -> dict:
    data = np.load(path)
    beat_ids = data["beat_ids"].astype(np.int64)
    num_beats = int(data["num_beats"]) if "num_beats" in data else int(beat_ids.max() + 1)
    boundary_len = int(data["boundary_probs"].shape[0]) if "boundary_probs" in data else 0
    beat_unit = float(data["beat_unit"]) if "beat_unit" in data else None
    valid = beat_ids >= 0
    if valid.any():
        unique_beats = np.unique(beat_ids[valid])
        min_beat = int(unique_beats.min())
        max_beat = int(unique_beats.max())
        uniq_count = int(unique_beats.size)
    else:
        min_beat = max_beat = -1
        uniq_count = 0
    missing = max(num_beats - uniq_count, 0)
    coverage = (uniq_count / num_beats) if num_beats > 0 else 0.0
    return {
        "num_beats": num_beats,
        "boundary_len": boundary_len,
        "min_beat": min_beat,
        "max_beat": max_beat,
        "uniq_count": uniq_count,
        "missing": missing,
        "coverage": coverage,
        "note_count": int(beat_ids.shape[0]),
        "beat_unit": beat_unit,
    }


def main():
    parser = argparse.ArgumentParser(description="Check Mazurka alignment across npz/boundary/beat_time.")
    parser.add_argument("--npz_dir", default=None, help="Directory with *.npz (default: beat_data_mazurka)")
    parser.add_argument("--boundary_dir", default=None, help="Directory with *_boundary_prob.csv")
    parser.add_argument("--beat_time_dir", default=None, help="Directory with *beat_time.csv")
    parser.add_argument("--show_all", action="store_true", help="Print all pieces, not only mismatches")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    mazurka_root = repo_root / "MazurkaBL-master"

    npz_dir = Path(args.npz_dir) if args.npz_dir else Path(__file__).resolve().parent / "beat_data_mazurka"
    boundary_dir = Path(args.boundary_dir) if args.boundary_dir else repo_root / "out" / "mazurka_boundary_probs"
    beat_time_dir = Path(args.beat_time_dir) if args.beat_time_dir else mazurka_root / "beat_time"

    npz_map = {}
    for p in npz_dir.glob("*.npz"):
        mid = extract_mazurka_id(p.stem)
        if mid:
            npz_map[mid] = p

    boundary_map = {}
    for p in boundary_dir.glob("*_boundary_prob.csv"):
        mid = extract_mazurka_id(p.stem)
        if mid:
            boundary_map[mid] = p

    beat_time_map = {}
    for p in beat_time_dir.glob("*beat_time.csv"):
        mid = extract_mazurka_id(p.stem)
        if mid:
            beat_time_map[mid] = p

    ids = sorted(set(npz_map) | set(boundary_map) | set(beat_time_map))
    if not ids:
        print("No matching Mazurka IDs found.")
        return

    mismatches = 0
    for mid in ids:
        npz_path = npz_map.get(mid)
        boundary_path = boundary_map.get(mid)
        beat_time_path = beat_time_map.get(mid)

        info = {}
        if npz_path and npz_path.exists():
            info.update(load_npz_info(npz_path))
        if boundary_path and boundary_path.exists():
            info["boundary_rows"] = count_csv_rows(boundary_path)
        if beat_time_path and beat_time_path.exists():
            info["beat_time_rows"] = count_csv_rows(beat_time_path)

        flags = []
        if "boundary_rows" in info and "num_beats" in info and info["boundary_rows"] != info["num_beats"]:
            flags.append("boundary!=num_beats")
        if "beat_time_rows" in info and "boundary_rows" in info and info["beat_time_rows"] != info["boundary_rows"]:
            flags.append("beat_time!=boundary")
        if "min_beat" in info and info["min_beat"] > 0:
            flags.append("min_beat>0")
        if "max_beat" in info and "num_beats" in info and info["max_beat"] >= info["num_beats"]:
            flags.append("max_beat>=num_beats")
        if "coverage" in info and info["coverage"] < 0.98:
            flags.append("low_coverage")

        if flags or args.show_all:
            print(
                f"{mid}: "
                f"npz_beats={info.get('num_beats')} "
                f"boundary_rows={info.get('boundary_rows')} "
                f"beat_time_rows={info.get('beat_time_rows')} "
                f"beat_id_range=[{info.get('min_beat')},{info.get('max_beat')}] "
                f"missing_beats={info.get('missing')} "
                f"coverage={info.get('coverage'):.3f} "
                f"beat_unit={info.get('beat_unit')} "
                f"notes={info.get('note_count')} "
                f"{' | '.join(flags) if flags else ''}"
            )
            if flags:
                mismatches += 1

    print(f"\nChecked {len(ids)} pieces. Mismatches: {mismatches}")


if __name__ == "__main__":
    main()
