import argparse
import ast
import re
from pathlib import Path

import numpy as np


def parse_cp_list(path: Path) -> list[list[int]]:
    cps = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                values = ast.literal_eval(line)
            except Exception:
                continue
            if isinstance(values, (list, tuple)):
                cps.append([int(v) for v in values])
    return cps


def get_mazurka_id(path: Path) -> str:
    m = re.search(r"(M\d{2}-\d+)", path.stem)
    return m.group(1) if m else ""


def get_performer_ids(sones_dir: Path, mazurka_id: str) -> list[str]:
    folder = sones_dir / mazurka_id
    if not folder.exists():
        return []
    ids = []
    for p in sorted(folder.glob("*.csv")):
        stem = p.stem
        if "Ntot" in stem:
            stem = stem.split("Ntot")[0]
        ids.append(stem)
    return ids


def build_boundary_mask(num_beats: int, cp_list: list[int]) -> np.ndarray:
    mask = np.zeros(num_beats, dtype=np.float32)
    for idx in cp_list:
        b = int(idx) - 1  # cp list is 1-based
        if 0 <= b < num_beats:
            mask[b] = 1.0
    return mask


def main():
    parser = argparse.ArgumentParser(
        description="Build per-performer Mazurka npz files using 0/1 boundary labels."
    )
    parser.add_argument(
        "--base_npz_dir",
        default=None,
        help="Directory with base *.npz (note_feats + beat_ids). Default: ./beat_data_mazurka",
    )
    parser.add_argument(
        "--cp_dir",
        default=None,
        help="Directory with cp_list_from_R_PELT_*.txt (per performer change points).",
    )
    parser.add_argument(
        "--sones_dir",
        default=None,
        help="Directory with per-mazurka sone files (used to get performer IDs).",
    )
    parser.add_argument(
        "--out_dir",
        default=None,
        help="Output directory for per-performer *.npz (default: ./beat_data_mazurka_performer).",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    base_npz_dir = (
        Path(args.base_npz_dir)
        if args.base_npz_dir
        else Path(__file__).resolve().parent / "beat_data_mazurka"
    )
    cp_dir = (
        Path(args.cp_dir)
        if args.cp_dir
        else repo_root / "MazurkaBL-master" / "change_points_data" / "cp_per_maz_rec_PELT"
    )
    sones_dir = (
        Path(args.sones_dir)
        if args.sones_dir
        else repo_root / "MazurkaBL-master" / "sones"
    )
    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else Path(__file__).resolve().parent / "beat_data_mazurka_performer"
    )

    if not base_npz_dir.exists():
        raise FileNotFoundError(f"base_npz_dir not found: {base_npz_dir}")
    if not cp_dir.exists():
        raise FileNotFoundError(f"cp_dir not found: {cp_dir}")
    if not sones_dir.exists():
        raise FileNotFoundError(f"sones_dir not found: {sones_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)

    cp_files = sorted(cp_dir.glob("cp_list_from_R_PELT_*.txt"))
    if not cp_files:
        raise FileNotFoundError(f"No cp_list_from_R_PELT_*.txt files found in {cp_dir}")

    total_written = 0
    total_pos = 0
    total_beats = 0
    skipped = []

    for cp_path in cp_files:
        mazurka_id = get_mazurka_id(cp_path)
        if not mazurka_id:
            skipped.append((cp_path.name, "bad_id"))
            continue

        base_npz = base_npz_dir / f"{mazurka_id}.npz"
        if not base_npz.exists():
            skipped.append((cp_path.name, "missing_base_npz"))
            continue

        cps = parse_cp_list(cp_path)
        performer_ids = get_performer_ids(sones_dir, mazurka_id)
        if not cps or not performer_ids:
            skipped.append((cp_path.name, "missing_cps_or_performers"))
            continue

        if len(cps) != len(performer_ids):
            print(
                f"Warning: {mazurka_id} cps={len(cps)} performers={len(performer_ids)}; using min length."
            )

        n = min(len(cps), len(performer_ids))
        data = np.load(base_npz)
        note_feats = data["note_feats"]
        beat_ids = data["beat_ids"]
        num_beats = int(data["num_beats"]) if "num_beats" in data else int(beat_ids.max() + 1)
        beat_unit = float(data["beat_unit"]) if "beat_unit" in data else 1.0

        for i in range(n):
            pid = performer_ids[i]
            boundary = build_boundary_mask(num_beats, cps[i])
            out_path = out_dir / f"{mazurka_id}_{pid}.npz"
            np.savez(
                out_path,
                note_feats=note_feats.astype(np.float32),
                beat_ids=beat_ids.astype(np.int32),
                boundary_probs=boundary.astype(np.float32),
                num_beats=int(num_beats),
                beat_unit=float(beat_unit),
            )
            total_written += 1
            total_pos += int(boundary.sum())
            total_beats += int(num_beats)

    print(f"Wrote {total_written} npz files to {out_dir}")
    if skipped:
        print("Skipped:")
        for name, reason in skipped:
            print(f"  - {name}: {reason}")
    if total_beats > 0:
        p = total_pos / float(total_beats)
        if total_pos > 0:
            pos_weight = (total_beats - total_pos) / float(total_pos)
        else:
            pos_weight = float("inf")
        print(f"pos_ratio={p:.6f} pos_weight={(pos_weight):.6f}")


if __name__ == "__main__":
    main()
