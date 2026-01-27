import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np

from tokenizer_beat import extract_score_tokens, build_note_features


def normalize_mazurka_id(raw_id: str) -> str:
    match = re.match(r"^M(\d+)-(\d+)$", raw_id)
    if not match:
        return ""
    opus = int(match.group(1))
    num = int(match.group(2))
    return f"M{opus:02d}-{num}"


def parse_beat_unit_arg(value: str) -> float:
    if value is None:
        return 1.0
    v = value.strip().lower()
    if v == "auto":
        return 1.0
    return float(value)


def build_xml_map(xml_dir: Path):
    xml_map = {}
    pattern = re.compile(r"(?i)mazurka0*(\d+)-(\d+)")
    for xml_path in xml_dir.glob("*.xml"):
        m = pattern.search(xml_path.stem)
        if not m:
            continue
        opus = int(m.group(1))
        num = int(m.group(2))
        key = f"M{opus:02d}-{num}"
        if key not in xml_map:
            xml_map[key] = xml_path
    return xml_map


def read_boundary_csv(csv_path: Path) -> np.ndarray:
    indices = []
    probs = []
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            indices.append(int(row["beat_index"]))
            probs.append(float(row["boundary_probability"]))
    if not indices:
        return np.zeros(0, dtype=np.float32)
    if indices == list(range(len(indices))):
        return np.asarray(probs, dtype=np.float32)
    max_idx = max(indices)
    arr = np.zeros(max_idx + 1, dtype=np.float32)
    for idx, prob in zip(indices, probs):
        if 0 <= idx <= max_idx:
            arr[idx] = prob
    return arr


def build_score_beat_features(note_feats: np.ndarray, beat_ids: np.ndarray, num_beats: int) -> np.ndarray:
    """
    Build score-only beat-level features from note-level features:
      - note_count_norm
      - mean_pitch_norm
      - pitch_range_norm
      - mean_duration_norm
      - accent_ratio
      - staccato_ratio
    """
    feats = np.zeros((num_beats, 6), dtype=np.float32)
    if num_beats <= 0 or note_feats.size == 0:
        return feats

    valid = (beat_ids >= 0) & (beat_ids < num_beats)
    if not np.any(valid):
        return feats

    b = beat_ids[valid]
    f = note_feats[valid]

    counts = np.zeros(num_beats, dtype=np.float32)
    sum_pitch = np.zeros(num_beats, dtype=np.float32)
    sum_dur = np.zeros(num_beats, dtype=np.float32)
    sum_acc = np.zeros(num_beats, dtype=np.float32)
    sum_stc = np.zeros(num_beats, dtype=np.float32)
    min_pitch = np.full(num_beats, np.inf, dtype=np.float32)
    max_pitch = np.full(num_beats, -np.inf, dtype=np.float32)

    np.add.at(counts, b, 1.0)
    np.add.at(sum_pitch, b, f[:, 0])
    np.add.at(sum_dur, b, f[:, 1])
    np.add.at(sum_acc, b, f[:, 4])
    np.add.at(sum_stc, b, f[:, 5])
    np.minimum.at(min_pitch, b, f[:, 0])
    np.maximum.at(max_pitch, b, f[:, 0])

    mean_pitch = np.divide(sum_pitch, counts, out=np.zeros_like(sum_pitch), where=counts > 0)
    mean_dur = np.divide(sum_dur, counts, out=np.zeros_like(sum_dur), where=counts > 0)
    acc_ratio = np.divide(sum_acc, counts, out=np.zeros_like(sum_acc), where=counts > 0)
    stc_ratio = np.divide(sum_stc, counts, out=np.zeros_like(sum_stc), where=counts > 0)

    pitch_range = np.where(counts > 0, max_pitch - min_pitch, 0.0)
    max_count = float(counts.max()) if counts.size > 0 else 1.0

    count_norm = counts / max(max_count, 1.0)
    mean_pitch_norm = mean_pitch / 127.0
    pitch_range_norm = pitch_range / 127.0
    mean_dur_norm = mean_dur / 8.0

    feats = np.stack(
        [
            np.clip(count_norm, 0.0, 1.0),
            np.clip(mean_pitch_norm, 0.0, 1.0),
            np.clip(pitch_range_norm, 0.0, 1.0),
            np.clip(mean_dur_norm, 0.0, 1.0),
            np.clip(acc_ratio, 0.0, 1.0),
            np.clip(stc_ratio, 0.0, 1.0),
        ],
        axis=1,
    ).astype(np.float32)
    return feats


def main():
    parser = argparse.ArgumentParser(
        description="Build beat-level training npz files for MazurkaBL (note_feats + beat_ids + boundary_probs)."
    )
    parser.add_argument(
        "--boundary_dir",
        default=None,
        help="Directory with *_boundary_prob.csv files (default: <repo>/out/mazurka_boundary_probs).",
    )
    parser.add_argument(
        "--xml_dir",
        default=None,
        help="Directory with Mazurka XML scores (default: <repo>/MazurkaBL-master/xml_scores).",
    )
    parser.add_argument(
        "--out_dir",
        default=None,
        help="Output directory for *.npz files (default: ./beat_data_mazurka).",
    )
    parser.add_argument(
        "--beat_unit",
        default="1.0",
        help="Beat unit in quarterLength (default: 1.0).",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    mazurka_root = repo_root / "MazurkaBL-master"

    boundary_dir = Path(args.boundary_dir) if args.boundary_dir else repo_root / "out" / "mazurka_boundary_probs"
    xml_dir = Path(args.xml_dir) if args.xml_dir else mazurka_root / "xml_scores"
    out_dir = Path(args.out_dir) if args.out_dir else Path(__file__).resolve().parent / "beat_data_mazurka"
    if not boundary_dir.exists():
        raise FileNotFoundError(f"boundary_dir not found: {boundary_dir}")
    if not xml_dir.exists():
        raise FileNotFoundError(f"xml_dir not found: {xml_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)

    xml_map = build_xml_map(xml_dir)
    boundary_files = sorted(boundary_dir.glob("*_boundary_prob.csv"))
    if not boundary_files:
        raise FileNotFoundError(f"No *_boundary_prob.csv files found in {boundary_dir}")

    feature_meta = {
        "base_features": [
            "pitch_midi",
            "duration",
            "position",
            "part_idx",
            "is_accent",
            "is_staccato",
        ],
        "score_beat_features": [
            "note_count_norm",
            "mean_pitch_norm",
            "pitch_range_norm",
            "mean_duration_norm",
            "accent_ratio",
            "staccato_ratio",
        ],
        "total_dim": 12,
    }
    (out_dir / "feature_meta.json").write_text(json.dumps(feature_meta, indent=2), encoding="utf-8")

    skipped = []
    mismatched = []
    total = 0

    beat_unit_arg = parse_beat_unit_arg(args.beat_unit)

    for csv_path in boundary_files:
        stem = csv_path.stem
        if not stem.endswith("_boundary_prob"):
            continue
        raw_id = stem[: -len("_boundary_prob")]
        mazurka_id = normalize_mazurka_id(raw_id)
        if not mazurka_id:
            skipped.append((csv_path.name, "bad_id"))
            continue
        xml_path = xml_map.get(mazurka_id)
        if xml_path is None:
            skipped.append((csv_path.name, "missing_xml"))
            continue

        tokens, _ = extract_score_tokens(xml_path, expand_repeats=True)
        if not tokens:
            skipped.append((csv_path.name, "no_tokens"))
            continue

        beat_unit = beat_unit_arg
        note_feats, beat_ids, num_beats = build_note_features(tokens, beat_unit=beat_unit)

        boundary_probs = read_boundary_csv(csv_path)
        if boundary_probs.size == 0:
            skipped.append((csv_path.name, "empty_boundary"))
            continue
        target_beats = int(boundary_probs.shape[0])
        if num_beats != target_beats:
            mismatched.append((mazurka_id, num_beats, target_beats, abs(num_beats - target_beats)))
        num_beats = target_beats

        valid = (beat_ids >= 0) & (beat_ids < num_beats)
        if not np.any(valid):
            skipped.append((csv_path.name, "no_valid_notes"))
            continue
        if not np.all(valid):
            beat_ids = beat_ids[valid]
            note_feats = note_feats[valid]

        beat_feats = build_score_beat_features(note_feats, beat_ids, num_beats)
        beat_ids_safe = np.clip(beat_ids, 0, num_beats - 1)
        note_feats = np.concatenate([note_feats, beat_feats[beat_ids_safe]], axis=1)

        out_path = out_dir / f"{mazurka_id}.npz"
        np.savez(
            out_path,
            note_feats=note_feats.astype(np.float32),
            beat_ids=beat_ids.astype(np.int32),
            boundary_probs=boundary_probs,
            num_beats=int(num_beats),
            beat_unit=float(beat_unit),
        )
        total += 1

    print(f"Wrote {total} npz files to {out_dir}")
    if skipped:
        print("Skipped:")
        for name, reason in skipped:
            print(f"  - {name}: {reason}")
    if mismatched:
        print("Length mismatches (note-derived beats vs boundary rows):")
        for mazurka_id, num_beats, boundary_rows, diff in mismatched:
            print(f"  - {mazurka_id}: num_beats={num_beats}, boundary_rows={boundary_rows}, diff={diff}")


if __name__ == "__main__":
    main()
