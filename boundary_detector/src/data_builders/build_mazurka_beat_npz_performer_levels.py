import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

BUILDERS = Path(__file__).resolve().parent
if str(BUILDERS) not in sys.path:
    sys.path.insert(0, str(BUILDERS))

from tokenizer_beat import extract_score_tokens, build_note_features


def normalize_mazurka_id(raw_id: str) -> str:
    m = re.match(r"^M(\d+)-(\d+)$", raw_id)
    if not m:
        return ""
    opus = int(m.group(1))
    num = int(m.group(2))
    return f"M{opus:02d}-{num}"


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


def load_beat_time(file_path: Path):
    df = pd.read_csv(file_path)
    meta_cols = {"Unnamed: 0", "measure_number", "beat_number"}
    performer_cols = [c for c in df.columns if c not in meta_cols]
    return df, performer_cols


def compute_tempo_curves(df, performer_cols, smooth_window=3, bpm_range=(0, 5000), clip_max=600):
    tempo_arrays = {}
    lo, hi = bpm_range
    for col in performer_cols:
        times = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        dt = np.diff(times, prepend=np.nan)
        tempo = 60.0 / dt
        invalid = (dt <= 0) | (~np.isfinite(tempo))
        tempo[invalid] = np.nan

        s = pd.Series(tempo)
        s = s.where((s > lo) & (s < hi))
        s = s.interpolate("linear", limit_direction="both")
        s = s.rolling(window=smooth_window, center=True, min_periods=1).mean()
        s = s.clip(upper=clip_max)
        tempo_arrays[col] = s.to_numpy()
    return tempo_arrays


def _pad_to_multiple(x, m):
    x = np.asarray(x, dtype=float).reshape(-1)
    pad = (-len(x)) % m
    if pad == 0:
        return x
    return np.concatenate([x, np.zeros(pad, dtype=float)])


def group_analysis_hierarchy(tempo_curve, str_vec, enforce_nested=True):
    """
    Block-hierarchy + trace:
      - Level 1 groups beats into blocks of size str_vec[0]
      - Level i>=2 computes eng2 per block: RMS(norm) - std(norm), finds valleys on eng2
      - Trace from top-level valleys down to beat indices
    Returns:
      results_raw: (L, N) bool, row t means boundaries traced from top=t (1..L)
      level_sets: dict level->np.array beat indices
                 if enforce_nested=True: B_L ⊆ ... ⊆ B_1, with L=coarsest, 1=finest
    """
    tempo = np.asarray(tempo_curve, dtype=float).reshape(-1)
    n = len(tempo)
    str_vec = np.asarray(str_vec, dtype=int).reshape(-1)
    L = len(str_vec)

    if n == 0:
        return np.zeros((L, 0), dtype=bool), {l: np.array([], dtype=int) for l in range(1, L + 1)}

    avg = np.nanmean(tempo)
    if not np.isfinite(avg) or avg == 0:
        avg = 1.0

    energy = [None] * (L + 1)
    energy2 = [None] * (L + 1)
    valleys = [None] * (L + 1)

    s1 = int(str_vec[0])
    pad_raw = _pad_to_multiple(tempo, s1)
    pad_norm = _pad_to_multiple(tempo / avg, s1)

    energy[1] = pad_raw.reshape((s1, -1), order="F")
    energy2[1] = pad_norm.reshape((s1, -1), order="F")

    v1, _ = find_peaks(-tempo)
    valleys[1] = v1 + 1

    for i in range(2, L + 1):
        t_eng = energy[i - 1].copy()
        t_eng[t_eng == 0] = np.nan

        t_norm = t_eng / avg
        eng2 = np.sqrt(np.nanmean(t_norm ** 2, axis=0)) - np.nanstd(t_norm, axis=0)

        vi, _ = find_peaks(-eng2)
        valleys[i] = vi + 1

        si = int(str_vec[i - 1])
        eng2_pad = _pad_to_multiple(eng2, si)

        energy[i] = _pad_to_multiple(np.sqrt(np.nanmean(t_eng ** 2, axis=0)), si).reshape(
            (si, -1), order="F"
        )
        energy2[i] = eng2_pad.reshape((si, -1), order="F")

    results_raw = np.zeros((L, n), dtype=bool)

    for top in range(L, 0, -1):
        roots = valleys[top]
        if roots is None or len(roots) == 0:
            continue

        trace = np.zeros((top, len(roots)), dtype=int)
        trace[0, :] = roots

        for rj in range(len(roots)):
            for i in range(2, top + 1):
                lvl = top + 1 - i
                parent_col = trace[i - 2, rj]
                colvec = energy2[lvl][:, parent_col - 1]
                row = int(np.nanargmin(colvec)) + 1
                trace[i - 1, rj] = int(str_vec[lvl - 1]) * (parent_col - 1) + row

        beats_1b = trace[top - 1, :]
        beats_0b = beats_1b - 1
        beats_0b = beats_0b[(beats_0b >= 0) & (beats_0b < n)]
        results_raw[top - 1, beats_0b] = True

    level_sets = {l: np.where(results_raw[l - 1])[0] for l in range(1, L + 1)}

    if enforce_nested:
        cum = np.zeros(n, dtype=bool)
        nested = {}
        for l in range(L, 0, -1):
            cum |= results_raw[l - 1]
            nested[l] = np.where(cum)[0]
        level_sets = nested

    return results_raw, level_sets


def boundaries_to_mask(n_beats, boundary_indices):
    mask = np.zeros(n_beats, dtype=np.float32)
    boundary_indices = np.asarray(boundary_indices, dtype=int)
    boundary_indices = boundary_indices[(boundary_indices >= 0) & (boundary_indices < n_beats)]
    mask[boundary_indices] = 1.0
    return mask


def build_score_beat_features(note_feats: np.ndarray, beat_ids: np.ndarray, num_beats: int) -> np.ndarray:
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


def parse_ws_list(text: str) -> list[int]:
    parts = [p.strip() for p in text.split(",") if p.strip()]
    return [int(p) for p in parts]


def main():
    parser = argparse.ArgumentParser(
        description="Build per-performer, per-level beat npz from beat_time tempo curves."
    )
    parser.add_argument(
        "--beat_time_dir",
        default=None,
        help="Directory with *beat_time.csv (default: <repo>/data/raw/MazurkaBL/beat_time).",
    )
    parser.add_argument(
        "--xml_dir",
        default=None,
        help="Directory with Mazurka XML scores (default: <repo>/data/raw/MazurkaBL/xml_scores).",
    )
    parser.add_argument(
        "--out_dir",
        default=None,
        help="Output dir for *.npz (default: ./beat_data_mazurka_performer_levels).",
    )
    parser.add_argument(
        "--csv_dir",
        default=None,
        help="Output dir for boundary CSVs (default: ./beat_data_mazurka_performer).",
    )
    parser.add_argument(
        "--beat_unit",
        type=float,
        default=1.0,
        help="Beat unit in quarterLength (default: 1.0).",
    )
    parser.add_argument(
        "--str_vec",
        default="3,2,2,2,2,2",
        help="Comma-separated hierarchy strides per level (default: 3,2,2,2,2,2).",
    )
    parser.add_argument(
        "--smooth_window",
        type=int,
        default=3,
        help="Rolling mean window for tempo smoothing (default: 3).",
    )
    parser.add_argument(
        "--bpm_min",
        type=float,
        default=0.0,
        help="Min BPM for filtering (default: 0).",
    )
    parser.add_argument(
        "--bpm_max",
        type=float,
        default=5000.0,
        help="Max BPM for filtering (default: 5000).",
    )
    parser.add_argument(
        "--clip_max",
        type=float,
        default=600.0,
        help="Clip tempo to max BPM (default: 600).",
    )
    parser.add_argument(
        "--append_beat_features",
        action="store_true",
        help="Append score-only beat features to note_feats (default: True).",
    )
    parser.add_argument(
        "--no_append_beat_features",
        action="store_true",
        help="Do not append beat features.",
    )
    parser.add_argument(
        "--expand_repeats",
        action="store_true",
        help="Expand repeats when parsing MusicXML.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    beat_time_dir = (
        Path(args.beat_time_dir)
        if args.beat_time_dir
        else repo_root / "data" / "raw" / "MazurkaBL" / "beat_time"
    )
    xml_dir = (
        Path(args.xml_dir)
        if args.xml_dir
        else repo_root / "data" / "raw" / "MazurkaBL" / "xml_scores"
    )
    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else Path(__file__).resolve().parent / "beat_data_mazurka_performer_levels"
    )
    csv_dir = (
        Path(args.csv_dir)
        if args.csv_dir
        else Path(__file__).resolve().parent / "beat_data_mazurka_performer"
    )

    if not beat_time_dir.exists():
        raise FileNotFoundError(f"beat_time_dir not found: {beat_time_dir}")
    if not xml_dir.exists():
        raise FileNotFoundError(f"xml_dir not found: {xml_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)
    str_vec = parse_ws_list(args.str_vec)
    if not str_vec:
        raise ValueError("str_vec is empty; provide --str_vec like '3,2,2,2,2,2'.")

    xml_map = build_xml_map(xml_dir)

    level_stats = {i + 1: {"pos": 0, "total": 0, "files": 0} for i in range(len(str_vec))}
    skipped = []
    total_written = 0

    for file_path in sorted(beat_time_dir.glob("*beat_time.csv")):
        raw_id = file_path.name.replace("beat_time.csv", "")
        mazurka_id = normalize_mazurka_id(raw_id)
        if not mazurka_id:
            skipped.append((file_path.name, "bad_id"))
            continue
        xml_path = xml_map.get(mazurka_id)
        if xml_path is None:
            skipped.append((file_path.name, "missing_xml"))
            continue

        df_bt, performer_cols = load_beat_time(file_path)
        if not performer_cols:
            skipped.append((file_path.name, "no_performers"))
            continue

        tempo_arrays = compute_tempo_curves(
            df_bt,
            performer_cols,
            smooth_window=args.smooth_window,
            bpm_range=(args.bpm_min, args.bpm_max),
            clip_max=args.clip_max,
        )
        num_beats = len(df_bt)
        if num_beats <= 0:
            skipped.append((file_path.name, "empty_beats"))
            continue

        tokens, _ = extract_score_tokens(xml_path, expand_repeats=args.expand_repeats)
        if not tokens:
            skipped.append((file_path.name, "no_tokens"))
            continue
        note_feats, beat_ids, _ = build_note_features(tokens, beat_unit=args.beat_unit)

        valid = (beat_ids >= 0) & (beat_ids < num_beats)
        if not np.any(valid):
            skipped.append((file_path.name, "no_valid_notes"))
            continue
        note_feats = note_feats[valid]
        beat_ids = beat_ids[valid]

        append_beat_features = args.append_beat_features and not args.no_append_beat_features
        if append_beat_features:
            beat_feats = build_score_beat_features(note_feats, beat_ids, num_beats)
            beat_ids_safe = np.clip(beat_ids, 0, num_beats - 1)
            note_feats = np.concatenate([note_feats, beat_feats[beat_ids_safe]], axis=1)

        avg_sum = np.zeros(num_beats, dtype=np.float64)
        performer_count = 0

        for perf_id, curve in tempo_arrays.items():
            _, level_sets = group_analysis_hierarchy(curve, str_vec=str_vec, enforce_nested=True)
            performer_count += 1
            for level_idx in range(1, len(str_vec) + 1):
                locs = level_sets.get(level_idx, np.array([], dtype=int))
                boundary = boundaries_to_mask(num_beats, locs)
                out_path = out_dir / f"{mazurka_id}_{perf_id}_L{level_idx}.npz"
                np.savez(
                    out_path,
                    note_feats=note_feats.astype(np.float32),
                    beat_ids=beat_ids.astype(np.int32),
                    boundary_probs=boundary.astype(np.float32),
                    num_beats=int(num_beats),
                    beat_unit=float(args.beat_unit),
                    level=int(level_idx),
                    level_ws=int(str_vec[level_idx - 1]),
                    performer_id=str(perf_id),
                    mazurka_id=str(mazurka_id),
                )
                total_written += 1
                stats = level_stats[level_idx]
                stats["pos"] += int(boundary.sum())
                stats["total"] += int(num_beats)
                stats["files"] += 1

                csv_path = csv_dir / f"{mazurka_id}_{perf_id}_L{level_idx}.csv"
                pd.DataFrame(
                    {
                        "beat_index": np.arange(num_beats, dtype=int),
                        "boundary_probability": boundary.astype(float),
                    }
                ).to_csv(csv_path, index=False)

            avg_sum += boundaries_to_mask(num_beats, level_sets.get(1, np.array([], dtype=int)))

        if performer_count > 0:
            avg_prob = avg_sum / float(performer_count)
            avg_path = csv_dir / f"{mazurka_id}_boundary_prob.csv"
            pd.DataFrame(
                {
                    "beat_index": np.arange(num_beats, dtype=int),
                    "boundary_probability": avg_prob.astype(float),
                }
            ).to_csv(avg_path, index=False)

    print(f"Wrote {total_written} npz files to {out_dir}")
    if skipped:
        print("Skipped:")
        for name, reason in skipped:
            print(f"  - {name}: {reason}")
    for level_idx, stats in sorted(level_stats.items()):
        if stats["total"] == 0:
            continue
        pos = stats["pos"]
        total = stats["total"]
        pos_ratio = pos / float(total)
        pos_weight = (total - pos) / float(pos) if pos > 0 else float("inf")
        print(f"Level {level_idx}: files={stats['files']} pos_ratio={pos_ratio:.6f} pos_weight={pos_weight:.6f}")


if __name__ == "__main__":
    main()
