from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from .table_io import assign_split, extract_level, extract_performer_id, extract_piece_id
from .xml_score_features import extract_xml_beat_features


@dataclass
class PeakConfig:
    distance: int = 6
    height: float = 0.15
    prominence: float = 0.05


BASE_NOTE_FEATURES = [
    "pitch_midi",
    "duration",
    "position",
    "part_idx",
    "is_accent",
    "is_staccato",
]

LEVEL_SUFFIX_RE = re.compile(r"_L(\d+)$")


def load_boundary_npz(path: Path, beat_unit_fallback: float = 1.0) -> dict:
    with np.load(path) as data:
        note_feats = np.asarray(data["note_feats"], dtype=np.float32)
        beat_ids = np.asarray(data["beat_ids"], dtype=np.int32)
        boundary_probs = np.asarray(data["boundary_probs"], dtype=np.float32)
        num_beats = int(data["num_beats"]) if "num_beats" in data else int(boundary_probs.shape[0])
        beat_unit = float(data["beat_unit"]) if "beat_unit" in data else float(beat_unit_fallback)
    return {
        "note_feats": note_feats,
        "beat_ids": beat_ids,
        "boundary_probs": boundary_probs[:num_beats],
        "num_beats": num_beats,
        "beat_unit": beat_unit,
    }


def replace_level_suffix(path: Path, level: int) -> Path:
    stem = path.stem
    match = LEVEL_SUFFIX_RE.search(stem)
    if match is None:
        raise ValueError(f"{path} does not end with a level suffix like _L5")
    base = stem[: match.start()]
    return path.with_name(f"{base}_L{int(level)}{path.suffix}")


def boundary_probs_to_binary(boundary_probs: np.ndarray, cfg: PeakConfig) -> np.ndarray:
    boundary_probs = np.asarray(boundary_probs, dtype=np.float32)
    labels = np.zeros(boundary_probs.shape[0], dtype=np.float32)
    if boundary_probs.size == 0:
        return labels
    finite = boundary_probs[np.isfinite(boundary_probs)]
    if finite.size > 0 and np.all((finite == 0.0) | (finite == 1.0)):
        return (boundary_probs > 0.5).astype(np.float32)
    peaks, _ = find_peaks(
        boundary_probs,
        distance=max(int(cfg.distance), 1),
        height=float(cfg.height),
        prominence=float(cfg.prominence),
    )
    if peaks.size > 0:
        labels[peaks.astype(int)] = 1.0
    return labels


def _safe_divide(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    return np.divide(num, den, out=np.zeros_like(num, dtype=np.float32), where=den > 0)


def _cosine_similarity(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < eps or nb < eps:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _cosine_adjacent(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    num_rows = matrix.shape[0]
    prev_sim = np.zeros(num_rows, dtype=np.float32)
    next_sim = np.zeros(num_rows, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1)
    for idx in range(num_rows):
        if idx > 0 and norms[idx] > 0 and norms[idx - 1] > 0:
            prev_sim[idx] = float(np.dot(matrix[idx], matrix[idx - 1]) / (norms[idx] * norms[idx - 1]))
        if idx + 1 < num_rows and norms[idx] > 0 and norms[idx + 1] > 0:
            next_sim[idx] = float(np.dot(matrix[idx], matrix[idx + 1]) / (norms[idx] * norms[idx + 1]))
    return prev_sim.astype(np.float32), next_sim.astype(np.float32)


def _barline_proximity(beat_idx: np.ndarray, cycle: int) -> np.ndarray:
    phase = np.mod(beat_idx, cycle).astype(np.float32)
    return np.minimum(phase, cycle - phase) / max(cycle / 2.0, 1.0)


def _rolling_mean(x: np.ndarray, window: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    out = np.zeros_like(x, dtype=np.float32)
    n = x.shape[0]
    for idx in range(n):
        start = max(0, idx - window)
        end = min(n, idx + window + 1)
        out[idx] = float(np.mean(x[start:end])) if end > start else 0.0
    return out


def _local_delta(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    out = np.zeros_like(x, dtype=np.float32)
    n = x.shape[0]
    for idx in range(n):
        prev = x[idx] - x[idx - 1] if idx - 1 >= 0 else 0.0
        nxt = x[idx + 1] - x[idx] if idx + 1 < n else 0.0
        out[idx] = 0.5 * (abs(prev) + abs(nxt))
    return out


def _window_symmetry(matrix: np.ndarray, window: int) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    num_rows = matrix.shape[0]
    out = np.zeros(num_rows, dtype=np.float32)
    for idx in range(num_rows):
        if idx - window < 0 or idx + window >= num_rows:
            continue
        pre = matrix[idx - window:idx].reshape(-1)
        post = matrix[idx + 1:idx + 1 + window].reshape(-1)
        out[idx] = _cosine_similarity(pre, post)
    return out


def build_beat_feature_frame(
    npz_path: Path,
    peak_cfg: PeakConfig,
    split_cfg: dict[str, set[str]] | None = None,
    long_note_threshold: float = 1.0,
    beat_unit_fallback: float = 1.0,
    symmetry_window: int = 4,
    deviation_window: int = 8,
    measure_cycle: int = 3,
    xml_score_dir: Path | None = None,
    xml_expand_repeats: bool = True,
) -> pd.DataFrame:
    loaded = load_boundary_npz(npz_path, beat_unit_fallback=beat_unit_fallback)
    note_feats = loaded["note_feats"]
    beat_ids = loaded["beat_ids"]
    boundary_probs = loaded["boundary_probs"]
    num_beats = loaded["num_beats"]
    beat_unit = loaded["beat_unit"]

    if note_feats.shape[1] < 6:
        raise ValueError(f"{npz_path} has feature_dim={note_feats.shape[1]} < 6")

    valid = (beat_ids >= 0) & (beat_ids < num_beats)
    note_feats = note_feats[valid]
    beat_ids = beat_ids[valid]

    pitch = note_feats[:, 0]
    duration = note_feats[:, 1]
    position = note_feats[:, 2]
    part_idx = note_feats[:, 3]
    is_accent = note_feats[:, 4]
    is_staccato = note_feats[:, 5]

    counts = np.zeros(num_beats, dtype=np.float32)
    sum_pitch = np.zeros(num_beats, dtype=np.float32)
    sum_duration = np.zeros(num_beats, dtype=np.float32)
    sum_pitch_sq = np.zeros(num_beats, dtype=np.float32)
    sum_duration_sq = np.zeros(num_beats, dtype=np.float32)
    sum_accent = np.zeros(num_beats, dtype=np.float32)
    sum_staccato = np.zeros(num_beats, dtype=np.float32)
    sum_part = np.zeros(num_beats, dtype=np.float32)
    min_pitch = np.full(num_beats, np.inf, dtype=np.float32)
    max_pitch = np.full(num_beats, -np.inf, dtype=np.float32)
    max_duration = np.zeros(num_beats, dtype=np.float32)
    onset_sum = np.zeros(num_beats, dtype=np.float32)
    onset_sq_sum = np.zeros(num_beats, dtype=np.float32)
    onset_min_local = np.full(num_beats, np.inf, dtype=np.float32)
    onset_max_local = np.full(num_beats, -np.inf, dtype=np.float32)
    onset_min_abs = np.full(num_beats, np.inf, dtype=np.float32)
    onset_max_end_abs = np.full(num_beats, -np.inf, dtype=np.float32)
    long_note_count = np.zeros(num_beats, dtype=np.float32)
    release_count = np.zeros(num_beats, dtype=np.float32)
    long_release_count = np.zeros(num_beats, dtype=np.float32)
    carry_diff = np.zeros(num_beats + 1, dtype=np.float32)
    pitchclass_hist = np.zeros((num_beats, 12), dtype=np.float32)
    unique_parts = [set() for _ in range(num_beats)]

    eps = 1e-9
    for idx, beat in enumerate(beat_ids.tolist()):
        p = float(pitch[idx])
        d = float(duration[idx])
        pos = float(position[idx])
        part = int(round(float(part_idx[idx])))
        beat_float = pos / max(beat_unit, eps)
        onset_local = beat_float - beat
        note_end = pos + d

        counts[beat] += 1.0
        sum_pitch[beat] += p
        sum_duration[beat] += d
        sum_pitch_sq[beat] += p * p
        sum_duration_sq[beat] += d * d
        sum_accent[beat] += float(is_accent[idx])
        sum_staccato[beat] += float(is_staccato[idx])
        sum_part[beat] += float(part)
        min_pitch[beat] = min(min_pitch[beat], p)
        max_pitch[beat] = max(max_pitch[beat], p)
        max_duration[beat] = max(max_duration[beat], d)
        onset_sum[beat] += onset_local
        onset_sq_sum[beat] += onset_local * onset_local
        onset_min_local[beat] = min(onset_min_local[beat], onset_local)
        onset_max_local[beat] = max(onset_max_local[beat], onset_local)
        onset_min_abs[beat] = min(onset_min_abs[beat], pos)
        onset_max_end_abs[beat] = max(onset_max_end_abs[beat], note_end)
        unique_parts[beat].add(part)
        pitchclass_hist[beat, int(p) % 12] += 1.0

        if d >= long_note_threshold:
            long_note_count[beat] += 1.0

        end_beat = int(math.floor(max(note_end - eps, 0.0) / max(beat_unit, eps)))
        end_beat = min(max(end_beat, 0), num_beats - 1)
        release_count[end_beat] += 1.0
        if d >= long_note_threshold:
            long_release_count[end_beat] += 1.0

        first_carry = beat + 1
        last_carry = int(math.ceil(max(note_end - eps, 0.0) / max(beat_unit, eps))) - 1
        last_carry = min(last_carry, num_beats - 1)
        if first_carry <= last_carry:
            carry_diff[first_carry] += 1.0
            carry_diff[last_carry + 1] -= 1.0

    carry_in_count = np.cumsum(carry_diff[:-1]).astype(np.float32)
    active_count = counts + carry_in_count

    mean_pitch = _safe_divide(sum_pitch, counts)
    mean_duration = _safe_divide(sum_duration, counts)
    mean_pitch_sq = _safe_divide(sum_pitch_sq, counts)
    mean_duration_sq = _safe_divide(sum_duration_sq, counts)
    accent_ratio = _safe_divide(sum_accent, counts)
    staccato_ratio = _safe_divide(sum_staccato, counts)
    mean_part_idx = _safe_divide(sum_part, counts)
    long_note_ratio = _safe_divide(long_note_count, counts)
    carry_in_ratio = _safe_divide(carry_in_count, np.maximum(active_count, 1.0))
    long_release_ratio = _safe_divide(long_release_count, np.maximum(active_count, 1.0))
    pitch_range = np.where(counts > 0, max_pitch - min_pitch, 0.0).astype(np.float32)
    pitch_std = np.sqrt(np.clip(mean_pitch_sq - mean_pitch * mean_pitch, 0.0, None)).astype(np.float32)
    duration_std = np.sqrt(np.clip(mean_duration_sq - mean_duration * mean_duration, 0.0, None)).astype(np.float32)
    min_pitch = np.where(counts > 0, min_pitch, 0.0).astype(np.float32)
    max_pitch = np.where(counts > 0, max_pitch, 0.0).astype(np.float32)
    unique_pitch_classes = (pitchclass_hist > 0).sum(axis=1).astype(np.float32)
    pc_unique_ratio = unique_pitch_classes / 12.0

    max_count = max(float(counts.max()) if counts.size > 0 else 1.0, 1.0)
    note_count_norm = counts / max_count
    mean_pitch_norm = mean_pitch / 127.0
    pitch_range_norm = pitch_range / 127.0
    mean_duration_norm = mean_duration / 8.0
    pitch_std_norm = pitch_std / 127.0
    duration_std_norm = duration_std / 8.0

    onset_mean = _safe_divide(onset_sum, counts)
    onset_var = _safe_divide(onset_sq_sum, counts) - onset_mean * onset_mean
    onset_std = np.sqrt(np.clip(onset_var, 0.0, None)).astype(np.float32)
    onset_span = np.where(counts > 0, onset_max_local - onset_min_local, 0.0).astype(np.float32)

    prev_counts = np.roll(counts, 1)
    prev_counts[0] = 0.0
    next_counts = np.roll(counts, -1)
    next_counts[-1] = 0.0

    prev_mean_pitch = np.roll(mean_pitch, 1)
    prev_mean_pitch[0] = 0.0
    next_mean_pitch = np.roll(mean_pitch, -1)
    next_mean_pitch[-1] = 0.0

    prev_mean_duration = np.roll(mean_duration, 1)
    prev_mean_duration[0] = 0.0
    next_mean_duration = np.roll(mean_duration, -1)
    next_mean_duration[-1] = 0.0

    prev_empty = np.roll((counts == 0).astype(np.float32), 1)
    prev_empty[0] = 1.0
    next_empty = np.roll((counts == 0).astype(np.float32), -1)
    next_empty[-1] = 1.0

    gap_before = np.zeros(num_beats, dtype=np.float32)
    gap_after = np.zeros(num_beats, dtype=np.float32)
    for beat in range(num_beats):
        if beat > 0 and np.isfinite(onset_min_abs[beat]) and np.isfinite(onset_max_end_abs[beat - 1]):
            gap_before[beat] = max(0.0, onset_min_abs[beat] - onset_max_end_abs[beat - 1])
        if beat + 1 < num_beats and np.isfinite(onset_min_abs[beat + 1]) and np.isfinite(onset_max_end_abs[beat]):
            gap_after[beat] = max(0.0, onset_min_abs[beat + 1] - onset_max_end_abs[beat])

    pitchclass_dist = _safe_divide(pitchclass_hist, np.maximum(counts[:, None], 1.0))
    pitchclass_sim_prev, pitchclass_sim_next = _cosine_adjacent(pitchclass_dist)
    repeat_end_score = pitchclass_sim_prev - pitchclass_sim_next
    pc_entropy = (-np.sum(pitchclass_dist * np.log(pitchclass_dist + 1e-8), axis=1)).astype(np.float32)
    pc_max = np.max(pitchclass_dist, axis=1).astype(np.float32)
    pc_window_sym = _window_symmetry(pitchclass_dist, symmetry_window)

    base_beat_feats = np.stack(
        [
            np.clip(note_count_norm, 0.0, 1.0),
            np.clip(mean_pitch_norm, 0.0, 1.0),
            np.clip(pitch_range_norm, 0.0, 1.0),
            np.clip(mean_duration_norm, 0.0, 1.0),
            np.clip(accent_ratio, 0.0, 1.0),
            np.clip(staccato_ratio, 0.0, 1.0),
        ],
        axis=1,
    ).astype(np.float32)
    symmetry_cosine = _window_symmetry(base_beat_feats, symmetry_window)
    contour_delta = _local_delta(mean_pitch_norm)
    density_delta = _local_delta(note_count_norm)
    pitch_dev = np.abs(mean_pitch_norm - _rolling_mean(mean_pitch_norm, deviation_window)).astype(np.float32)
    density_dev = np.abs(note_count_norm - _rolling_mean(note_count_norm, deviation_window)).astype(np.float32)

    beat_idx = np.arange(num_beats, dtype=np.int32)
    denom = max(num_beats - 1, 1)
    beat_progress = beat_idx.astype(np.float32) / float(denom)
    beats_to_end_norm = (num_beats - 1 - beat_idx).astype(np.float32) / float(denom)
    phase3 = np.mod(beat_idx, 3).astype(np.float32) / 3.0
    phase4 = np.mod(beat_idx, 4).astype(np.float32) / 4.0

    rows = {
        "beat_idx": beat_idx,
        "num_beats": np.full(num_beats, num_beats, dtype=np.int32),
        "boundary_prob": boundary_probs.astype(np.float32),
        "boundary_peak": boundary_probs_to_binary(boundary_probs, peak_cfg),
        "note_count": counts,
        "note_count_norm": np.clip(note_count_norm, 0.0, 1.0),
        "is_empty": (counts == 0).astype(np.float32),
        "mean_pitch": mean_pitch,
        "mean_pitch_norm": np.clip(mean_pitch_norm, 0.0, 1.0),
        "min_pitch": min_pitch,
        "max_pitch": max_pitch,
        "pitch_range": pitch_range,
        "pitch_range_norm": np.clip(pitch_range_norm, 0.0, 1.0),
        "unique_pitch_classes": unique_pitch_classes,
        "pc_unique_ratio": np.clip(pc_unique_ratio, 0.0, 1.0),
        "mean_duration": mean_duration,
        "mean_duration_norm": np.clip(mean_duration_norm, 0.0, 1.0),
        "max_duration": max_duration,
        "pitch_std_norm": np.clip(pitch_std_norm, 0.0, 1.0),
        "duration_std_norm": np.clip(duration_std_norm, 0.0, 1.0),
        "long_note_ratio": long_note_ratio,
        "accent_ratio": accent_ratio,
        "staccato_ratio": staccato_ratio,
        "part_count": np.asarray([len(parts) for parts in unique_parts], dtype=np.float32),
        "mean_part_idx": mean_part_idx,
        "onset_mean": onset_mean,
        "onset_std": onset_std,
        "onset_span": onset_span,
        "carry_in_count": carry_in_count,
        "carry_in_ratio": carry_in_ratio,
        "release_count": release_count,
        "long_release_count": long_release_count,
        "long_release_ratio": long_release_ratio,
        "gap_before": gap_before,
        "gap_after": gap_after,
        "prev_empty": prev_empty,
        "next_empty": next_empty,
        "prev_count_delta": counts - prev_counts,
        "next_count_delta": counts - next_counts,
        "density_delta": density_delta,
        "prev_pitch_delta": mean_pitch - prev_mean_pitch,
        "next_pitch_delta": mean_pitch - next_mean_pitch,
        "contour_delta": contour_delta,
        "prev_duration_delta": mean_duration - prev_mean_duration,
        "next_duration_delta": mean_duration - next_mean_duration,
        "pitch_dev": pitch_dev,
        "density_dev": density_dev,
        "pitchclass_sim_prev": pitchclass_sim_prev,
        "pitchclass_sim_next": pitchclass_sim_next,
        "pc_cos_prev": pitchclass_sim_prev,
        "pc_cos_next": pitchclass_sim_next,
        "pc_entropy": pc_entropy,
        "pc_max": pc_max,
        "pc_window_sym": pc_window_sym,
        "repeat_end_score": repeat_end_score,
        "symmetry_cosine": symmetry_cosine,
        "beat_progress": beat_progress,
        "beats_to_end_norm": beats_to_end_norm,
        "beat_pos_in_measure": (np.mod(beat_idx, max(measure_cycle, 1)).astype(np.float32) / max(float(measure_cycle), 1.0)),
        "phase3_sin": np.sin(2.0 * np.pi * phase3).astype(np.float32),
        "phase3_cos": np.cos(2.0 * np.pi * phase3).astype(np.float32),
        "phase4_sin": np.sin(2.0 * np.pi * phase4).astype(np.float32),
        "phase4_cos": np.cos(2.0 * np.pi * phase4).astype(np.float32),
        "downbeat_mod3": (np.mod(beat_idx, 3) == 0).astype(np.float32),
        "downbeat_mod4": (np.mod(beat_idx, 4) == 0).astype(np.float32),
        "barline_proximity_mod3": _barline_proximity(beat_idx, 3).astype(np.float32),
        "barline_proximity_mod4": _barline_proximity(beat_idx, 4).astype(np.float32),
    }

    for pc_idx in range(12):
        rows[f"pc_{pc_idx}"] = pitchclass_dist[:, pc_idx].astype(np.float32)

    frame = pd.DataFrame(rows)
    sample_id = npz_path.stem
    piece_id = extract_piece_id(sample_id)
    performer_id = extract_performer_id(sample_id)
    level = extract_level(sample_id)
    split = assign_split(piece_id, split_cfg)

    frame.insert(0, "split", split)
    frame.insert(0, "level", -1 if level is None else int(level))
    frame.insert(0, "performer_id", performer_id)
    frame.insert(0, "piece_id", piece_id)
    frame.insert(0, "sample_id", sample_id)
    frame.insert(0, "source_path", str(npz_path.resolve()))
    if xml_score_dir is not None:
        xml_frame = extract_xml_beat_features(
            piece_id=piece_id,
            xml_dir=xml_score_dir,
            num_beats=num_beats,
            beat_unit=beat_unit,
            expand_repeats=xml_expand_repeats,
        )
        frame = frame.merge(xml_frame, on="beat_idx", how="left", validate="one_to_one")
    return frame


def build_weighted_salience_frame(
    npz_path: Path,
    levels: list[int],
    level_weights: list[float],
    peak_cfg: PeakConfig,
    split_cfg: dict[str, set[str]] | None = None,
    long_note_threshold: float = 1.0,
    beat_unit_fallback: float = 1.0,
    symmetry_window: int = 4,
    deviation_window: int = 8,
    measure_cycle: int = 3,
    xml_score_dir: Path | None = None,
    xml_expand_repeats: bool = True,
) -> pd.DataFrame:
    if len(levels) != len(level_weights):
        raise ValueError("levels and level_weights must have the same length")

    ref_frame = build_beat_feature_frame(
        npz_path=npz_path,
        peak_cfg=peak_cfg,
        split_cfg=split_cfg,
        long_note_threshold=long_note_threshold,
        beat_unit_fallback=beat_unit_fallback,
        symmetry_window=symmetry_window,
        deviation_window=deviation_window,
        measure_cycle=measure_cycle,
        xml_score_dir=xml_score_dir,
        xml_expand_repeats=xml_expand_repeats,
    )
    num_beats = int(ref_frame["num_beats"].iloc[0])
    weights = np.asarray(level_weights, dtype=np.float32)
    weight_sum = float(weights.sum())
    if weight_sum <= 0:
        raise ValueError("level_weights must sum to a positive value")
    weights = weights / weight_sum

    weighted_prob = np.zeros(num_beats, dtype=np.float32)
    weighted_peak = np.zeros(num_beats, dtype=np.float32)

    for level, weight in zip(levels, weights.tolist()):
        level_path = replace_level_suffix(npz_path, level=level)
        if not level_path.exists():
            raise FileNotFoundError(f"Missing level file for salience target: {level_path}")

        loaded = load_boundary_npz(level_path, beat_unit_fallback=beat_unit_fallback)
        if int(loaded["num_beats"]) != num_beats:
            raise ValueError(
                f"num_beats mismatch for {level_path}: expected {num_beats}, got {loaded['num_beats']}"
            )

        level_prob = np.asarray(loaded["boundary_probs"], dtype=np.float32)
        level_peak = boundary_probs_to_binary(level_prob, peak_cfg)

        ref_frame[f"target_prob_L{int(level)}"] = level_prob
        ref_frame[f"target_peak_L{int(level)}"] = level_peak

        weighted_prob += float(weight) * level_prob
        weighted_peak += float(weight) * level_peak

    ref_frame["salience_prob"] = weighted_prob.astype(np.float32)
    ref_frame["salience_peak"] = weighted_peak.astype(np.float32)
    ref_frame["boundary_prob"] = ref_frame["salience_prob"].astype(np.float32)
    ref_frame["boundary_peak"] = ref_frame["salience_peak"].astype(np.float32)

    sample_stem = npz_path.stem
    match = LEVEL_SUFFIX_RE.search(sample_stem)
    base_stem = sample_stem[: match.start()] if match else sample_stem
    ref_frame["sample_id"] = f"{base_stem}_SAL"
    ref_frame["level"] = 0
    return ref_frame


def build_grouped_salience_frame(
    npz_path: Path,
    target_groups: list[list[int]],
    group_weights: list[float],
    group_merge: str,
    peak_cfg: PeakConfig,
    split_cfg: dict[str, set[str]] | None = None,
    long_note_threshold: float = 1.0,
    beat_unit_fallback: float = 1.0,
    symmetry_window: int = 4,
    deviation_window: int = 8,
    measure_cycle: int = 3,
    xml_score_dir: Path | None = None,
    xml_expand_repeats: bool = True,
) -> pd.DataFrame:
    if len(target_groups) != len(group_weights):
        raise ValueError("target_groups and group_weights must have the same length")
    merge_mode = str(group_merge).lower()
    if merge_mode not in {"max", "mean"}:
        raise ValueError(f"Unsupported group_merge: {group_merge}")

    ref_frame = build_beat_feature_frame(
        npz_path=npz_path,
        peak_cfg=peak_cfg,
        split_cfg=split_cfg,
        long_note_threshold=long_note_threshold,
        beat_unit_fallback=beat_unit_fallback,
        symmetry_window=symmetry_window,
        deviation_window=deviation_window,
        measure_cycle=measure_cycle,
        xml_score_dir=xml_score_dir,
        xml_expand_repeats=xml_expand_repeats,
    )
    num_beats = int(ref_frame["num_beats"].iloc[0])
    weights = np.asarray(group_weights, dtype=np.float32)
    weight_sum = float(weights.sum())
    if weight_sum <= 0:
        raise ValueError("group_weights must sum to a positive value")
    weights = weights / weight_sum

    weighted_prob = np.zeros(num_beats, dtype=np.float32)
    weighted_peak = np.zeros(num_beats, dtype=np.float32)
    merged_peak_per_group: list[np.ndarray] = []

    for group_idx, (levels, weight) in enumerate(zip(target_groups, weights.tolist()), start=1):
        if not levels:
            raise ValueError("target_groups cannot contain empty groups")

        prob_stack = []
        peak_stack = []
        for level in levels:
            level_path = replace_level_suffix(npz_path, level=int(level))
            if not level_path.exists():
                raise FileNotFoundError(f"Missing level file for grouped salience target: {level_path}")
            loaded = load_boundary_npz(level_path, beat_unit_fallback=beat_unit_fallback)
            if int(loaded["num_beats"]) != num_beats:
                raise ValueError(
                    f"num_beats mismatch for {level_path}: expected {num_beats}, got {loaded['num_beats']}"
                )
            level_prob = np.asarray(loaded["boundary_probs"], dtype=np.float32)
            level_peak = boundary_probs_to_binary(level_prob, peak_cfg)
            prob_stack.append(level_prob)
            peak_stack.append(level_peak)

        prob_arr = np.stack(prob_stack, axis=0)
        peak_arr = np.stack(peak_stack, axis=0)
        if merge_mode == "max":
            merged_prob = prob_arr.max(axis=0)
            merged_peak = peak_arr.max(axis=0)
        else:
            merged_prob = prob_arr.mean(axis=0)
            merged_peak = peak_arr.mean(axis=0)

        ref_frame[f"target_prob_G{group_idx}"] = merged_prob.astype(np.float32)
        ref_frame[f"target_peak_G{group_idx}"] = merged_peak.astype(np.float32)
        merged_peak_per_group.append(merged_peak.astype(np.float32))
        weighted_prob += float(weight) * merged_prob
        weighted_peak += float(weight) * merged_peak

    ref_frame["salience_prob"] = weighted_prob.astype(np.float32)
    ref_frame["salience_peak"] = weighted_peak.astype(np.float32)
    ref_frame["boundary_prob"] = ref_frame["salience_prob"].astype(np.float32)
    ref_frame["boundary_peak"] = ref_frame["salience_peak"].astype(np.float32)

    stage_class = np.zeros(num_beats, dtype=np.int32)
    for class_idx, merged_peak in enumerate(merged_peak_per_group, start=1):
        stage_class = np.where(merged_peak > 0.5, class_idx, stage_class)
    ref_frame["stage_class"] = stage_class.astype(np.int32)
    ref_frame["stage_class_midhigh"] = np.where(stage_class >= 2, stage_class - 1, 0).astype(np.int32)

    sample_stem = npz_path.stem
    match = LEVEL_SUFFIX_RE.search(sample_stem)
    base_stem = sample_stem[: match.start()] if match else sample_stem
    ref_frame["sample_id"] = f"{base_stem}_G3SAL"
    ref_frame["level"] = 0
    return ref_frame
