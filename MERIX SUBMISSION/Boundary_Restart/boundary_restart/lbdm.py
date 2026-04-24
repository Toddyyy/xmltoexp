from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from .features import load_boundary_npz


def _positive_scale(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float32)
    positive = values[values > 1e-8]
    if positive.size == 0:
        return 1.0
    return float(max(np.quantile(positive, 0.95), 1e-6))


def _normalize_unit(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    vmax = float(np.max(values)) if values.size else 0.0
    if vmax <= 1e-8:
        return np.zeros_like(values, dtype=np.float32)
    return (values / vmax).astype(np.float32)


def _degree_of_change(prev_value: float, cur_value: float, scale: float) -> float:
    prev_abs = abs(float(prev_value))
    cur_abs = abs(float(cur_value))
    denom = prev_abs + cur_abs
    if denom <= 1e-8:
        return 0.0
    magnitude = min(math.sqrt(max(prev_abs, cur_abs) / max(float(scale), 1e-8)), 1.0)
    return float((abs(cur_abs - prev_abs) / denom) * magnitude)


def _surface_events(
    note_feats: np.ndarray,
    beat_ids: np.ndarray,
    surface: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if note_feats.size == 0:
        empty = np.zeros(0, dtype=np.float32)
        return empty, empty.astype(np.int32), empty

    onset = np.round(np.asarray(note_feats[:, 2], dtype=np.float32), 6)
    pitch = np.asarray(note_feats[:, 0], dtype=np.float32)
    duration = np.asarray(note_feats[:, 1], dtype=np.float32)
    beat_ids = np.asarray(beat_ids, dtype=np.int32)

    order = np.lexsort((pitch, onset))
    onset_sorted = onset[order]
    unique_onsets, first_idx = np.unique(onset_sorted, return_index=True)
    selected_idx: list[int] = []
    for group_idx, start in enumerate(first_idx.tolist()):
        end = first_idx[group_idx + 1] if group_idx + 1 < len(first_idx) else len(order)
        group = order[start:end]
        if surface == "highest":
            chosen = int(group[np.argmax(pitch[group])])
        elif surface == "lowest":
            chosen = int(group[np.argmin(pitch[group])])
        else:
            raise ValueError(f"Unsupported surface: {surface}")
        selected_idx.append(chosen)

    idx = np.asarray(selected_idx, dtype=np.int64)
    return (
        onset[idx].astype(np.float32),
        beat_ids[idx].astype(np.int32),
        np.stack([pitch[idx], duration[idx]], axis=1).astype(np.float32),
    )


def _surface_salience(
    onset: np.ndarray,
    beat_ids: np.ndarray,
    surface_values: np.ndarray,
    num_beats: int,
) -> np.ndarray:
    beat_scores = np.zeros(int(num_beats), dtype=np.float32)
    if onset.size < 2:
        return beat_scores

    pitch = surface_values[:, 0].astype(np.float32)
    duration = surface_values[:, 1].astype(np.float32)

    prev_interval = np.zeros_like(pitch, dtype=np.float32)
    cur_interval = np.zeros_like(pitch, dtype=np.float32)
    for idx in range(1, len(pitch)):
        cur_interval[idx] = abs(float(pitch[idx] - pitch[idx - 1]))
    if len(pitch) > 2:
        prev_interval[2:] = cur_interval[1:-1]

    rest_before = np.zeros_like(onset, dtype=np.float32)
    for idx in range(1, len(onset)):
        rest_before[idx] = max(0.0, float(onset[idx] - (onset[idx - 1] + duration[idx - 1])))

    interval_scale = _positive_scale(cur_interval)
    duration_scale = _positive_scale(duration)
    rest_scale = _positive_scale(rest_before)

    event_scores = np.zeros_like(onset, dtype=np.float32)
    for idx in range(1, len(onset)):
        pitch_term = _degree_of_change(prev_interval[idx], cur_interval[idx], interval_scale)
        duration_term = _degree_of_change(float(duration[idx - 1]), float(duration[idx]), duration_scale)
        rest_term = _degree_of_change(float(rest_before[idx - 1]), float(rest_before[idx]), rest_scale)
        rest_bonus = min(float(rest_before[idx]) / max(rest_scale, 1e-8), 1.0) if rest_before[idx] > 0 else 0.0
        event_scores[idx] = float(
            0.5 * pitch_term
            + 0.3 * duration_term
            + 0.2 * max(rest_term, rest_bonus)
        )

    event_scores = _normalize_unit(event_scores)
    valid = (beat_ids >= 0) & (beat_ids < int(num_beats))
    if np.any(valid):
        np.maximum.at(beat_scores, beat_ids[valid], event_scores[valid])
    return beat_scores.astype(np.float32)


def compute_lbdm_beat_salience(
    note_feats: np.ndarray,
    beat_ids: np.ndarray,
    num_beats: int,
) -> np.ndarray:
    num_beats = int(num_beats)
    if num_beats <= 0:
        return np.zeros(0, dtype=np.float32)
    note_feats = np.asarray(note_feats, dtype=np.float32)
    beat_ids = np.asarray(beat_ids, dtype=np.int32)
    valid = (beat_ids >= 0) & (beat_ids < num_beats)
    note_feats = note_feats[valid]
    beat_ids = beat_ids[valid]
    if note_feats.size == 0:
        return np.zeros(num_beats, dtype=np.float32)

    highest_onset, highest_beats, highest_values = _surface_events(note_feats, beat_ids, surface="highest")
    lowest_onset, lowest_beats, lowest_values = _surface_events(note_feats, beat_ids, surface="lowest")
    melody_scores = _surface_salience(highest_onset, highest_beats, highest_values, num_beats=num_beats)
    bass_scores = _surface_salience(lowest_onset, lowest_beats, lowest_values, num_beats=num_beats)
    combined = np.maximum(0.65 * melody_scores + 0.35 * bass_scores, np.maximum(melody_scores, 0.85 * bass_scores))
    return _normalize_unit(combined)


def compute_lbdm_beat_salience_from_npz(
    npz_path: Path,
    beat_unit_fallback: float = 1.0,
) -> np.ndarray:
    loaded = load_boundary_npz(Path(npz_path), beat_unit_fallback=beat_unit_fallback)
    return compute_lbdm_beat_salience(
        note_feats=np.asarray(loaded["note_feats"], dtype=np.float32),
        beat_ids=np.asarray(loaded["beat_ids"], dtype=np.int32),
        num_beats=int(loaded["num_beats"]),
    )
