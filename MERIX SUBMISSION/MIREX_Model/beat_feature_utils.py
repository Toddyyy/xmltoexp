from __future__ import annotations

from typing import Dict, List

import numpy as np


SELECTED_SCORE_BEAT_FEATURES = [
    "density_delta",
    "density_dev",
    "contour_delta",
    "pitch_dev",
    "beat_pos_in_measure",
    "note_count_norm",
    "pc_unique_ratio",
    "mean_duration_norm",
]


def build_selected_score_beat_features(
    tokens: List[Dict],
    note_feats: np.ndarray,
    beat_ids: np.ndarray,
    num_beats: int,
    beat_unit: float,
) -> np.ndarray:
    feats = np.zeros((num_beats, len(SELECTED_SCORE_BEAT_FEATURES)), dtype=np.float32)
    if num_beats <= 0 or note_feats.size == 0 or len(tokens) == 0:
        return feats

    valid = (beat_ids >= 0) & (beat_ids < num_beats)
    if not np.any(valid):
        return feats

    f = note_feats[valid]
    b = beat_ids[valid].astype(np.int64)
    valid_idx = np.flatnonzero(valid)
    valid_tokens = [tokens[i] for i in valid_idx if i < len(tokens)]
    if len(valid_tokens) != len(b):
        limit = min(len(valid_tokens), len(b))
        valid_tokens = valid_tokens[:limit]
        b = b[:limit]
        f = f[:limit]
    if len(valid_tokens) == 0:
        return feats

    counts = np.zeros(num_beats, dtype=np.float32)
    sum_pitch = np.zeros(num_beats, dtype=np.float32)
    sum_dur = np.zeros(num_beats, dtype=np.float32)
    beat_pos_sum = np.zeros(num_beats, dtype=np.float32)
    unique_pcs = [set() for _ in range(num_beats)]

    np.add.at(counts, b, 1.0)
    np.add.at(sum_pitch, b, f[:, 0])
    np.add.at(sum_dur, b, f[:, 1])

    for beat_idx, tok in zip(b, valid_tokens):
        pitch_midi = float(tok.get("pitch_midi", 0.0))
        unique_pcs[int(beat_idx)].add(int(round(pitch_midi)) % 12)

        measure_progress = tok.get("measure_progress")
        if measure_progress is None:
            pos = float(tok.get("position", 0.0))
            denom = max(float(beat_unit), 1e-6)
            measure_progress = (pos / denom) % 1.0
        beat_pos_sum[int(beat_idx)] += float(np.clip(measure_progress, 0.0, 1.0))

    mean_pitch = np.divide(sum_pitch, counts, out=np.zeros_like(sum_pitch), where=counts > 0)
    mean_dur = np.divide(sum_dur, counts, out=np.zeros_like(sum_dur), where=counts > 0)
    beat_pos_in_measure = np.divide(
        beat_pos_sum,
        counts,
        out=np.zeros_like(beat_pos_sum),
        where=counts > 0,
    )

    max_count = float(counts.max()) if counts.size > 0 else 1.0
    note_count_norm = np.clip(counts / max(max_count, 1.0), 0.0, 1.0)
    mean_pitch_norm = np.clip(mean_pitch / 127.0, 0.0, 1.0)
    mean_duration_norm = np.clip(mean_dur / 8.0, 0.0, 1.0)
    pc_unique_ratio = np.array(
        [len(unique_pcs[i]) / counts[i] if counts[i] > 0 else 0.0 for i in range(num_beats)],
        dtype=np.float32,
    )

    density_delta = np.zeros(num_beats, dtype=np.float32)
    if num_beats > 1:
        density_delta[1:] = np.abs(np.diff(note_count_norm))

    local_density = _moving_average(note_count_norm, window=3)
    density_dev = np.abs(note_count_norm - local_density)

    contour_delta = np.zeros(num_beats, dtype=np.float32)
    if num_beats > 1:
        contour_delta[1:] = np.abs(np.diff(mean_pitch_norm))

    global_pitch = (
        float(sum_pitch.sum()) / float(counts.sum())
        if float(counts.sum()) > 0
        else 0.0
    )
    pitch_dev = np.abs(mean_pitch - global_pitch) / 127.0
    pitch_dev = np.clip(pitch_dev, 0.0, 1.0)

    feats = np.stack(
        [
            density_delta,
            density_dev,
            contour_delta,
            pitch_dev,
            np.clip(beat_pos_in_measure, 0.0, 1.0),
            note_count_norm,
            np.clip(pc_unique_ratio, 0.0, 1.0),
            mean_duration_norm,
        ],
        axis=1,
    ).astype(np.float32)
    return feats


def _moving_average(x: np.ndarray, window: int = 3) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0 or window <= 1:
        return x.copy()
    pad = window // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(xp, kernel, mode="valid").astype(np.float32)
