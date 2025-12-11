"""
Beat-level embedding extraction for ATEPP v1.2 metadata, following the
padding/sequence handling style in final.py.

The script:
- loads ATEPP-metadata-1.2.csv,
- filters out low-quality rows,
- samples a small subset for a dry-run,
- parses each MIDI and computes beat-level feature vectors,
- pads/truncates to a fixed length with a padding mask,
- saves embeddings, masks, and lightweight metadata.
- Additionally outputs beat-level tempo curves (smoothed, mean-normalized, no phrase split).

Dependencies (not installed here): numpy, pandas, pretty_midi.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import pretty_midi


logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


@dataclass
class BeatEmbeddingConfig:
    atepp_root: Path = Path("ATEPP-master")
    metadata_filename: str = "ATEPP-metadata-1.2.csv"
    midi_root: Path = Path("ATEPP-1.2/ATEPP-1.2")  # folder that contains the midi_path entries
    output_dir: Path = Path("beat_embeddings_v12")
    subset_size: int = 24
    max_seq_len: int = 2048
    random_seed: int = 42
    quality_exclude: Tuple[str, ...] = (
        "low quality",
        "background noise",
        "applause",
        "corrupted",
    )


def load_metadata(config: BeatEmbeddingConfig) -> pd.DataFrame:
    csv_path = config.atepp_root / config.metadata_filename
    df = pd.read_csv(csv_path)
    if "quality" in df.columns:
        df["quality"] = df["quality"].fillna("").str.lower()
    return df


def filter_metadata(df: pd.DataFrame, config: BeatEmbeddingConfig) -> pd.DataFrame:
    if "quality" not in df.columns:
        return df
    mask = ~df["quality"].isin(config.quality_exclude)
    kept = df[mask].reset_index(drop=True)
    logging.info("Filtered by quality: kept %d / %d rows", len(kept), len(df))
    return kept


def sample_subset(df: pd.DataFrame, config: BeatEmbeddingConfig) -> pd.DataFrame:
    if len(df) <= config.subset_size:
        return df.copy()
    return df.sample(n=config.subset_size, random_state=config.random_seed).reset_index(drop=True)


def _collect_note_arrays(pm: pretty_midi.PrettyMIDI):
    starts, ends, pitches, velocities = [], [], [], []
    for inst in pm.instruments:
        for note in inst.notes:
            starts.append(note.start)
            ends.append(note.end)
            pitches.append(note.pitch)
            velocities.append(note.velocity)
    if not starts:
        return (
            np.zeros(0, dtype=np.float32),
            np.zeros(0, dtype=np.float32),
            np.zeros(0, dtype=np.float32),
            np.zeros(0, dtype=np.float32),
        )
    return (
        np.asarray(starts, dtype=np.float32),
        np.asarray(ends, dtype=np.float32),
        np.asarray(pitches, dtype=np.float32),
        np.asarray(velocities, dtype=np.float32),
    )


def _tempo_at_time(tempo_times: np.ndarray, tempos: np.ndarray, t: float) -> float:
    if tempos.size == 0:
        return 0.0
    idx = np.searchsorted(tempo_times, t, side="right") - 1
    idx = np.clip(idx, 0, len(tempos) - 1)
    return float(tempos[idx])


def _beat_grid(pm: pretty_midi.PrettyMIDI) -> np.ndarray:
    beats = pm.get_beats()
    if len(beats) >= 2:
        return beats
    tempo = pm.estimate_tempo()
    step = 60.0 / max(tempo, 1e-6)
    total_time = max(pm.get_end_time(), step)
    grid = np.arange(0.0, total_time + step, step)
    if len(grid) < 2:
        grid = np.array([0.0, total_time + 1e-3], dtype=np.float32)
    return grid


def pad_1d(seq: np.ndarray, max_seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
    length = min(len(seq), max_seq_len)
    padded = np.zeros(max_seq_len, dtype=np.float32)
    mask = np.zeros(max_seq_len, dtype=np.float32)
    if length > 0:
        padded[:length] = seq[:length]
        mask[:length] = 1.0
    return padded, mask


def compute_tempo_curve(
    pm: pretty_midi.PrettyMIDI,
    max_seq_len: int,
    smooth_window: int = 3,
    normalize_mean: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    beats = _beat_grid(pm)
    if len(beats) < 2:
        return np.zeros(max_seq_len, dtype=np.float32), np.zeros(max_seq_len, dtype=np.float32)
    deltas = np.diff(beats)
    deltas = np.clip(deltas, 1e-4, None)
    tempo_raw = 60.0 / deltas
    if len(tempo_raw) >= smooth_window and smooth_window > 1:
        window = np.ones(smooth_window, dtype=np.float32) / smooth_window
        tempo_smooth = np.convolve(tempo_raw, window, mode="same")
    else:
        tempo_smooth = tempo_raw
    if normalize_mean and tempo_smooth.mean() > 1e-8:
        tempo_smooth = tempo_smooth / tempo_smooth.mean()
    tempo_padded, tempo_mask = pad_1d(tempo_smooth.astype(np.float32), max_seq_len)
    return tempo_padded, tempo_mask


def beat_level_embedding(
    pm: pretty_midi.PrettyMIDI, max_seq_len: int
) -> Tuple[np.ndarray, np.ndarray]:
    beats = _beat_grid(pm)
    tempo_values, tempo_times = pm.get_tempo_changes()
    note_starts, note_ends, note_pitches, note_velocities = _collect_note_arrays(pm)
    if len(beats) < 2:
        return np.zeros((max_seq_len, 7), dtype=np.float32), np.zeros(max_seq_len, dtype=np.float32)

    features = []
    for start, end in zip(beats[:-1], beats[1:]):
        beat_dur = max(end - start, 1e-4)
        onset_mask = (note_starts >= start) & (note_starts < end)
        active_mask = (note_starts < end) & (note_ends > start)
        onset_count = onset_mask.sum()

        mean_vel = float(note_velocities[onset_mask].mean()) if onset_count else 0.0
        mean_pitch = float(note_pitches[onset_mask].mean()) if onset_count else 0.0
        durations = note_ends[onset_mask] - note_starts[onset_mask]
        dur_ratio = float(durations.mean() / beat_dur) if onset_count else 0.0
        onset_pos = (note_starts[onset_mask] - start) / beat_dur if onset_count else np.zeros(0, dtype=np.float32)
        onset_spread = float(onset_pos.std()) if onset_pos.size else 0.0
        tempo = _tempo_at_time(tempo_times, tempo_values, start)

        features.append(
            [
                tempo,
                float(onset_count),
                mean_vel / 127.0,
                mean_pitch / 127.0,
                float(active_mask.sum()),
                dur_ratio,
                onset_spread,
            ]
        )

    feats = np.asarray(features, dtype=np.float32)
    padded, mask = pad_features(feats, max_seq_len)
    return padded, mask


def pad_features(features: np.ndarray, max_seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
    if features.ndim != 2:
        raise ValueError("features should be 2D (beats x dim)")
    dim = features.shape[1]
    length = min(len(features), max_seq_len)
    padded = np.zeros((max_seq_len, dim), dtype=np.float32)
    mask = np.zeros(max_seq_len, dtype=np.float32)
    if length > 0:
        padded[:length] = features[:length]
        mask[:length] = 1.0
    return padded, mask


def midi_path_for_row(row: pd.Series, config: BeatEmbeddingConfig) -> Path:
    midi_rel = str(row["midi_path"])
    return (config.midi_root / midi_rel).resolve()


def build_subset_embeddings(config: BeatEmbeddingConfig):
    df = load_metadata(config)
    df = filter_metadata(df, config)
    df = sample_subset(df, config)

    embeddings = []
    padding_masks = []
    tempo_curves = []
    tempo_masks = []
    meta_rows = []
    config.output_dir.mkdir(parents=True, exist_ok=True)

    for _, row in df.iterrows():
        midi_file = midi_path_for_row(row, config)
        if not midi_file.exists():
            logging.warning("Missing MIDI: %s", midi_file)
            continue
        try:
            pm = pretty_midi.PrettyMIDI(midi_file)
        except Exception as exc:  # noqa: BLE001
            logging.warning("Failed to parse %s: %s", midi_file, exc)
            continue

        embedding, padding_mask = beat_level_embedding(pm, config.max_seq_len)
        embeddings.append(embedding)
        padding_masks.append(padding_mask)
        tempo_curve, tempo_mask = compute_tempo_curve(pm, config.max_seq_len)
        tempo_curves.append(tempo_curve)
        tempo_masks.append(tempo_mask)
        meta_rows.append(
            {
                "artist": row.get("artist"),
                "track": row.get("track"),
                "composer": row.get("composer"),
                "composition_id": row.get("composition_id"),
                "perf_id": row.get("perf_id"),
                "midi_path": str(midi_file),
                "beat_steps": int(padding_mask.sum()),
            }
        )

    if not embeddings:
        logging.warning("No embeddings were produced. Verify midi_root and subset filters.")
        return

    embeddings_arr = np.stack(embeddings)
    masks_arr = np.stack(padding_masks)
    tempo_arr = np.stack(tempo_curves)
    tempo_masks_arr = np.stack(tempo_masks)
    np.save(config.output_dir / "beat_embeddings.npy", embeddings_arr)
    np.save(config.output_dir / "beat_padding_masks.npy", masks_arr)
    np.save(config.output_dir / "beat_tempo_curves.npy", tempo_arr)
    np.save(config.output_dir / "beat_tempo_masks.npy", tempo_masks_arr)
    pd.DataFrame(meta_rows).to_csv(config.output_dir / "beat_embedding_metadata.csv", index=False)
    logging.info(
        "Saved embeddings: %s, masks: %s, tempo curves: %s, tempo masks: %s, metadata: %s",
        config.output_dir / "beat_embeddings.npy",
        config.output_dir / "beat_padding_masks.npy",
        config.output_dir / "beat_tempo_curves.npy",
        config.output_dir / "beat_tempo_masks.npy",
        config.output_dir / "beat_embedding_metadata.csv",
    )


if __name__ == "__main__":
    cfg = BeatEmbeddingConfig(
        midi_root=Path("ATEPP-1.2/ATEPP-1.2"),  # replace if needed
        output_dir=Path("outputs/atepp_v12_subset"),
        subset_size=16,
        max_seq_len=2048,
    )
    build_subset_embeddings(cfg)
