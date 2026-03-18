from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from boundary_restart.features import PeakConfig, boundary_probs_to_binary, load_boundary_npz, replace_level_suffix

COMPONENT_RAW_LEVELS: dict[str, tuple[int, ...]] = {
    "level1": (1,),
    "level2": (2,),
    "level3": (3,),
    "level4": (4,),
    "level5": (5,),
    "level6": (6,),
    "level56": (5, 6),
}

CUMULATIVE_COMPONENTS: dict[str, tuple[str, ...]] = {
    "level1plus_boundary": ("level56", "level4", "level3", "level2", "level1"),
    "level2plus_boundary": ("level56", "level4", "level3", "level2"),
    "level3plus_boundary": ("level56", "level4", "level3"),
    "level4plus_boundary": ("level56", "level4"),
    "level5plus_split56_boundary": ("level6", "level5"),
    "level1plus_split56_boundary": ("level6", "level5", "level4", "level3", "level2", "level1"),
    "level2plus_split56_boundary": ("level6", "level5", "level4", "level3", "level2"),
    "level3plus_split56_boundary": ("level6", "level5", "level4", "level3"),
    "level4plus_split56_boundary": ("level6", "level5", "level4"),
}


def cumulative_components_for_target(target_mode: str) -> tuple[str, ...] | None:
    components = CUMULATIVE_COMPONENTS.get(str(target_mode))
    return tuple(components) if components is not None else None


def build_piece_frequency_for_raw_levels(
    frame: pd.DataFrame,
    raw_levels: tuple[int, ...],
    peak_cfg: PeakConfig,
    beat_unit_fallback: float,
) -> pd.DataFrame:
    work = frame.copy()
    detector_binary = np.zeros(len(work), dtype=np.float32)
    beat_idx = work["beat_idx"].to_numpy(dtype=np.int32)
    for source_path, positions in work.groupby("source_path", sort=False).indices.items():
        pos = np.asarray(positions, dtype=np.int64)
        boundary_binary = None
        for raw_level in raw_levels:
            level_path = replace_level_suffix(Path(str(source_path)), level=raw_level)
            loaded = load_boundary_npz(level_path, beat_unit_fallback=beat_unit_fallback)
            current_binary = boundary_probs_to_binary(
                np.asarray(loaded["boundary_probs"], dtype=np.float32),
                peak_cfg,
            ).astype(np.float32)
            boundary_binary = current_binary if boundary_binary is None else np.maximum(boundary_binary, current_binary)
        sample_beat_idx = beat_idx[pos]
        detector_binary[pos] = boundary_binary[sample_beat_idx].astype(np.float32)
    work["detector_binary"] = detector_binary.astype(np.float32)
    piece = (
        work.sort_values(["piece_id", "beat_idx", "sample_id"])
        .groupby(["piece_id", "beat_idx"], sort=False)
        .agg({"detector_binary": "mean"})
        .rename(columns={"detector_binary": "frequency_target"})
        .reset_index()
    )
    piece["frequency_target"] = piece["frequency_target"].astype(np.float32)
    return piece


def _topdown_merge_piece(
    piece_beats: np.ndarray,
    component_frames: dict[str, pd.DataFrame],
    component_order: tuple[str, ...],
    tolerance: int,
    component_weights: dict[str, float] | None = None,
) -> np.ndarray:
    beat_to_pos = {int(beat): idx for idx, beat in enumerate(piece_beats.tolist())}
    merged_freq = np.zeros(piece_beats.shape[0], dtype=np.float32)
    kept_events: list[dict[str, float | int]] = []

    for component_name in component_order:
        frame = component_frames.get(component_name)
        if frame is None or frame.empty:
            continue
        component_weight = float((component_weights or {}).get(component_name, 1.0))
        current_events = [
            (int(row.beat_idx), float(row.frequency_target) * component_weight)
            for row in frame.itertuples(index=False)
            if float(row.frequency_target) > 0.0
        ]
        current_events.sort(key=lambda item: item[0])
        new_events: list[dict[str, float | int]] = []
        higher_events = list(kept_events)
        for beat_idx, freq in current_events:
            matched = None
            matched_dist = None
            for event_idx, event in enumerate(higher_events):
                dist = abs(int(event["beat_idx"]) - beat_idx)
                if dist <= int(tolerance):
                    if matched is None or dist < matched_dist or (
                        dist == matched_dist and int(event["beat_idx"]) < int(higher_events[matched]["beat_idx"])
                    ):
                        matched = event_idx
                        matched_dist = dist
            if matched is not None:
                higher_events[matched]["frequency_target"] = max(float(higher_events[matched]["frequency_target"]), freq)
            else:
                new_events.append({"beat_idx": beat_idx, "frequency_target": freq})
        kept_events = higher_events + new_events
        kept_events.sort(key=lambda item: int(item["beat_idx"]))

    for event in kept_events:
        pos = beat_to_pos.get(int(event["beat_idx"]))
        if pos is not None:
            merged_freq[pos] = max(merged_freq[pos], float(event["frequency_target"]))
    return merged_freq.astype(np.float32)


def build_topdown_cumulative_frequency(
    base_piece: pd.DataFrame,
    component_map: dict[str, pd.DataFrame],
    component_order: tuple[str, ...],
    tolerance: int,
    component_weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for piece_id, piece_frame in base_piece.groupby("piece_id", sort=False):
        piece_frame = piece_frame.sort_values("beat_idx").reset_index(drop=True)
        piece_beats = piece_frame["beat_idx"].to_numpy(dtype=np.int32)
        per_component: dict[str, pd.DataFrame] = {}
        for component_name in component_order:
            component_df = component_map.get(component_name)
            if component_df is None:
                continue
            per_component[component_name] = component_df[component_df["piece_id"] == piece_id][
                ["beat_idx", "frequency_target"]
            ].copy()
        merged_freq = _topdown_merge_piece(
            piece_beats,
            per_component,
            component_order,
            tolerance=int(tolerance),
            component_weights=component_weights,
        )
        rows.append(
            pd.DataFrame(
                {
                    "piece_id": piece_id,
                    "beat_idx": piece_beats,
                    "frequency_target": merged_freq,
                }
            )
        )
    if not rows:
        return pd.DataFrame(columns=["piece_id", "beat_idx", "frequency_target"])
    merged = pd.concat(rows, ignore_index=True)
    merged["frequency_target"] = merged["frequency_target"].astype(np.float32)
    return merged


def merge_event_frames_topdown(
    component_frames: dict[str, pd.DataFrame],
    component_order: tuple[str, ...],
    tolerance: int,
) -> pd.DataFrame:
    kept_rows: list[pd.DataFrame] = []
    higher_beats: list[int] = []
    for component_name in component_order:
        frame = component_frames.get(component_name)
        if frame is None or frame.empty:
            continue
        frame = frame.sort_values(["beat_idx", "detector_score"], ascending=[True, False]).reset_index(drop=True)
        keep_mask = []
        for row in frame.itertuples(index=False):
            beat_idx = int(row.beat_idx)
            near_higher = any(abs(beat_idx - higher_beat) <= int(tolerance) for higher_beat in higher_beats)
            keep_mask.append(not near_higher)
        kept = frame[np.asarray(keep_mask, dtype=bool)].copy()
        if not kept.empty:
            higher_beats.extend(int(x) for x in kept["beat_idx"].tolist())
            kept_rows.append(kept)
    if not kept_rows:
        return pd.DataFrame(columns=["beat_idx", "detector_score"])
    merged = pd.concat(kept_rows, ignore_index=True)
    return merged.sort_values(["beat_idx", "detector_score"], ascending=[True, False]).reset_index(drop=True)
