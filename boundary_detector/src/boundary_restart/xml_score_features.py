from __future__ import annotations

import math
import re
from functools import lru_cache
from pathlib import Path

import music21
import numpy as np
import pandas as pd


XML_NAME_RE = re.compile(r"(?i)mazurka0*(\d+)-(\d+)")
TOKENIZER_VELOCITY_TEXTS = {"cresc.", "molto marcato", "molto ritard."}
TOKENIZER_TEMPO_TEXTS = {"rit.", "accel.", "rit", "accel"}
DEFAULT_TEMPO_DURATION = 4.0
DYNAMIC_LEVEL_MAP = {
    "pppp": -4.0,
    "ppp": -3.0,
    "pp": -2.0,
    "p": -1.0,
    "mp": -0.5,
    "mf": 0.5,
    "f": 1.0,
    "ff": 2.0,
    "fff": 3.0,
    "ffff": 4.0,
    "sf": 1.5,
    "sfz": 1.5,
    "fz": 1.5,
    "rfz": 1.5,
    "fp": 0.0,
    "other-dynamics": 0.0,
}


def _normalize_piece_id(piece_id: str) -> str:
    match = re.match(r"^M(\d+)-(\d+)$", str(piece_id))
    if not match:
        return ""
    return f"M{int(match.group(1)):02d}-{int(match.group(2))}"


def build_xml_map(xml_dir: Path) -> dict[str, Path]:
    xml_map: dict[str, Path] = {}
    for xml_path in sorted(xml_dir.glob("*.xml")):
        match = XML_NAME_RE.search(xml_path.stem)
        if not match:
            continue
        key = f"M{int(match.group(1)):02d}-{int(match.group(2))}"
        xml_map.setdefault(key, xml_path)
    return xml_map


@lru_cache(maxsize=4)
def cached_xml_map(xml_dir_str: str) -> dict[str, Path]:
    return build_xml_map(Path(xml_dir_str))


def resolve_xml_score_path(piece_id: str, xml_dir: Path) -> Path:
    normalized = _normalize_piece_id(piece_id)
    xml_map = cached_xml_map(str(xml_dir.resolve()))
    xml_path = xml_map.get(normalized)
    if xml_path is None:
        raise FileNotFoundError(f"No XML score found for piece {piece_id} in {xml_dir}")
    return xml_path


def _safe_divide(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    return np.divide(num, den, out=np.zeros_like(num, dtype=np.float32), where=den > 0)


def _parse_measure_number(raw_value) -> int:
    match = re.search(r"\d+", str(raw_value))
    return int(match.group(0)) if match else 0


def _absolute_offset(el, score, part) -> float:
    try:
        return float(el.getOffsetInHierarchy(score))
    except Exception:
        try:
            return float(el.getOffsetInHierarchy(part))
        except Exception:
            return float(part.flatten().elementOffset(el))


def _barline_repeat_flags(barline) -> tuple[float, float]:
    if barline is None:
        return 0.0, 0.0
    is_final = 0.0
    is_repeat = 0.0
    classes = {cls.lower() for cls in getattr(barline, "classes", [])}
    direction = str(getattr(barline, "direction", "")).lower()
    bar_type = str(getattr(barline, "type", "")).lower()
    if "repeat" in classes or direction in {"start", "end"}:
        is_repeat = 1.0
    if bar_type in {"final", "light-heavy", "heavy-light"}:
        is_final = 1.0
    return is_repeat, is_final


def _slur_flags(el) -> tuple[float, float]:
    slur_start = 0.0
    slur_end = 0.0
    try:
        spanners = el.getSpannerSites()
    except Exception:
        spanners = []
    for spanner in spanners:
        classes = {cls.lower() for cls in getattr(spanner, "classes", [])}
        if "slur" not in classes:
            continue
        try:
            if spanner.getFirst() is el:
                slur_start = 1.0
            if spanner.getLast() is el:
                slur_end = 1.0
        except Exception:
            continue
    return slur_start, slur_end


def _score_to_key_pcs(score) -> tuple[int | None, int | None]:
    try:
        key_obj = score.analyze("key")
    except Exception:
        return None, None
    tonic = getattr(getattr(key_obj, "tonic", None), "pitchClass", None)
    if tonic is None:
        return None, None
    dominant = int((int(tonic) + 7) % 12)
    return int(tonic), dominant


def _map_dynamic_level(value: str | None) -> float:
    if value is None:
        return 0.0
    return float(DYNAMIC_LEVEL_MAP.get(str(value).strip().lower(), 0.0))


@lru_cache(maxsize=128)
def _extract_xml_beat_features_cached(
    xml_path_str: str,
    num_beats: int,
    beat_unit: float,
    expand_repeats: bool,
) -> pd.DataFrame:
    xml_path = Path(xml_path_str)
    score = music21.converter.parse(str(xml_path))
    if expand_repeats:
        try:
            score = score.expandRepeats()
        except Exception:
            pass

    eps = 1e-9
    beat_unit = float(beat_unit)
    num_beats = int(num_beats)

    measure_number = np.zeros(num_beats, dtype=np.float32)
    measure_start = np.zeros(num_beats, dtype=np.float32)
    measure_end = np.zeros(num_beats, dtype=np.float32)
    measure_progress = np.zeros(num_beats, dtype=np.float32)
    beat_in_measure_norm = np.zeros(num_beats, dtype=np.float32)
    beats_in_measure_norm = np.zeros(num_beats, dtype=np.float32)
    time_sig_numerator_norm = np.zeros(num_beats, dtype=np.float32)
    time_sig_denominator_norm = np.zeros(num_beats, dtype=np.float32)
    key_sharps_norm = np.full(num_beats, 0.5, dtype=np.float32)
    downbeat_actual = np.zeros(num_beats, dtype=np.float32)
    repeat_start = np.zeros(num_beats, dtype=np.float32)
    repeat_end = np.zeros(num_beats, dtype=np.float32)
    final_barline = np.zeros(num_beats, dtype=np.float32)

    primary_part = score.parts[0] if len(score.parts) > 0 else score
    measures = list(primary_part.getElementsByClass(music21.stream.Measure))
    max_measure_number = 1
    for measure in measures:
        measure_offset = _absolute_offset(measure, score, primary_part)
        measure_len = float(getattr(measure.barDuration, "quarterLength", 0.0) or measure.duration.quarterLength or 0.0)
        if measure_len <= 0:
            continue
        start = int(math.floor((measure_offset + eps) / beat_unit))
        end = min(num_beats, int(math.ceil((measure_offset + measure_len - eps) / beat_unit)))
        if end <= start:
            continue
        number = _parse_measure_number(getattr(measure, "number", 0))
        max_measure_number = max(max_measure_number, number)
        time_sig = measure.timeSignature or measure.getContextByClass(music21.meter.TimeSignature)
        key_sig = measure.keySignature or measure.getContextByClass(music21.key.KeySignature)
        beats_in_measure = max(int(round(measure_len / max(beat_unit, eps))), 1)

        rep_start, _ = _barline_repeat_flags(getattr(measure, "leftBarline", None))
        rep_end, is_final = _barline_repeat_flags(getattr(measure, "rightBarline", None))

        if start < num_beats:
            repeat_start[start] = max(repeat_start[start], rep_start)
        if end - 1 < num_beats:
            repeat_end[end - 1] = max(repeat_end[end - 1], rep_end)
            final_barline[end - 1] = max(final_barline[end - 1], is_final)

        for beat_idx in range(start, end):
            local_idx = beat_idx - start
            measure_number[beat_idx] = float(number)
            measure_start[beat_idx] = 1.0 if beat_idx == start else measure_start[beat_idx]
            measure_end[beat_idx] = 1.0 if beat_idx == end - 1 else measure_end[beat_idx]
            measure_progress[beat_idx] = float(local_idx / max(beats_in_measure - 1, 1))
            beat_in_measure_norm[beat_idx] = float((local_idx + 1) / beats_in_measure)
            beats_in_measure_norm[beat_idx] = min(float(beats_in_measure) / 8.0, 1.0)
            downbeat_actual[beat_idx] = 1.0 if beat_idx == start else downbeat_actual[beat_idx]
            if time_sig is not None:
                time_sig_numerator_norm[beat_idx] = min(float(time_sig.numerator) / 12.0, 1.0)
                time_sig_denominator_norm[beat_idx] = min(float(time_sig.denominator) / 16.0, 1.0)
            if key_sig is not None and getattr(key_sig, "sharps", None) is not None:
                key_sharps_norm[beat_idx] = float((float(key_sig.sharps) + 7.0) / 14.0)

    rest_count = np.zeros(num_beats, dtype=np.float32)
    rest_duration = np.zeros(num_beats, dtype=np.float32)
    beat_strength_sum = np.zeros(num_beats, dtype=np.float32)
    beat_strength_count = np.zeros(num_beats, dtype=np.float32)
    note_event_count = np.zeros(num_beats, dtype=np.float32)
    tie_start_count = np.zeros(num_beats, dtype=np.float32)
    tie_stop_count = np.zeros(num_beats, dtype=np.float32)
    tie_continue_count = np.zeros(num_beats, dtype=np.float32)
    fermata_count = np.zeros(num_beats, dtype=np.float32)
    slur_start_count = np.zeros(num_beats, dtype=np.float32)
    slur_end_count = np.zeros(num_beats, dtype=np.float32)
    tokenizer_slur_active_count = np.zeros(num_beats, dtype=np.float32)
    upper_staff_count = np.zeros(num_beats, dtype=np.float32)
    lower_staff_count = np.zeros(num_beats, dtype=np.float32)
    upper_staff_diff = np.zeros(num_beats + 1, dtype=np.float32)
    lower_staff_diff = np.zeros(num_beats + 1, dtype=np.float32)
    fingering_count = np.zeros(num_beats, dtype=np.float32)
    fingering_sum = np.zeros(num_beats, dtype=np.float32)
    active_pc_diff = np.zeros((num_beats + 1, 12), dtype=np.float32)
    part_active_matrix = []
    parts = list(score.parts)
    last_part_idx = max(len(parts) - 1, 0)
    dynamic_mark_count = np.zeros(num_beats, dtype=np.float32)
    dynamic_level_sum = np.zeros(num_beats, dtype=np.float32)
    dynamic_level_updates = np.full(num_beats, np.nan, dtype=np.float32)
    tokenizer_text_count = np.zeros(num_beats, dtype=np.float32)
    tokenizer_velocity_text_count = np.zeros(num_beats, dtype=np.float32)
    tokenizer_tempo_text_count = np.zeros(num_beats, dtype=np.float32)
    tokenizer_tempo_active = np.zeros(num_beats, dtype=np.float32)
    tokenizer_rit_active = np.zeros(num_beats, dtype=np.float32)
    tokenizer_accel_active = np.zeros(num_beats, dtype=np.float32)

    for el in score.recurse():
        offset = _absolute_offset(el, score, score)
        beat_idx = int(math.floor((offset + eps) / beat_unit))
        if beat_idx < 0 or beat_idx >= num_beats:
            continue

        if isinstance(el, music21.dynamics.Dynamic):
            dynamic_mark_count[beat_idx] += 1.0
            dynamic_level = _map_dynamic_level(getattr(el, "value", None))
            dynamic_level_sum[beat_idx] += dynamic_level
            dynamic_level_updates[beat_idx] = dynamic_level
            continue

        if not isinstance(el, music21.expressions.TextExpression):
            continue

        text = str(getattr(el, "content", "") or "").strip().lower()
        if not text:
            continue
        tokenizer_text_count[beat_idx] += 1.0
        if text in TOKENIZER_VELOCITY_TEXTS:
            tokenizer_velocity_text_count[beat_idx] += 1.0
        if text in TOKENIZER_TEMPO_TEXTS:
            tokenizer_tempo_text_count[beat_idx] += 1.0
            duration = float(getattr(getattr(el, "duration", None), "quarterLength", DEFAULT_TEMPO_DURATION) or DEFAULT_TEMPO_DURATION)
            end_idx = int(math.ceil((offset + duration - eps) / beat_unit))
            end_idx = min(max(end_idx, beat_idx + 1), num_beats)
            tokenizer_tempo_active[beat_idx:end_idx] = 1.0
            if text.startswith("rit"):
                tokenizer_rit_active[beat_idx:end_idx] = 1.0
            if text.startswith("accel"):
                tokenizer_accel_active[beat_idx:end_idx] = 1.0

    for part_idx, part in enumerate(parts):
        part_diff = np.zeros(num_beats + 1, dtype=np.float32)
        for el in part.recurse().notesAndRests:
            offset = _absolute_offset(el, score, part)
            duration = float(el.quarterLength)
            if duration <= 0:
                continue
            beat_idx = int(math.floor((offset + eps) / beat_unit))
            if beat_idx < 0 or beat_idx >= num_beats:
                continue

            beat_strength = float(getattr(el, "beatStrength", 0.0) or 0.0)
            beat_strength_sum[beat_idx] += beat_strength
            beat_strength_count[beat_idx] += 1.0

            if el.isRest:
                rest_count[beat_idx] += 1.0
                rest_duration[beat_idx] += duration
                if any("fermata" in type(expr).__name__.lower() for expr in getattr(el, "expressions", [])):
                    fermata_count[beat_idx] += 1.0
                continue

            note_objects = list(el.notes) if el.isChord else [el]
            pitches = [n.pitch for n in note_objects]
            note_event_count[beat_idx] += float(len(note_objects))

            if part_idx == 0:
                upper_staff_count[beat_idx] += float(len(note_objects))
            if part_idx == last_part_idx and len(parts) > 1:
                lower_staff_count[beat_idx] += float(len(note_objects))

            tie_obj = getattr(el, "tie", None)
            if tie_obj is not None:
                tie_type = str(getattr(tie_obj, "type", "")).lower()
                if tie_type == "start":
                    tie_start_count[beat_idx] += 1.0
                elif tie_type == "stop":
                    tie_stop_count[beat_idx] += 1.0
                elif tie_type == "continue":
                    tie_continue_count[beat_idx] += 1.0

            if any("fermata" in type(expr).__name__.lower() for expr in getattr(el, "expressions", [])):
                fermata_count[beat_idx] += 1.0

            slur_start, slur_end = _slur_flags(el)
            slur_start_count[beat_idx] += slur_start
            slur_end_count[beat_idx] += slur_end

            for note_obj in note_objects:
                if any(isinstance(expr, music21.articulations.Fingering) for expr in getattr(note_obj, "expressions", [])):
                    fingering_count[beat_idx] += 1.0
                    finger_numbers = [
                        int(expr.number)
                        for expr in getattr(note_obj, "expressions", [])
                        if isinstance(expr, music21.articulations.Fingering) and getattr(expr, "number", None) is not None
                    ]
                    if finger_numbers:
                        fingering_sum[beat_idx] += float(np.mean(finger_numbers))
                try:
                    if any(isinstance(spanner, music21.spanner.Slur) for spanner in note_obj.getSpannerSites()):
                        tokenizer_slur_active_count[beat_idx] += 1.0
                except Exception:
                    continue

            end_idx = int(math.ceil((offset + duration - eps) / beat_unit)) - 1
            end_idx = min(max(end_idx, beat_idx), num_beats - 1)
            part_diff[beat_idx] += 1.0
            part_diff[end_idx + 1] -= 1.0
            if part_idx == 0:
                upper_staff_diff[beat_idx] += float(len(note_objects))
                upper_staff_diff[end_idx + 1] -= float(len(note_objects))
            if part_idx == last_part_idx and len(parts) > 1:
                lower_staff_diff[beat_idx] += float(len(note_objects))
                lower_staff_diff[end_idx + 1] -= float(len(note_objects))
            for pitch in pitches:
                pc = int(pitch.pitchClass)
                active_pc_diff[beat_idx, pc] += 1.0
                active_pc_diff[end_idx + 1, pc] -= 1.0

        part_active = np.cumsum(part_diff[:-1]) > 0
        part_active_matrix.append(part_active.astype(np.float32))

    active_part_count = np.zeros(num_beats, dtype=np.float32)
    part_entry_count = np.zeros(num_beats, dtype=np.float32)
    part_exit_count = np.zeros(num_beats, dtype=np.float32)
    if part_active_matrix:
        part_active_arr = np.stack(part_active_matrix, axis=0)
        active_part_count = part_active_arr.sum(axis=0).astype(np.float32)
        prev_state = np.pad(part_active_arr[:, :-1], ((0, 0), (1, 0)), constant_values=False)
        next_state = np.pad(part_active_arr[:, 1:], ((0, 0), (0, 1)), constant_values=False)
        part_entry_count = np.logical_and(part_active_arr, np.logical_not(prev_state)).sum(axis=0).astype(np.float32)
        part_exit_count = np.logical_and(part_active_arr, np.logical_not(next_state)).sum(axis=0).astype(np.float32)

    active_pc_hist = np.cumsum(active_pc_diff[:-1], axis=0).astype(np.float32)
    active_pc_total = active_pc_hist.sum(axis=1)
    active_pc_dist = _safe_divide(active_pc_hist, np.maximum(active_pc_total[:, None], 1.0))
    upper_staff_active = np.cumsum(upper_staff_diff[:-1]).astype(np.float32)
    lower_staff_active = np.cumsum(lower_staff_diff[:-1]).astype(np.float32)
    prev_sim = np.zeros(num_beats, dtype=np.float32)
    next_sim = np.zeros(num_beats, dtype=np.float32)
    norms = np.linalg.norm(active_pc_dist, axis=1)
    for idx in range(num_beats):
        if idx > 0 and norms[idx] > 1e-8 and norms[idx - 1] > 1e-8:
            prev_sim[idx] = float(np.dot(active_pc_dist[idx], active_pc_dist[idx - 1]) / (norms[idx] * norms[idx - 1]))
        if idx + 1 < num_beats and norms[idx] > 1e-8 and norms[idx + 1] > 1e-8:
            next_sim[idx] = float(np.dot(active_pc_dist[idx], active_pc_dist[idx + 1]) / (norms[idx] * norms[idx + 1]))
    active_pc_entropy = (-np.sum(active_pc_dist * np.log(active_pc_dist + 1e-8), axis=1)).astype(np.float32)
    active_pc_unique_ratio = ((active_pc_hist > 0).sum(axis=1).astype(np.float32) / 12.0).astype(np.float32)
    harmonic_change = (1.0 - prev_sim).astype(np.float32)

    tonic_pc, dominant_pc = _score_to_key_pcs(score)
    tonic_weight = np.zeros(num_beats, dtype=np.float32)
    dominant_weight = np.zeros(num_beats, dtype=np.float32)
    cadence_dom_tonic = np.zeros(num_beats, dtype=np.float32)
    if tonic_pc is not None and dominant_pc is not None:
        tonic_weight = active_pc_dist[:, tonic_pc].astype(np.float32)
        dominant_weight = active_pc_dist[:, dominant_pc].astype(np.float32)
        prev_dominant = np.roll(dominant_weight, 1)
        prev_dominant[0] = 0.0
        cadence_dom_tonic = (prev_dominant * tonic_weight).astype(np.float32)

    beat_strength = _safe_divide(beat_strength_sum, beat_strength_count)
    strongbeat_actual = (beat_strength >= 0.5).astype(np.float32)
    rest_duration_norm = np.clip(rest_duration / max(beat_unit, 1.0), 0.0, 4.0).astype(np.float32)
    prev_rest_duration = np.roll(rest_duration_norm, 1)
    prev_rest_duration[0] = 0.0
    next_rest_duration = np.roll(rest_duration_norm, -1)
    next_rest_duration[-1] = 0.0
    tie_start_ratio = _safe_divide(tie_start_count, np.maximum(note_event_count, 1.0))
    tie_stop_ratio = _safe_divide(tie_stop_count, np.maximum(note_event_count, 1.0))
    tie_continue_ratio = _safe_divide(tie_continue_count, np.maximum(note_event_count, 1.0))
    slur_start_ratio = _safe_divide(slur_start_count, np.maximum(note_event_count, 1.0))
    slur_end_ratio = _safe_divide(slur_end_count, np.maximum(note_event_count, 1.0))
    tokenizer_slur_active_ratio = _safe_divide(tokenizer_slur_active_count, np.maximum(note_event_count, 1.0))
    upper_staff_ratio = _safe_divide(upper_staff_count, np.maximum(note_event_count, 1.0))
    lower_staff_ratio = _safe_divide(lower_staff_count, np.maximum(note_event_count, 1.0))
    staff_balance = np.abs(upper_staff_ratio - lower_staff_ratio).astype(np.float32)
    both_staff_active = np.logical_and(upper_staff_active > 0.0, lower_staff_active > 0.0).astype(np.float32)
    fingering_ratio = _safe_divide(fingering_count, np.maximum(note_event_count, 1.0))
    fingering_mean_norm = np.clip(_safe_divide(fingering_sum, np.maximum(fingering_count, 1.0)) / 5.0, 0.0, 1.0).astype(np.float32)
    dynamic_level_avg = _safe_divide(dynamic_level_sum, np.maximum(dynamic_mark_count, 1.0))
    dynamic_level_active = np.zeros(num_beats, dtype=np.float32)
    current_dynamic = 0.0
    for beat_idx in range(num_beats):
        if dynamic_mark_count[beat_idx] > 0:
            current_dynamic = float(dynamic_level_avg[beat_idx])
            dynamic_level_updates[beat_idx] = current_dynamic
        dynamic_level_active[beat_idx] = current_dynamic
    dynamic_level_norm = np.clip((dynamic_level_active + 4.0) / 8.0, 0.0, 1.0).astype(np.float32)
    phrase_stop_proxy = np.clip(
        np.maximum.reduce(
            [
                (fermata_count > 0).astype(np.float32),
                slur_end_ratio,
                repeat_end,
                final_barline,
                np.clip(next_rest_duration / 2.0, 0.0, 1.0),
            ]
        ),
        0.0,
        1.0,
    ).astype(np.float32)

    if np.any(measure_number > 0):
        measure_index_norm = measure_number / max(float(measure_number.max()), 1.0)
    else:
        measure_index_norm = np.zeros(num_beats, dtype=np.float32)

    rows = {
        "beat_idx": np.arange(num_beats, dtype=np.int32),
        "xml_measure_number_norm": measure_index_norm.astype(np.float32),
        "xml_measure_start": measure_start.astype(np.float32),
        "xml_measure_end": measure_end.astype(np.float32),
        "xml_measure_progress": measure_progress.astype(np.float32),
        "xml_beat_in_measure_norm": beat_in_measure_norm.astype(np.float32),
        "xml_beats_in_measure_norm": beats_in_measure_norm.astype(np.float32),
        "xml_downbeat_actual": downbeat_actual.astype(np.float32),
        "xml_strongbeat_actual": strongbeat_actual.astype(np.float32),
        "xml_beat_strength": np.clip(beat_strength, 0.0, 1.0).astype(np.float32),
        "xml_time_sig_numerator_norm": time_sig_numerator_norm.astype(np.float32),
        "xml_time_sig_denominator_norm": time_sig_denominator_norm.astype(np.float32),
        "xml_key_sharps_norm": np.clip(key_sharps_norm, 0.0, 1.0).astype(np.float32),
        "xml_rest_count": rest_count.astype(np.float32),
        "xml_rest_duration_norm": rest_duration_norm.astype(np.float32),
        "xml_prev_rest_duration_norm": prev_rest_duration.astype(np.float32),
        "xml_next_rest_duration_norm": next_rest_duration.astype(np.float32),
        "xml_tie_start_ratio": np.clip(tie_start_ratio, 0.0, 1.0).astype(np.float32),
        "xml_tie_stop_ratio": np.clip(tie_stop_ratio, 0.0, 1.0).astype(np.float32),
        "xml_tie_continue_ratio": np.clip(tie_continue_ratio, 0.0, 1.0).astype(np.float32),
        "xml_fermata_count": fermata_count.astype(np.float32),
        "xml_slur_start_ratio": np.clip(slur_start_ratio, 0.0, 1.0).astype(np.float32),
        "xml_slur_end_ratio": np.clip(slur_end_ratio, 0.0, 1.0).astype(np.float32),
        "xml_phrase_stop_proxy": phrase_stop_proxy.astype(np.float32),
        "xml_repeat_start": repeat_start.astype(np.float32),
        "xml_repeat_end": repeat_end.astype(np.float32),
        "xml_final_barline": final_barline.astype(np.float32),
        "xml_active_part_count_norm": _safe_divide(active_part_count, np.full_like(active_part_count, max(active_part_count.max(), 1.0))).astype(np.float32),
        "xml_part_entry_count_norm": _safe_divide(part_entry_count, np.maximum(active_part_count, 1.0)).astype(np.float32),
        "xml_part_exit_count_norm": _safe_divide(part_exit_count, np.maximum(active_part_count, 1.0)).astype(np.float32),
        "xml_active_pc_entropy": active_pc_entropy.astype(np.float32),
        "xml_active_pc_unique_ratio": np.clip(active_pc_unique_ratio, 0.0, 1.0).astype(np.float32),
        "xml_active_pc_cos_prev": np.clip(prev_sim, -1.0, 1.0).astype(np.float32),
        "xml_active_pc_cos_next": np.clip(next_sim, -1.0, 1.0).astype(np.float32),
        "xml_harmonic_change": np.clip(harmonic_change, 0.0, 1.0).astype(np.float32),
        "xml_tonic_weight": np.clip(tonic_weight, 0.0, 1.0).astype(np.float32),
        "xml_dominant_weight": np.clip(dominant_weight, 0.0, 1.0).astype(np.float32),
        "xml_cadence_dom_tonic": np.clip(cadence_dom_tonic, 0.0, 1.0).astype(np.float32),
        "xml_tok_dynamic_mark_count": dynamic_mark_count.astype(np.float32),
        "xml_tok_dynamic_level_norm": dynamic_level_norm.astype(np.float32),
        "xml_tok_text_expr_count": tokenizer_text_count.astype(np.float32),
        "xml_tok_velocity_text_count": tokenizer_velocity_text_count.astype(np.float32),
        "xml_tok_tempo_text_count": tokenizer_tempo_text_count.astype(np.float32),
        "xml_tok_tempo_active": tokenizer_tempo_active.astype(np.float32),
        "xml_tok_rit_active": tokenizer_rit_active.astype(np.float32),
        "xml_tok_accel_active": tokenizer_accel_active.astype(np.float32),
        "xml_tok_upper_staff_ratio": np.clip(upper_staff_ratio, 0.0, 1.0).astype(np.float32),
        "xml_tok_lower_staff_ratio": np.clip(lower_staff_ratio, 0.0, 1.0).astype(np.float32),
        "xml_tok_staff_balance": np.clip(staff_balance, 0.0, 1.0).astype(np.float32),
        "xml_tok_both_staff_active": both_staff_active.astype(np.float32),
        "xml_tok_fingering_ratio": np.clip(fingering_ratio, 0.0, 1.0).astype(np.float32),
        "xml_tok_fingering_mean_norm": fingering_mean_norm.astype(np.float32),
        "xml_tok_slur_active_ratio": np.clip(tokenizer_slur_active_ratio, 0.0, 1.0).astype(np.float32),
    }
    return pd.DataFrame(rows)


def extract_xml_beat_features(
    piece_id: str,
    xml_dir: Path,
    num_beats: int,
    beat_unit: float = 1.0,
    expand_repeats: bool = True,
) -> pd.DataFrame:
    xml_path = resolve_xml_score_path(piece_id=piece_id, xml_dir=xml_dir)
    return _extract_xml_beat_features_cached(
        xml_path_str=str(xml_path.resolve()),
        num_beats=int(num_beats),
        beat_unit=float(beat_unit),
        expand_repeats=bool(expand_repeats),
    ).copy()


def extract_xml_beat_features_from_path(
    xml_path: Path,
    num_beats: int,
    beat_unit: float = 1.0,
    expand_repeats: bool = True,
) -> pd.DataFrame:
    return _extract_xml_beat_features_cached(
        xml_path_str=str(Path(xml_path).resolve()),
        num_beats=int(num_beats),
        beat_unit=float(beat_unit),
        expand_repeats=bool(expand_repeats),
    ).copy()
