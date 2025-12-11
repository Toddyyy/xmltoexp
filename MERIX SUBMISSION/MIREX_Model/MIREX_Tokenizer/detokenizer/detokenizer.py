import logging
from dataclasses import dataclass
from typing import Literal
import mido
import numpy as np
from music21.pitch import Pitch
from scipy.interpolate import CubicSpline
from scipy.special import expit
from tokenizer.base_score_tokenizer import ScoreNoteToken

logger = logging.getLogger(__name__)

TICKS_PER_BEAT_DEFAULT = 480
ONE_MINUTE = 60.0
MIDI_CONTROL_VALUE_DEFAULT = 64
MIDI_TEMPO_DEFAULT = 120.0
FIRST_EVENT_SECOND = 0.0
GAP_BETWEEN_NOTES = 0.1
MAX_NOTE_DURATION_SECONDS = 8.0
SAFE_EPSILON_MULTIPLIER = 1.1
MINIMUM_DURATION_RATIO = 0.25
OFFSET_ESTIMATION_WINDOW_SIZE = 5

type TPitch = int
type TIndex = int


@dataclass(frozen=True)
class RenderingPerformanceNoteToken:
    velocity: int
    onset_deviation_in_sec: float
    duration_deviation_in_sec: float
    local_tempo: float
    sustain_level: int
    segment_avg_tempo: float


@dataclass(frozen=True)
class PerformanceEvent:
    time: float
    type: Literal["control_change", "note_on", "note_off"]
    pitch: int | None = None
    velocity: int | None = None
    control: int | None = None
    value: int | None = None


@dataclass(frozen=True)
class ScoreNoteTokenWithLocalTempoContext:
    note: ScoreNoteToken

    def estimate_score_onset_seconds(
        self,
        predicted_chunk_local_tempo: float,
        last_chunk_score_offset_beats: float,
        last_chunk_perf_offset_seconds: float,
    ) -> float:
        seconds_per_beat = ONE_MINUTE / predicted_chunk_local_tempo
        onset = (
            self.note.position - last_chunk_score_offset_beats
        ) * seconds_per_beat + last_chunk_perf_offset_seconds
        return onset

    def estimate_score_duration_seconds(
        self,
        predicted_chunk_local_tempo: float,
        last_chunk_score_offset_beats: float,
        last_chunk_perf_offset_seconds: float,
    ) -> float:
        seconds_per_beat = ONE_MINUTE / predicted_chunk_local_tempo
        duration = self.note.duration * seconds_per_beat
        return duration


def convert_pitch_str_to_midi_int(pitch_str: str) -> int:
    return Pitch(pitch_str).midi


def apply_safe_duration_deviation(
    score_duration_sec: float, predicted_deviation_sec: float
) -> float:
    if predicted_deviation_sec >= 0:
        return predicted_deviation_sec

    gate = 2 * expit(predicted_deviation_sec)
    gated_deviation = gate * predicted_deviation_sec

    min_allowable_duration = score_duration_sec * MINIMUM_DURATION_RATIO
    final_duration = score_duration_sec + gated_deviation

    if final_duration < min_allowable_duration:
        final_safe_deviation = min_allowable_duration - score_duration_sec
        return final_safe_deviation
    else:
        return gated_deviation


def map_prediction_to_performance_time(
    notes: list[ScoreNoteTokenWithLocalTempoContext],
    performance_predictions: list[RenderingPerformanceNoteToken],
    chunk_offsets: tuple[float, float],
    average_tempo: float,
) -> tuple[list[float], list[float]]:
    eps = 0.001

    system_offset_sec = 0.0
    if performance_predictions:
        system_offset_sec = performance_predictions[0].onset_deviation_in_sec
    adjusted_chunk_offsets = (chunk_offsets[0], chunk_offsets[1] + system_offset_sec)

    prev_offset = adjusted_chunk_offsets[1]
    note_onset_in_seconds: list[float] = []
    note_offset_in_seconds: list[float] = []

    for i, (note, prediction) in enumerate(zip(notes, performance_predictions)):

        local_onset_deviation = prediction.onset_deviation_in_sec if i > 0 else 0.0
        final_onset_seconds = max(
            eps,
            note.estimate_score_onset_seconds(average_tempo, *adjusted_chunk_offsets)
            + local_onset_deviation,
        )

        baseline_duration_sec = note.estimate_score_duration_seconds(
            average_tempo, *adjusted_chunk_offsets
        )
        safe_duration_deviation_sec = apply_safe_duration_deviation(
            score_duration_sec=baseline_duration_sec,
            predicted_deviation_sec=prediction.duration_deviation_in_sec,
        )
        final_duration_seconds = max(
            eps,
            baseline_duration_sec + safe_duration_deviation_sec,
        )

        final_offset_seconds = max(
            final_onset_seconds + final_duration_seconds, prev_offset + eps
        )
        prev_offset = final_offset_seconds

        note_onset_in_seconds.append(final_onset_seconds)
        note_offset_in_seconds.append(final_offset_seconds)

    return note_onset_in_seconds, note_offset_in_seconds


def generate_performance_events_in_seconds(
    notes: list[ScoreNoteTokenWithLocalTempoContext],
    performance_predictions: list[RenderingPerformanceNoteToken],
    average_tempo: float,
    chunk_offsets: tuple[float, float],
    chunk_number: int,
) -> list[PerformanceEvent]:

    eps = 0.001
    events_in_seconds: list[PerformanceEvent] = []
    prev_sustain: int = 0

    note_onset_in_seconds, note_offset_in_seconds = map_prediction_to_performance_time(
        notes, performance_predictions, chunk_offsets, average_tempo
    )

    # remove overlap
    pitches = [convert_pitch_str_to_midi_int(note.note.pitch) for note in notes]
    original_offsets_before_correction = note_offset_in_seconds.copy()
    note_offset_in_seconds = remove_overlaps(
        pitches,
        note_onset_in_seconds,
        note_offset_in_seconds,
        average_tempo=average_tempo,
    )

    for i, (note, prediction) in enumerate(zip(notes, performance_predictions)):
        onset = note_onset_in_seconds[i]
        offset = note_offset_in_seconds[i]

        # Sustain
        if prediction.sustain_level != prev_sustain:
            sustain_time = max(0.0, onset - eps)
            events_in_seconds.append(
                PerformanceEvent(
                    time=sustain_time,
                    type="control_change",
                    control=MIDI_CONTROL_VALUE_DEFAULT,
                    value=prediction.sustain_level,
                )
            )
            prev_sustain = prediction.sustain_level

        # Split a note into note_on and note_off events
        events_in_seconds.extend(
            [
                PerformanceEvent(
                    time=onset,
                    type="note_on",
                    pitch=convert_pitch_str_to_midi_int(note.note.pitch),
                    velocity=prediction.velocity,
                ),
                PerformanceEvent(
                    time=offset,
                    type="note_off",
                    pitch=convert_pitch_str_to_midi_int(note.note.pitch),
                    velocity=0,
                ),
            ]
        )
    events_in_seconds.sort(key=lambda e: e.time)
    return events_in_seconds


def assemble_messages(
    events_in_seconds: list[PerformanceEvent],
    target_track: mido.MidiTrack,
    ticks_per_beat: int,
    first_event_offset_in_seconds: float,
    chunk_average_tempo: float,
) -> None:
    midi_tempo = mido.bpm2tempo(chunk_average_tempo)

    def second_to_tick(
        seconds: float, ticks_per_quarter_note: int, tempo_bpm: float
    ) -> int:
        return int(seconds * ticks_per_quarter_note * 1000000 / tempo_bpm + 0.5)

    for event in events_in_seconds:
        delta_seconds = max(0.0, event.time - first_event_offset_in_seconds)
        first_event_offset_in_seconds = event.time
        delta_ticks = second_to_tick(delta_seconds, ticks_per_beat, midi_tempo)
        msg = None
        event_type = event.type
        if event_type == "note_on":
            msg = mido.Message(
                "note_on",
                note=event.pitch,
                velocity=event.velocity,
                time=delta_ticks,
            )
        elif event_type == "note_off":
            msg = mido.Message(
                "note_off",
                note=event.pitch,
                velocity=event.velocity,
                time=delta_ticks,
            )
        elif event_type == "control_change":
            msg = mido.Message(
                "control_change",
                control=event.control,
                value=event.value,
                time=delta_ticks,
            )
        if msg:
            target_track.append(msg)


def estimate_chunk_tempo(performance_predictions, score_note_tokens):
    average_tempo = performance_predictions[0].segment_avg_tempo
    return average_tempo


def detokenize_chunk(
    track: mido.MidiTrack,
    performance_predictions: list[RenderingPerformanceNoteToken],
    score_note_tokens: list[ScoreNoteToken],
    chunk_offsets: tuple[float, float],
    chunk_number: int,
) -> list[PerformanceEvent]:
    # Estimate average tempo for the chunk
    average_tempo = estimate_chunk_tempo(performance_predictions, score_note_tokens)
    notes_with_predicted_tempo: list[ScoreNoteTokenWithLocalTempoContext] = [
        ScoreNoteTokenWithLocalTempoContext(note=note) for note in score_note_tokens
    ]
    # Write events
    events_in_seconds = generate_performance_events_in_seconds(
        notes_with_predicted_tempo,
        performance_predictions,
        average_tempo=average_tempo,
        chunk_offsets=chunk_offsets,
        chunk_number=chunk_number,
    )
    track.append(
        mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(average_tempo), time=0)
    )
    chunk_start_second = chunk_offsets[1]
    assemble_messages(
        events_in_seconds,
        track,
        TICKS_PER_BEAT_DEFAULT,
        chunk_start_second,
        average_tempo,
    )

    return events_in_seconds


def detokenize_to_midi(
    score_note_tokens: list[ScoreNoteToken],
    performance_predictions: list[RenderingPerformanceNoteToken],
    output_midi_path: str,
    chunk_size: int,
):
    # Set up a MIDI track
    track = mido.MidiTrack()
    midi_file = mido.MidiFile(ticks_per_beat=TICKS_PER_BEAT_DEFAULT)
    midi_file.tracks.append(track)

    all_generated_events: list[PerformanceEvent] = []

    total_beat_offset = 0.0
    total_second_offset = 0.0
    for i in range(0, len(performance_predictions), chunk_size):
        chunk_number = i // chunk_size + 1

        current_chunk_predictions = performance_predictions[i : i + chunk_size]
        current_chunk_score_tokens = score_note_tokens[i : i + chunk_size]

        next_chunk_start_index = i + chunk_size
        if next_chunk_start_index < len(score_note_tokens):
            next_chunk_first_note = score_note_tokens[next_chunk_start_index]
        else:
            next_chunk_first_note = None
        next_chunk_offsets = get_chunk_offset_beats_and_seconds(
            current_chunk_predictions,
            current_chunk_score_tokens,
            next_chunk_first_note,
            (total_beat_offset, total_second_offset),
        )

        current_chunk_offsets = (total_beat_offset, total_second_offset)

        chunk_events = detokenize_chunk(
            track,
            current_chunk_predictions,
            current_chunk_score_tokens,
            chunk_offsets=current_chunk_offsets,
            chunk_number=chunk_number,
        )
        all_generated_events.extend(chunk_events)

        total_beat_offset = next_chunk_offsets[0]
        total_second_offset = next_chunk_offsets[1]

    # Tempo smoothing
    tempo_smoothed_track = smooth_tempo(track)
    midi_file.tracks[0] = tempo_smoothed_track

    midi_file.save(output_midi_path)
    logger.info(f"MIDI file successfully generated at: {output_midi_path}")


def get_chunk_offset_beats_and_seconds(
    current_chunk_predictions: list[RenderingPerformanceNoteToken],
    current_chunk_score_tokens: list[ScoreNoteToken],
    next_chunk_first_score_note: ScoreNoteToken,
    current_chunk_offset: tuple[float, float],
) -> tuple[float, float]:

    if next_chunk_first_score_note is None:
        return 0.0, 0.0

    b2 = next_chunk_first_score_note.position
    current_chunk_average_tempo = estimate_chunk_tempo(
        current_chunk_predictions, current_chunk_score_tokens
    )
    seconds_per_beat = ONE_MINUTE / current_chunk_average_tempo

    window_size = min(len(current_chunk_score_tokens), OFFSET_ESTIMATION_WINDOW_SIZE)
    window_score_notes = current_chunk_score_tokens[-window_size:]
    window_predictions = current_chunk_predictions[-window_size:]

    all_t2_estimates: list[float] = []

    for note, predictions in zip(window_score_notes, window_predictions):
        note_with_context = ScoreNoteTokenWithLocalTempoContext(note)
        final_onset_sec = (
            note_with_context.estimate_score_onset_seconds(
                current_chunk_average_tempo, *current_chunk_offset
            )
            + predictions.onset_deviation_in_sec
        )

        t2_estimate = final_onset_sec + (b2 - note.position) * seconds_per_beat
        all_t2_estimates.append(t2_estimate)

    t2 = float(np.mean(all_t2_estimates))
    return b2, t2


def remove_overlaps(
    pitches: list[int],
    note_onset_in_seconds: list[float],
    note_offset_in_seconds: list[float],
    average_tempo: float,
) -> list[float]:
    """Remove overlapping onsets and offsets. Truncate long notes to right before the
        next note onset.

    Returns:
        (list[float]): A list of updated offsets in seconds.
    """
    seconds_per_beat = ONE_MINUTE / average_tempo
    seconds_per_tick = seconds_per_beat / TICKS_PER_BEAT_DEFAULT

    safe_epsilon = seconds_per_tick * SAFE_EPSILON_MULTIPLIER
    min_note_duration_seconds = seconds_per_tick

    # Get the original index of each note
    onset_events = sorted(
        zip(note_onset_in_seconds, pitches, range(len(pitches))), key=lambda x: x[0]
    )
    updated_offsets = note_offset_in_seconds.copy()
    pitch_states: dict[TPitch, TIndex] = {}

    for current_onset, current_pitch, current_index in onset_events:
        # Check if the note was played before
        if current_pitch in pitch_states:
            previous_note_index = pitch_states[current_pitch]
            previous_note_onset = note_onset_in_seconds[previous_note_index]
            overlap_limit_offset = current_onset - safe_epsilon

            if updated_offsets[previous_note_index] > overlap_limit_offset:
                min_duration_limit_offset = (
                    previous_note_onset + min_note_duration_seconds
                )
                safe_new_offset = max(overlap_limit_offset, min_duration_limit_offset)
                updated_offsets[previous_note_index] = min(
                    updated_offsets[previous_note_index], safe_new_offset
                )
        # Update states list
        pitch_states[current_pitch] = current_index

    for i in range(len(updated_offsets)):
        onset = note_onset_in_seconds[i]
        max_allowed_offset = onset + MAX_NOTE_DURATION_SECONDS
        updated_offsets[i] = min(updated_offsets[i], max_allowed_offset)

    return updated_offsets


# spline curve
def smooth_tempo(
    track: mido.MidiTrack,
    *,
    smoothing_steps: int = 10,
    bpm_min: float = 10.0,
    bpm_max: float = 400.0,
    min_bpm_jump: float = 0.5,  # Minimum BPM change to keep (skip tiny changes)
    min_tick_gap: int = 1,  # Minimum tick gap between set_tempo events
) -> mido.MidiTrack:
    """
    Keep the first set_tempo event, smooth subsequent set_tempo events, and merge back into the track.
    - Non-tempo events remain unchanged
    - Use CubicSpline interpolation for smooth tempo transitions
    - Optionally enforce a set_tempo at tick=0 to avoid falling back to default 120 BPM

    """
    # 1) Collect all tempo events (absolute ticks), deduplicate same-tick entries
    tempo_events = []
    abs_time = 0
    track_end_time = 0
    for msg in track:
        abs_time += msg.time
        track_end_time = abs_time
        if msg.type == "set_tempo":
            tempo_events.append((abs_time, float(mido.tempo2bpm(msg.tempo))))
    if len(tempo_events) < 2:
        return track

    dedup = []
    for t, b in tempo_events:
        if dedup and t == dedup[-1][0]:
            dedup[-1] = (t, b)  # keep the last tempo at this tick
        else:
            dedup.append((t, b))
    tempo_events = dedup

    # 2) Save the first tempo event, smooth the rest
    t0, bpm0 = tempo_events[0]

    # 3) Build control points (midpoints) for spline
    control_points = []
    for i in range(len(tempo_events)):
        ct, cb = tempo_events[i]
        if i == 0:
            nt = tempo_events[i + 1][0] if i + 1 < len(tempo_events) else track_end_time
            mid = ct + (nt - ct) * 0.5
        elif i == len(tempo_events) - 1:
            mid = ct + (track_end_time - ct) * 0.5
        else:
            pt = tempo_events[i - 1][0]
            nt = tempo_events[i + 1][0]
            mid = pt + (nt - pt) * 0.5
        control_points.append((mid, cb))

    times, tempos = zip(*control_points)

    # Ensure strictly increasing times (CubicSpline requirement)
    times_np = np.array(times, dtype=float)
    tempos_np = np.array(tempos, dtype=float)
    keep = np.concatenate(([True], np.diff(times_np) > 0))
    times_np = times_np[keep]
    tempos_np = tempos_np[keep]

    cs = CubicSpline(times_np, tempos_np, bc_type="natural")

    # 4) Generate new tempo points
    new_tempo_points = []
    new_tempo_points.append((float(t0), float(bpm0)))

    for i in range(len(times_np) - 1):
        start_time = times_np[i]
        end_time = times_np[i + 1]
        if end_time <= t0:
            continue
        seg_start = max(start_time, t0)
        t_seg = np.linspace(seg_start, end_time, smoothing_steps + 2)
        y_seg = np.clip(cs(t_seg), bpm_min, bpm_max)
        new_tempo_points.extend((float(t), float(y)) for t, y in zip(t_seg, y_seg))

    # 5) Compress tempo points (remove redundant ones)
    compressed = []
    last_t, last_b = None, None
    for t, b in sorted(new_tempo_points, key=lambda x: x[0]):
        if last_t is None:
            compressed.append((t, b))
            last_t, last_b = t, b
            continue
        if (t - last_t) >= min_tick_gap and abs(b - last_b) >= min_bpm_jump:
            compressed.append((t, b))
            last_t, last_b = t, b
    new_tempo_points = compressed

    # 6) Merge non-tempo events + new tempo events, rebuild delta times
    all_events = []
    abs_time = 0
    for msg in track:
        abs_time += msg.time
        if msg.type != "set_tempo":
            all_events.append((float(abs_time), msg))
    for t, b in new_tempo_points:
        all_events.append(
            (float(t), mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(b), time=0))
        )

    all_events.sort(key=lambda x: x[0])

    # Deduplicate: keep only the last set_tempo at the same tick
    merged = []
    for t, msg in all_events:
        if (
            merged
            and int(round(t)) == int(round(merged[-1][0]))
            and msg.type == "set_tempo"
            and merged[-1][1].type == "set_tempo"
        ):
            merged[-1] = (t, msg)
        else:
            merged.append((t, msg))

    # Rebuild track with delta times
    new_track = mido.MidiTrack()
    prev = 0
    for t, msg in merged:
        dt = max(0, int(round(t - prev)))
        new_track.append(msg.copy(time=dt))
        prev = int(round(t))

    # Ensure track starts with set_tempo
    if not new_track or new_track[0].type != "set_tempo":
        new_track.insert(
            0, mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(bpm0), time=0)
        )

    return new_track
