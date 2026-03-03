import logging
from collections.abc import Callable
from dataclasses import dataclass
from itertools import pairwise
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_smoothing_spline

from tokenizer.base_performance_tokenizer import (
    PerformanceControl,
    PerformanceMetadata,
    PerformanceNoteToken,
)
from tokenizer.base_score_tokenizer import (
    ScoreExpression,
    ScoreMetadata,
    ScoreNoteToken,
)

logger = logging.getLogger(__name__)


@dataclass
class FullPerformanceToken:
    performance_note_token: PerformanceNoteToken
    score_note_token: ScoreNoteToken
    onset_deviation_in_beats: float
    duration_deviation_in_beats: float
    local_tempo: float
    sustain_level: int = 0


class MissingPerformanceInformation(Exception): ...


class ScoreExplainer:
    def __init__(
        self,
        /,  # NOTE: kw only so that the order does't matter.
        score_tokens: list[ScoreNoteToken],
        score_expressions: list[ScoreExpression],
        score_meta: ScoreMetadata | None = None,
        performance_meta: PerformanceMetadata | None = None,
        performance_tokens: list[PerformanceNoteToken] | None = None,
        performance_controls: list[PerformanceControl] | None = None,
        tempo_estimation_time_window: float = 5,
    ) -> None:
        self.score_tokens: list[ScoreNoteToken] = score_tokens
        self.score_expressions: list[ScoreExpression] = score_expressions
        self.score_meta: ScoreMetadata | None = score_meta
        self.performance_meta: PerformanceMetadata | None = performance_meta
        self.performance_tokens: list[PerformanceNoteToken] | None = performance_tokens
        self.performance_controls: list[PerformanceControl] | None = (
            performance_controls
        )

        self.score_tokens_by_id: dict[str, ScoreNoteToken]
        self.perf_to_score: dict[PerformanceNoteToken, ScoreNoteToken]
        self.perf_tokens_by_id: dict[str, PerformanceNoteToken]
        self.note_ids: list[str] = sorted(
            list(n.xml_note_id for n in score_tokens if n.xml_note_id is not None)
        )

        # Stats
        self.total_score_time_beats: float = 0.0
        self.total_performance_time_seconds: float | None

        # Sorted performance tokens
        self.perf_tokens_time_ascending: list[PerformanceNoteToken]
        self.onset_deviation_in_beats: dict[PerformanceNoteToken, float] = {}
        self.duration_deviation_in_beats: dict[PerformanceNoteToken, float] = {}
        self.tempo_per_note: dict[PerformanceNoteToken, float] = {}
        self.sustain_level_at_notes: dict[PerformanceNoteToken, int] = {}

        self.segments_time_range: list[tuple[float, float]] | None = []

        self._build_note_id_maps()
        self._sort_notes()
        self._segmentation()
        self._fit_tempo_curve(tempo_estimation_time_window)
        self._downsample_sustains()

    def _build_note_id_maps(self):
        self.score_tokens_by_id = {
            note.xml_note_id: note for note in self.score_tokens if note.xml_note_id
        }

        if self.performance_tokens is None:
            logger.warning("Performance tokens are not provided.")
            return

        self.perf_to_score = {
            perf_note: self.score_tokens_by_id[perf_note.xml_note_id]
            for perf_note in self.performance_tokens
            if perf_note.xml_note_id in self.score_tokens_by_id
        }

    def _sort_notes(self):
        # NOTE: Sorting score notes is barely useful. So we only sort performance tokens.

        # Performance tokens
        if self.performance_tokens is None:
            logger.warning("Performance tokens are not provided.")
            return
        self.perf_tokens_time_ascending = sorted(
            self.performance_tokens, key=lambda note: note.onset_sec
        )
        if len(self.performance_tokens) > 0:
            last_note = self.performance_tokens[-1]
            self.total_performance_time_seconds = (
                last_note.onset_sec + last_note.duration_sec
            )

    def _downsample_sustains(self):
        """
        This method down samples the sustain events to as many as midi tokens.

        O(n log(n)) of sorting by time is required for the events.
        Downsampling takes O(n) time.
        """

        if self.performance_tokens is None:
            logger.warning("Performance is not provided.")
            return
        if self.performance_controls is None:
            logger.warning("Sustains are not provided.")
            return

        i, j = 0, 0
        while i < len(self.performance_tokens) and j < len(self.performance_controls):
            note, control = self.performance_tokens[i], self.performance_controls[j]
            t_i, t_j = note.onset_sec, control.time_in_seconds

            # As long as the sustian is earlier than the note, it's a possible value
            if t_i > t_j:
                self.sustain_level_at_notes[note] = control.data_value
                j += 1
            # Otherwise we try the next note.
            else:
                i += 1

    def _segmentation(self):
        if self.performance_tokens is None:
            raise MissingPerformanceInformation(
                "self.performance_tokens are missing for tempo estimation"
            )

        xs = np.array([n.onset_sec for n in self.perf_tokens_time_ascending])
        ys = np.array(
            [self.perf_to_score[n].position for n in self.perf_tokens_time_ascending]
        )

        # plt.plot(xs, ys, "x-", markersize=2)
        # plt.xlabel("sec")
        # plt.ylabel("beats")
        # plt.title("Expressive time mapping")
        # plt.show()

        # 1. split them into segments that don't have drastic change in
        # score position because of repetition

        # time secs when the score position suddenly changes
        seg_boundaries: list[float]

        threshold_beat_gap = 8  # let's say a minimal gap has 8 beats
        gaps = np.diff(ys)
        splits = np.nonzero((abs(gaps) > threshold_beat_gap))
        seg_boundaries = [xs[i] for i in splits[0]]  # type: ignore

        assert self.total_performance_time_seconds is not None
        seg_boundaries = [0] + seg_boundaries + [self.total_performance_time_seconds]
        # NOTE: Even if nothing happens in between, the entire piece is a segment.
        self.segments_time_range = []
        for start, end in pairwise(seg_boundaries):
            self.segments_time_range.append((start, end + 1e-7))

        assert len(self.segments_time_range) > 0

    def _fit_tempo_curve(self, time_window: float) -> None:
        assert self.segments_time_range is not None
        assert self.performance_tokens is not None

        self._segment_models: list[dict[str, Any]] = []

        # Collect notes for each segment and fit a smoothing spline
        for seg_id, (start, end) in enumerate(self.segments_time_range, start=1):
            notes: list[PerformanceNoteToken] = [
                n for n in self.perf_tokens_time_ascending
                if start <= n.onset_sec <= end
            ]
            notes_monotonic: list[PerformanceNoteToken] = []
            last_t = None
            for n in notes:
                if last_t is None or n.onset_sec > last_t:
                    notes_monotonic.append(n)
                    last_t = n.onset_sec
            if len(notes_monotonic) < 2:
                continue

            xs = np.array([n.onset_sec for n in notes_monotonic], dtype=float)
            ys = np.array([self.perf_to_score[n].position for n in notes_monotonic], dtype=float)

            spl = make_smoothing_spline(xs, ys, lam=0.1)

            self._segment_models.append({
                "id": seg_id,
                "start": float(start),
                "end": float(end),
                "spl": spl,
                "notes": notes,  
            })


        if not self._segment_models:
            raise ValueError("No valid segment models found.")

        # Check the first onset time, if the first segment starts too late, raise
        first_onset_sec = self._segment_models[0]["notes"][0].onset_sec
        if first_onset_sec > 20.0:
            raise ValueError(
                f"First onset is too late, Check segmentation: {first_onset_sec}"
            )

        # Plot the splines 
        # self._plot_segments()

        # Calculate local tempo for each note and the deviations
        for seg in self._segment_models:
            start = seg["start"]
            end = seg["end"]
            spl = seg["spl"]
            notes: list[PerformanceNoteToken] = seg["notes"]

            for note in notes:
                # Local window around the note for tempo estimation
                self._estimate_note_local_tempo(note, time_window, start, end, spl)
                # Fix zero tempo by global expansion
                self.tempo_per_note[note] = self._fix_zero_tempo(note)

            self._median_smooth_tempi(notes,outlier_only=True)

            for note in notes:
                t = note.onset_sec
                beat_estimated = float(spl(t))
                local_tempo_estimated = float(self.tempo_per_note.get(note, 0.0))
                self._estimate_deviations(note, beat_estimated, local_tempo_estimated)

    def _estimate_note_local_tempo(
        self,
        note: PerformanceNoteToken,
        time_window: float,
        start: float,
        end: float,
        spl: Callable,
    ):
        t = note.onset_sec
        t_left = t_right = t
        if t_left - time_window / 2 < start:
            t_left, t_right = start, start + time_window
        elif t_right + time_window / 2 > end:
            t_left, t_right = end - time_window, end
        else:
            t_left = t - time_window / 2
            t_right = t + time_window / 2
        beat_left, beat_right = spl(t_left), spl(t_right) 
        tempo = abs(60 * (beat_right - beat_left) / (t_right - t_left))
        self.tempo_per_note[note] = float(tempo)

    def _estimate_deviations(
        self,
        note: PerformanceNoteToken,
        beat_estimated: float,
        local_tempo_estimated: float,
    ):
        beat = beat_estimated
        tempo = local_tempo_estimated

        # Onset deviation: actual estiamted beat - score beat
        self.onset_deviation_in_beats[note] = float(
            beat - self.perf_to_score[note].position
        )
        actual_duration_in_beats = note.duration_sec * tempo / 60

        # Duration deviation: actual estiamted duration - score duration
        self.duration_deviation_in_beats[note] = float(
            actual_duration_in_beats - self.perf_to_score[note].duration
        )

    def _get_segment_index(self, time_in_seconds: float) -> int:
        assert self.total_performance_time_seconds is not None
        assert self.segments_time_range is not None

        answer = 0
        if time_in_seconds > self.total_performance_time_seconds:
            answer = len(self.segments_time_range) - 1

        for i, (start, end) in enumerate(self.segments_time_range):
            if start < time_in_seconds < end:
                return i

        return answer

    def get_full_tokens(self) -> list[FullPerformanceToken]:
        """
        Returns a list of FullPerformanceToken objects that contain
        performance and score tokens, onset and duration deviations,
        and local tempo for each note.

        NOTE: The tokens are sorted in ascending order by performance time.
        """
        if self.performance_tokens is None:
            raise MissingPerformanceInformation(
                "self.performance_tokens are missing for full token generation"
            )

        full_tokens: list[FullPerformanceToken] = []
        for perf_token in self.perf_tokens_time_ascending:
            score_token = self.perf_to_score.get(perf_token)
            assert score_token is not None

            onset_deviation = self.onset_deviation_in_beats[perf_token]
            duration_deviation = self.duration_deviation_in_beats[perf_token]
            local_tempo = self.tempo_per_note[perf_token]
            sustain_level = self.sustain_level_at_notes.get(perf_token, 0)

            full_tokens.append(
                FullPerformanceToken(
                    performance_note_token=perf_token,
                    score_note_token=score_token,
                    onset_deviation_in_beats=onset_deviation,
                    duration_deviation_in_beats=duration_deviation,
                    local_tempo=local_tempo,
                    sustain_level=sustain_level,
                )
            )

        return full_tokens

    def _plot_segments(self) -> None:
        if not self._segment_models:
            return
        cmap = plt.get_cmap("tab10")

        for i, seg in enumerate(self._segment_models):
            start, end, spl = seg["start"], seg["end"], seg["spl"]
            xs_plot = np.linspace(start, end, max(100, int((end - start) * 50)))
            ys_plot = spl(xs_plot)
            plt.plot(xs_plot, ys_plot, "-", label=f"Seg {seg['id']}", color=cmap(i % 10))

            note_xs = [n.onset_sec for n in seg["notes"]]
            note_ys = [float(spl(x)) for x in note_xs]
            plt.plot(note_xs, note_ys, "x", color=cmap(i % 10), markersize=2, markeredgewidth=1)

        plt.title("Time → Beat curves by segment")
        plt.xlabel("Time (sec)")
        plt.ylabel("Beat position")
        plt.legend()
        plt.grid(True, alpha=0.2)
        plt.show()

    def _fix_zero_tempo(
        self,
        note: PerformanceNoteToken,
        eps_beat: float = 1e-6,
        max_expand: int = 2000,
    ) -> float:
        """
        If initial_tempo is 0, expand globally from the note to both sides.
        If expansion reaches the next segment, use the next segment's spline for br.
        """
        initial_tempo = float(self.tempo_per_note[note])
        if initial_tempo != 0.0:
            return initial_tempo

        all_notes = self.perf_tokens_time_ascending
        ri = gidx = all_notes.index(note)
        amount = len(all_notes)
        steps = 0

        # Find current segment id
        seg_idx = self._get_segment_index(note.onset_sec)
        curr_seg = self._segment_models[seg_idx]
        curr_spl = curr_seg["spl"]
        curr_end = curr_seg["end"]

        # Try symmetric expansion; only expand right
        while steps < max_expand:
            moved = False
            if ri < amount - 1:
                ri += 1
                moved = True
            if not moved:
                break  # Both ends reached

            t_c = all_notes[gidx].onset_sec
            t_r = all_notes[ri].onset_sec

            # If right expansion crosses segment boundary, use next segment's spline
            if t_r > curr_end and seg_idx + 1 < len(self._segment_models):
                next_seg = self._segment_models[seg_idx + 1]
                next_spl = next_seg["spl"]
                b_r = float(next_spl(t_r))
            else:
                b_r = float(curr_spl(t_r))

            b_l = float(curr_spl(t_c))
            if abs(b_r - b_l) > eps_beat:
                denom = max(t_r - t_c, 1e-6)
                return abs(60.0 * (b_r - b_l) / denom)

            steps += 1

        # No distinguishable beat change found, keep 0
        return initial_tempo
    
    def _median_smooth_tempi(
        self,
        notes: list[PerformanceNoteToken],
        window_size: int = 5,
        outlier_only: bool = True,
        rel_tol: float = 0.35,
        abs_tol: float = 5.0,
    ) -> None:
        """
        Median smoothing for per-note tempo in the current segment:
        - outlier_only=True: only replace outliers (more conservative)
        - outlier_only=False: replace all with window median (smoother)
        """
        if not notes:
            return
        window_size = max(1, window_size | 1)  # Force odd window size
        half = window_size // 2

        # Extract tempo list in the order of notes
        tempi = [float(self.tempo_per_note.get(n, 0.0)) for n in notes]
        smoothed: list[float] = []

        for i in range(len(notes)):
            l = max(0, i - half)
            r = min(len(notes), i + half + 1)
            med = float(np.median(tempi[l:r]))
            x = tempi[i]
            

            if outlier_only:
                if (abs(x - med) > (abs_tol + rel_tol * max(med, 1e-6))):
                    print(f"Debug: NOTE {notes[i].onset_sec}, ORIGINAL: {x}, MEDIAN: {med}")
                    smoothed.append(med)
                else:
                    smoothed.append(x)
            else:
                smoothed.append(med)

        # Write back smoothed values
        for n, v in zip(notes, smoothed):
            self.tempo_per_note[n] = float(v)