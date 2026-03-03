import logging
import os
import re
from collections import defaultdict
from typing import Protocol, runtime_checkable

import numpy as np
import partitura as pt

from tokenizer.base_performance_tokenizer import (
    PerformanceControl,
    PerformanceMetadata,
    PerformanceNoteToken,
)
from tokenizer.musicxml_utils import safe_load_match

logger = logging.getLogger(__name__)


@runtime_checkable
class PerformanceTokenizer(Protocol):
    """Protocol for tokenizing performance data."""

    def tokenize(
        self,
    ) -> tuple[
        "PerformanceMetadata", list["PerformanceNoteToken"], list["PerformanceControl"]
    ]:
        """Tokenize the performance data."""
        raise NotImplementedError("Subclasses must implement this method.")

    def tokenize_notes(self) -> list["PerformanceNoteToken"]:
        """Tokenize the notes in the performance."""
        raise NotImplementedError("Subclasses must implement this method.")

    def parse_controls(self) -> list["PerformanceControl"]:
        """Parse control changes in the performance."""
        raise NotImplementedError("Subclasses must implement this method.")

    def parse_metadata(self) -> "PerformanceMetadata":
        """Parse metadata from the performance."""
        raise NotImplementedError("Subclasses must implement this method.")


class MIDITokenizer(PerformanceTokenizer):
    """Tokenizer for MIDI file paired with score.

    If you need to load a MIDI file without a score nor match file, please implement a
    different tokenizer.
    """

    def __init__(
        self,
        xml_score_path: str,
        midi_file: str,
        match_file: str,
    ):
        """Initialise the MIDI tokenizer.

        Args:
            xml_score_path (str | None): Path to the MusicXML score file.
            midi_file (str): Path to the matched MIDI file.
            match_file (str | None): Path to the match file that aligns score and performance.
            add_note_suffix (bool): If True, suffixes would be added to note IDs
                to disambiguate repeated notes with the same base ID.
        """
        self.midi_file = midi_file  # maybe useful for sustain
        assert midi_file.endswith(".mid"), "MIDI file must have .mid extension"
        self.match_file = match_file
        assert match_file.endswith(".match"), "Match file must have .match extension"
        self.xml_score_path = xml_score_path
        self.note_aligned = bool(match_file) and bool(xml_score_path)

    def tokenize(
        self,
    ) -> tuple[
        "PerformanceMetadata", "list[PerformanceNoteToken]", "list[PerformanceControl]"
    ]:
        metadata = self.parse_metadata()
        notes = self.tokenize_notes()
        controls = self.parse_controls()
        return metadata, notes, controls

    def tokenize_notes(self) -> list["PerformanceNoteToken"]:
        """
        Extract and tokenize MIDI notes.

        If the score and performance are aligned (using a match file),
        this method returns a list of matched performance notes, each wrapped
        in a `PerformanceNoteToken`. The score notes are aligned using unique suffixed IDs.

        Returns:
            List[PerformanceNoteToken]: A list of aligned performance notes with metadata.
        """
        if not self.note_aligned:
            raise NotImplementedError(
                "MIDI note alignment is not implemented for this tokenizer."
            )

        if not self.xml_score_path or not self.match_file:
            raise FileNotFoundError(
                "Both XML score path and match file must be provided for note alignment."
            )

        # Load score and match alignment
        score_part = pt.load_musicxml(self.xml_score_path)[0]
        performance, alignment = safe_load_match(
            self.match_file
        )  # pyright: ignore[reportAssignmentType]

        # Extract note arrays from both score and performance

        # FIX: We need to tokenize these notes in reference to the part object
        # as how the score tokenizer uses the score object.
        # Currently score tokenizer calls its own '_tokenize_note' method
        # and we should share the same logic here. NOT copy-pasting it here,
        # but rather using the same method from the score part.
        # Basically, you need to move the `_tokenize_note` method
        # from the `MusicXMLTokenizer` to outside of that class
        # so that it can be reused here.

        snote_array = score_part.note_array()
        pnote_array = (
            performance.note_array()  # pyright: ignore[reportAttributeAccessIssue]
        )

        # Get matched note indices from alignment file
        matched_note_idxs = pt.musicanalysis.performance_codec.get_matched_notes(
            spart_note_array=snote_array,
            ppart_note_array=pnote_array,
            alignment=alignment,
        )

        if matched_note_idxs.size == 0:
            raise ValueError("No matched notes found.")

        # Wrap matched notes into token objects
        midi_tokens = []
        for score_idx, perf_idx in matched_note_idxs:
            snote = snote_array[score_idx]
            pnote = pnote_array[perf_idx]

            midi_token = PerformanceNoteToken(
                pitch=int(pnote["pitch"]),
                velocity=int(pnote["velocity"]),
                onset_sec=float(pnote["onset_sec"]),
                duration_sec=float(pnote["duration_sec"]),
                xml_note_id=str(snote[-1]),
                score_note_token=None,
            )
            midi_tokens.append(midi_token)

        return midi_tokens

    def parse_controls(self) -> list["PerformanceControl"]:
        """Parse MIDI control changes (e.g., sustain pedal)"""
        if not self.match_file:
            raise ValueError("No match file provided to extract controls.")

        control_events = []
        with open(self.match_file, "r") as f:
            for line in f:
                if not line.startswith("sustain("):
                    continue

                match = re.match(r"sustain\((\d+),\s*(\d+)\)", line)
                if not match:
                    continue

                time_in_secs = self.convert_tick_to_seconds(int(match.group(1)))
                value = int(match.group(2))
                control = PerformanceControl(
                    control_type="sustain",
                    data_value=value,
                    time_in_seconds=time_in_secs,
                )
                control_events.append(control)

        return control_events

    def parse_metadata(self) -> "PerformanceMetadata":
        """Extract metadata from the score."""
        if not self.match_file:
            raise ValueError("No match file provided to extract metadata.")

        # Use the base name of the match file as performer
        performer = os.path.splitext(os.path.basename(self.match_file))[0]

        return PerformanceMetadata(
            performer=performer,
            note_aligned=self.note_aligned,
        )

    def convert_tick_to_seconds(self, tick: int) -> float:
        """
        Convert a MIDI tick value into real-world time in seconds.

        The conversion is based on two parameters extracted from the `.match` file:
        - `midiClockUnits`: the number of MIDI ticks per quarter note (TPQN).
        - `midiClockRate`: the number of microseconds per quarter note (tempo).

        Args:
            tick (int): The tick value to convert.

        Returns:
            float: Time in seconds corresponding to the given tick.

        Raises:
            ValueError: If `midiClockUnits` or `midiClockRate` cannot be found in the match file.

        Formula:
            seconds = (tick * midiClockRate) / (midiClockUnits * 1_000_000)
        """

        if not self.match_file:
            raise ValueError("No match file provided to extract timing info.")

        # midi_clock_units: int | None = None
        # midi_clock_rate: int | None = None

        with open(self.match_file, "r") as f:
            content = f.read()

        match_units = re.search(r"info\(midiClockUnits,\s*(\d+)\)", content)
        match_rate = re.search(r"info\(midiClockRate,\s*(\d+)\)", content)

        if not match_units or not match_rate:
            raise ValueError(
                "Could not determine MIDI clock parameters from match file."
            )

        midi_clock_units = int(match_units.group(1))
        midi_clock_rate = int(match_rate.group(1))

        return (tick * midi_clock_rate) / (midi_clock_units * 1_000_000)
