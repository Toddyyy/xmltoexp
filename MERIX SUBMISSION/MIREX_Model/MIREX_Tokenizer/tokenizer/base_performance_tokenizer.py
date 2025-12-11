from dataclasses import dataclass
from typing import Literal

from tokenizer.base_score_tokenizer import ScoreNoteToken


@dataclass(frozen=True)
class PerformanceMetadata:
    """Represents metadata for a MIDI performance."""

    performer: str
    note_aligned: bool


@dataclass(frozen=True)
class PerformanceControl:
    """Represents a MIDI control change event with its properties."""

    control_type: Literal["sustain"]
    data_value: int  # Value of the control change, from 0 to 127
    time_in_seconds: float  # Time in second

    def __post_init__(self):
        if not (0 <= self.data_value <= 127):
            raise ValueError("MIDI control data value must be between 0 and 127.")


@dataclass(frozen=True)
class PerformanceNoteToken:
    """Represents a note from a MIDI performance with all its aligned and raw properties."""

    # Non-default fields (aligned or raw data)
    pitch: int  # MIDI pitch number
    velocity: int  # MIDI velocity
    onset_sec: float  # Onset time in seconds
    duration_sec: float  # Duration in seconds
    # TODO: modify to xml_note_id
    xml_note_id: str  # Unique note alignment ID

    score_note_token: ScoreNoteToken | None = None
