from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

# Define expression types using Literal for type safety
ExpressionType = Literal[
    "velocity", "tempo", "slur_start", "slur_end", "octave_shift", "tie", "unknown"
]

import logging

logging.basicConfig(
    level=logging.INFO, format="%(filename)s:%(lineno)d:%(levelname)s: %(message)s"
)


# Protocol for Tokenizer interface
@runtime_checkable
class ScoreTokenizer(Protocol):
    """API for score tokenizers. Not applicable to MIDI files."""

    def tokenize_notes(self) -> list["ScoreNoteToken"]:
        """Extract note tokens from the score."""
        ...

    def parse_expressions(self) -> list["ScoreExpression"]:
        """Parse musical expressions from the score."""
        ...

    def parse_metadata(self) -> "ScoreMetadata":
        """Extract metadata from the score."""
        ...


@dataclass(frozen=True)
class ScoreNoteToken:
    """Represents a musical note with all its properties."""

    # Non-default fields first
    pitch: str
    duration: float  # Duration in beats
    position: float  # Absolute onset in beats
    part_id: str  # Maybe useful to determine the hand

    # Fields with defaults follow
    tie: str | None = None
    is_staccato: bool = False
    is_accent: bool = False
    fingering: int | None = None
    xml_note_id: str | None = None  # ID from the MusicXML file if matched or expanded


@dataclass(frozen=True)
class ScoreExpression:
    """Represents a musical expression (dynamic, tempo, articulation, or slur)."""

    text: str
    type: ExpressionType
    absolute_onset_in_beats: float
    duration: float = 0.0
    data_value: float | None = None  # For sustain pedal related expressions

    @property
    def has_duration(self) -> bool:
        """True if this expression has meaningful duration."""
        return self.duration > 0.0


@dataclass(frozen=True)
class ScoreMetadata:
    """Contains metadata information about a musical score."""

    composer: str | None = None
    genre: str | None = None
    major_time_sig: str | None = None

    # Fields with defaults follow
    year: int | None = None
    title: str | None = None
