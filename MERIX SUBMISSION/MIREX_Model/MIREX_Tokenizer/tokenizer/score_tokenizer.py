from dataclasses import replace
from typing import cast

from music21 import articulations, chord, converter, note
from music21.stream import Score

from tokenizer.base_score_tokenizer import (
    ScoreExpression,
    ScoreMetadata,
    ScoreNoteToken,
    ScoreTokenizer,
)
from tokenizer.musicxml_utils import (
    parse_expressions,
    parse_metadata,
    parse_title_from_xml,
)


class MusicXMLTokenizer(ScoreTokenizer):
    """Tokenizer implementation for MusicXML files using music21."""

    def __init__(self, file_path: str):
        """Initialize the tokenizer.score_tokenizer with a MusicXML file path."""
        self.file_path = file_path
        self.score: Score = cast(Score, converter.parse(file_path))
        assert isinstance(
            self.score, Score
        ), "Parsed score must be a music21 Score object."

    def tokenize(
        self,
    ) -> tuple[list[ScoreNoteToken], list[ScoreExpression], ScoreMetadata]:
        """
        Tokenize the MusicXML score into note tokens, expressions, and metadata.

        Returns:
            A tuple containing:
            - List of ScoreNoteToken objects
            - List of ScoreExpression objects
            - ScoreMetadata object
        """
        notes = self.tokenize_notes()
        expressions = self.parse_expressions()
        metadata = self.parse_metadata()
        return notes, expressions, metadata

    def tokenize_notes(self) -> list[ScoreNoteToken]:
        """Extract note tokens from _all_ parts of the MusicXML file."""
        tokens = []
        for part in self.score.parts:
            flat_part = part.flatten()
            for el in flat_part.notes:
                # Handle the single note case
                if isinstance(el, note.Note):
                    # Calculate absolute position in beats
                    abs_offset = float(flat_part.elementOffset(el))
                    tokens.append(self._tokenize_note(el, abs_offset, str(part.id)))
                # Handle chords
                elif isinstance(el, chord.Chord):
                    abs_offset = float(flat_part.elementOffset(el))
                    # Iterate over each note in the chord
                    for single_note in el.notes:
                        tokens.append(
                            self._tokenize_note(single_note, abs_offset, str(part.id))
                        )
        return tokens

    @staticmethod
    def _tokenize_note(
        n: note.Note, absolute_offset: float, part_id: str
    ) -> ScoreNoteToken:
        """
        Convert a music21 Note object to a NoteToken.

        Args:
            n: music21 Note object
            absolute_offset: Absolute onset time in beats
            part_id: Identifier of the musical part

        Returns:
            Structured NoteToken representation
        """

        return ScoreNoteToken(
            pitch=str(n.pitch.nameWithOctave),  # FIXED: use str() directly
            duration=float(n.duration.quarterLength),
            tie=n.tie.type if n.tie else None,
            is_staccato=any(
                isinstance(a, articulations.Staccato) for a in n.articulations
            ),
            is_accent=any(isinstance(a, articulations.Accent) for a in n.articulations),
            fingering=next(
                (
                    int(expr.number)  # pyright: ignore[reportAttributeAccessIssue]
                    for expr in n.expressions
                    if isinstance(expr, articulations.Fingering)
                ),
                None,
            ),
            position=absolute_offset,
            part_id=part_id,
            # xml_note_id=note_id if note_id else None,
        )

    def parse_expressions(self) -> list[ScoreExpression]:
        """Parse musical expressions from the score as a list."""
        return parse_expressions(self.score)

    def parse_metadata(self) -> ScoreMetadata:
        """Extract metadata from the score."""
        metadata = parse_metadata(self.score)

        # If title is missing, try to parse it from XML credit words
        if not metadata.title:
            title = parse_title_from_xml(self.file_path)
            if title:
                metadata = replace(metadata, title=title)

        return metadata
