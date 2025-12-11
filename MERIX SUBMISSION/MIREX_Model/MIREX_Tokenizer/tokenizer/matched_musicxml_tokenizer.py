import logging
from dataclasses import replace
from typing import cast

from music21 import chord, converter, note
from music21.stream import Score

from .base_score_tokenizer import (
    ScoreExpression,
    ScoreMetadata,
    ScoreNoteToken,
    ScoreTokenizer,
)
from .musicxml_parser_patch import PatchedMusicXMLParser
from .score_tokenizer import MusicXMLTokenizer

logger = logging.getLogger(__name__)


class MatchedMusicXMLScoreTokenizer(ScoreTokenizer):
    """
    A specialized tokenizer for MusicXML files that include `note id` attributes
    for score-performance alignment.
    """

    def __init__(self, musicxml_path: str):
        self.musicxml_path = musicxml_path

        # Immediately parse the score within a patched context to capture note IDs.
        with PatchedMusicXMLParser() as patched_context:
            self.score: Score = cast(
                Score, converter.parse(self.musicxml_path, forceSource=True)
            )
            self._note_id_map: dict[int, str] = patched_context.get_note_id_map()

        assert isinstance(self.score, Score), "Parsed object must be a music21 Score."

        # Create a delegate tokenizer for non-note-related parsing tasks.
        self._non_note_tokenizer = MusicXMLTokenizer(file_path=self.musicxml_path)
        self._non_note_tokenizer.score = self.score

    def tokenize_notes(self) -> list[ScoreNoteToken]:
        """
        Extracts all notes from the score, breaking down chords into individual
        note tokens, and annotates each with its original XML ID.
        """
        tokens: list[ScoreNoteToken] = []
        for part in self.score.parts:
            flat_part = part.flatten()
            for el in flat_part.notes:  # Iterates over both Note and Chord objects.
                abs_offset = float(flat_part.elementOffset(el))

                if isinstance(el, note.Note):
                    # For a single Note, tokenize and append its captured ID.
                    base_token = MusicXMLTokenizer._tokenize_note(
                        el, abs_offset, str(part.id)
                    )
                    xml_id = self._note_id_map.get(id(el))
                    tokens.append(replace(base_token, xml_note_id=xml_id))

                elif isinstance(el, chord.Chord):
                    # For a Chord, "unroll" it into its constituent Note objects.
                    # This ensures a uniform output of NoteTokens and preserves all details.
                    for note_in_chord in el.notes:
                        base_token = MusicXMLTokenizer._tokenize_note(
                            note_in_chord, abs_offset, str(part.id)
                        )
                        xml_id = self._note_id_map.get(id(note_in_chord))
                        tokens.append(replace(base_token, xml_note_id=xml_id))
        return tokens

    def parse_expressions(self) -> list[ScoreExpression]:
        """Delegates expression parsing to the standard MusicXMLTokenizer."""
        return self._non_note_tokenizer.parse_expressions()

    def parse_metadata(self) -> ScoreMetadata:
        """Same."""
        return self._non_note_tokenizer.parse_metadata()
