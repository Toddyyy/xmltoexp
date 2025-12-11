import logging
from typing import cast

from tokenizer.base_score_tokenizer import (
    ScoreExpression,
    ScoreMetadata,
    ScoreNoteToken,
    ScoreTokenizer,
)
from tokenizer.matched_musicxml_tokenizer import MatchedMusicXMLScoreTokenizer
from tokenizer.midi_tokenizer import (
    MIDITokenizer,
    PerformanceControl,
    PerformanceMetadata,
    PerformanceNoteToken,
)

logger = logging.getLogger(__name__)


class PairedTokenizer:
    """
    Tokenizer for aligned score–performance data.
    Combines score and MIDI tokenizers into a unified interface.
    """

    def __init__(self, xml_score_path: str, midi_path: str, match_path: str):
        """
        Initialize the paired tokenizer with (score, MIDI, match) file paths.

        Args:
            xml_score_path: Path to the MusicXML score file
            midi_path: Path to the performance MIDI file
            match_path: Path to the match alignment file
        """
        self.xml_score_path = xml_score_path
        self.midi_path = midi_path
        self.match_path = match_path

        self.score_tokenizer: ScoreTokenizer = MatchedMusicXMLScoreTokenizer(
            xml_score_path
        )

        self.midi_tokenizer: MIDITokenizer = MIDITokenizer(
            xml_score_path=xml_score_path,
            midi_file=midi_path,
            match_file=match_path,
        )

    def tokenize(self) -> tuple[
        ScoreMetadata,
        PerformanceMetadata,
        list[ScoreNoteToken],
        list[PerformanceNoteToken],
        list[ScoreExpression],
        list[PerformanceControl],
    ]:
        """
        Perform full tokenization on score and MIDI side, including alignment.

        Returns:
            PairedPerformanceTokens: Combined structured token data from both domains
        """
        # Score tokens
        score_note_tokens = self.score_tokenizer.tokenize_notes()
        score_expressions = self.score_tokenizer.parse_expressions()
        score_metadata = self.score_tokenizer.parse_metadata()

        # MIDI tokens (note-aligned)
        midi_metadata, midi_note_tokens, midi_controls = self.midi_tokenizer.tokenize()

        common_note_idxes = set(
            map(lambda x: cast(str, x.xml_note_id), score_note_tokens)
        ) & set(map(lambda x: x.xml_note_id, midi_note_tokens))
        score_note_tokens = [
            token
            for token in score_note_tokens
            if token.xml_note_id in common_note_idxes
        ]
        midi_note_tokens = [
            token for token in midi_note_tokens if token.xml_note_id in common_note_idxes
        ]
        logger.info(
            f"Note alignment by common indexes: score={len(score_note_tokens)}, "
            f"midi={len(midi_note_tokens)} -> aligned {len(common_note_idxes)}"
        )

        return (
            score_metadata,
            midi_metadata,
            score_note_tokens,
            midi_note_tokens,
            score_expressions,
            midi_controls,
        )
