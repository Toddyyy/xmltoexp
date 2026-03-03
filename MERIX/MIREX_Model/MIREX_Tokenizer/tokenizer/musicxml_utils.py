import logging
import xml.etree.ElementTree as ET
from itertools import chain
from typing import Optional
import re
import tempfile
from pathlib import Path

import music21
import partitura as pt
from music21 import dynamics, expressions, note

from tokenizer.base_score_tokenizer import ScoreExpression, ScoreMetadata

# Configure logging
logger = logging.getLogger(__name__)

# Define expression constants
VELOCITY_EXPRESSIONS = {"cresc.", "molto marcato", "molto ritard."}
TEMPO_EXPRESSIONS = {"rit.", "accel.", "rit", "accel"}
OCTAVE_SHIFT_TYPES = {"8va", "8vb", "15ma", "15mb"}  # I am not sure about these
DEFAULT_TEMPO_DURATION = 4.0  # Default duration for tempo expressions


def get_absolute_offset(
    element: music21.base.Music21Object, score: music21.stream.Score
) -> Optional[float]:
    """
    Calculate the absolute offset of an element within a score.

    Args:
        element: Music21 element to locate
        score: Parent score containing the element

    Returns:
        Absolute offset in beats from score start, or None if calculation fails
    """
    try:
        if hasattr(element, "getOffsetInHierarchy"):
            return float(element.getOffsetInHierarchy(score))
        return float(element.offset)  # Fallback to simple offset
    except Exception as e:
        logger.error(f"Offset calculation failed for {element}: {e}")
        return None


def parse_expressions(score: music21.stream.Score) -> list[ScoreExpression]:
    """
    Extract musical expressions from a score.

    Handles:
    - Dynamics (mf, f, p, etc.)
    - Tempo markings (rit., accel., etc.)
    - Slurs
    - Octave shifts (8va, 8vb, etc.)
    - Tie

    Args:
        score: Parsed music21 score object

    Returns:
        List of Expression objects found in the score
    """
    expressions_list = []

    # Process all word elements in the score
    for el in score.recurse():
        # Handle text expressions (dynamics, tempo, etc.)
        if not isinstance(el, expressions.TextExpression):
            continue

        text = el.content.strip().lower()

        if text in VELOCITY_EXPRESSIONS:
            expressions_list.append(
                ScoreExpression(
                    text=text,
                    type="velocity",
                    absolute_onset_in_beats=get_absolute_offset(el, score) or 0.0,
                )
            )
        elif text in TEMPO_EXPRESSIONS:
            duration = (
                el.duration.quarterLength if el.duration else DEFAULT_TEMPO_DURATION
            )
            expressions_list.append(
                ScoreExpression(
                    text=text,
                    type="tempo",
                    absolute_onset_in_beats=get_absolute_offset(el, score) or 0.0,
                    duration=float(duration),
                )
            )
        # Handle octave shift expressions
        elif any(octave_type in text for octave_type in OCTAVE_SHIFT_TYPES):
            expressions_list.append(
                ScoreExpression(
                    text=text,
                    type="octave_shift",
                    absolute_onset_in_beats=get_absolute_offset(el, score) or 0.0,
                    duration=(float(el.duration.quarterLength) if el.duration else 0.0),
                )
            )

    # Handle dynamic symbols
    for el in score.recurse():
        if not isinstance(el, dynamics.Dynamic):
            continue
        expressions_list.append(
            ScoreExpression(
                text=el.value.strip().lower(),
                type="velocity",
                absolute_onset_in_beats=get_absolute_offset(el, score) or 0.0,
            )
        )

    # Handle tie symbols and slur in the score
    for n in score.recurse().notes:
        # Process tie symbols
        if isinstance(n, note.Note) and n.tie:
            expressions_list.append(
                ScoreExpression(
                    text=f"tie_{n.id}_{n.pitch}_{n.tie.type}",
                    type="tie",
                    absolute_onset_in_beats=get_absolute_offset(n, score) or 0.0,
                )
            )

        # Process slurs in the score
        for spanner in n.getSpannerSites():
            if not isinstance(spanner, music21.spanner.Slur):
                continue
            try:
                if spanner.getFirst() != n:
                    continue
                start_offset = get_absolute_offset(n, score) or 0.0
                end_offset = get_absolute_offset(spanner.getLast(), score) or 0.0
                duration = end_offset - start_offset

                expressions_list.append(
                    ScoreExpression(
                        text=f"slur_{spanner.id}",
                        type="slur_start",
                        absolute_onset_in_beats=start_offset,
                        duration=duration,
                    )
                )
            except Exception as e:
                logger.error(f"Failed to process slur: {e}")

    # Sort expressions by their onset time
    expressions_list.sort(key=lambda expr: expr.absolute_onset_in_beats)
    return expressions_list


def parse_title_from_xml(xml_file: str) -> str | None:
    """
    Try to extract the title from the XML file's credit words.

    Returns:
        Title string extracted from the credit words, or an empty string if not found
    """

    tree = ET.parse(xml_file)
    root = tree.getroot()
    # filter justify=center and valign=top
    title_strings: list[str] = []
    finders = chain(
        root.findall(".//work-number"),
        root.findall(".//work-title"),
        root.findall(".//movement-number"),
        root.findall(".//movement-title"),
        root.findall(".//credit-words[@justify='center'][@valign='top']"),
    )
    for text in finders:
        if text.text:
            title_strings.append(text.text.strip())

    title = " ".join(title_strings).strip()

    return title or None


def parse_metadata(score: music21.stream.Score) -> ScoreMetadata:
    """Extract metadata from a music21 score object."""
    metadata = score.metadata

    # Extract required fields with fallbacks
    composer = metadata.composer or None

    # Genre extraction - music21 doesn't have direct genre property
    genre = None
    # Try to find genre in text expressions
    for text_expr in score.recurse().getElementsByClass(expressions.TextExpression):
        content = text_expr.content.lower()
        if "genre:" in content:
            genre = content.split("genre:")[1].strip()
            break

    # Extract major time signature
    time_sig = None  # Default time signature
    try:
        # Get the first time signature in the score
        time_sigs = score.flatten().getElementsByClass("TimeSignature")
        if time_sigs:
            time_sig_obj = time_sigs[0]
            time_sig = f"{time_sig_obj.numerator}/{time_sig_obj.denominator}"
    except Exception as e:
        logger.error(f"Error getting time signature: {e}")

    # Extract optional fields
    year = None
    if metadata.dateCreated:
        if hasattr(metadata.dateCreated, "year"):
            year = metadata.dateCreated.year
        elif isinstance(metadata.dateCreated, str):
            try:
                year = int(metadata.dateCreated[:4])
            except (ValueError, TypeError):
                pass

    return ScoreMetadata(
        composer=composer,
        genre=genre,
        major_time_sig=time_sig,
        year=year,
        title=metadata.title,
    )


class InvalidMatchFile(Exception):
    """Raised when a .match file has invalid structure or timing errors."""

    ...


def safe_load_match(match_path: str) -> tuple:
    """
    Load a match file and raise a custom InvalidMatchFile if 'sound_off < note_off' in Batik dataset.
    For the dataset 4x22, no suffix removal is needed, since it already has the correct format.
    For the other two datasets, reads the file, removes '-1' suffixes in match file,
    and loads from the modified content.
    """
    try:
        # Dataset 4x22 do not need to fix repetition suffixes
        if "Vienna4x22" in match_path or "4x22" in match_path:
            return pt.load_match(match_path)

        # Only the other two datasets need suffix removal
        fixed_content = remove_repetition_suffix(match_path)

        # Write fixed content to a temporary file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".match", delete=False) as tmp:
            tmp.write(fixed_content)
            tmp_path = tmp.name

        match_result = pt.load_match(tmp_path) # pt.load_match can only accept file paths

        # Clean up temporary file
        Path(tmp_path).unlink()

        return match_result
    except ValueError as e:
        if "sound_off must be greater or equal to note_off" in str(e):
            raise InvalidMatchFile(f"[Invalid Timing] {match_path}") from e
        raise  # propagate other unexpected errors

def remove_repetition_suffix(match_file: str) -> str:
    """
    Read a match file and remove '-1' suffixes *only* from snote IDs (e.g., "n4-1" → "n4").
    If a snote ID has a '-2' suffix, raise an InvalidMatchFile error.
    """
    with open(match_file, "r") as f:
        content = f.read()
    
    # Check for -2 in snote IDs first
    if re.search(r"snote\([^,]+-2,", content):
        raise ValueError("Found '-2' in snote ID - this is not allowed")

    # Only proceed with -1 replacement if no -2 was found
    fixed_content = re.sub(r"snote\(([^,]+)-1,", r"snote(\1,", content)

    return fixed_content
