import dataclasses
import hashlib
import logging
import os
import pickle
import re
from dataclasses import replace
from typing import List

from constants import COMPOSER_NORMALIZATION
from tokenizer.base_score_tokenizer import (
    ScoreExpression,
    ScoreMetadata,
    ScoreNoteToken,
)
from tokenizer.midi_tokenizer import (
    PerformanceControl,
    PerformanceMetadata,
    PerformanceNoteToken,
)

logger = logging.getLogger(__name__)


type T_score_file = str  # Type alias for score file path
type T_midi_file = str  # Type alias for MIDI file path
type T_match_file = str  # Type alias for match file path

type Triple = tuple[T_score_file, T_midi_file, T_match_file]


def save_triple_cache(
    triple,
    score_metadata: ScoreMetadata,
    midi_metadata: PerformanceMetadata,
    score_tokens: List[ScoreNoteToken],
    midi_tokens: List[PerformanceNoteToken],
    expressions: List[ScoreExpression],
    controls: List[PerformanceControl],
    cache_root: str,
) -> str:
    """
    Save tokenized representation of a score-performance triple to disk cache.

    Each triple is hashed to create a unique cache folder under `cache_root`.
    Saves all components: metadata, token lists, expressions, and controls.
    """
    triple_hash = triple_hash_fn(triple)
    cache_folder = os.path.join(cache_root, triple_hash)
    os.makedirs(cache_folder, exist_ok=True)

    with open(os.path.join(cache_folder, "score_metadata.pkl"), "wb") as f:
        pickle.dump(score_metadata, f)
    with open(os.path.join(cache_folder, "midi_metadata.pkl"), "wb") as f:
        pickle.dump(midi_metadata, f)
    with open(os.path.join(cache_folder, "score_tokens.pkl"), "wb") as f:
        pickle.dump(score_tokens, f)
    with open(os.path.join(cache_folder, "midi_tokens.pkl"), "wb") as f:
        pickle.dump(midi_tokens, f)
    with open(os.path.join(cache_folder, "expressions.pkl"), "wb") as f:
        pickle.dump(expressions, f)
    with open(os.path.join(cache_folder, "controls.pkl"), "wb") as f:
        pickle.dump(controls, f)

    logger.info(f"Saved new cache for triple {triple_hash}")

    return cache_folder


def normalize_path(path: str) -> str:
    return os.path.abspath(path).replace("\\", "/")


def triple_hash_fn(triple: Triple) -> str:
    """
    Calculate the SHA-1 hash for a score-performance triple.

    This ensures each score-MIDI-match set has a unique cache directory.
    """
    normed = [normalize_path(p) for p in triple]
    return hashlib.sha1("||".join(normed).encode()).hexdigest()


def load_pickle(path: str, name: str):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        logger.error(f"[Cache Error] Failed to load {name} from {path}: {e}")
        raise


def normalize_composer(composer: str) -> str:
    """Normalize composer names to a consistent format"""
    if not composer:
        return ""

    # Remove non-alphabet characters, lowercase, and strip
    normalized = re.sub(r"[^a-zA-Z\s]", "", composer).lower()
    # Collapse multiple whitespaces into single space and trim
    normalized = re.sub(r"\s+", " ", normalized).strip()

    if not normalized:
        return composer.title()  # Fallback if string becomes empty after cleaning

    # Check full normalized string
    if normalized in COMPOSER_NORMALIZATION:
        return COMPOSER_NORMALIZATION[normalized]

    # Extract the last word and check the mapping
    last_word = normalized.split()[-1]
    if last_word in COMPOSER_NORMALIZATION:
        return COMPOSER_NORMALIZATION[last_word]

    # Fallback to title-cased original
    return composer.title()


def try_fixing_missing_composer(
    triple: Triple, original_metadata: ScoreMetadata, fallback: str = "Unknown Composer"
) -> ScoreMetadata:
    """Attempt to fix missing composer metadata using the file path."""
    score_file, _, _ = triple

    # If composer already exists, return original
    if original_metadata.composer:
        return original_metadata

    # Extract from path
    normalized_path = score_file.replace("/", os.sep).replace("\\", os.sep)
    path_parts = normalized_path.split(os.sep)
    possible_composer = ""

    # Try to find composer name in path
    for part in path_parts:
        normalized = normalize_composer(part)
        if normalized in COMPOSER_NORMALIZATION.values():
            possible_composer = normalized
            break

    # If not found in path, try from filename
    if not possible_composer:
        filename = os.path.basename(score_file)
        name_without_ext = os.path.splitext(filename)[0]

        # Try extracting the first part before underscore
        if "_" in name_without_ext:
            candidate = name_without_ext.split("_")[0].strip()
            normalized = normalize_composer(candidate)
            if normalized in COMPOSER_NORMALIZATION.values():
                possible_composer = normalized

    # Update metadata
    amended_score_metadata = dataclasses.replace(
        original_metadata, composer=possible_composer or fallback
    )

    logger.info(f"Fix the composer to {possible_composer}")
    return amended_score_metadata


def normalize_composer_metadata(original_metadata: ScoreMetadata) -> ScoreMetadata:
    """
    Normalize the composer field in ScoreMetadata:
    - If it's in COMPOSER_NORMALIZATION, replace it with the standard version.
    Returns a new ScoreMetadata object.
    """
    composer = original_metadata.composer

    # Exact match in normalization dict
    if composer in COMPOSER_NORMALIZATION:
        normalized = COMPOSER_NORMALIZATION[composer]
        return replace(original_metadata, composer=normalized)

    return original_metadata
