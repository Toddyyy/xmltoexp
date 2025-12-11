import dataclasses
import logging
import os

import torch

from dataset_utils import Triple, load_pickle, triple_hash_fn
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


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
CACHE_ROOT = os.path.join(PROJECT_ROOT, "dataset_cache")


class ScorePerformanceDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        triples: list[Triple],
        cache_directory: str = "./dataset_cache",
        num_workers: int = 8,
        **kwargs,
    ):
        """Initialize the dataloader with a list of (score, MIDI, match) file name triples.

        Args:
            triples (list[Triple]): List of triples containing paths to score, MIDI, and match files.
        """
        self.triples: list[Triple] = triples
        self.triples_with_inconsistent_metadata: list[Triple] = []
        self.cache_directory = cache_directory
        self.failed_triples: set[str] = set()
        self.filter_triples_by_cache()

        self.num_workers = num_workers
        super().__init__(
            **kwargs,
        )

    def filter_triples_by_cache(self):
        """Filter out triples that have valid cache entries."""
        assert os.path.isdir(self.cache_directory), "Cache directory does not exist."
        failed_triple_hashes: set[str] = set()
        cache = os.path.join(self.cache_directory, "failed_triple_hashes.txt")
        assert os.path.isfile(
            cache
        ), f"Cache file {cache} does not exist. Please run the cache building script first."
        with open(cache, "r") as f:
            lines = f.readlines()
            failed_triple_hashes = set(
                str(line.strip()) for line in lines if line.strip()
            )

        original_count = len(self.triples)
        self.triples = [
            tp for tp in self.triples if triple_hash_fn(tp) not in failed_triple_hashes
        ]
        now_count = len(self.triples)
        logger.info(
            f"Filtered triples: {original_count} -> {now_count} (removed {original_count - now_count} failed triples)"
        )

    def __getitem__(self, index: int) -> tuple[
        ScoreMetadata,
        PerformanceMetadata,
        list[ScoreNoteToken],
        list[PerformanceNoteToken],
        list[ScoreExpression],
        list[PerformanceControl],
    ]:
        """Retrieve tokenized score-performance pair at a given index."""
        # Generate cache path for the current triple using its hash
        triple = self.triples[index]
        triple_hash = triple_hash_fn(triple)
        cache_folder = os.path.join(self.cache_directory, triple_hash)

        # Load from cache if available
        sextuple = try_load_sextuplet_from_cache(cache_folder)
        if sextuple is not None:
            return sextuple

        # If cache not found, It must have failed during cache building.
        # So the data example is unusable.
        raise IndexError(f"Cache missed for triple {triple_hash}. Can't be used")

    def __len__(self) -> int:
        """Return the number of triples in the dataset."""
        return len(self.triples)

    def get_item_by_triple(self, triple: Triple):
        if triple not in self.triples:
            raise KeyError(
                f"Triple {triple} not found in dataset cache. It's either not processed or invalid."
            )
        index = self.triples.index(triple)
        return self[index]


def try_load_sextuplet_from_cache(
    cache_folder: str,
) -> (
    tuple[
        ScoreMetadata,
        PerformanceMetadata,
        list[ScoreNoteToken],
        list[PerformanceNoteToken],
        list[ScoreExpression],
        list[PerformanceControl],
    ]
    | None
):
    """Try to load a sextuplet from the cache.

    Args:
        cache_folder (str): Path to the cache folder.
    Returns:
        sextuplet (tuple | None): The loaded sextuplet containing metadata and tokens,
        or None if the cache is not found or invalid.
    """

    if not os.path.isdir(cache_folder):
        raise FileNotFoundError(
            f"Cache folder {cache_folder} does not exist or is not a directory."
        )

    try:
        score_metadata = load_pickle(
            os.path.join(cache_folder, "score_metadata.pkl"), "score_metadata"
        )
        midi_metadata = load_pickle(
            os.path.join(cache_folder, "midi_metadata.pkl"), "midi_metadata"
        )
        score_tokens = load_pickle(
            os.path.join(cache_folder, "score_tokens.pkl"), "score_tokens"
        )
        midi_tokens = load_pickle(
            os.path.join(cache_folder, "midi_tokens.pkl"), "midi_tokens"
        )
        expressions = load_pickle(
            os.path.join(cache_folder, "expressions.pkl"), "expressions"
        )
        controls = load_pickle(os.path.join(cache_folder, "controls.pkl"), "controls")

        # map the performance tokens to their corresponding score tokens
        reverse_map: dict[str, ScoreNoteToken] = {
            token.xml_note_id: token
            for token in score_tokens
            if token.xml_note_id is not None
        }
        for i in range(len(midi_tokens)):
            midi_tokens[i] = dataclasses.replace(
                midi_tokens[i], score_note_token=reverse_map.get(midi_tokens[i].xml_note_id)
            )

    except Exception as e:
        raise ValueError(f"Failed to load sextuplet from cache {cache_folder}: ") from e

    return (
        score_metadata,
        midi_metadata,
        score_tokens,
        midi_tokens,
        expressions,
        controls,
    )
