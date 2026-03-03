import logging
import os
from functools import partial

from alive_progress import alive_bar

from dataset_utils import (
    Triple,
    normalize_composer_metadata,
    save_triple_cache,
    triple_hash_fn,
    try_fixing_missing_composer,
)
from tokenizer.musicxml_utils import InvalidMatchFile
from tokenizer.paired_tokenizer import PairedTokenizer

logger = logging.getLogger("BuildDataset")

DEFAULT_CACHE_ROOT = "./dataset_cache"


def _build_cache_task(cache_root: str, triple: Triple):
    ok = build_cache_for_triple(triple, cache_directory=cache_root)
    return ok, triple


def build_cache_for_triple(
    triple: Triple, cache_directory: str = DEFAULT_CACHE_ROOT
) -> bool:
    """
    Process and cache the tokenized data and metadata for a given triple.
    If the cache already exists, skip processing.

    Args:
        triple (Triple): A tuple of (score_path, midi_path, match_path)
        cache_directory (str): Root directory for cache files

    Returns:
        bool: True if the triple was successfully cached (or existing), False otherwise.
    """
    triple_hash = triple_hash_fn(triple)
    out_dir = os.path.join(cache_directory, triple_hash)

    # Skip if already cached
    if os.path.exists(os.path.join(out_dir, "score_metadata.pkl")):
        logger.info(f"[SKIP] Already cached: {triple_hash}")
        return True

    os.makedirs(out_dir, exist_ok=True)

    try:
        tokenizer = PairedTokenizer(*triple)
        (
            score_metadata,
            midi_metadata,
            score_tokens,
            midi_tokens,
            expressions,
            controls,
        ) = tokenizer.tokenize()

        # Normalize and fix missing composer metadata
        score_metadata = normalize_composer_metadata(score_metadata)
        if not score_metadata.composer:
            score_metadata = try_fixing_missing_composer(triple, score_metadata)

        # Save processed data to disk cache
        save_triple_cache(
            triple,
            score_metadata,
            midi_metadata,
            score_tokens,
            midi_tokens,
            expressions,
            controls,
            cache_root=cache_directory,
        )
        logger.info(f"[✓] Cached triple: {triple_hash}")
        return True

    except InvalidMatchFile as e:
        logger.warning(f"[SKIP] Invalid match file for {triple}: {e}")
    except Exception as e:
        logger.error(f"[ERROR] Failed to process {triple}: {e}")

    return False


def build_dataset_cache(
    triples: list[Triple], cache_root: str = DEFAULT_CACHE_ROOT, num_workers: int = 1
):
    """
    Build dataset cache for a list of score-performance triples.

    Args:
        triples (list[Triple]): List of (score, midi, match) file path tuples.
        cache_root (str): Root directory where cache files will be written.
        num_workers (int): Number of worker threads to use for processing.
    """
    from multiprocessing import Pool

    os.makedirs(cache_root, exist_ok=True)
    failed_triple_hashes: set[str] = set()
    with alive_bar(total=len(triples)) as bar, Pool(processes=num_workers) as pool:
        for ok, triple in pool.imap_unordered(
            partial(_build_cache_task, cache_root), triples
        ):
            if not ok:
                failed_triple_hashes.add(triple_hash_fn(triple))
            bar()
            bar.text(f"fail count: {len(failed_triple_hashes)}")

    failed_triples_text = "\n".join(str(t) for t in failed_triple_hashes)

    fail_file = os.path.join(cache_root, "failed_triple_hashes.txt")
    if failed_triple_hashes:
        logger.error(
            f"Failed to process {len(failed_triple_hashes)}/{len(triples)} triples. See {fail_file} for details."
        )
        with open(fail_file, "w") as f:
            f.write(failed_triples_text)
    else:
        logger.info("All triples processed successfully.")
        with open(fail_file, "w") as f:
            f.write("No failed triples.")


if __name__ == "__main__":
    # Collect triples from desired dataset(s)
    from dataset_collectors import asap_collector, batik_collector, four_22_collector

    all_triples = []
    all_triples += asap_collector("datasets/asap_dataset/")
    all_triples += batik_collector("datasets/batik_plays_mozart/")
    all_triples += four_22_collector("datasets/vienna4x22")

    logger.info(f"Total {len(all_triples)} triples to process.")

    build_dataset_cache(all_triples, cache_root=DEFAULT_CACHE_ROOT, num_workers=8)
