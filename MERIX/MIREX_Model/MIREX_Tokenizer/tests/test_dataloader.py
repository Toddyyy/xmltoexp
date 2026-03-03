import warnings

from dataloader import ScorePerformanceDataset
from dataset_collectors import asap_collector, batik_collector, four_22_collector

warnings.filterwarnings("ignore", category=UserWarning)

import logging
import os
from itertools import chain

from alive_progress import alive_it

logger = logging.getLogger(__name__)

custom_cache = "./dataset_cache"
base_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(base_dir, ".."))

asap_dir = os.path.join(root_dir, "datasets", "asap_dataset")
batik_dir = os.path.join(root_dir, "datasets", "batik_plays_mozart")
four_22_dir = os.path.join(root_dir, "datasets", "vienna4x22")


def iterate_dataset(dl: ScorePerformanceDataset):
    count = 0
    ok_count = 0
    for i in range(len(dl)):
        count += 1
        try:
            (
                score_metadata,
                midi_metadata,
                score_tokens,
                midi_tokens,
                expressions,
                controls,
            ) = dl[i]

            # Process the data here...
            print(f"Sample {i}:")
            print(f"  Score file: {dl.triples[i][0]}")
            print(f"  Composer: {score_metadata.composer}")
            print(f"  Performer: {midi_metadata.performer}")
            print(f"  Title: {score_metadata.title}")
            print(f"  Number of score tokens: {len(score_tokens)}")
            print(f"  Number of MIDI tokens: {len(midi_tokens)}")
            print(f"  Number of expression tokens: {len(expressions)}")
            print(f"  Number of control tokens: {len(controls)}")
            ok_count += 1
        except ValueError as e:
            raise e
            # print(f"Sample {i} skipped: {e}")
            # continue

    logger.warning(f"{ok_count} out of {count} samples are successfully loaded.")


def test_data_loading():
    triples = []
    triples.extend(asap_collector(asap_dir)[:2])
    triples.extend(batik_collector(batik_dir)[:5])
    triples.extend(four_22_collector(four_22_dir)[:2])
    ds = ScorePerformanceDataset(triples, cache_directory=custom_cache)
    iterate_dataset(ds)


def test_asap_loading():
    triples = asap_collector(asap_dir)
    dl = ScorePerformanceDataset(triples, cache_directory=custom_cache)
    iterate_dataset(dl)


def test_batik_loading():
    triples = batik_collector(batik_dir)
    dl = ScorePerformanceDataset(triples, cache_directory=custom_cache)
    iterate_dataset(dl)


def test_four_22_loading():
    triples = four_22_collector(four_22_dir)
    dl = ScorePerformanceDataset(triples, cache_directory=custom_cache)
    iterate_dataset(dl)


def get_token_sequence_length_distributions():
    datasets = (
        asap_collector(asap_dir),
        batik_collector(batik_dir),
        four_22_collector(four_22_dir),
    )
    triples = list(chain(*datasets))
    dl = ScorePerformanceDataset(triples, cache_directory=custom_cache)
    # Collecting length data for each dataset
    length_data = []

    for i, dataset in enumerate(datasets):
        dataset_name = ["asap", "batik", "four_22"][i]
        for triple in alive_it(
            dataset, total=len(dataset), title=f"Processing {dataset_name} dataset"
        ):
            tokenized = None
            try:
                dl.get_item_by_triple(triple)
            except (KeyError, ValueError):
                logger.warning(f"Skipping triple {triple} in dataset {dataset_name}")
                continue

            tokenized = dl.get_item_by_triple(triple)
            score_tokens = tokenized[2]
            midi_tokens = tokenized[3]
            mapped_score_tokens = [
                tk.score_note_token
                for tk in midi_tokens
                if tk.score_note_token is not None
            ]
            unique_score_tokens = set(mapped_score_tokens)
            lookup = set(score_tokens)
            mapped_back_tokens_exist = sum(
                int(tk in lookup) for tk in unique_score_tokens
            )

            length_data.append(
                dict(
                    dataset=dataset_name,
                    score_tokens=len(score_tokens),
                    midi_tokens=len(
                        midi_tokens,
                    ),
                    mapped_score_tokens=len(mapped_score_tokens),
                    unique_score_tokens=len(unique_score_tokens),
                    mapped_back_tokens_exist=mapped_back_tokens_exist,
                )
            )

    import pandas as pd

    df = pd.DataFrame(length_data)
    print(df.groupby("dataset").mean())


if __name__ == "__main__":
    # test_data_loading()
    # test_asap_loading()
    # test_batik_loading()
    # test_four_22_loading()
    get_token_sequence_length_distributions()
