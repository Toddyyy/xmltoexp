# MusicXML and MIDI Tokenizer for Matched Score and Performance

## Overview

This repository contains tokenizers and data loading modules for MusicXML and
performance MIDI files, specifically designed for classical music, supporting
notes, rests, and score elements like metadata, slurs, tempo, articulation and
dynamics.

The concept of `token` in this repository refers to a structured representation
of musical notes or elements, with their associated metadata and attributes,
usually as one time step. This is different from the `token` in the context of
natural language processing, which typically refers to a word or sub-word unit.

Therefore, the tokens are NOT directly represented as integer IDs, and this step
is expected to be handled in the feature encoding step (e.g. discretization,
binning, one-hot encoding, etc.).

## Usage: data preparation

The aligned score-performance dataset is indexed by a triple
`(score_file, performance_midi_file, match_file)`.

First, prepare the paired score (`.musicxml`, `.xml` or `.mxl`), match files and
MIDI files in the `./datasets/` directory, following the hierarchy of
`./datasets/dataset_name/composer/piece/[subpiece/]/` In each piece directory
there should be one score file, and different pairs of (match, MIDI) files by
different performers.

### Cache building

For different datasets, you may need to implement custom `collector` functions
that returns a list of all the triples
`(score_file, performance_midi_file, match_file)`.

To speed up data loading and prevent re-computation, you should pre-process the
dataset and build a cache by running the following command:

```bash
python ./build_dataset.py
```

This would create a cache folder `./dataset_cache/` indexed by the SHA-1 hash of
the triple.

The reason we don't separately store the score file is because each match file
may refer to a different subset of the score (as only the matched notes are
present)

### Preparing dataset for training

If you would like to train using our model and training code, please then run:

```bash
python ./load_dataset.py
```

This script will read the previously created cache folder ./dataset_cache/
and generate the full tokenized JSON versions for all samples.

The output is a folder containing all JSON files (one per data triple), which
serves as the unified training input.
Our training code expects this JSON folder as input, rather than re-parsing raw
score/performance files each time.

> [!NOTE]
> This means during training the score can be actually only a subset of
> the full score because the system works as a tagging system rather than a
> sequence generator. Maybe future versions would cache the full score using a
> separate index, but now a `Triple` is more efficient.

## Usage: Tokenization

The most useful tokenizer is the `PairedScoreTokenizer`, which tokenizes both
the score and aligned performance data by one call. It consists of two
sub-tokenizers for MusicXML `.xml` (or `.mxl`) and performance file (`.match`),
respectively. Then can also be used individually.

The tokenizers convert the score and performance data into a structured
representation of `Metadata`, `Note`s, and `Expression`s/`Control`s.

> [!TIP]
> For extension, you can implement your own parser that implements the
> `ScoreTokenizer` or `PerformanceTokenizer` interface. `PerformanceTokenizer`,
> for example, loading a MIDI file without a paired score, or loading a
> non-MusicXML score.

## Basic Features and Advanced Features

| Type / Feature                                   | from Score?           | from MIDI?     | Implemented |
| ------------------------------------------------ | --------------------- | -------------- | ----------- |
| **Metadata**                                     |                       |                |             |
| Composer                                         | ✅                    | -              | ✅          |
| Performer                                        | -                     | ✅             | ✅          |
| **Note**                                         |                       |                |             |
| - Pitch                                          | ✅                    | Note Alignment | ✅          |
| -- Pitch Class Name                              | ✅                    | Note Alignment | ✅          |
| - Absolute onset in beats                        | ✅                    | Note Alignment |             |
| * Absolute onset in seconds                      | No, Non-deterministic | ✅             |             |
| - Relative onset to the bar                      | ✅                    | Note Alignment |             |
| - Bar number                                     | ✅                    | Bar Alignment  |             |
| - Duration in beats                              | ✅                    | Note Alignment | ✅          |
| * Duration in seconds                            | No, Non-deterministic | ✅             | ✅          |
| * Note Alignment ID                              | Note Alignment        | Note Alignment | ✅          |
| * Velocity (requires note-level alignment)       | ✅                    | ✅             | ✅          |
| - Onset deviation                                | -                     | Note Alignment | ✅          |
| **# Expression, score texts, structural marks.** |                       |                |             |
| **Expressions**                                  |                       |                |             |
| - Dynamic expressions                            | ✅                    | -              |             |
| **Tempo**                                        |                       |                |             |
| - Notated absolute tempo                         | ✅                    | -              |             |
| - Tempo changes                                  | ✅                    | Beat Alignment |             |
| - Time signature                                 | ✅                    | -              |             |
| * Estimated local tempo                          | -                     | Beat Alignment | ✅          |
| **Key**                                          |                       |                |             |
| - Key signature                                  | ✅                    | ✅             |             |
| **Articulations**                                |                       |                |             |
| - Accents, staccato                              | ✅                    | -              | ✅          |
| **Phrases**                                      |                       |                |             |
| - Slur                                           | ✅                    | -              |             |
| - Fermata                                        | ✅                    | -              |             |
| **Ornamentation**                                | ✅                    | Note Alignment |             |

For advanced features (e.g. those that require note-alignment to extract /
derive), use the `ScoreExplainer` to obtain `FullPerformanceToken`.

## Example Usage

```python
from score_explainer import FullPerformanceToken, ScoreExplainer
from tokenizer.base_performance_tokenizer import PerformanceMetadata
from tokenizer.base_score_tokenizer import ScoreMetadata
from tokenizer.paired_tokenizer import PairedTokenizer

paired_tokenizer = PairedTokenizer(
    xml_score_path="./datasets/asap_dataset/Bach/Fugue/bwv_846/xml_score.musicxml",
    midi_path="./datasets/asap_dataset/Bach/Fugue/bwv_846/Shi05M.mid",
    match_path="./datasets/asap_dataset/Bach/Fugue/bwv_846/Shi05M.match",
)

(
    score_metadata,
    midi_metadata,
    score_tokens,
    midi_tokens,
    score_expressions,
    midi_controls,
) = paired_tokenizer.tokenize()

se = ScoreExplainer(
    score_tokens=score_tokens,
    score_expressions=score_expressions,
    performance_tokens=midi_tokens,
    performance_controls=midi_controls,
)

full_tokens = se.get_full_tokens()

# Result
>>> full_tokens[0]
FullPerformanceToken(
    performance_note_token=PerformanceNoteToken(pitch=64, velocity=54, onset_sec=1.7052083015441895, duration_sec=0.7489583492279053, note_id='n2', score_note_token=None),
    score_note_token=ScoreNoteToken(pitch='C4', duration=0.5, position=0.5, part_id='P1-Staff1', tie=None, is_staccato=False, is_accent=False, fingering=None, xml_note_id='n2'),
    onset_deviation_in_beats=-0.018399233477087162,
    duration_deviation_in_beats=0.12521935895929015,
    local_tempo=50.087113090106286,
    sustain_level=43
)
```


MIREX/
├── detokenizer/
│   ├── detokenizer.py
│   └── run_detokenizer.py  <-- This is the script we will run
│
├── requirements.txt      <-- The file for installing dependencies
│
├── example_data/
│   └── ...
│
└── tokenizer/
    └── ...
