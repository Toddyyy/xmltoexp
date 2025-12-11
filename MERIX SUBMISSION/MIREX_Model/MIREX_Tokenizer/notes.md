## Minor problems

- [x] is the tick information (`onset_tick`, `duration_tick`) itself useful
      without knowing the resolution of the midi file? - see header of the match
      file (`midiClockUnits`)
  - No, the tick-based values like onset_tick and duration_tick are not
    independently meaningful unless you know the MIDI resolution — which is
    specified by:(`midiClockUnits`)

  - (`midiClockUnits` is the number of ticks per quarter note, or TPQN: tick /
    midiClockUnits = time in beats)
  - (`midiClockRate` is the number of ticks per second, or TPT: tempo =
    60,000,000(1min) / 500,000(0.5s) = 120 BPM) 1 tick = (`midiClockRate` /
    `midiClockUnits`) / 1,000,000 seconds

- [x] How do we properly model the relationship between `PerformanceNoteToken`
      and `ScoreNoteToken` ?
  - For paired data, a `PerformanceNoteToken` is always associated with a
    `NoteToken`. So maybe model it as
    `PerformanceNoteToken --has-> ScoreNoteToken`?

  - For unpaired MIDI files, a `PerformanceNoteToken` has only basic pitch and
    duration features. Beat-related features are NOT available.
    - [x] (further discussion needed to confirm if we are using them)
    - How are they going to be loaded?
    - How is such data going to be used? Metadata about the performer?

- [x] support of sustain pedal
  - (`sustain(<tick>, <value>)`) could use (`midiClockUnits`) to convert to
    seconds or beats

- [x] If the note is marked as accented or staccato, is the duration as noted or
      is it already shortened?
  - No it only represents the origin duration, we need to manually shorten it.

- [x] How are tied notes represented? A single entity?
  - Both tied and slur has a start and an end entity

- [x] Could not read some metadata from MUSICXML files in nASAP, e.g. Composer,
      Genre, Title.
  - [ ] amend

### Clues so far about the ties

They are represented as multiple notes and annotated by `tied` and `tie`
objects.

However, in `patitura` and `music21`, only the first note would be encoded, and
in `pt`'s `snote_array` the duration is already correctly set to the total
duration.

### Generator Output Features

- Pitch, duration, onset and velocity

### Tokenizer Interface

- [ ] `MIDITokenizer` should have an interface similar to `ScoreTokenizer`
      because at the moment the `PairedScoreTokenizer` only uses a special MIDI
      tokenizer to load MIDI from match files. Hence, the interface should be
      left for users to develop their own MIDI tokenizer in case the MIDI file
      is not paired.
  - [ ] Also the current `MIDITokenizer` should be renamed to
        `PairedMIDITokenizer`

## Major problems / questions

- Data loader
  - only useful for training and validation, not for inference because inference
    requires a different tokenization process (transforming the generated steps
    back to the input features)

- What does a Tokenizer return for paired score-MIDI data?
  - `ScoreMetadata`, `MIDIMetadata`
  - `list[NoteToken]`, `list[ScoreExpression]`
  - `list[PerformanceNoteToken]`, `list[PerformanceControl]`

- A `ScoreExplainer`, initialised by tokenizer.score_tokenizer output, provides
  the following utility functions for advanced analysis and feature extraction.
  - selection by beats or bars,
    - selection by region (e.g. by slur)
    - selection by filter (custom function)
  - time conversion (e.g. seconds to beats, or (bar, beats))
  - stats about the score, or the performance
  - [ ] derived features (e.g. local tempo)
  - query of onsets, active/sustained notes
    - voice information (e.g. onsets)

- [x] configure formatter `black`
- [ ] configure linter e.g. `pyright`

### Tokenizer Life Cycle

1. During training, the tokenizer is only used once. A tokenizer should not be
   repetitively called over the same data (overhead)

2. During inference, the tokenizer object is required to tokenize the generated
   steps back to input representation.

## Data Pipeline

Build dataset -> Data loader -> tokenizer -> feature encoding -> embedding ->
model --> output, loss, metrics
