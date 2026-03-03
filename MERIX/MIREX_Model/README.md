# MIREX Score-to-Performance Model

## Overview

This repository contains the model and training code for the RenCon 2025: Expressive Performance Rendering Competition. Our model transforms MusicXML score files into expressive MIDI performances with realistic musical interpretation.


## Quick Start

### 1. Environment Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Set up MIREX integration (run once, see setup_mirex.py)
python setup.py
```

### 2. Generation with our pre-trained model

**For very quick validation on direct score XML to Performanced MIDI Generation:**

```bash
python generate_from_xml.py \
  --model_path ./check/sustain_classify_20250815_073720_best.pt \
  --xml_path xml_data/With_Dog-teams.mxl \
  --output_midi generate_results/With_Dog-teams.mid  \
  --composer_id 2 \
  --config_path config.yaml \
  --sequence_length 512 \
  --overlap_length 256
```


## Generation Pipeline

Our generation process consists of two main stages:

1. **Score Tokens → Performance Tokens**: Our model predicts expressive performance tokens from score tokens
2. **Performance Tokens → MIDI**: Our detokenizer converts performance tokens to playable MIDI

### Direct XML to MIDI Generation (Recommended)

**One-step conversion for convenience** (see [`generate_from_xml.py`](./generate_from_xml.py)):

```bash
python generate_from_xml.py \
  --model_path ./check/sustain_classify_20250815_073720_best.pt \
  --xml_path input.xml \
  --output_midi output.mid \
  --composer_id 2 \
  --config_path config.yaml \
  --sequence_length 512 \
  --overlap_length 256
```

### Step-by-Step Generation (For Research/Debugging)

**Generate performance tokens from score tokens** (see [`generate_overlap.py`](./generate_overlap.py)):

```bash
python generate_overlap.py \
  --model_path ./check/sustain_classify_20250815_073720_best.pt \
  --input_json score_tokens.json \
  --output_json performance_tokens.json \
  --composer_id 14 \
  --config_path config.yaml \
  --sequence_length 512 \
  --overlap_length 256
```

This script reads JSON score tokens and generates predicted performance tokens, saving results in JSON format for further analysis. And you can use (see [`run_detokenizer.py`](./MIREX_Tokenizer/detokenizer/run_detokenizer.py)) to convert json performance data to MIDI.

## Generation Options

```bash
# Basic generation
python generate_from_xml.py \
  --xml_path input.xml \
  --model_path model.pt \
  --output_midi output.mid \
  --composer_id 3

# Long pieces with overlapping inference
python generate_from_xml.py \
  --xml_path input.xml \
  --model_path model.pt \
  --output_midi output.mid \
  --sequence_length 512 \
  --overlap_length 256 \
  --composer_id 3

# Save intermediate files for debugging
python generate_from_xml.py \
  --xml_path input.xml \
  --model_path model.pt \
  --output_midi output.mid \
  --save_intermediate \
  --save_performance

# Adjust sampling parameters
python generate_from_xml.py \
  --xml_path input.xml \
  --model_path model.pt \
  --output_midi output.mid \
  --temperature 1.2 \
  --top_p 0.8 \
  --no_tempo_prediction
```

## Supported Composers

```
Bach: 0          Liszt: 7         Schubert: 12
Beethoven: 1     Mozart: 8        Schumann: 13  
Brahms: 2        Prokofiev: 9     Scriabin: 14
Chopin: 3        Rachmaninoff: 10
Debussy: 4       Ravel: 11
Glinka: 5        
Haydn: 6         
```
## Training from scratch
### 1. Data Preparation

This model requires training data prepared using our MIREX tokenizer. Please first follow the data preparation steps from the [MIREX Tokenizer repository](./MIREX_Tokenizer/README.md).

**Build Dataset Cache:**
Prepare your datasets in `./datasets/` following the hierarchy:
```
./datasets/dataset_name/composer/piece/[subpiece/]/
```

Run the data preparation scripts (see [`build_dataset.py`](./MIREX_Tokenizer/build_dataset.py) and [`load_dataset.py`](./MIREX_Tokenizer/load_dataset.py)):

```bash
# Build tokenized cache
python ./build_dataset.py

# Generate training JSON files
python ./load_dataset.py
```

Update the `data_dir` path in your [`config.yaml`](./config.yaml) to point to the generated JSON folder.

### 2. Training

For detailed training configuration options, see [`train.py`](./train.py):

```bash
# Train from scratch
python train.py --config config.yaml

# Resume from checkpoint
python train.py --config config.yaml --resume ./check/model_checkpoint.pt

# Use pretrained weights
python train.py --config config.yaml --pretrained ./check/pretrained_model.pt
```

## Performance Features

The model predicts comprehensive performance parameters:

- **Velocity**: MIDI dynamics (0-127)
- **Onset Deviation**: Timing flexibility (±4 seconds)
- **Duration Deviation**: Note length modifications (±3 seconds) 
- **Local Tempo**: Beat-level tempo variations (20-420 BPM)
- **Chunk-level Average Tempo**: Chunk-level average tempo variations (20-420 BPM)
- **Sustain Level**: Pedal control (binary)


## Input/Output Format

**Input:**
- MusicXML files (`.xml`, `.mxl`, `.musicxml`)
- Automatic score tokenization and note sorting
- Multi-staff piece support

**Output:**
- Standard MIDI files (`.mid`) ready for playback
- Expressive performance features included
- Natural tempo variations and articulation

## Configuration

Key parameters in [`config.yaml`](./config.yaml):



## Directory Structure

```
./
├── MIREX/                    # MIREX tokenizer/detokenizer (submodule)
├── model/                    # Model architecture
├── dataset.py               # Dataset loading
├── train.py                 # Training script
├── generate_from_xml.py     # XML to MIDI generation (one-step)
├── generate_overlap.py      # Score tokens to performance tokens
├── setup_mirex.py          # MIREX integration setup
└── config.yaml             # Configuration
```

## Requirements

- Python 3.12
- Dependencies: `pip install -r requirements.txt`