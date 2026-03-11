# Boundary Restart

This is a clean restart for the Mazurka boundary task. It does not reuse the old note-to-beat Transformer training code. The workflow is:

1. Build a beat-level feature table from the existing `*.npz` files.
2. Train strong non-neural baselines first: logistic regression and XGBoost.
3. Evaluate as sparse event detection, not just per-beat loss.
4. Then train a small beat-level sequence model: `BiLSTM` or `TCN`.

## Scope

- Current line: score-only boundary prediction.
- Current unit: one row per beat.
- Current default target: `L5` performer-level files from `MIREX_Model/beat_data_mazurka_performer_levels`.
- Current split: fixed piece split from `MIREX_Model/splits/mazurka_piece_split_v4_35_9_0.yaml`.

This line reports:

- `average_precision`
- `event_precision`
- `event_recall`
- `event_f1`
- `mean_offset`
- best threshold on validation data

Event metrics use a beat tolerance and minimum event spacing.

## Layout

```text
Boundary_Restart/
├── boundary_restart/
│   ├── config.py
│   ├── features.py
│   ├── metrics.py
│   ├── models.py
│   └── table_io.py
├── configs/
│   └── level5_score_only.yaml
├── build_beat_table.py
├── train_baselines.py
├── train_sequence.py
└── requirements.txt
```

## Setup

Use the same environment as `MIREX_Model`, then add:

```bash
pip install -r "MERIX SUBMISSION/Boundary_Restart/requirements.txt"
```

## Step 1: Build the beat table

```bash
python "MERIX SUBMISSION/Boundary_Restart/build_beat_table.py" \
  --config "MERIX SUBMISSION/Boundary_Restart/configs/level5_score_only.yaml"
```

Output:

- beat table: `outputs/level5/beat_table_L5.csv.gz`
- metadata: `outputs/level5/beat_table_L5.meta.json`

## Step 2: Train non-neural baselines

```bash
python "MERIX SUBMISSION/Boundary_Restart/train_baselines.py" \
  --config "MERIX SUBMISSION/Boundary_Restart/configs/level5_score_only.yaml"
```

Outputs per model:

- metrics JSON
- per-beat validation predictions
- model pickle
- coefficient or importance table

## Step 3: Train a small sequence model

BiLSTM:

```bash
python "MERIX SUBMISSION/Boundary_Restart/train_sequence.py" \
  --config "MERIX SUBMISSION/Boundary_Restart/configs/level5_score_only.yaml" \
  --model bilstm
```

TCN:

```bash
python "MERIX SUBMISSION/Boundary_Restart/train_sequence.py" \
  --config "MERIX SUBMISSION/Boundary_Restart/configs/level5_score_only.yaml" \
  --model tcn
```

Outputs:

- best checkpoint
- scaler statistics
- validation predictions
- training summary JSON

## Feature Set

The beat table includes:

- note density and local change
- pitch summary and pitch-class profile
- duration and long-note release features
- accent and staccato ratios
- empty-beat and short-gap cues
- simple metrical proxies: mod-3 and mod-4 phase features
- local repetition continuation vs ending score

## Notes

- `performer_cond` is intentionally removed from this line.
- This line assumes the current score-only `*.npz` files with 6 base note features.
- Performer-specific boundary prediction is not implemented here yet. That should be a separate performance-conditioned line using tempo and dynamics curves.
