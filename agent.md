# Current Boundary Prediction Experiment Notes

## Maintenance Rule

根据每次实验进展和用户提出的新要求，实时更新本文件中的实验配置记录。不要删除之前的实验记录；如果配置、数据集、标签构造、训练流程、验证方式或结果发生变化，应追加新记录或明确标注版本差异。

最主要的实验结果需要同步记录到仓库根目录的 `report.md`，用于快速查看当前结论和关键数值。

This document records the current experiment configuration used in this repo for beat-level phrase/boundary prediction from tempo-derived labels and score features.

## Current Selected Configuration

当前主要采用的配置是：

- Dataset label source: performer tempo curves
- Boundary levels: `L1-L6`
- Training target: `L2+`, using levels `L2, L3, L4, L5, L6`
- Target construction:

```text
target(b) = max_l weight_l * consensus_l(b), l in {L2, L3, L4, L5, L6}
```

- Weight rule: sqrt-mass compensation, normalized to `L6 = 1`

```python
SQRT_MASS_WEIGHTS = {
    2: 0.205,
    3: 0.284,
    4: 0.408,
    5: 0.613,
    6: 1.000,
}
```

- True event threshold:

```text
true_event(b) = target(b) >= 0.01
```

- Event matching tolerance:

```text
±1 beat
```

- Density evaluation:

```text
expected_pred_count(piece) = round(num_beats(piece) / 6)
```

This corresponds to approximately one predicted boundary every 6 beats.

- Density extraction minimum distance:

```text
min_distance = 1
```

This means the previous `min_distance=6` constraint is removed.

## Structural Vector

For MazurkaBL, the structural vector is:

```python
STR_VEC = [beats_per_measure, 2, 2, 2, 2, 2]
```

For 3/4 Mazurka, this becomes:

```python
STR_VEC = [3, 2, 2, 2, 2, 2]
```

For ASAP, `beats_per_measure` is inferred from the ASAP time-signature metadata. Examples:

- `4/4`: `beats_per_measure = 4`, `STR_VEC = [4, 2, 2, 2, 2, 2]`
- `3/4`: `beats_per_measure = 3`, `STR_VEC = [3, 2, 2, 2, 2, 2]`
- `6/8`: treated as compound meter with `beats_per_measure = 2`, `STR_VEC = [2, 2, 2, 2, 2, 2]`
- `12/8`: treated as compound meter with `beats_per_measure = 4`, `STR_VEC = [4, 2, 2, 2, 2, 2]`

The purpose is to make `L1` correspond to one musical measure, then each higher level doubles the previous one.

## Beat Unit

`beats_per_measure` only controls how many beat-grid steps make one measure. The actual duration of one beat-grid step is controlled by `beat_unit`.

For ASAP score feature and note feature generation:

```text
beat_unit = measure_quarter_length / beats_per_measure
```

Here `beat_unit` is measured in `music21` quarterLength units:

```text
quarter note = 1.0
eighth note = 0.5
dotted quarter = 1.5
half note = 2.0
```

Examples:

```text
4/4:
measure_quarter_length = 4.0
beats_per_measure = 4
beat_unit = 1.0

3/4:
measure_quarter_length = 3.0
beats_per_measure = 3
beat_unit = 1.0

6/8:
measure_quarter_length = 3.0
beats_per_measure = 2
beat_unit = 1.5
```

For `6/8`, `beat_unit = 1.5` means one beat-grid step is a dotted quarter note, i.e. `3` eighth notes. Therefore:

```text
6/8:
each beat = 3 eighths
L1 = 2 beats = 1 measure
STR_VEC = [2, 2, 2, 2, 2, 2]
```

For `12/8`:

```text
measure_quarter_length = 6.0
beats_per_measure = 4
beat_unit = 1.5
each beat = 3 eighths
L1 = 4 beats = 1 measure
STR_VEC = [4, 2, 2, 2, 2, 2]
```

## Label Construction Flow

For each piece and each performance:

1. Read beat-aligned performance timestamps.
2. Convert beat timestamps into a tempo curve:

```text
tempo_bpm[i] = 60 / (performance_beat_time[i] - performance_beat_time[i-1])
```

3. Interpolate invalid tempo values.
4. Smooth tempo curve with rolling window size `3`.
5. Clip extreme tempo values to avoid outliers.
6. Run hierarchical local-minimum boundary extraction using `STR_VEC`.
7. Save per-performance binary boundary arrays for `L1-L6`.
8. For each piece and level, average across performances:

```text
consensus_l(b) = mean_performer boundary_l_performer(b)
```

9. Build final training target from `L2-L6`:

```text
target(b) = max_l weight_l * consensus_l(b)
```

## Model Settings

Two main settings are currently compared.

### A. baseline_cnn

Input:

```text
handcrafted score beat features
```

Model:

```text
CNN sequence tagger
```

This is the main non-embedding baseline.

### B. handcrafted_plus_branchwise

Input:

```text
handcrafted score beat features
+ MidiBERT rich beat embedding
```

Rich beat embedding is built per beat from six note-pooling branches:

1. onset-note mean
2. sustain-note mean
3. all-note mean
4. highest-note vector
5. lowest-note vector
6. overlap-duration-weighted mean
7. scalar stats:
   - note count
   - onset count
   - sustain count
   - all count
   - pitch span
   - rest / empty flags or related beat stats

Raw rich feature dimension:

```text
6 * 768 + 7 = 4615
```

Branchwise bottleneck:

```text
onset:   768 -> 24
sustain: 768 -> 24
all:     768 -> 24
top:     768 -> 24
bass:    768 -> 24
dur:     768 -> 24
```

Then:

```text
6 * 24 + 7 = 151
151 -> 128 beat embedding
concat with handcrafted features
CNN sequence tagger
```

Important: `handcrafted_plus_branchwise` uses MidiBERT-derived embeddings, not Aria embeddings.

## Training Parameters

The current runner inherits the CNN sequence settings from:

```text
MERIX SUBMISSION/Boundary_Restart/configs/mazurkabl_l2plus_weighted_auto_meter.yaml
```

Main parameters:

```yaml
model_type: cnn
epochs: 18
lr: 0.001
weight_decay: 0.0001
grad_clip: 1.0
early_stop_patience: 5
hidden_dim: 64
num_layers: 2
dropout: 0.2
kernel_size: 3
tcn_channels: [64, 64, 64]
```

Loss:

```text
BCEWithLogitsLoss(pos_weight = negative_mass / positive_mass)
```

Best epoch logic:

```text
Within each fold, the model keeps the epoch with the lowest mean training loss.
Early stopping triggers after no sufficient training-loss improvement for 5 epochs.
```

Threshold selection:

```text
Threshold is selected on train pieces by sweeping threshold grid and maximizing train F1 under ±1 beat tolerance.
```

Density mode does not use the threshold. It selects the top scoring beats according to:

```text
round(num_beats / 6)
```

## Validation Protocol

Cross validation:

```text
5-fold piece-level split
seed = 42
```

Important: folds are by piece, not by performance. This avoids putting the same score piece in both train and validation.

Metrics:

- `pred_events`: number of predicted boundary beats
- `true_events`: number of true boundary beats where `target >= 0.01`
- `matches_tol1`: one-to-one matches under `±1 beat`
- `UP`: unweighted precision

```text
UP = matches / pred_events
```

- `recall`:

```text
recall = matches / true_events
```

- `WR`: weighted recall

```text
WR = sum(target at matched true events) / sum(target at all true events)
```

- `F1`:

```text
F1 = 2 * UP * recall / (UP + recall)
```

Current reporting preference:

```text
Primary metrics to display and discuss are UP and WR.
F1 should not be the main reported metric in future summaries unless explicitly requested.
```

## MazurkaBL Configuration

Main current output directory:

```text
MERIX SUBMISSION/MIREX_Model/mazurkabl_l2plus_sqrtmass_branchwise_cnn_event0p01_densityfixed_2bars_mindist1/
```

Important files:

```text
aggregate_totals.csv
fold_summary.csv
metadata.json
label_stats.csv
```

Current MazurkaBL result:

| setting | threshold_pred | true | threshold_match | threshold_UP | threshold_WR | threshold_F1 | density_pred | density_match | density_UP | density_WR | density_F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline_cnn | 1747 | 6252 | 1475 | 0.8443 | 0.3081 | 0.3688 | 2435 | 2013 | 0.8267 | 0.3977 | 0.4635 |
| handcrafted_plus_branchwise | 1754 | 6252 | 1514 | 0.8632 | 0.3272 | 0.3782 | 2435 | 2059 | 0.8456 | 0.4064 | 0.4740 |

Interpretation:

```text
On MazurkaBL, handcrafted_plus_branchwise gives a real but modest gain over baseline_cnn.
The gain appears in both precision and weighted recall.
Density F1 improves from 0.4635 to 0.4740.
```

## ASAP30 Configuration

ASAP dataset root:

```text
asap-dataset-master/
```

ASAP expansion constraint check:

```text
Requested expansion to 40 pieces with >=25 performances is not feasible.
Using usable aligned performances, ASAP has only 2 pieces with >=25:
1. Liszt_Gran_Etudes_de_Paganini_2_La_campanella: 29
2. Chopin_Etudes_op_10_8: 28

Using raw metadata performance count, ASAP has only 3 pieces with >=25:
1. Liszt/Gran_Etudes_de_Paganini/2_La_campanella: 29
2. Chopin/Etudes_op_10/8: 28
3. Chopin/Etudes_op_10/1: 25 raw, but only 24 usable aligned

Therefore, ASAP40 cannot satisfy the >=25 usable aligned performance requirement.
To build 40 ASAP pieces, the usable aligned performance threshold must be relaxed to about >=7.
```

ASAP has beat-level score/performance alignment:

```text
performance_beats[i] <-> midi_score_beats[i]
```

It does not provide the same kind of explicit note-level match used in ATEPP-style pipelines. For this experiment, ASAP is used at beat level.

ASAP30 label/output directory:

```text
MERIX SUBMISSION/MIREX_Model/asap30_tempo_boundary_labels/
```

Generated label files:

```text
asap_piece_inventory.csv
asap_gt25_manifest.csv
asap_top30_manifest.csv
asap_top30_tempo_curves_long.csv.gz
asap_top30_level_consensus_long.csv.gz
asap_top30_boundary_summary.csv
asap_top30_performance_summary.csv
asap_top30_aggregate_summary.csv
metadata.json
beat_data_asap_top30_performer_levels/*.npz
```

ASAP strict filter:

```text
Only 2 ASAP pieces have more than 25 usable aligned performances.
```

Therefore the practical `asap30` set is the top 30 pieces ranked by usable aligned performance count.

ASAP30 aggregate:

```text
selected_top30_pieces = 30
processed_performances = 441
total_beats = 20387
target >= 0.01 = 8281 true events
target >= 0.03 = 5391
target >= 0.05 = 3392
```

ASAP30 score feature generation:

```text
MERIX SUBMISSION/MIREX_Model/build_asap30_score_features.py
```

Outputs:

```text
MERIX SUBMISSION/MIREX_Model/asap30_tempo_boundary_labels/asap_top30_beat_table.csv.gz
MERIX SUBMISSION/MIREX_Model/asap30_score_note_feats/
```

For ASAP score features, the beat grid uses:

```text
beat_unit = measure_quarter_length / beats_per_measure
```

This is important for compound meters:

- `6/8`: `beat_unit = 1.5` quarterLength
- `12/8`: `beat_unit = 1.5` quarterLength
- `3/8`: `beat_unit = 0.5` quarterLength
- `2/2`: `beat_unit = 2.0` quarterLength

ASAP30 MidiBERT rich feature generation:

```text
MERIX SUBMISSION/MIREX_Model/build_asap30_midibert_rich_beat_features.py
```

Output:

```text
MERIX SUBMISSION/MIREX_Model/asap30_midibert_rich_beat_features/
```

ASAP30 training script:

```text
MERIX SUBMISSION/MIREX_Model/run_asap30_l2plus_sqrtmass_baseline_branchwise.py
```

ASAP30 result directory:

```text
MERIX SUBMISSION/MIREX_Model/asap30_l2plus_sqrtmass_baseline_branchwise/
```

Current ASAP30 result:

| setting | threshold_pred | true | threshold_match | threshold_UP | threshold_WR | threshold_F1 | density_pred | density_match | density_UP | density_WR | density_F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline_cnn | 2296 | 8281 | 1800 | 0.7840 | 0.2609 | 0.3404 | 3322 | 2559 | 0.7703 | 0.3544 | 0.4411 |
| handcrafted_plus_branchwise | 2328 | 8281 | 1822 | 0.7826 | 0.2507 | 0.3435 | 3336 | 2509 | 0.7521 | 0.3503 | 0.4320 |

Interpretation:

```text
On ASAP30, handcrafted_plus_branchwise does not improve over baseline_cnn.
Threshold F1 is only slightly higher: 0.3435 vs 0.3404.
Density F1 is worse: 0.4320 vs 0.4411.
The main degradation is precision-side: density UP drops from 0.7703 to 0.7521.
```

## ASAP40 Configuration

ASAP40 was built by selecting the top 40 ASAP pieces ranked by usable aligned performance count.

Important constraint:

```text
ASAP40 does not satisfy >=25 usable aligned performances per piece.
Only the first 2 pieces satisfy >=25.
The 40th piece has 7 usable aligned performances.
```

ASAP40 output directories:

```text
MERIX SUBMISSION/MIREX_Model/asap40_tempo_boundary_labels/
MERIX SUBMISSION/MIREX_Model/asap40_score_note_feats/
MERIX SUBMISSION/MIREX_Model/asap40_midibert_rich_beat_features/
MERIX SUBMISSION/MIREX_Model/asap40_l2plus_sqrtmass_baseline_branchwise/
```

ASAP40 aggregate:

```text
selected_top40_pieces = 40
processed_performances = 525
total_beats = 25466
target >= 0.01 = 10024 true events
target >= 0.03 = 6675
target >= 0.05 = 4350
```

ASAP40 selected pieces ranked 31-40:

| rank | piece | usable aligned | beats |
|---:|---|---:|---:|
| 31 | Chopin_Sonata_3_3rd | 9 | 476 |
| 32 | Chopin_Barcarolle | 9 | 463 |
| 33 | Chopin_Etudes_op_25_12 | 9 | 331 |
| 34 | Bach_Prelude_bwv_848 | 9 | 310 |
| 35 | Bach_Fugue_bwv_848 | 9 | 217 |
| 36 | Ravel_Miroirs_4_Alborada_del_gracioso | 8 | 547 |
| 37 | Beethoven_Piano_Sonatas_3-1 | 8 | 1064 |
| 38 | Prokofiev_Toccata | 8 | 452 |
| 39 | Beethoven_Piano_Sonatas_30-1 | 8 | 216 |
| 40 | Scriabin_Sonatas_5 | 7 | 1003 |

Current ASAP40 result:

| setting | threshold_pred | true | threshold_match | threshold_UP | threshold_WR | threshold_F1 | density_pred | density_match | density_UP | density_WR | density_F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline_cnn | 2918 | 10024 | 2226 | 0.7629 | 0.2554 | 0.3440 | 4164 | 3110 | 0.7469 | 0.3504 | 0.4384 |
| handcrafted_plus_branchwise | 2994 | 10024 | 2280 | 0.7615 | 0.2527 | 0.3503 | 4185 | 3111 | 0.7434 | 0.3409 | 0.4379 |

Interpretation:

```text
On ASAP40, handcrafted_plus_branchwise slightly improves threshold F1,
but density-mode performance is essentially unchanged/slightly worse.
Density F1: 0.4379 vs baseline 0.4384.
Density WR drops from 0.3504 to 0.3409.
```

## Combined MazurkaBL + ASAP40 Configuration

Combined dataset experiment:

```text
MazurkaBL + ASAP40
Mazurka pieces = 44
ASAP40 pieces = 40
Total pieces = 84
True events target >= 0.01 = 16276
```

Output directory:

```text
MERIX SUBMISSION/MIREX_Model/combined_mazurkabl_asap40_l2plus_sqrtmass_baseline_branchwise/
```

Training/evaluation configuration:

```text
target = max_{L2..L6}(sqrt-mass weight_L * performer consensus_L)
true_event = target >= 0.01
density = round(num_beats / 6)
min_distance = 1
tolerance = ±1 beat
5-fold piece-level split, seed = 42
```

Important feature compatibility note:

```text
MazurkaBL handcrafted feature table has 104 selected columns.
ASAP40 handcrafted feature table has 43 selected columns.
For the combined experiment, only the 43 common handcrafted XML/score features are used.
This keeps baseline_cnn input dimensions consistent across datasets.
```

Rich branchwise embeddings:

```text
MazurkaBL: mazurkabl_midibert_rich_beat_features_meter34/
ASAP40: asap40_midibert_rich_beat_features/
rich_dim = 4615
branchwise bottleneck = 6 * 24 + 7 -> 128
```

Combined result:

| setting | threshold_pred | true | threshold_match | threshold_UP | threshold_WR | threshold_F1 | density_pred | density_match | density_UP | density_WR | density_F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline_cnn | 4820 | 16276 | 3800 | 0.7884 | 0.2681 | 0.3603 | 6615 | 5120 | 0.7740 | 0.3570 | 0.4473 |
| handcrafted_plus_branchwise | 4729 | 16276 | 3782 | 0.7997 | 0.2789 | 0.3601 | 6618 | 5198 | 0.7854 | 0.3670 | 0.4541 |

Interpretation:

```text
On the combined MazurkaBL+ASAP40 dataset, handcrafted_plus_branchwise improves density-mode performance modestly.
Density F1 improves from 0.4473 to 0.4541.
Density UP improves from 0.7740 to 0.7854.
Density WR improves from 0.3570 to 0.3670.

Threshold F1 is essentially unchanged: 0.3601 vs 0.3603.
```

Per-piece validation results for the combined experiment:

```text
MERIX SUBMISSION/MIREX_Model/combined_mazurkabl_asap40_l2plus_sqrtmass_baseline_branchwise/per_piece_validation_summary.csv
MERIX SUBMISSION/MIREX_Model/combined_mazurkabl_asap40_l2plus_sqrtmass_baseline_branchwise/per_piece_validation_compact.csv
```

Per-piece result columns include:

```text
setting
fold
dataset
piece
num_beats
threshold_pred_events
threshold_true_events
threshold_matches_tol1
threshold_UP
threshold_recall
threshold_WR
threshold_f1
density_pred_events
density_true_events
density_matches_tol1
density_UP
density_recall
density_WR
density_f1
```

Per-dataset totals computed from per-piece validation rows:

| setting | dataset | density_pred | density_true | density_match | density_UP | density_recall | density_F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| baseline_cnn | mazurka | 2435 | 6252 | 1983 | 0.8144 | 0.3172 | 0.4565 |
| baseline_cnn | asap40 | 4180 | 10024 | 3137 | 0.7505 | 0.3129 | 0.4417 |
| handcrafted_plus_branchwise | mazurka | 2435 | 6252 | 2049 | 0.8415 | 0.3277 | 0.4717 |
| handcrafted_plus_branchwise | asap40 | 4183 | 10024 | 3149 | 0.7528 | 0.3141 | 0.4433 |

## Combined L23 / L456 Split-Target Experiment

Experiment request:

```text
Group L2 and L3 into one target.
Group L4, L5, and L6 into another target.
Train the two targets separately.
```

Dataset:

```text
MazurkaBL + ASAP40
pieces = 84
handcrafted features = 43 common columns
rich_dim = 4615
```

Target definitions:

```text
L23 target:
target_L23(b) = max(weight_2 * consensus_L2(b), weight_3 * consensus_L3(b))

L456 target:
target_L456(b) = max(weight_4 * consensus_L4(b), weight_5 * consensus_L5(b), weight_6 * consensus_L6(b))
```

Weights:

```text
L2 = 0.205
L3 = 0.284
L4 = 0.408
L5 = 0.613
L6 = 1.000
```

Evaluation configuration:

```text
true_event = target >= 0.01
density = round(num_beats / 6)
min_distance = 1
tolerance = ±1 beat
```

Output directory:

```text
MERIX SUBMISSION/MIREX_Model/combined_mazurkabl_asap40_l23_l456_targets/
```

Main result, density mode:

| target | setting | true | pred | match | UP | WR |
|---|---|---:|---:|---:|---:|---:|
| L23 | baseline_cnn | 15977 | 6612 | 5085 | 0.7691 | 0.3608 |
| L23 | handcrafted_plus_branchwise | 15977 | 6620 | 5168 | 0.7807 | 0.3692 |
| L456 | baseline_cnn | 6370 | 6605 | 2414 | 0.3655 | 0.4059 |
| L456 | handcrafted_plus_branchwise | 6370 | 6617 | 2468 | 0.3730 | 0.4116 |

Interpretation:

```text
Splitting targets into L23 and L456 preserves a small positive branchwise effect in density mode.

L23:
UP improves 0.7691 -> 0.7807.
WR improves 0.3608 -> 0.3692.

L456:
UP improves 0.3655 -> 0.3730.
WR improves 0.4059 -> 0.4116.

L456 has much lower UP because the high-level target is sparse but density mode still predicts about one event per 6 beats.
For L456, a lower density setting may be more appropriate in future experiments.
```

## Reproducibility Commands

Build ASAP30 labels:

```bash
python "MERIX SUBMISSION/MIREX_Model/build_asap30_tempo_boundary_labels.py"
```

Build ASAP40 labels:

```bash
ASAP_TOP_N=40 python "MERIX SUBMISSION/MIREX_Model/build_asap30_tempo_boundary_labels.py"
```

Build ASAP30 score features and note features:

```bash
python "MERIX SUBMISSION/MIREX_Model/build_asap30_score_features.py"
```

Build ASAP40 score features and note features:

```bash
ASAP_TOP_N=40 python "MERIX SUBMISSION/MIREX_Model/build_asap30_score_features.py"
```

Build ASAP30 MidiBERT rich features:

```bash
python "MERIX SUBMISSION/MIREX_Model/build_asap30_midibert_rich_beat_features.py"
```

Build ASAP40 MidiBERT rich features:

```bash
ASAP_TOP_N=40 python "MERIX SUBMISSION/MIREX_Model/build_asap30_midibert_rich_beat_features.py"
```

Run ASAP30 current experiment:

```bash
python -u "MERIX SUBMISSION/MIREX_Model/run_asap30_l2plus_sqrtmass_baseline_branchwise.py"
```

Run ASAP40 current experiment:

```bash
ASAP_TOP_N=40 python -u "MERIX SUBMISSION/MIREX_Model/run_asap30_l2plus_sqrtmass_baseline_branchwise.py"
```

Run combined MazurkaBL + ASAP40 current experiment:

```bash
python -u "MERIX SUBMISSION/MIREX_Model/run_combined_mazurkabl_asap40_l2plus_sqrtmass.py"
```

Run MazurkaBL current experiment:

```bash
python -u "MERIX SUBMISSION/MIREX_Model/run_mazurkabl_l2plus_sqrtmass_branchwise_cnn.py"
```

Note: the Mazurka current result used the event/density/min-distance variant saved under:

```text
mazurkabl_l2plus_sqrtmass_branchwise_cnn_event0p01_densityfixed_2bars_mindist1/
```

## Current Practical Conclusion

MazurkaBL:

```text
handcrafted_plus_branchwise is useful.
It improves density F1 and WR over baseline_cnn.
```

ASAP30:

```text
handcrafted_plus_branchwise is not useful in the current setup.
The rich MidiBERT branch does not transfer cleanly to ASAP30 under this label construction.
```

Working hypothesis:

```text
The rich MidiBERT beat embedding may encode useful score context on MazurkaBL,
but on ASAP30 the label noise, repertoire spread, meter differences, and beat-only alignment
make the added embedding less aligned with the tempo-derived boundary target.
```

Therefore, for paper/report purposes:

```text
Keep baseline_cnn as the stable reference.
Use handcrafted_plus_branchwise as an embedding augmentation only where it shows controlled improvement.
Always report both baseline_cnn and handcrafted_plus_branchwise to show the effect is not just model capacity.
```

## 2026-06-26 L456 Density Sweep

Request:

```text
For the split L23 / L456 target setup, test L456 with lower density:
one predicted boundary per 12 beats and one predicted boundary per 24 beats.
```

Script:

```text
MERIX SUBMISSION/MIREX_Model/run_combined_mazurkabl_asap40_l23_l456_targets.py
```

Change:

```text
The script now accepts:
COMBINED_L456_DENSITY_BEATS
COMBINED_L23_L456_RUN_NAME
```

This keeps the original run intact and writes separate output folders.

Output folders:

```text
MERIX SUBMISSION/MIREX_Model/combined_mazurkabl_asap40_l23_l456_targets_density12/
MERIX SUBMISSION/MIREX_Model/combined_mazurkabl_asap40_l23_l456_targets_density24/
```

Common setup:

```text
Dataset: MazurkaBL + ASAP40
Targets:
  L23 = max(weight_L2 * consensus_L2, weight_L3 * consensus_L3)
  L456 = max(weight_L4 * consensus_L4, weight_L5 * consensus_L5, weight_L6 * consensus_L6)
Weights: L2=0.205, L3=0.284, L4=0.408, L5=0.613, L6=1.000
True event threshold: target >= 0.01
Tolerance: +/- 1 beat
min_distance: 1
5-fold piece-level CV, seed 42
Main metrics: UP and WR
```

L456 density-mode results:

| density rule | setting | pred | true | match | UP | WR |
|---|---|---:|---:|---:|---:|---:|
| every 6 beats | baseline_cnn | 6605 | 6370 | 2414 | 0.3655 | 0.4059 |
| every 6 beats | handcrafted_plus_branchwise | 6617 | 6370 | 2468 | 0.3730 | 0.4116 |
| every 12 beats | baseline_cnn | 3344 | 6370 | 1432 | 0.4282 | 0.2558 |
| every 12 beats | handcrafted_plus_branchwise | 3344 | 6370 | 1514 | 0.4528 | 0.2806 |
| every 24 beats | baseline_cnn | 1672 | 6370 | 843 | 0.5042 | 0.1655 |
| every 24 beats | handcrafted_plus_branchwise | 1672 | 6370 | 854 | 0.5108 | 0.1801 |

Interpretation:

```text
Lowering L456 density increases UP but strongly reduces WR.
Every 12 beats is a more balanced sparse-L456 setting than every 24 beats.
Every 24 beats is high-precision / low-coverage.
handcrafted_plus_branchwise improves UP and WR over baseline_cnn at all three L456 density settings.
```

## 2026-06-26 L456 Per-Piece UP Ranking

Request:

```text
List the validation pieces with the highest UP for the L456 experiment.
```

Script:

```text
MERIX SUBMISSION/MIREX_Model/run_combined_mazurkabl_asap40_l456_per_piece_density_sweep.py
```

Output:

```text
MERIX SUBMISSION/MIREX_Model/combined_mazurkabl_asap40_l456_per_piece_density_sweep/
```

The script trains the L456 target once per setting/fold, then evaluates each validation piece at density rules:

```text
every 6 beats
every 12 beats
every 24 beats
```

Top 5 pieces by density UP, `handcrafted_plus_branchwise`:

| density rule | dataset | piece | pred | true | match | UP | WR |
|---|---|---|---:|---:|---:|---:|---:|
| every 6 beats | mazurka | M68-1 | 42 | 70 | 33 | 0.7857 | 0.4237 |
| every 6 beats | mazurka | M41-3 | 39 | 75 | 27 | 0.6923 | 0.3516 |
| every 6 beats | mazurka | M67-1 | 30 | 38 | 18 | 0.6000 | 0.3735 |
| every 6 beats | mazurka | M59-2 | 56 | 77 | 33 | 0.5893 | 0.5674 |
| every 6 beats | mazurka | M17-3 | 84 | 116 | 47 | 0.5595 | 0.5193 |
| every 12 beats | mazurka | M41-3 | 20 | 75 | 17 | 0.8500 | 0.2544 |
| every 12 beats | mazurka | M68-1 | 21 | 70 | 16 | 0.7619 | 0.2378 |
| every 12 beats | mazurka | M67-1 | 15 | 38 | 11 | 0.7333 | 0.2829 |
| every 12 beats | mazurka | M50-3 | 52 | 128 | 38 | 0.7308 | 0.2867 |
| every 12 beats | mazurka | M56-3 | 55 | 156 | 40 | 0.7273 | 0.2981 |
| every 24 beats | mazurka | M24-3 | 10 | 42 | 10 | 1.0000 | 0.3642 |
| every 24 beats | mazurka | M67-4 | 14 | 74 | 13 | 0.9286 | 0.3003 |
| every 24 beats | mazurka | M68-2 | 11 | 40 | 10 | 0.9091 | 0.4137 |
| every 24 beats | mazurka | M68-1 | 10 | 70 | 9 | 0.9000 | 0.1847 |
| every 24 beats | mazurka | M41-3 | 10 | 75 | 9 | 0.9000 | 0.1846 |

Observation:

```text
The highest-UP L456 validation pieces are all MazurkaBL pieces.
Lower density increases UP sharply, but some high-UP cases have low WR because very few points are predicted.
```
