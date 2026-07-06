# Boundary Prediction Main Report

This file records the main experiment results and current conclusions. Detailed configuration and experiment history are kept in `agent.md`.

## Current Main Configuration

- Target: `L2+`
- Beat unit for ASAP score features:

```text
beat_unit = measure_quarter_length / beats_per_measure
```

For compound meters, this means:

```text
6/8:  beats_per_measure = 2, beat_unit = 1.5 quarterLength = 3 eighths
12/8: beats_per_measure = 4, beat_unit = 1.5 quarterLength = 3 eighths
```

- Label construction:

```text
target(b) = max_l weight_l * consensus_l(b), l in {L2, L3, L4, L5, L6}
```

- Weights:

```python
{2: 0.205, 3: 0.284, 4: 0.408, 5: 0.613, 6: 1.000}
```

- True event: `target >= 0.01`
- Tolerance: `±1 beat`
- Density mode: `round(num_beats / 6)`
- Density `min_distance = 1`
- Main reporting metrics: `UP` and `WR`
- `F1` is secondary and should not be emphasized unless explicitly requested.
- Main model comparison:
  - `baseline_cnn`
  - `handcrafted_plus_branchwise`

## MazurkaBL Result

Output directory:

```text
MERIX SUBMISSION/MIREX_Model/mazurkabl_l2plus_sqrtmass_branchwise_cnn_event0p01_densityfixed_2bars_mindist1/
```

| setting | threshold_pred | true | threshold_match | threshold_UP | threshold_WR | threshold_F1 | density_pred | density_match | density_UP | density_WR | density_F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline_cnn | 1747 | 6252 | 1475 | 0.8443 | 0.3081 | 0.3688 | 2435 | 2013 | 0.8267 | 0.3977 | 0.4635 |
| handcrafted_plus_branchwise | 1754 | 6252 | 1514 | 0.8632 | 0.3272 | 0.3782 | 2435 | 2059 | 0.8456 | 0.4064 | 0.4740 |

Main conclusion:

```text
On MazurkaBL, handcrafted_plus_branchwise improves over baseline_cnn.
The main density-mode F1 improves from 0.4635 to 0.4740.
Weighted recall improves from 0.3977 to 0.4064.
```

## ASAP30 Result

Output directory:

```text
MERIX SUBMISSION/MIREX_Model/asap30_l2plus_sqrtmass_baseline_branchwise/
```

ASAP expansion note:

```text
ASAP cannot be expanded to 40 pieces while requiring >=25 usable aligned performances.
Only 2 pieces satisfy >=25 usable aligned performances.
To reach 40 pieces, the threshold must be relaxed to about >=7 usable aligned performances.
```

| setting | threshold_pred | true | threshold_match | threshold_UP | threshold_WR | threshold_F1 | density_pred | density_match | density_UP | density_WR | density_F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline_cnn | 2296 | 8281 | 1800 | 0.7840 | 0.2609 | 0.3404 | 3322 | 2559 | 0.7703 | 0.3544 | 0.4411 |
| handcrafted_plus_branchwise | 2328 | 8281 | 1822 | 0.7826 | 0.2507 | 0.3435 | 3336 | 2509 | 0.7521 | 0.3503 | 0.4320 |

Main conclusion:

```text
On ASAP30, handcrafted_plus_branchwise does not improve over baseline_cnn.
Density-mode F1 drops from 0.4411 to 0.4320.
The main problem is precision-side: density UP drops from 0.7703 to 0.7521.
```

## ASAP40 Result

Output directory:

```text
MERIX SUBMISSION/MIREX_Model/asap40_l2plus_sqrtmass_baseline_branchwise/
```

Dataset summary:

```text
selected_top40_pieces = 40
processed_performances = 525
total_beats = 25466
target >= 0.01 = 10024
min usable aligned performances = 7
max usable aligned performances = 29
```

| setting | threshold_pred | true | threshold_match | threshold_UP | threshold_WR | threshold_F1 | density_pred | density_match | density_UP | density_WR | density_F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline_cnn | 2918 | 10024 | 2226 | 0.7629 | 0.2554 | 0.3440 | 4164 | 3110 | 0.7469 | 0.3504 | 0.4384 |
| handcrafted_plus_branchwise | 2994 | 10024 | 2280 | 0.7615 | 0.2527 | 0.3503 | 4185 | 3111 | 0.7434 | 0.3409 | 0.4379 |

Main conclusion:

```text
On ASAP40, handcrafted_plus_branchwise does not materially improve over baseline_cnn.
Threshold F1 improves slightly: 0.3503 vs 0.3440.
Density F1 is effectively the same/slightly worse: 0.4379 vs 0.4384.
Weighted recall in density mode drops from 0.3504 to 0.3409.
```

## Combined MazurkaBL + ASAP40 Result

Output directory:

```text
MERIX SUBMISSION/MIREX_Model/combined_mazurkabl_asap40_l2plus_sqrtmass_baseline_branchwise/
```

Dataset summary:

```text
MazurkaBL pieces = 44
ASAP40 pieces = 40
Total pieces = 84
target >= 0.01 = 16276
```

Feature note:

```text
MazurkaBL has 104 selected handcrafted columns.
ASAP40 has 43 selected handcrafted columns.
The combined experiment uses only the 43 common handcrafted features.
```

| setting | threshold_pred | true | threshold_match | threshold_UP | threshold_WR | threshold_F1 | density_pred | density_match | density_UP | density_WR | density_F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline_cnn | 4820 | 16276 | 3800 | 0.7884 | 0.2681 | 0.3603 | 6615 | 5120 | 0.7740 | 0.3570 | 0.4473 |
| handcrafted_plus_branchwise | 4729 | 16276 | 3782 | 0.7997 | 0.2789 | 0.3601 | 6618 | 5198 | 0.7854 | 0.3670 | 0.4541 |

Main conclusion:

```text
On the combined MazurkaBL+ASAP40 dataset, handcrafted_plus_branchwise gives a modest density-mode gain.
Density F1 improves from 0.4473 to 0.4541.
Density UP improves from 0.7740 to 0.7854.
Density WR improves from 0.3570 to 0.3670.
Threshold-mode F1 is essentially unchanged.
```

## Current Overall Conclusion

```text
The branchwise MidiBERT augmentation is useful on MazurkaBL but not stable across ASAP30.
For reporting, baseline_cnn should remain the main stable reference.
handcrafted_plus_branchwise should be reported as an embedding augmentation with dataset-dependent benefit.
```

ASAP40 note:

```text
Expanding ASAP from 30 to 40 pieces does not make the branchwise embedding clearly useful.
The additional 10 pieces have only 7-9 usable aligned performances each, so their consensus labels are weaker.
```

Combined dataset note:

```text
Combining MazurkaBL with ASAP40 recovers a small positive density-mode effect from branchwise embeddings.
However, the improvement is modest and depends on using common 43-dimensional handcrafted features.
```

## L23 / L456 Split-Target Result

Output directory:

```text
MERIX SUBMISSION/MIREX_Model/combined_mazurkabl_asap40_l23_l456_targets/
```

Density mode only:

| target | setting | true | pred | match | UP | WR |
|---|---|---:|---:|---:|---:|---:|
| L23 | baseline_cnn | 15977 | 6612 | 5085 | 0.7691 | 0.3608 |
| L23 | handcrafted_plus_branchwise | 15977 | 6620 | 5168 | 0.7807 | 0.3692 |
| L456 | baseline_cnn | 6370 | 6605 | 2414 | 0.3655 | 0.4059 |
| L456 | handcrafted_plus_branchwise | 6370 | 6617 | 2468 | 0.3730 | 0.4116 |

Main conclusion:

```text
The split-target setup keeps the branchwise improvement in UP and WR for both groups.
L23 behaves similarly to the full L2+ target.
L456 is much sparser and has low UP under the current density rule.
Future L456 experiments should reduce prediction density.
```

## L456 Density Sweep

Output directories:

```text
MERIX SUBMISSION/MIREX_Model/combined_mazurkabl_asap40_l23_l456_targets_density12/
MERIX SUBMISSION/MIREX_Model/combined_mazurkabl_asap40_l23_l456_targets_density24/
```

Density mode, L456 only:

| density rule | setting | pred | true | match | UP | WR |
|---|---|---:|---:|---:|---:|---:|
| every 6 beats | baseline_cnn | 6605 | 6370 | 2414 | 0.3655 | 0.4059 |
| every 6 beats | handcrafted_plus_branchwise | 6617 | 6370 | 2468 | 0.3730 | 0.4116 |
| every 12 beats | baseline_cnn | 3344 | 6370 | 1432 | 0.4282 | 0.2558 |
| every 12 beats | handcrafted_plus_branchwise | 3344 | 6370 | 1514 | 0.4528 | 0.2806 |
| every 24 beats | baseline_cnn | 1672 | 6370 | 843 | 0.5042 | 0.1655 |
| every 24 beats | handcrafted_plus_branchwise | 1672 | 6370 | 854 | 0.5108 | 0.1801 |

Main takeaway:

```text
For L456, lower density trades WR for UP.
Every 12 beats is the better compromise.
Every 24 beats gives the highest UP but WR becomes too low.
handcrafted_plus_branchwise is consistently better than baseline_cnn in both UP and WR.
```

## L456 Per-Piece Highest UP

Output directory:

```text
MERIX SUBMISSION/MIREX_Model/combined_mazurkabl_asap40_l456_per_piece_density_sweep/
```

Top 5 validation pieces by density UP, `handcrafted_plus_branchwise`:

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

Main takeaway:

```text
The highest-UP L456 pieces are all from MazurkaBL.
Every 12 beats gives a more usable balance than every 24 beats.
```

## Performer-Level Boundary Density

This uses the original per-performance boundaries, not consensus `target >= 0.01` events.

Source:

```text
MERIX SUBMISSION/MIREX_Model/combined_mazurkabl_asap40_performer_boundary_density_stats/
```

| target group | performances | pooled beats / boundary | mean performer beats / boundary | median performer beats / boundary |
|---|---:|---:|---:|---:|
| L2+ | 2524 | 10.49 | 10.74 | 10.59 |
| L23 | 2524 | 10.49 | 10.74 | 10.59 |
| L456 | 2524 | 40.98 | 44.55 | 42.00 |

Main takeaway:

```text
At the per-performance boundary level, L2+ and L23 both average about one boundary per 10.7 beats.
L456 averages about one boundary per 44.6 beats.
```
