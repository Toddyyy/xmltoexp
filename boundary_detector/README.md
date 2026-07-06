# Boundary Detector

This folder is a self-contained package for the current boundary prediction experiments recorded in `agent.md` and `report.md`.

## Current Configuration

- Dataset: MazurkaBL + ASAP40
- Targets: L2+, L23, L456
- Models: `baseline_cnn`, `handcrafted_plus_branchwise`
- Label rule: `target(b) = max_l weight_l * consensus_l(b)`
- Weights: L2=0.205, L3=0.284, L4=0.408, L5=0.613, L6=1.000
- True event threshold: `target >= 0.01`
- Tolerance: +/- 1 beat
- Density: default `round(num_beats / 6)`, plus L456 sweeps at 12 and 24 beats

## Directory Layout

```text
boundary_detector/
  README.md
  agent.md
  report.md
  config/
    mazurkabl_l2plus_weighted_auto_meter.yaml
  src/
    boundary_restart/        # CNN model, metrics, table loading, feature utilities
    data_builders/           # optional scripts for regenerating labels/features
    experiments/             # current training/evaluation runners
  data/
    raw/
      MazurkaBL -> ../../../datasets/MazurkaBL
    labels/
      mazurkabl_performer_levels/
      asap40_tempo_boundary_labels/
    features/
      mazurkabl_handcrafted_beat_table.csv.gz
      mazurkabl_midibert_rich_beat_features/
      asap40_score_note_feats/
      asap40_midibert_rich_beat_features/
  results/
    mazurkabl_l2plus_sqrtmass_branchwise_cnn_event0p01_densityfixed_2bars_mindist1/
    asap30_l2plus_sqrtmass_baseline_branchwise/
    asap40_l2plus_sqrtmass_baseline_branchwise/
    combined_l2plus_sqrtmass_baseline_branchwise/
    combined_l23_l456_density6/
    combined_l23_l456_density12/
    combined_l23_l456_density24/
    combined_l456_per_piece_density_sweep/
    performer_boundary_density_stats/
  project_inventory/
```

## Main Commands

Run from this directory:

```bash
cd /Users/toddywang/Documents/VsCodeProjects/xmltoexp/boundary_detector
python -u src/experiments/run_combined_mazurkabl_asap40_l2plus_sqrtmass.py
python -u src/experiments/run_combined_mazurkabl_asap40_l23_l456_targets.py
python -u src/experiments/run_combined_mazurkabl_asap40_l456_per_piece_density_sweep.py
```

For L456 density variants:

```bash
env COMBINED_L456_DENSITY_BEATS=12 COMBINED_L23_L456_RUN_NAME=combined_mazurkabl_asap40_l23_l456_targets_density12 python -u src/experiments/run_combined_mazurkabl_asap40_l23_l456_targets.py

env COMBINED_L456_DENSITY_BEATS=24 COMBINED_L23_L456_RUN_NAME=combined_mazurkabl_asap40_l23_l456_targets_density24 python -u src/experiments/run_combined_mazurkabl_asap40_l23_l456_targets.py
```

## Notes

- `src/experiments/` contains runnable training/evaluation scripts.
- `src/data_builders/` contains optional regeneration scripts. Some regeneration scripts need external resources such as ASAP raw data or MidiBERT checkpoints.
- Raw datasets are centralized under `/Users/toddywang/Documents/VsCodeProjects/xmltoexp/datasets/`.
- The existing labels/features/results are already copied under `data/` and `results/`, so the current experiments do not require the old `MERIX SUBMISSION/MIREX_Model` folder.
- Full copied-file inventory is in `project_inventory/`.
