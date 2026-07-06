# Boundary Prediction Model Related Files

Scope: files referenced by `agent.md` and `report.md` for the current boundary prediction model: MazurkaBL + ASAP40, L2+/L23/L456, `baseline_cnn` and `handcrafted_plus_branchwise`.

- Full per-file CSV: `/Users/toddywang/Documents/VsCodeProjects/xmltoexp/project_inventory/boundary_prediction_related_files.csv.gz`
- Total related files listed in CSV: 20551

## Current Core Files

| category | path | files | size | role |
|---|---|---:|---:|---|
| documentation | `agent.md` | 1 | 24.4KB | 实验配置、流程和历史记录 |
| documentation | `report.md` | 1 | 9.9KB | 主要结果汇总和结论 |
| config | `MERIX SUBMISSION/Boundary_Restart/configs/mazurkabl_l2plus_weighted_auto_meter.yaml` | 1 | 1.3KB | 当前 CNN/feature 配置 |
| code_dependency | `MERIX SUBMISSION/Boundary_Restart/boundary_restart/models.py` | 1 | 8.8KB | CNN sequence model builder |
| code_dependency | `MERIX SUBMISSION/Boundary_Restart/boundary_restart/table_io.py` | 1 | 3.2KB | beat table loading / feature column selection |
| code_dependency | `MERIX SUBMISSION/Boundary_Restart/boundary_restart/metrics.py` | 1 | 31.6KB | event/tolerance metrics |
| code_dependency | `MERIX SUBMISSION/Boundary_Restart/boundary_restart/features.py` | 1 | 24.9KB | feature utilities |
| code_dependency | `MERIX SUBMISSION/Boundary_Restart/boundary_restart/config.py` | 1 | 982B | config helper |
| code_dependency | `MERIX SUBMISSION/Boundary_Restart/boundary_restart/__init__.py` | 1 | 32B | package marker |
| build_script | `MERIX SUBMISSION/MIREX_Model/build_mazurka_beat_npz_performer_levels.py` | 1 | 15.2KB | MazurkaBL per-performance L1-L6 boundary label builder |
| build_script | `MERIX SUBMISSION/MIREX_Model/build_mazurkabl_midibert_rich_beat_features.py` | 1 | 7.8KB | MazurkaBL rich MidiBERT beat features |
| build_script | `MERIX SUBMISSION/MIREX_Model/build_asap30_tempo_boundary_labels.py` | 1 | 16.1KB | ASAP30/40 tempo labels and per-performance L1-L6 boundaries; parameterized by ASAP_TOP_N |
| build_script | `MERIX SUBMISSION/MIREX_Model/build_asap30_score_features.py` | 1 | 6.6KB | ASAP30/40 score note/handcrafted beat features; parameterized by ASAP_TOP_N |
| build_script | `MERIX SUBMISSION/MIREX_Model/build_asap30_midibert_rich_beat_features.py` | 1 | 5.7KB | ASAP30/40 rich MidiBERT beat features; parameterized by ASAP_TOP_N |
| run_script | `MERIX SUBMISSION/MIREX_Model/run_mazurkabl_l2plus_weighted_target_experiment.py` | 1 | 17.5KB | base MazurkaBL L2+ runner used indirectly for labels/metrics |
| run_script | `MERIX SUBMISSION/MIREX_Model/run_mazurkabl_l2plus_rich_midibert_mlp_cnn.py` | 1 | 15.7KB | rich MidiBERT MLP-CNN base runner, provides load_config/load_features/evaluate helpers |
| run_script | `MERIX SUBMISSION/MIREX_Model/run_mazurkabl_l2plus_branchwise_bottleneck_cnn.py` | 1 | 10.4KB | branchwise bottleneck model implementation |
| run_script | `MERIX SUBMISSION/MIREX_Model/run_mazurkabl_l2plus_sqrtmass_branchwise_cnn.py` | 1 | 8.4KB | current MazurkaBL sqrt-mass L2+ experiment |
| run_script | `MERIX SUBMISSION/MIREX_Model/run_asap30_l2plus_sqrtmass_baseline_branchwise.py` | 1 | 11.6KB | ASAP30/40 L2+ sqrt-mass experiment runner |
| run_script | `MERIX SUBMISSION/MIREX_Model/run_combined_mazurkabl_asap40_l2plus_sqrtmass.py` | 1 | 8.3KB | current combined MazurkaBL+ASAP40 L2+ experiment |
| run_script | `MERIX SUBMISSION/MIREX_Model/run_combined_mazurkabl_asap40_per_piece_eval.py` | 1 | 8.0KB | combined per-piece validation evaluation |
| run_script | `MERIX SUBMISSION/MIREX_Model/run_combined_mazurkabl_asap40_l23_l456_targets.py` | 1 | 8.7KB | L23/L456 split-target experiment |
| run_script | `MERIX SUBMISSION/MIREX_Model/run_combined_mazurkabl_asap40_l456_per_piece_density_sweep.py` | 1 | 7.3KB | L456 per-piece density sweep |
| data_dir | `MERIX SUBMISSION/MIREX_Model/beat_data_mazurka_performer_levels_recomputed_sw3_20260524` | 11994 | 461.5MB | MazurkaBL per-performance L1-L6 binary labels used for consensus |
| data_dir | `MERIX SUBMISSION/MIREX_Model/mazurkabl_midibert_rich_beat_features_meter34` | 46 | 140.2MB | MazurkaBL rich MidiBERT beat features |
| data_dir | `MERIX SUBMISSION/MIREX_Model/asap40_tempo_boundary_labels` | 3160 | 17.7MB | ASAP40 selected pieces, tempo labels, per-performance labels, consensus |
| data_dir | `MERIX SUBMISSION/MIREX_Model/asap40_score_note_feats` | 42 | 723.8KB | ASAP40 score handcrafted/note features |
| data_dir | `MERIX SUBMISSION/MIREX_Model/asap40_midibert_rich_beat_features` | 42 | 244.8MB | ASAP40 rich MidiBERT beat features |
| raw_dataset | `MazurkaBL-master` | 2438 | 622.9MB | MazurkaBL raw beat_time/beat_dyn/xml_scores source dataset |
| result_dir | `MERIX SUBMISSION/MIREX_Model/mazurkabl_l2plus_sqrtmass_branchwise_cnn_event0p01_densityfixed_2bars_mindist1` | 6 | 15.6KB | MazurkaBL current L2+ result |
| result_dir | `MERIX SUBMISSION/MIREX_Model/asap30_l2plus_sqrtmass_baseline_branchwise` | 6 | 13.7KB | ASAP30 comparison result |
| result_dir | `MERIX SUBMISSION/MIREX_Model/asap40_l2plus_sqrtmass_baseline_branchwise` | 6 | 16.0KB | ASAP40 comparison result |
| result_dir | `MERIX SUBMISSION/MIREX_Model/combined_mazurkabl_asap40_l2plus_sqrtmass_baseline_branchwise` | 9 | 143.4KB | main combined L2+ result and per-piece validation |
| result_dir | `MERIX SUBMISSION/MIREX_Model/combined_mazurkabl_asap40_l23_l456_targets` | 4 | 23.7KB | split L23/L456 result, density every 6 beats |
| result_dir | `MERIX SUBMISSION/MIREX_Model/combined_mazurkabl_asap40_l23_l456_targets_density12` | 4 | 23.7KB | split L23/L456 result, density every 12 beats |
| result_dir | `MERIX SUBMISSION/MIREX_Model/combined_mazurkabl_asap40_l23_l456_targets_density24` | 4 | 23.8KB | split L23/L456 result, density every 24 beats |
| result_dir | `MERIX SUBMISSION/MIREX_Model/combined_mazurkabl_asap40_l456_per_piece_density_sweep` | 2 | 119.6KB | L456 per-piece top-UP/density sweep result |
| result_dir | `MERIX SUBMISSION/MIREX_Model/combined_mazurkabl_asap40_performer_boundary_density_stats` | 2 | 232.8KB | per-performance boundary density statistics |

## Historical / Diagnostic Related Files

| category | path | files | size | role |
|---|---|---:|---:|---|
| historical_related_dir | `MERIX SUBMISSION/MIREX_Model/mazurkabl_l2plus_branchwise_bottleneck_cnn` | 3 | 8.6KB | branchwise bottleneck ablation |
| historical_related_dir | `MERIX SUBMISSION/MIREX_Model/mazurkabl_l2plus_rich_midibert_mlp_cnn` | 3 | 10.2KB | rich MLP-CNN ablation |
| historical_related_dir | `MERIX SUBMISSION/MIREX_Model/mazurkabl_l2plus_rich_branch_ablation` | 3 | 10.1KB | single-branch rich feature ablation |
| historical_related_dir | `MERIX SUBMISSION/MIREX_Model/mazurkabl_l2plus_diff_bottleneck_cnn` | 3 | 8.6KB | difference-feature bottleneck ablation |
| historical_related_dir | `MERIX SUBMISSION/MIREX_Model/mazurkabl_midibert_target_association` | 6 | 650.6KB | MidiBERT-target association diagnostics |
| historical_related_dir | `MERIX SUBMISSION/MIREX_Model/mazurkabl_performer_boundary_commonality` | 5 | 1.3MB | performer commonality diagnostics |
| historical_related_dir | `MERIX SUBMISSION/MIREX_Model/mazurkabl_boundary_patterns` | 8 | 6.7MB | boundary spacing/nesting/pattern diagnostics |
| historical_related_dir | `MERIX SUBMISSION/MIREX_Model/mazurkabl_level_boundary_distribution` | 3 | 96.8KB | L1-L6 boundary distribution diagnostics |
| historical_related_dir | `MERIX SUBMISSION/MIREX_Model/mazurkabl_sqrtmass_prediction_example_plots` | 9 | 2.1MB | example prediction plots |
| historical_related_dir | `MERIX SUBMISSION/MIREX_Model/asap30_tempo_boundary_labels` | 2656 | 15.2MB | ASAP30 derived labels |
| historical_related_dir | `MERIX SUBMISSION/MIREX_Model/asap30_score_note_feats` | 32 | 565.5KB | ASAP30 score features |
| historical_related_dir | `MERIX SUBMISSION/MIREX_Model/asap30_midibert_rich_beat_features` | 32 | 193.6MB | ASAP30 rich features |

## Practical Keep List

Keep these to reproduce the current reported model/results:

- `agent.md`
- `report.md`
- `MERIX SUBMISSION/Boundary_Restart/configs/mazurkabl_l2plus_weighted_auto_meter.yaml`
- `MERIX SUBMISSION/Boundary_Restart/boundary_restart/models.py`
- `MERIX SUBMISSION/Boundary_Restart/boundary_restart/table_io.py`
- `MERIX SUBMISSION/Boundary_Restart/boundary_restart/metrics.py`
- `MERIX SUBMISSION/Boundary_Restart/boundary_restart/features.py`
- `MERIX SUBMISSION/Boundary_Restart/boundary_restart/config.py`
- `MERIX SUBMISSION/Boundary_Restart/boundary_restart/__init__.py`
- `MERIX SUBMISSION/MIREX_Model/build_mazurka_beat_npz_performer_levels.py`
- `MERIX SUBMISSION/MIREX_Model/build_mazurkabl_midibert_rich_beat_features.py`
- `MERIX SUBMISSION/MIREX_Model/build_asap30_tempo_boundary_labels.py`
- `MERIX SUBMISSION/MIREX_Model/build_asap30_score_features.py`
- `MERIX SUBMISSION/MIREX_Model/build_asap30_midibert_rich_beat_features.py`
- `MERIX SUBMISSION/MIREX_Model/run_mazurkabl_l2plus_weighted_target_experiment.py`
- `MERIX SUBMISSION/MIREX_Model/run_mazurkabl_l2plus_rich_midibert_mlp_cnn.py`
- `MERIX SUBMISSION/MIREX_Model/run_mazurkabl_l2plus_branchwise_bottleneck_cnn.py`
- `MERIX SUBMISSION/MIREX_Model/run_mazurkabl_l2plus_sqrtmass_branchwise_cnn.py`
- `MERIX SUBMISSION/MIREX_Model/run_asap30_l2plus_sqrtmass_baseline_branchwise.py`
- `MERIX SUBMISSION/MIREX_Model/run_combined_mazurkabl_asap40_l2plus_sqrtmass.py`
- `MERIX SUBMISSION/MIREX_Model/run_combined_mazurkabl_asap40_per_piece_eval.py`
- `MERIX SUBMISSION/MIREX_Model/run_combined_mazurkabl_asap40_l23_l456_targets.py`
- `MERIX SUBMISSION/MIREX_Model/run_combined_mazurkabl_asap40_l456_per_piece_density_sweep.py`
- `MERIX SUBMISSION/MIREX_Model/beat_data_mazurka_performer_levels_recomputed_sw3_20260524`
- `MERIX SUBMISSION/MIREX_Model/mazurkabl_midibert_rich_beat_features_meter34`
- `MERIX SUBMISSION/MIREX_Model/asap40_tempo_boundary_labels`
- `MERIX SUBMISSION/MIREX_Model/asap40_score_note_feats`
- `MERIX SUBMISSION/MIREX_Model/asap40_midibert_rich_beat_features`
- `MazurkaBL-master`
- `MERIX SUBMISSION/MIREX_Model/mazurkabl_l2plus_sqrtmass_branchwise_cnn_event0p01_densityfixed_2bars_mindist1`
- `MERIX SUBMISSION/MIREX_Model/asap30_l2plus_sqrtmass_baseline_branchwise`
- `MERIX SUBMISSION/MIREX_Model/asap40_l2plus_sqrtmass_baseline_branchwise`
- `MERIX SUBMISSION/MIREX_Model/combined_mazurkabl_asap40_l2plus_sqrtmass_baseline_branchwise`
- `MERIX SUBMISSION/MIREX_Model/combined_mazurkabl_asap40_l23_l456_targets`
- `MERIX SUBMISSION/MIREX_Model/combined_mazurkabl_asap40_l23_l456_targets_density12`
- `MERIX SUBMISSION/MIREX_Model/combined_mazurkabl_asap40_l23_l456_targets_density24`
- `MERIX SUBMISSION/MIREX_Model/combined_mazurkabl_asap40_l456_per_piece_density_sweep`
- `MERIX SUBMISSION/MIREX_Model/combined_mazurkabl_asap40_performer_boundary_density_stats`

## Notes

- `asap30_*` outputs are historical comparison files; the current combined result uses ASAP40.
- ATEPP-related files are not included in the current keep list because the current report has moved to MazurkaBL + ASAP40.
- `MazurkaBL-master/` is listed because MazurkaBL labels/features can be regenerated from it.
- The exact per-file listing is in the compressed CSV because data directories contain thousands of `.npz` files.
