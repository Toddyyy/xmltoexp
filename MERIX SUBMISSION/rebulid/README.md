# rebulid

This folder collects the current clean-outer merged `L5+6` reconstruction materials in one place.

Contents:

- `plot_clean_outer_merge56_reconstruction.py`
  - reconstruction script
  - current logic uses per-performer relative tempo `((tempo / median) - 1)` with robust IQR scaling
  - prediction uses the actual clean-outer predicted event counts
- `clean_outer_reconstruction_merge56_seed44_robust_relative_predcount/`
  - latest generated CSV, JSON, PNG, and PDF outputs
  - includes reconstruction summaries and per-piece selected breakpoint lists
- `clean_outer_reconstruction_merge56_seed44_per_performer_zscore_predcount/`
  - per-performer z-score variant
- `merge56_per_performer_zscore_train_avg_topk_seed44/`
  - per-performer z-score + train-average top-K selection
- `clean_outer_reconstruction_merge56_seed44_centered_predcount/`
  - previous mean-centered variant
- `clean_outer_reconstruction_merge56_seed44_standardized_predcount/`
  - earlier per-piece z-score variant

Key result files:

- `clean_outer_reconstruction_merge56_seed44_robust_relative_predcount/reconstruction_summary.csv`
- `clean_outer_reconstruction_merge56_seed44_robust_relative_predcount/reconstruction_metadata.json`
- `clean_outer_reconstruction_merge56_seed44_robust_relative_predcount/clean_outer_merge56_robust_relative_reconstruction_seed44.pdf`
- `clean_outer_reconstruction_merge56_seed44_robust_relative_predcount/clean_outer_merge56_robust_relative_pred_vs_all_true_tempo_seed44.pdf`
