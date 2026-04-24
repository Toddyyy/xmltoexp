#!/bin/bash
#
# Local experiment runner - runs experiments one-by-one with progress tracking
# Usage: ./run_experiments_locally.sh [options]
#
# Options:
#   --seed SEED            Run only specific seed (default: 42 43 44 45 46)
#   --skip-tcn            Skip TCN experiments
#   --skip-baselines      Skip baseline experiments
#   --skip-bilstm         Skip BiLSTM experiments
#   --skip-summary        Skip summary generation
#   --resume-from SEED    Resume from specific seed

set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$PROJECT_DIR"

VENV_PATH="${VENV_PATH:-$PROJECT_DIR/../MIREX_Model/.venv}"
if [ ! -f "$VENV_PATH/bin/activate" ]; then
  echo "Missing virtualenv activate script: $VENV_PATH/bin/activate" >&2
  exit 1
fi
source "$VENV_PATH/bin/activate"

# Parse command-line arguments
SEEDS=(${SEEDS:-42 43 44 45 46})
RUN_TCN="${RUN_TCN:-1}"
RUN_BASELINES="${RUN_BASELINES:-1}"
RUN_BILSTM="${RUN_BILSTM:-1}"
RUN_SUMMARY="${RUN_SUMMARY:-1}"
RESUME_FROM=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --seed)
      SEEDS=("$2")
      shift 2
      ;;
    --skip-tcn)
      RUN_TCN=0
      shift
      ;;
    --skip-baselines)
      RUN_BASELINES=0
      shift
      ;;
    --skip-bilstm)
      RUN_BILSTM=0
      shift
      ;;
    --skip-summary)
      RUN_SUMMARY=0
      shift
      ;;
    --resume-from)
      RESUME_FROM="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

CONFIG="${CONFIG:-configs/salience_grouped3_hi8_score_only_xml_curated.yaml}"
OUTER_HELDOUT=(${OUTER_HELDOUT:-M06-1 M06-2 M06-3})
MAX_INNER_FOLDS="${MAX_INNER_FOLDS:-}"
REUSE_EXISTING="${REUSE_EXISTING:-1}"
DEVICE="${DEVICE:-cpu}"
BATCH_SIZE="${BATCH_SIZE:-}"
EPOCHS="${EPOCHS:-60}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-10}"
MIN_TRAIN_FREQUENCY_TARGET="${MIN_TRAIN_FREQUENCY_TARGET:-0.05}"
CUMULATIVE_MERGE_TOLERANCE="${CUMULATIVE_MERGE_TOLERANCE:-2}"
WEIGHTS_JSON_DEFAULT='{"level56":1.0,"level4":0.64,"level3":0.46,"level2":0.28,"level1":0.16}'
CUMULATIVE_COMPONENT_WEIGHTS_JSON="${CUMULATIVE_COMPONENT_WEIGHTS_JSON:-$WEIGHTS_JSON_DEFAULT}"

BILSTM_DEVICE="${BILSTM_DEVICE:-$DEVICE}"
# For CPU: reduce batch size and epochs for reasonable runtime
if [ "$DEVICE" = "cpu" ]; then
  BILSTM_BATCH_SIZE="${BILSTM_BATCH_SIZE:-32}"    # Reduced from 128
  BILSTM_EPOCHS="${BILSTM_EPOCHS:-30}"            # Reduced from 60
else
  BILSTM_BATCH_SIZE="${BILSTM_BATCH_SIZE:-128}"
  BILSTM_EPOCHS="${BILSTM_EPOCHS:-60}"
fi
BILSTM_EARLY_STOP_PATIENCE="${BILSTM_EARLY_STOP_PATIENCE:-10}"

# Log file for tracking
LOG_FILE="experiment_run.log"
COMPLETED_FILE=".experiments_completed"

# Initialize completed tracking if not exists
if [ ! -f "$COMPLETED_FILE" ]; then
  echo "=== Starting new experiment run ===" > "$COMPLETED_FILE"
fi

echo "============================================"
echo "Local Experiment Runner"
echo "============================================"
echo "PROJECT_DIR=$PROJECT_DIR"
echo "CONFIG=$CONFIG"
echo "DEVICE=$DEVICE"
echo "SEEDS=${SEEDS[*]}"
echo "RUN_TCN=$RUN_TCN"
echo "RUN_BASELINES=$RUN_BASELINES"
echo "RUN_BILSTM=$RUN_BILSTM"
echo "RUN_SUMMARY=$RUN_SUMMARY"
if [ -n "$RESUME_FROM" ]; then
  echo "RESUME_FROM=$RESUME_FROM"
fi
echo "============================================"

TCN_TARGETS=(
  "level1plus_boundary"
  "level2plus_boundary"
  "level3plus_boundary"
  "level4plus_boundary"
  "level56_boundary"
)
TCN_MIN_PRECISIONS=("0.85" "0.85" "0.85" "0.85" "0.80")

run_tcn_seed() {
  local seed="$1"
  echo ""
  echo ">>> Running TCN for seed $seed"
  for idx in "${!TCN_TARGETS[@]}"; do
    local detector_target="${TCN_TARGETS[$idx]}"
    local min_precision="${TCN_MIN_PRECISIONS[$idx]}"
    local output_dir="reports/nested_piece_cv/weighted_topdown_merge56_${detector_target}_seed${seed}"
    echo "  ├─ Target: $detector_target (min_precision=$min_precision)"

    python run_nested_piece_cv.py \
      --config "$CONFIG" \
      --outer_heldout_piece "${OUTER_HELDOUT[@]}" \
      --inner_mode leave_one \
      --model tcn \
      --detector_target "$detector_target" \
      --selection_metric weighted_recall \
      --precision_metric union_precision \
      --min_precision "$min_precision" \
      --loss_type bce \
      --min_train_frequency_target "$MIN_TRAIN_FREQUENCY_TARGET" \
      --cumulative_merge_tolerance "$CUMULATIVE_MERGE_TOLERANCE" \
      --cumulative_component_weights_json "$CUMULATIVE_COMPONENT_WEIGHTS_JSON" \
      --device "$DEVICE" \
      --epochs "$EPOCHS" \
      --early_stop_patience "$EARLY_STOP_PATIENCE" \
      --seed "$seed" \
      --skip_stage_grading \
      --run_outer_fit \
      --output_dir "$output_dir" \
      ${MAX_INNER_FOLDS:+--max_inner_folds "$MAX_INNER_FOLDS"} \
      ${BATCH_SIZE:+--batch_size "$BATCH_SIZE"} \
      ${REUSE_EXISTING:+--reuse_existing}
  done
}

run_baselines_seed() {
  local seed="$1"
  echo ""
  echo ">>> Running Baselines for seed $seed"

  echo "  ├─ LogReg + weighted_topdown (all features)"
  python run_outer_score_baselines.py \
    --config "$CONFIG" \
    --models logreg \
    --outer_heldout_piece "${OUTER_HELDOUT[@]}" \
    --target_design weighted_topdown \
    --feature_family all \
    --seed "$seed" \
    --output_dir "reports/paper_outer_baselines_weighted_topdown_all_seed${seed}_logreg" \
    ${MAX_INNER_FOLDS:+--max_inner_folds "$MAX_INNER_FOLDS"} \
    ${REUSE_EXISTING:+--reuse_existing}

  echo "  ├─ LogReg + simple_union (all features)"
  python run_outer_score_baselines.py \
    --config "$CONFIG" \
    --models logreg \
    --outer_heldout_piece "${OUTER_HELDOUT[@]}" \
    --target_design simple_union \
    --feature_family all \
    --seed "$seed" \
    --output_dir "reports/paper_outer_baselines_simple_union_all_seed${seed}_logreg" \
    ${MAX_INNER_FOLDS:+--max_inner_folds "$MAX_INNER_FOLDS"} \
    ${REUSE_EXISTING:+--reuse_existing}

  echo "  ├─ LogReg + weighted_topdown (note only)"
  python run_outer_score_baselines.py \
    --config "$CONFIG" \
    --models logreg \
    --outer_heldout_piece "${OUTER_HELDOUT[@]}" \
    --target_design weighted_topdown \
    --feature_family note_only \
    --seed "$seed" \
    --output_dir "reports/paper_outer_baselines_weighted_topdown_note_only_seed${seed}_logreg" \
    ${MAX_INNER_FOLDS:+--max_inner_folds "$MAX_INNER_FOLDS"} \
    ${REUSE_EXISTING:+--reuse_existing}

  echo "  ├─ LogReg + weighted_topdown (xml only)"
  python run_outer_score_baselines.py \
    --config "$CONFIG" \
    --models logreg \
    --outer_heldout_piece "${OUTER_HELDOUT[@]}" \
    --target_design weighted_topdown \
    --feature_family xml_only \
    --seed "$seed" \
    --output_dir "reports/paper_outer_baselines_weighted_topdown_xml_only_seed${seed}_logreg" \
    ${MAX_INNER_FOLDS:+--max_inner_folds "$MAX_INNER_FOLDS"} \
    ${REUSE_EXISTING:+--reuse_existing}

  echo "  ├─ LBDM + weighted_topdown (all features)"
  python run_outer_score_baselines.py \
    --config "$CONFIG" \
    --models lbdm \
    --outer_heldout_piece "${OUTER_HELDOUT[@]}" \
    --target_design weighted_topdown \
    --feature_family all \
    --seed "$seed" \
    --output_dir "reports/paper_outer_baselines_weighted_topdown_all_seed${seed}_lbdm_only" \
    ${MAX_INNER_FOLDS:+--max_inner_folds "$MAX_INNER_FOLDS"} \
    ${REUSE_EXISTING:+--reuse_existing}

  echo "  └─ Missing baselines (multiple models)"
  python run_outer_score_baselines.py \
    --config "$CONFIG" \
    --models all_boundary periodic downbeat logreg_window7 \
    --outer_heldout_piece "${OUTER_HELDOUT[@]}" \
    --target_design weighted_topdown \
    --feature_family all \
    --seed "$seed" \
    --output_dir "reports/paper_outer_missing_baselines_seed${seed}" \
    ${MAX_INNER_FOLDS:+--max_inner_folds "$MAX_INNER_FOLDS"} \
    ${REUSE_EXISTING:+--reuse_existing}
}

run_bilstm_seed() {
  local seed="$1"
  echo ""
  echo ">>> Running BiLSTM for seed $seed"
  python run_outer_bilstm_baseline.py \
    --config "$CONFIG" \
    --outer_heldout_piece "${OUTER_HELDOUT[@]}" \
    --seed "$seed" \
    --device "$BILSTM_DEVICE" \
    --batch_size "$BILSTM_BATCH_SIZE" \
    --epochs "$BILSTM_EPOCHS" \
    --early_stop_patience "$BILSTM_EARLY_STOP_PATIENCE" \
    --output_dir "reports/paper_outer_baselines_weighted_topdown_all_seed${seed}_bilstm" \
    ${MAX_INNER_FOLDS:+--max_inner_folds "$MAX_INNER_FOLDS"} \
    ${REUSE_EXISTING:+--reuse_existing}
}

# Check if we should skip to a specific seed
SHOULD_RUN=1
if [ -n "$RESUME_FROM" ]; then
  SHOULD_RUN=0
fi

for SEED in "${SEEDS[@]}"; do
  # Check if we should resume from this seed
  if [ "$SHOULD_RUN" = "0" ] && [ "$SEED" = "$RESUME_FROM" ]; then
    SHOULD_RUN=1
  fi

  if [ "$SHOULD_RUN" = "0" ]; then
    echo "⊘ Skipping seed $SEED (waiting for resume point)"
    continue
  fi

  echo ""
  echo "╔══════════════════════════════════════╗"
  echo "║  Processing Seed: $SEED"
  echo "╚══════════════════════════════════════╝"

  if [ "$RUN_TCN" = "1" ]; then
    run_tcn_seed "$SEED" 2>&1 | tee -a "$LOG_FILE"
  fi

  if [ "$RUN_BASELINES" = "1" ]; then
    run_baselines_seed "$SEED" 2>&1 | tee -a "$LOG_FILE"
  fi

  if [ "$RUN_BILSTM" = "1" ]; then
    run_bilstm_seed "$SEED" 2>&1 | tee -a "$LOG_FILE"
  fi

  echo "✓ Seed $SEED completed" | tee -a "$COMPLETED_FILE"
  echo "" | tee -a "$LOG_FILE"
done

if [ "$RUN_SUMMARY" = "1" ]; then
  echo ""
  echo "╔══════════════════════════════════════╗"
  echo "║  Generating Summary"
  echo "╚══════════════════════════════════════╝"
  SEED_SLUG="$(printf '%s' "${SEEDS[@]}")"
  python summarize_outer_all_baselines.py \
    --seeds "${SEEDS[@]}" \
    --report_root reports \
    --output_prefix "paper_outer_all_baseline_summary_seed${SEED_SLUG}" \
    2>&1 | tee -a "$LOG_FILE"
fi

echo ""
echo "╔══════════════════════════════════════╗"
echo "║  All experiments completed!"
echo "╚══════════════════════════════════════╝"
echo "Logs saved to: $LOG_FILE"
