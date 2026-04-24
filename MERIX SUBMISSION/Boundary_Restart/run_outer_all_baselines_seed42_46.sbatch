#!/bin/bash
#SBATCH --job-name=br_outer_all
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --time=168:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu3090
#SBATCH --qos=1gpu
#SBATCH --gres=gpu:1

set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$PROJECT_DIR"

VENV_PATH="${VENV_PATH:-$PROJECT_DIR/../MIREX_Model/.venv}"
if [ ! -f "$VENV_PATH/bin/activate" ]; then
  echo "Missing virtualenv activate script: $VENV_PATH/bin/activate" >&2
  exit 1
fi
source "$VENV_PATH/bin/activate"

SEEDS=(${SEEDS:-42 43 44 45 46})
SEED_SLUG="$(printf '%s' "${SEEDS[@]}")"

CONFIG="${CONFIG:-configs/salience_grouped3_hi8_score_only_xml_curated.yaml}"
OUTER_HELDOUT=(${OUTER_HELDOUT:-M06-1 M06-2 M06-3})
MAX_INNER_FOLDS="${MAX_INNER_FOLDS:-}"

REUSE_EXISTING="${REUSE_EXISTING:-1}"

RUN_TCN="${RUN_TCN:-1}"
RUN_RETAINED_BASELINES="${RUN_RETAINED_BASELINES:-1}"
RUN_NEW_BASELINES="${RUN_NEW_BASELINES:-1}"
RUN_BILSTM="${RUN_BILSTM:-1}"
RUN_SUMMARY="${RUN_SUMMARY:-1}"

DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-}"
EPOCHS="${EPOCHS:-60}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-10}"
MIN_TRAIN_FREQUENCY_TARGET="${MIN_TRAIN_FREQUENCY_TARGET:-0.05}"
CUMULATIVE_MERGE_TOLERANCE="${CUMULATIVE_MERGE_TOLERANCE:-2}"
WEIGHTS_JSON_DEFAULT='{"level56":1.0,"level4":0.64,"level3":0.46,"level2":0.28,"level1":0.16}'
CUMULATIVE_COMPONENT_WEIGHTS_JSON="${CUMULATIVE_COMPONENT_WEIGHTS_JSON:-$WEIGHTS_JSON_DEFAULT}"

BILSTM_DEVICE="${BILSTM_DEVICE:-$DEVICE}"
BILSTM_BATCH_SIZE="${BILSTM_BATCH_SIZE:-128}"
BILSTM_EPOCHS="${BILSTM_EPOCHS:-60}"
BILSTM_EARLY_STOP_PATIENCE="${BILSTM_EARLY_STOP_PATIENCE:-10}"

GROUPER_LEVELWISE_CSV="${GROUPER_LEVELWISE_CSV:-}"
SUMMARY_OUTPUT_PREFIX="${SUMMARY_OUTPUT_PREFIX:-paper_outer_all_baseline_summary_seed${SEED_SLUG}}"

TCN_TARGETS=(
  "level1plus_boundary"
  "level2plus_boundary"
  "level3plus_boundary"
  "level4plus_boundary"
  "level56_boundary"
)
TCN_MIN_PRECISIONS=("0.85" "0.85" "0.85" "0.85" "0.80")

echo "PROJECT_DIR=$PROJECT_DIR"
echo "VENV_PATH=$VENV_PATH"
echo "CONFIG=$CONFIG"
echo "OUTER_HELDOUT=${OUTER_HELDOUT[*]}"
echo "SEEDS=${SEEDS[*]}"
echo "REUSE_EXISTING=$REUSE_EXISTING"
echo "RUN_TCN=$RUN_TCN"
echo "RUN_RETAINED_BASELINES=$RUN_RETAINED_BASELINES"
echo "RUN_NEW_BASELINES=$RUN_NEW_BASELINES"
echo "RUN_BILSTM=$RUN_BILSTM"
echo "RUN_SUMMARY=$RUN_SUMMARY"

run_tcn_seed() {
  local seed="$1"
  local idx=""
  for idx in "${!TCN_TARGETS[@]}"; do
    local detector_target="${TCN_TARGETS[$idx]}"
    local min_precision="${TCN_MIN_PRECISIONS[$idx]}"
    local output_dir="reports/nested_piece_cv/weighted_topdown_merge56_${detector_target}_seed${seed}"
    local -a cmd=(
      python run_nested_piece_cv.py
      --config "$CONFIG"
      --outer_heldout_piece "${OUTER_HELDOUT[@]}"
      --inner_mode leave_one
      --model tcn
      --detector_target "$detector_target"
      --selection_metric weighted_recall
      --precision_metric union_precision
      --min_precision "$min_precision"
      --loss_type bce
      --min_train_frequency_target "$MIN_TRAIN_FREQUENCY_TARGET"
      --cumulative_merge_tolerance "$CUMULATIVE_MERGE_TOLERANCE"
      --cumulative_component_weights_json "$CUMULATIVE_COMPONENT_WEIGHTS_JSON"
      --device "$DEVICE"
      --epochs "$EPOCHS"
      --early_stop_patience "$EARLY_STOP_PATIENCE"
      --seed "$seed"
      --skip_stage_grading
      --run_outer_fit
      --output_dir "$output_dir"
    )
    if [ -n "$MAX_INNER_FOLDS" ]; then
      cmd+=(--max_inner_folds "$MAX_INNER_FOLDS")
    fi
    if [ -n "$BATCH_SIZE" ]; then
      cmd+=(--batch_size "$BATCH_SIZE")
    fi
    if [ "$REUSE_EXISTING" = "1" ]; then
      cmd+=(--reuse_existing)
    fi
    printf 'Command: %q ' "${cmd[@]}"
    echo
    "${cmd[@]}"
  done
}

run_score_baseline() {
  local seed="$1"
  local output_dir="$2"
  local target_design="$3"
  local feature_family="$4"
  shift 4
  local -a models=("$@")
  local -a cmd=(
    python run_outer_score_baselines.py
    --config "$CONFIG"
    --models "${models[@]}"
    --outer_heldout_piece "${OUTER_HELDOUT[@]}"
    --target_design "$target_design"
    --feature_family "$feature_family"
    --seed "$seed"
    --output_dir "$output_dir"
  )
  if [ -n "$MAX_INNER_FOLDS" ]; then
    cmd+=(--max_inner_folds "$MAX_INNER_FOLDS")
  fi
  if [ "$REUSE_EXISTING" = "1" ]; then
    cmd+=(--reuse_existing)
  fi
  printf 'Command: %q ' "${cmd[@]}"
  echo
  "${cmd[@]}"
}

run_bilstm_seed() {
  local seed="$1"
  local output_dir="reports/paper_outer_baselines_weighted_topdown_all_seed${seed}_bilstm"
  local -a cmd=(
    python run_outer_bilstm_baseline.py
    --config "$CONFIG"
    --outer_heldout_piece "${OUTER_HELDOUT[@]}"
    --seed "$seed"
    --device "$BILSTM_DEVICE"
    --batch_size "$BILSTM_BATCH_SIZE"
    --epochs "$BILSTM_EPOCHS"
    --early_stop_patience "$BILSTM_EARLY_STOP_PATIENCE"
    --output_dir "$output_dir"
  )
  if [ -n "$MAX_INNER_FOLDS" ]; then
    cmd+=(--max_inner_folds "$MAX_INNER_FOLDS")
  fi
  if [ "$REUSE_EXISTING" = "1" ]; then
    cmd+=(--reuse_existing)
  fi
  printf 'Command: %q ' "${cmd[@]}"
  echo
  "${cmd[@]}"
}

for SEED in "${SEEDS[@]}"; do
  echo
  echo "=== Seed ${SEED} ==="

  if [ "$RUN_TCN" = "1" ]; then
    run_tcn_seed "$SEED"
  fi

  if [ "$RUN_RETAINED_BASELINES" = "1" ]; then
    run_score_baseline \
      "$SEED" \
      "reports/paper_outer_baselines_weighted_topdown_all_seed${SEED}_logreg" \
      weighted_topdown \
      all \
      logreg

    run_score_baseline \
      "$SEED" \
      "reports/paper_outer_baselines_simple_union_all_seed${SEED}_logreg" \
      simple_union \
      all \
      logreg

    run_score_baseline \
      "$SEED" \
      "reports/paper_outer_baselines_weighted_topdown_note_only_seed${SEED}_logreg" \
      weighted_topdown \
      note_only \
      logreg

    run_score_baseline \
      "$SEED" \
      "reports/paper_outer_baselines_weighted_topdown_xml_only_seed${SEED}_logreg" \
      weighted_topdown \
      xml_only \
      logreg

    run_score_baseline \
      "$SEED" \
      "reports/paper_outer_baselines_weighted_topdown_all_seed${SEED}_lbdm_only" \
      weighted_topdown \
      all \
      lbdm
  fi

  if [ "$RUN_NEW_BASELINES" = "1" ]; then
    run_score_baseline \
      "$SEED" \
      "reports/paper_outer_missing_baselines_seed${SEED}" \
      weighted_topdown \
      all \
      all_boundary periodic downbeat logreg_window7
  fi

  if [ "$RUN_BILSTM" = "1" ]; then
    run_bilstm_seed "$SEED"
  fi
done

if [ "$RUN_SUMMARY" = "1" ]; then
  if [ -z "$GROUPER_LEVELWISE_CSV" ]; then
    echo "GROUPER_LEVELWISE_CSV not set; Grouper rows will not be included."
  fi
  SUMMARY_CMD=(
    python summarize_outer_all_baselines.py
    --seeds "${SEEDS[@]}"
    --report_root reports
    --output_prefix "$SUMMARY_OUTPUT_PREFIX"
  )
  if [ -n "$GROUPER_LEVELWISE_CSV" ]; then
    SUMMARY_CMD+=(--grouper_levelwise_csv "$GROUPER_LEVELWISE_CSV")
  fi
  echo
  printf 'Command: %q ' "${SUMMARY_CMD[@]}"
  echo
  "${SUMMARY_CMD[@]}"
fi
