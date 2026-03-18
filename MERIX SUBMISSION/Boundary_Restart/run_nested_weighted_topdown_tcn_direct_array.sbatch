#!/bin/bash
#SBATCH --job-name=br_nested_wt_tcn
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err
#SBATCH --time=72:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpua800
#SBATCH --qos=1a800
#SBATCH --gres=gpu:1
#SBATCH --array=0-2

set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$PROJECT_DIR"

VENV_PATH="${VENV_PATH:-$PROJECT_DIR/../MIREX_Model/.venv}"
if [ ! -f "$VENV_PATH/bin/activate" ]; then
  echo "Missing virtualenv activate script: $VENV_PATH/bin/activate" >&2
  exit 1
fi
source "$VENV_PATH/bin/activate"

SEEDS=(42 43 44)
SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"

CONFIG="${CONFIG:-configs/salience_grouped3_hi8_score_only_xml_curated.yaml}"
OUTER_HELDOUT="${OUTER_HELDOUT:-M06-1 M06-2 M06-3}"
INNER_MODE="${INNER_MODE:-leave_one}"
MAX_INNER_FOLDS="${MAX_INNER_FOLDS:-}"

MODEL="${MODEL:-tcn}"
SELECTION_METRIC="${SELECTION_METRIC:-weighted_recall}"
PRECISION_METRIC="${PRECISION_METRIC:-union_precision}"
LOSS_TYPE="${LOSS_TYPE:-bce}"

MIN_TRAIN_FREQUENCY_TARGET="${MIN_TRAIN_FREQUENCY_TARGET:-0.05}"
CUMULATIVE_MERGE_TOLERANCE="${CUMULATIVE_MERGE_TOLERANCE:-2}"
WEIGHTS_JSON_DEFAULT='{"level6":1.0,"level5":0.82,"level4":0.64,"level3":0.46,"level2":0.28,"level1":0.16}'
CUMULATIVE_COMPONENT_WEIGHTS_JSON="${CUMULATIVE_COMPONENT_WEIGHTS_JSON:-$WEIGHTS_JSON_DEFAULT}"

DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-}"
EPOCHS="${EPOCHS:-60}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-10}"
RUN_OUTER_FIT="${RUN_OUTER_FIT:-1}"
REUSE_EXISTING="${REUSE_EXISTING:-1}"

TARGETS=(
  "level1plus_split56_boundary"
  "level2plus_split56_boundary"
  "level3plus_split56_boundary"
  "level4plus_split56_boundary"
  "level5plus_split56_boundary"
  "level6_boundary"
)
MIN_PRECISIONS=("0.85" "0.85" "0.85" "0.85" "0.80" "0.70")

if [ -n "${DETECTOR_TARGET:-}" ]; then
  TARGETS=("${DETECTOR_TARGET}")
  MIN_PRECISIONS=("${MIN_PRECISION:-0.85}")
fi

for idx in "${!TARGETS[@]}"; do
  DETECTOR_TARGET="${TARGETS[$idx]}"
  MIN_PRECISION="${MIN_PRECISIONS[$idx]}"
  TARGET_SLUG="${DETECTOR_TARGET//[^A-Za-z0-9_]/_}"
  OUTPUT_DIR="${OUTPUT_DIR_BASE:-reports/nested_piece_cv/weighted_topdown_${TARGET_SLUG}_seed${SEED}}"

  CMD=(
    python3 run_nested_piece_cv.py
    --config "$CONFIG"
    --outer_heldout_piece ${OUTER_HELDOUT}
    --inner_mode "$INNER_MODE"
    --model "$MODEL"
    --detector_target "$DETECTOR_TARGET"
    --selection_metric "$SELECTION_METRIC"
    --precision_metric "$PRECISION_METRIC"
    --min_precision "$MIN_PRECISION"
    --loss_type "$LOSS_TYPE"
    --min_train_frequency_target "$MIN_TRAIN_FREQUENCY_TARGET"
    --cumulative_merge_tolerance "$CUMULATIVE_MERGE_TOLERANCE"
    --cumulative_component_weights_json "$CUMULATIVE_COMPONENT_WEIGHTS_JSON"
    --device "$DEVICE"
    --epochs "$EPOCHS"
    --early_stop_patience "$EARLY_STOP_PATIENCE"
    --seed "$SEED"
    --skip_stage_grading
    --output_dir "$OUTPUT_DIR"
  )

  if [ -n "$MAX_INNER_FOLDS" ]; then
    CMD+=(--max_inner_folds "$MAX_INNER_FOLDS")
  fi
  if [ -n "$BATCH_SIZE" ]; then
    CMD+=(--batch_size "$BATCH_SIZE")
  fi
  if [ "$RUN_OUTER_FIT" = "1" ]; then
    CMD+=(--run_outer_fit)
  fi
  if [ "$REUSE_EXISTING" = "1" ]; then
    CMD+=(--reuse_existing)
  fi

  echo "Running weighted-topdown nested direct seed=$SEED target=$DETECTOR_TARGET min_precision=$MIN_PRECISION"
  printf 'Command: %q ' "${CMD[@]}"
  echo
  "${CMD[@]}"
done
