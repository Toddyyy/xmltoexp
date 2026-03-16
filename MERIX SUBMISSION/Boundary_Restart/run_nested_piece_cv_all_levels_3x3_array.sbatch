#!/bin/bash
#SBATCH --job-name=br_nested_alllvl
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err
#SBATCH --time=96:00:00
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
N_SPLITS="${N_SPLITS:-5}"
MAX_INNER_FOLDS="${MAX_INNER_FOLDS:-}"
MODEL="${MODEL:-tcn}"
SELECTION_METRIC="${SELECTION_METRIC:-union_recall}"
LOSS_TYPE="${LOSS_TYPE:-bce}"
REST_SPAN_LABEL_MODE="${REST_SPAN_LABEL_MODE:-none}"
REST_SPAN_TOLERANCE_NEGATIVE_WEIGHT="${REST_SPAN_TOLERANCE_NEGATIVE_WEIGHT:-1.0}"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-}"
EPOCHS="${EPOCHS:-}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-}"
RUN_OUTER_FIT="${RUN_OUTER_FIT:-1}"
REUSE_EXISTING="${REUSE_EXISTING:-1}"
CANDIDATE_JSON="${CANDIDATE_JSON:-configs/nested_candidates_tcn_085_095_generic.json}"

LEVEL_TARGETS=(level1_boundary level2_boundary level3_boundary level4_boundary level56_boundary)
LEVEL_SLUGS=(level1 level2 level3 level4 level56)

echo "Running nested all-levels on seed=$SEED from $PROJECT_DIR"

for idx in "${!LEVEL_TARGETS[@]}"; do
  DETECTOR_TARGET="${LEVEL_TARGETS[$idx]}"
  LEVEL_SLUG="${LEVEL_SLUGS[$idx]}"
  OUTPUT_DIR="reports/nested_piece_cv/M06_outer_${LEVEL_SLUG}_seed${SEED}"

  CMD=(
    python3 run_nested_piece_cv.py
    --config "$CONFIG"
    --outer_heldout_piece ${OUTER_HELDOUT}
    --inner_mode "$INNER_MODE"
    --n_splits "$N_SPLITS"
    --candidate_json "$CANDIDATE_JSON"
    --model "$MODEL"
    --detector_target "$DETECTOR_TARGET"
    --selection_metric "$SELECTION_METRIC"
    --loss_type "$LOSS_TYPE"
    --rest_span_label_mode "$REST_SPAN_LABEL_MODE"
    --rest_span_tolerance_negative_weight "$REST_SPAN_TOLERANCE_NEGATIVE_WEIGHT"
    --device "$DEVICE"
    --seed "$SEED"
    --output_dir "$OUTPUT_DIR"
  )

  if [ -n "$MAX_INNER_FOLDS" ]; then
    CMD+=(--max_inner_folds "$MAX_INNER_FOLDS")
  fi
  if [ -n "$BATCH_SIZE" ]; then
    CMD+=(--batch_size "$BATCH_SIZE")
  fi
  if [ -n "$EPOCHS" ]; then
    CMD+=(--epochs "$EPOCHS")
  fi
  if [ -n "$EARLY_STOP_PATIENCE" ]; then
    CMD+=(--early_stop_patience "$EARLY_STOP_PATIENCE")
  fi
  if [ "$RUN_OUTER_FIT" = "1" ]; then
    CMD+=(--run_outer_fit)
  fi
  if [ "$REUSE_EXISTING" = "1" ]; then
    CMD+=(--reuse_existing)
  fi

  echo
  echo "=== SEED=$SEED TARGET=$DETECTOR_TARGET OUTPUT=$OUTPUT_DIR ==="
  printf 'Command: %q ' "${CMD[@]}"
  echo
  "${CMD[@]}"
done
