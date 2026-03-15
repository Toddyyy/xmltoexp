#!/bin/bash
#SBATCH --job-name=br_nested_cv
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --time=72:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$PROJECT_DIR"

VENV_PATH="${VENV_PATH:-$PROJECT_DIR/../MIREX_Model/.venv}"
if [ ! -f "$VENV_PATH/bin/activate" ]; then
  echo "Missing virtualenv activate script: $VENV_PATH/bin/activate" >&2
  exit 1
fi
source "$VENV_PATH/bin/activate"

CONFIG="${CONFIG:-configs/salience_grouped3_hi8_score_only_xml_curated.yaml}"
OUTER_HELDOUT="${OUTER_HELDOUT:-M06-1 M06-2 M06-3}"
INNER_MODE="${INNER_MODE:-leave_one}"
N_SPLITS="${N_SPLITS:-5}"
MAX_INNER_FOLDS="${MAX_INNER_FOLDS:-}"
MODEL="${MODEL:-tcn}"
DETECTOR_TARGET="${DETECTOR_TARGET:-level56_boundary}"
SELECTION_METRIC="${SELECTION_METRIC:-union_recall}"
MIN_PRECISION="${MIN_PRECISION:-0.85}"
LOSS_TYPE="${LOSS_TYPE:-bce}"
REST_SPAN_LABEL_MODE="${REST_SPAN_LABEL_MODE:-none}"
REST_SPAN_TOLERANCE_NEGATIVE_WEIGHT="${REST_SPAN_TOLERANCE_NEGATIVE_WEIGHT:-1.0}"
DEVICE="${DEVICE:-cpu}"
BATCH_SIZE="${BATCH_SIZE:-}"
EPOCHS="${EPOCHS:-}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-}"
SEED="${SEED:-42}"
CANDIDATE_JSON="${CANDIDATE_JSON:-}"
RUN_OUTER_FIT="${RUN_OUTER_FIT:-0}"
REUSE_EXISTING="${REUSE_EXISTING:-1}"
OUTPUT_DIR="${OUTPUT_DIR:-reports/nested_piece_cv/M06_outer}"

CMD=(
  python3 run_nested_piece_cv.py
  --config "$CONFIG"
  --outer_heldout_piece ${OUTER_HELDOUT}
  --inner_mode "$INNER_MODE"
  --n_splits "$N_SPLITS"
  --model "$MODEL"
  --detector_target "$DETECTOR_TARGET"
  --selection_metric "$SELECTION_METRIC"
  --min_precision "$MIN_PRECISION"
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
if [ -n "$CANDIDATE_JSON" ]; then
  CMD+=(--candidate_json "$CANDIDATE_JSON")
fi
if [ "$RUN_OUTER_FIT" = "1" ]; then
  CMD+=(--run_outer_fit)
fi
if [ "$REUSE_EXISTING" = "1" ]; then
  CMD+=(--reuse_existing)
fi

echo "Running nested CV from $PROJECT_DIR"
printf 'Command: %q ' "${CMD[@]}"
echo
"${CMD[@]}"
