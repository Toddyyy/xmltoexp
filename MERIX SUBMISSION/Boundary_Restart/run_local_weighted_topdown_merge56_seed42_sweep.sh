#!/bin/bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

if [ "${SKIP_VENV:-0}" != "1" ]; then
  VENV_PATH="${VENV_PATH:-$PROJECT_DIR/../MIREX_Model/.venv}"
  if [ ! -f "$VENV_PATH/bin/activate" ]; then
    echo "Missing virtualenv activate script: $VENV_PATH/bin/activate" >&2
    exit 1
  fi
  source "$VENV_PATH/bin/activate"
fi
PYTHON_EXEC="${PYTHON_EXEC:-python3}"

CONFIG="${CONFIG:-configs/salience_grouped3_hi8_score_only_xml_curated.yaml}"
OUTER_HELDOUT="${OUTER_HELDOUT:-M06-1 M06-2 M06-3}"
INNER_MODE="${INNER_MODE:-leave_one}"
DEVICE="${DEVICE:-mps}"
SEED="${SEED:-42}"
EPOCHS="${EPOCHS:-60}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-10}"
MAX_INNER_FOLDS="${MAX_INNER_FOLDS:-}"
RUN_OUTER_FIT="${RUN_OUTER_FIT:-1}"
REUSE_EXISTING="${REUSE_EXISTING:-1}"

TARGETS=(
  "level1plus_boundary"
  "level2plus_boundary"
  "level3plus_boundary"
  "level4plus_boundary"
  "level56_boundary"
)
MIN_PRECISIONS=("0.85" "0.85" "0.85" "0.85" "0.80")

WEIGHT_NAMES=(
  "baseline"
  "middle"
  "aggressive"
)
WEIGHT_JSONS=(
  '{"level56":1.0,"level4":0.64,"level3":0.46,"level2":0.28,"level1":0.16}'
  '{"level56":1.0,"level4":0.75,"level3":0.55,"level2":0.35,"level1":0.20}'
  '{"level56":1.0,"level4":0.50,"level3":0.30,"level2":0.15,"level1":0.08}'
)

if [ -n "${WEIGHT_JSON:-}" ]; then
  WEIGHT_NAMES=("custom")
  WEIGHT_JSONS=("$WEIGHT_JSON")
fi

for weight_idx in "${!WEIGHT_NAMES[@]}"; do
  WEIGHT_NAME="${WEIGHT_NAMES[$weight_idx]}"
  WEIGHT_JSON_VALUE="${WEIGHT_JSONS[$weight_idx]}"
  echo
  echo "=== Running weight preset: $WEIGHT_NAME ==="
  echo "WEIGHTS=$WEIGHT_JSON_VALUE"

  for idx in "${!TARGETS[@]}"; do
    DETECTOR_TARGET="${TARGETS[$idx]}"
    MIN_PRECISION="${MIN_PRECISIONS[$idx]}"
    TARGET_SLUG="${DETECTOR_TARGET//[^A-Za-z0-9_]/_}"
    OUTPUT_DIR="reports/nested_piece_cv/weighted_topdown_merge56_${WEIGHT_NAME}_${TARGET_SLUG}_seed${SEED}"

    CMD=(
      "$PYTHON_EXEC" run_nested_piece_cv.py
      --config "$CONFIG"
      --outer_heldout_piece ${OUTER_HELDOUT}
      --inner_mode "$INNER_MODE"
      --model tcn
      --detector_target "$DETECTOR_TARGET"
      --selection_metric weighted_recall
      --precision_metric union_precision
      --min_precision "$MIN_PRECISION"
      --loss_type bce
      --min_train_frequency_target 0.05
      --cumulative_merge_tolerance 2
      --cumulative_component_weights_json "$WEIGHT_JSON_VALUE"
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
    if [ "$RUN_OUTER_FIT" = "1" ]; then
      CMD+=(--run_outer_fit)
    fi
    if [ "$REUSE_EXISTING" = "1" ]; then
      CMD+=(--reuse_existing)
    fi

    echo
    echo "Running weight=$WEIGHT_NAME target=$DETECTOR_TARGET min_precision=$MIN_PRECISION"
    printf 'Command: %q ' "${CMD[@]}"
    echo
    "${CMD[@]}"
  done
done
