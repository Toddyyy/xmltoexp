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
DEVICE="${DEVICE:-mps}"
SEED="${SEED:-42}"
EPOCHS="${EPOCHS:-60}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-10}"
HELDOUT_PIECES=(${HELDOUT_PIECES:-M06-1 M17-1 M30-1})

TARGETS=(
  "level1plus_boundary"
  "level2plus_boundary"
  "level3plus_boundary"
  "level4plus_boundary"
  "level56_boundary"
)
MIN_PRECISIONS=("0.85" "0.85" "0.85" "0.85" "0.80")

WEIGHT_NAMES=("baseline" "middle" "aggressive")
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
  echo "=== Weight preset: $WEIGHT_NAME ==="
  echo "WEIGHTS=$WEIGHT_JSON_VALUE"

  for heldout in "${HELDOUT_PIECES[@]}"; do
    for idx in "${!TARGETS[@]}"; do
      DETECTOR_TARGET="${TARGETS[$idx]}"
      MIN_PRECISION="${MIN_PRECISIONS[$idx]}"
      TARGET_SLUG="${DETECTOR_TARGET//[^A-Za-z0-9_]/_}"
      OUTPUT_DIR="outputs/local_runs/weight_benchmark_merge56/${WEIGHT_NAME}/${heldout}/${TARGET_SLUG}_seed${SEED}"

      CMD=(
        "$PYTHON_EXEC" train_piece_union_protocol.py
        --config "$CONFIG"
        --heldout_piece "$heldout"
        --model tcn
        --device "$DEVICE"
        --seed "$SEED"
        --detector_target "$DETECTOR_TARGET"
        --selection_metric weighted_recall
        --precision_metric union_precision
        --min_precision "$MIN_PRECISION"
        --loss_type bce
        --min_train_frequency_target 0.05
        --cumulative_merge_tolerance 2
        --cumulative_component_weights_json "$WEIGHT_JSON_VALUE"
        --epochs "$EPOCHS"
        --early_stop_patience "$EARLY_STOP_PATIENCE"
        --skip_stage_grading
        --output_dir "$OUTPUT_DIR"
      )

      echo
      echo "Running weight=$WEIGHT_NAME heldout=$heldout target=$DETECTOR_TARGET"
      printf 'Command: %q ' "${CMD[@]}"
      echo
      "${CMD[@]}"
    done
  done
done
