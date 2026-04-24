#!/bin/bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

if [ "${SKIP_VENV:-1}" != "1" ]; then
  VENV_PATH="${VENV_PATH:-$PROJECT_DIR/../MIREX_Model/.venv}"
  if [ ! -f "$VENV_PATH/bin/activate" ]; then
    echo "Missing virtualenv activate script: $VENV_PATH/bin/activate" >&2
    exit 1
  fi
  source "$VENV_PATH/bin/activate"
fi

PYTHON_EXEC="${PYTHON_EXEC:-/opt/miniconda3/bin/python}"
CONFIG="${CONFIG:-configs/salience_grouped3_hi8_score_only_xml_curated.yaml}"
OUTER_HELDOUT="${OUTER_HELDOUT:-M06-1 M06-2 M06-3}"
DEVICE="${DEVICE:-mps}"
SEEDS=(${SEEDS:-42 43 44})
OUTPUT_PREFIX="${OUTPUT_PREFIX:-weighted_topdown_merge56}"

TARGETS=(
  "level1plus_boundary"
  "level2plus_boundary"
  "level3plus_boundary"
  "level4plus_boundary"
  "level56_boundary"
)

for SEED in "${SEEDS[@]}"; do
  echo
  echo "=== Clean outer seed ${SEED} ==="

  for DETECTOR_TARGET in "${TARGETS[@]}"; do
    TARGET_SLUG="${DETECTOR_TARGET//[^A-Za-z0-9_]/_}"
    NESTED_REPORT_DIR="reports/nested_piece_cv/${OUTPUT_PREFIX}_${TARGET_SLUG}_seed${SEED}"
    OUTPUT_DIR="reports/clean_outer_test/${OUTPUT_PREFIX}_${TARGET_SLUG}_seed${SEED}"

    CMD=(
      "$PYTHON_EXEC" run_clean_outer_test_from_nested.py
      --config "$CONFIG"
      --nested_report_dir "$NESTED_REPORT_DIR"
      --outer_heldout_piece ${OUTER_HELDOUT}
      --device "$DEVICE"
      --output_dir "$OUTPUT_DIR"
    )

    echo
    echo "Running clean outer seed=$SEED target=$DETECTOR_TARGET"
    printf 'Command: %q ' "${CMD[@]}"
    echo
    "${CMD[@]}"
  done
done
