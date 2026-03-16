#!/bin/bash
#SBATCH --job-name=br_clean_alllvl
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err
#SBATCH --partition=gpua800
#SBATCH --qos=1a800
#SBATCH --gres=gpu:1
#SBATCH --array=0-1
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --export=ALL

set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$PROJECT_DIR"

VENV_PATH="${VENV_PATH:-$PROJECT_DIR/../MIREX_Model/.venv}"
if [ ! -f "$VENV_PATH/bin/activate" ]; then
  echo "Missing virtualenv activate script: $VENV_PATH/bin/activate" >&2
  exit 1
fi
source "$VENV_PATH/bin/activate"

SEEDS=(42 44)
SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"

CONFIG="${CONFIG:-configs/salience_grouped3_hi8_score_only_xml_curated.yaml}"
OUTER_PIECES="${OUTER_PIECES:-M06-1 M06-2 M06-3}"
DEVICE="${DEVICE:-cuda}"

LEVEL_SLUGS=(level1 level2 level3 level4 level56)

echo "PROJECT_DIR=$PROJECT_DIR"
echo "SEED=$SEED"
echo "CONFIG=$CONFIG"
echo "OUTER_PIECES=$OUTER_PIECES"
echo "DEVICE=$DEVICE"

for LEVEL_SLUG in "${LEVEL_SLUGS[@]}"; do
  NESTED_DIR="reports/nested_piece_cv/M06_outer_${LEVEL_SLUG}_seed${SEED}"
  OUTPUT_DIR="reports/clean_outer_test/M06_outer_${LEVEL_SLUG}_seed${SEED}"

  echo
  echo "=== SEED=$SEED LEVEL=$LEVEL_SLUG ==="
  echo "NESTED_DIR=$NESTED_DIR"
  echo "OUTPUT_DIR=$OUTPUT_DIR"

  python run_clean_outer_test_from_nested.py \
    --config "$CONFIG" \
    --nested_report_dir "$NESTED_DIR" \
    --outer_heldout_piece $OUTER_PIECES \
    --device "$DEVICE" \
    --output_dir "$OUTPUT_DIR"
done
