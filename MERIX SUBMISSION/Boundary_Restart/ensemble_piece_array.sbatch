#!/bin/bash
#SBATCH --job-name=br_ens_arr
#SBATCH --array=0-2
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --export=ALL

set -euo pipefail

# On Slurm clusters, the script may execute from a spool copy. Prefer the
# original submission directory so relative paths resolve against the repo.
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$PROJECT_DIR"

VENV_PATH="${VENV_PATH:-$PROJECT_DIR/../MIREX_Model/.venv}"
if [ ! -f "$VENV_PATH/bin/activate" ]; then
  echo "Missing virtualenv activate script: $VENV_PATH/bin/activate" >&2
  exit 1
fi
source "$VENV_PATH/bin/activate"

CONFIG="${CONFIG:-configs/salience_grouped3_hi8_score_only_xml_curated.yaml}"
MIN_PRECISION="${MIN_PRECISION:-0.95}"
HELDOUT_PIECES=("M06-2" "M17-1" "M30-1")
IDX="${SLURM_ARRAY_TASK_ID:-0}"
HELDOUT_PIECE="${HELDOUT_PIECES[$IDX]}"

TCN_RUN_NAME="${TCN_RUN_NAME:-tcn_midhigh_boundary_xml_curated_p95_cuda}"
BILSTM_RUN_NAME="${BILSTM_RUN_NAME:-bilstm_midhigh_boundary_xml_curated_p95_cuda}"
ENSEMBLE_RUN_NAME="${ENSEMBLE_RUN_NAME:-ensemble_tcn_bilstm_xml_curated_p95}"

TCN_PRED="${TCN_PRED:-$PROJECT_DIR/outputs/piece_protocol/${HELDOUT_PIECE}/${TCN_RUN_NAME}/val_predictions.csv.gz}"
BILSTM_PRED="${BILSTM_PRED:-$PROJECT_DIR/outputs/piece_protocol/${HELDOUT_PIECE}/${BILSTM_RUN_NAME}/val_predictions.csv.gz}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/outputs/piece_protocol/${HELDOUT_PIECE}/${ENSEMBLE_RUN_NAME}}"

echo "PROJECT_DIR=$PROJECT_DIR"
echo "VENV_PATH=$VENV_PATH"
echo "CONFIG=$CONFIG"
echo "SLURM_ARRAY_TASK_ID=$IDX"
echo "HELDOUT_PIECE=$HELDOUT_PIECE"
echo "MIN_PRECISION=$MIN_PRECISION"
echo "TCN_PRED=$TCN_PRED"
echo "BILSTM_PRED=$BILSTM_PRED"
echo "OUTPUT_DIR=$OUTPUT_DIR"

if [ ! -f "$TCN_PRED" ]; then
  echo "Missing TCN prediction file: $TCN_PRED" >&2
  exit 1
fi
if [ ! -f "$BILSTM_PRED" ]; then
  echo "Missing BiLSTM prediction file: $BILSTM_PRED" >&2
  exit 1
fi

python ensemble_piece_protocol.py \
  --config "$CONFIG" \
  --tcn_pred "$TCN_PRED" \
  --bilstm_pred "$BILSTM_PRED" \
  --output_dir "$OUTPUT_DIR" \
  --min_precision "$MIN_PRECISION"
