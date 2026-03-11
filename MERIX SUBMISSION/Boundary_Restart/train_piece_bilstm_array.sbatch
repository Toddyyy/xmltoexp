#!/bin/bash
#SBATCH --job-name=br_bilstm_arr
#SBATCH --partition=gpua800
#SBATCH --gres=gpu:1
#SBATCH --qos=1a800
#SBATCH --array=0-2
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --export=ALL

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

VENV_PATH="${VENV_PATH:-$PROJECT_DIR/../MIREX_Model/.venv}"
source "$VENV_PATH/bin/activate"

CONFIG="${CONFIG:-configs/salience_grouped3_hi8_score_only_xml_curated.yaml}"
MODEL="${MODEL:-bilstm}"
DETECTOR_TARGET="${DETECTOR_TARGET:-midhigh_boundary}"
MIN_PRECISION="${MIN_PRECISION:-0.95}"
BATCH_SIZE="${BATCH_SIZE:-128}"
EPOCHS="${EPOCHS:-30}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-5}"
SEED="${SEED:-42}"

HELDOUT_PIECES=("M06-2" "M17-1" "M30-1")
IDX="${SLURM_ARRAY_TASK_ID:-0}"
HELDOUT_PIECE="${HELDOUT_PIECES[$IDX]}"

OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/outputs/piece_protocol/${HELDOUT_PIECE}/${MODEL}_${DETECTOR_TARGET}_xml_curated_p95_cuda}"

echo "PROJECT_DIR=$PROJECT_DIR"
echo "VENV_PATH=$VENV_PATH"
echo "CONFIG=$CONFIG"
echo "SLURM_ARRAY_TASK_ID=$IDX"
echo "HELDOUT_PIECE=$HELDOUT_PIECE"
echo "MODEL=$MODEL"
echo "DETECTOR_TARGET=$DETECTOR_TARGET"
echo "MIN_PRECISION=$MIN_PRECISION"
echo "BATCH_SIZE=$BATCH_SIZE"
echo "EPOCHS=$EPOCHS"
echo "EARLY_STOP_PATIENCE=$EARLY_STOP_PATIENCE"
echo "SEED=$SEED"
echo "OUTPUT_DIR=$OUTPUT_DIR"

python train_piece_protocol.py \
  --config "$CONFIG" \
  --heldout_piece "$HELDOUT_PIECE" \
  --model "$MODEL" \
  --detector_target "$DETECTOR_TARGET" \
  --min_precision "$MIN_PRECISION" \
  --batch_size "$BATCH_SIZE" \
  --epochs "$EPOCHS" \
  --early_stop_patience "$EARLY_STOP_PATIENCE" \
  --seed "$SEED" \
  --device cuda \
  --output_dir "$OUTPUT_DIR"
