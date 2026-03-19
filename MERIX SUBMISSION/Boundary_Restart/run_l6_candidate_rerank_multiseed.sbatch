#!/bin/bash
#SBATCH --job-name=br_l6_rerank_ms
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --time=72:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpua800
#SBATCH --qos=1a800
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

export DEVICE="${DEVICE:-cuda}"
export OUTER_PIECES="${OUTER_PIECES:-M06-1 M06-2 M06-3}"
export OUTPUT_TAG="${OUTPUT_TAG:-server_cuda}"

echo "PROJECT_DIR=$PROJECT_DIR"
echo "DEVICE=$DEVICE"
echo "OUTER_PIECES=$OUTER_PIECES"
echo "OUTPUT_TAG=$OUTPUT_TAG"

python3 run_l6_candidate_rerank_multiseed.py
