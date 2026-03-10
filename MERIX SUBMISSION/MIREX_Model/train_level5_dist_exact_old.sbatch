#!/bin/bash
#SBATCH --job-name=beat_L5_old
#SBATCH --partition=gpua800
#SBATCH --gres=gpu:1
#SBATCH --qos=1a800
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate

python train_beat.py \
  --config config_beat_mazurka_level5_exact_old.yaml \
  --level 5 \
  --early_stop_patience 5 \
  --device cuda
