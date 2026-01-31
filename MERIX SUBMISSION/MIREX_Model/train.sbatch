#!/bin/bash
#SBATCH --job-name=1
#SBATCH --partition=gpu3090
#SBATCH --gres=gpu:1
#SBATCH --qos=1gpu
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -e
cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate

python train_beat.py \
  --config config_beat_mazurka_level6_ratio.yaml \
  --level 6 \
  --pos_weight 239.59 \
  --device cuda
