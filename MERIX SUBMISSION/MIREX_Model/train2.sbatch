#!/bin/bash
#SBATCH --job-name=beat_sal
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

python train_beat_salience.py \
  --config config_beat_mazurka_salience.yaml \
  --split_file splits/mazurka_piece_split_v1.yaml \
  --aux_split_file splits/mazurka_aux_performer_split_v1.yaml \
  --aux_mode heldout_pianists \
  --aux_targets Ashkenazy,Cohen \
  --levels 1,2,3,4,5 \
  --level_weights 2.158,3.505,5.692,9.243,15.012 \
  --device cuda
