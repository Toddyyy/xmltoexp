#!/bin/bash
#SBATCH --job-name=beat_L1to5
#SBATCH --partition=gpua800
#SBATCH --gres=gpu:1
#SBATCH --qos=1a800
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -e
cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate

for level in 1 2 3 4 5; do
  python train_beat.py \
    --config "config_beat_mazurka_level${level}.yaml" \
    --split_file splits/mazurka_piece_split_v1.yaml \
    --aux_split_file splits/mazurka_aux_performer_split_v1.yaml \
    --aux_mode heldout_pianists \
    --aux_targets Ashkenazy,Cohen \
    --level "${level}" \
    --device cuda
done
