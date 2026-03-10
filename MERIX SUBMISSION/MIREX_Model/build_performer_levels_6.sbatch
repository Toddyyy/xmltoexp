#!/bin/bash
#SBATCH --job-name=build_L1to6
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

echo "Working directory: $PWD"
echo "Rebuilding beat_data_mazurka_performer_levels with str_vec=3,2,2,2,2,2"

rm -rf beat_data_mazurka_performer_levels

python build_mazurka_beat_npz_performer_levels.py \
  --str_vec 3,2,2,2,2,2

