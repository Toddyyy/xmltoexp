#!/bin/bash
#SBATCH --job-name=beat_L1to6_old
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

for level in 1 2 3 4 5 6; do
  cfg="config_beat_mazurka_level${level}.yaml"
  if [ ! -f "$cfg" ]; then
    echo "Missing config: $cfg" >&2
    exit 1
  fi

  if [ "$level" -eq 6 ]; then
    python train_beat.py \
      --config "$cfg" \
      --level "$level" \
      --early_stop_patience 5 \
      --pos_weight 239.6 \
      --device cuda
  else
    python train_beat.py \
      --config "$cfg" \
      --level "$level" \
      --early_stop_patience 5 \
      --device cuda
  fi
done
