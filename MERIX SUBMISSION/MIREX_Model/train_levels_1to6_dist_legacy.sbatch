#!/bin/bash
#SBATCH --job-name=beat_L1to6_old_x5
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

seeds=(42 101 202 303 404)

for seed in "${seeds[@]}"; do
  echo "=== Legacy random split run with seed=${seed} ==="
  for level in 1 2 3 4 5 6; do
    cfg="config_beat_mazurka_level${level}.yaml"
    if [ ! -f "$cfg" ]; then
      echo "Missing config: $cfg" >&2
      exit 1
    fi

    tmp_cfg="$(mktemp "${TMPDIR:-/tmp}/beat_level${level}_seed${seed}_XXXX.yaml")"
    python - "$cfg" "$tmp_cfg" "$seed" <<'PYCFG'
import sys, yaml
src, dst, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
with open(src, 'r') as f:
    cfg = yaml.safe_load(f)
cfg.setdefault('training', {})['seed'] = seed
with open(dst, 'w') as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
PYCFG

    if [ "$level" -eq 6 ]; then
      python train_beat.py \
        --config "$tmp_cfg" \
        --level "$level" \
        --early_stop_patience 5 \
        --pos_weight 239.6 \
        --device cuda
    else
      python train_beat.py \
        --config "$tmp_cfg" \
        --level "$level" \
        --early_stop_patience 5 \
        --device cuda
    fi

    rm -f "$tmp_cfg"
  done
done
