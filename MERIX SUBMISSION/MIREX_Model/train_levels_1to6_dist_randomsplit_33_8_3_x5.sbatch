#!/bin/bash
#SBATCH --job-name=beat_L1to6_rand33_8_3x5
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

split_dir="splits/generated_random_33_8_3_x5"
mkdir -p "$split_dir"

python - <<'PY'
from pathlib import Path
import random
import yaml

all_pieces = [
    'M06-1', 'M06-2', 'M06-3', 'M07-1', 'M07-2', 'M07-3',
    'M17-1', 'M17-2', 'M17-3', 'M17-4',
    'M24-1', 'M24-2', 'M24-3', 'M24-4',
    'M30-1', 'M30-2', 'M30-3', 'M30-4',
    'M33-1', 'M33-2', 'M33-3', 'M33-4',
    'M41-1', 'M41-2', 'M41-3', 'M41-4',
    'M50-1', 'M50-2', 'M50-3',
    'M56-1', 'M56-2', 'M56-3',
    'M59-1', 'M59-2', 'M59-3',
    'M63-1', 'M63-3',
    'M67-1', 'M67-2', 'M67-3', 'M67-4',
    'M68-1', 'M68-2', 'M68-3',
]

out_dir = Path("splits/generated_random_33_8_3_x5")
seeds = [1101, 1102, 1103, 1104, 1105]

for idx, seed in enumerate(seeds, start=1):
    rng = random.Random(seed)
    pieces = list(all_pieces)
    rng.shuffle(pieces)
    train = sorted(pieces[:33])
    val = sorted(pieces[33:41])
    test = sorted(pieces[41:44])
    obj = {
        "version": 1,
        "notes": "Random reproducible 33/8/3 split for repeated fixed-split comparison",
        "seed": seed,
        "train": train,
        "val": val,
        "test": test,
    }
    path = out_dir / f"run{idx:02d}.yaml"
    path.write_text(yaml.safe_dump(obj, sort_keys=False, allow_unicode=False), encoding="utf-8")
    print(path)
    print(f"  train={len(train)} val={len(val)} test={len(test)}")
PY

for run_idx in 01 02 03 04 05; do
  split_file="${split_dir}/run${run_idx}.yaml"
  if [ ! -f "$split_file" ]; then
    echo "Missing split file: $split_file" >&2
    exit 1
  fi

  echo "=== Random split run ${run_idx} using ${split_file} ==="

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
        --split_file "$split_file" \
        --early_stop_patience 5 \
        --pos_weight 239.6 \
        --device cuda
    else
      python train_beat.py \
        --config "$cfg" \
        --level "$level" \
        --split_file "$split_file" \
        --early_stop_patience 5 \
        --device cuda
    fi
  done
done
