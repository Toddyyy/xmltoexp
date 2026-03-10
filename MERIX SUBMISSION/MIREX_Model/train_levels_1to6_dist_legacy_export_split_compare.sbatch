#!/bin/bash
#SBATCH --job-name=beat_L1to6_cmp_samepieces
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

seed=42
split_dir="splits/generated_legacy_replay_samepieces"
mkdir -p "$split_dir"

for level in 1 2 3 4 5 6; do
  cfg="config_beat_mazurka_level${level}.yaml"
  if [ ! -f "$cfg" ]; then
    echo "Missing config: $cfg" >&2
    exit 1
  fi

  legacy_prefix="cmp_samepieces_legacy_seed${seed}_lvl${level}"
  fixed_prefix="cmp_samepieces_fixed_seed${seed}_lvl${level}"
  legacy_cfg="$(mktemp "${TMPDIR:-/tmp}/beat_legacy_lvl${level}_XXXX.yaml")"
  fixed_cfg="$(mktemp "${TMPDIR:-/tmp}/beat_fixed_lvl${level}_XXXX.yaml")"

  python - "$cfg" "$legacy_cfg" "$seed" "$legacy_prefix" <<'PYCFG'
import sys, yaml
src, dst, seed, exp = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
with open(src, 'r') as f:
    cfg = yaml.safe_load(f)
cfg.setdefault('training', {})['seed'] = seed
cfg.setdefault('trainer', {})['experiment_name'] = exp
with open(dst, 'w') as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
PYCFG

  echo "=== Level ${level} | legacy run (seed=${seed}) ==="
  if [ "$level" -eq 6 ]; then
    python train_beat.py \
      --config "$legacy_cfg" \
      --level "$level" \
      --early_stop_patience 5 \
      --pos_weight 239.6 \
      --device cuda
  else
    python train_beat.py \
      --config "$legacy_cfg" \
      --level "$level" \
      --early_stop_patience 5 \
      --device cuda
  fi

  legacy_dir=$(ls -dt "check/beat_mazurka/level_${level}/${legacy_prefix}_L${level}_"* 2>/dev/null | head -n 1 || true)
  if [ -z "$legacy_dir" ]; then
    echo "Failed to locate legacy experiment dir for level ${level}" >&2
    exit 1
  fi
  split_file="${split_dir}/level${level}_seed${seed}.yaml"
  python - "$legacy_dir/split_summary.yaml" "$split_file" <<'PYSPLIT'
import sys, yaml
src, dst = sys.argv[1], sys.argv[2]
with open(src, 'r') as f:
    s = yaml.safe_load(f) or {}
out = {
    'version': 1,
    'source': 'legacy_random_piece_split_replay',
    'train': s.get('train_pieces', []),
    'val': s.get('val_pieces', []),
    'test': [],
}
with open(dst, 'w') as f:
    yaml.safe_dump(out, f, sort_keys=False)
print(dst)
PYSPLIT

  python - "$cfg" "$fixed_cfg" "$seed" "$fixed_prefix" <<'PYCFG'
import sys, yaml
src, dst, seed, exp = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
with open(src, 'r') as f:
    cfg = yaml.safe_load(f)
cfg.setdefault('training', {})['seed'] = seed
cfg.setdefault('trainer', {})['experiment_name'] = exp
with open(dst, 'w') as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
PYCFG

  echo "=== Level ${level} | fixed split replay using ${split_file} ==="
  if [ "$level" -eq 6 ]; then
    python train_beat.py \
      --config "$fixed_cfg" \
      --level "$level" \
      --split_file "$split_file" \
      --early_stop_patience 5 \
      --pos_weight 239.6 \
      --device cuda
  else
    python train_beat.py \
      --config "$fixed_cfg" \
      --level "$level" \
      --split_file "$split_file" \
      --early_stop_patience 5 \
      --device cuda
  fi

  rm -f "$legacy_cfg" "$fixed_cfg"
done
