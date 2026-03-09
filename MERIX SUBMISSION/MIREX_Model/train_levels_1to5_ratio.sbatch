#!/bin/bash
#SBATCH --job-name=beat_L1to5_ratio
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

tmp_cfg_dir="$(mktemp -d)"
trap 'rm -rf "$tmp_cfg_dir"' EXIT

for level in 1 2 3 4 5; do
  base_cfg="config_beat_mazurka_level${level}_ratio.yaml"
  run_cfg="${tmp_cfg_dir}/config_beat_mazurka_level${level}_ratio_feat14_pcond.yaml"

  python - "$base_cfg" "$run_cfg" <<'PY'
import sys
import yaml

src, dst = sys.argv[1], sys.argv[2]
with open(src, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

cfg["data"]["use_base_features_only"] = False
cfg["model"]["input_dim"] = 14
cfg["model"]["performer_cond"] = True
cfg["trainer"]["experiment_name"] = f'{cfg["trainer"]["experiment_name"]}_ratio'

with open(dst, "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
PY

  python train_beat.py \
    --config "$run_cfg" \
    --split_file splits/mazurka_piece_split_v1.yaml \
    --early_stop_patience 5 \
    --level "${level}" \
    --device cuda

  exp_dir=$(python - "$run_cfg" "$level" <<'PY'
import sys
from pathlib import Path
import yaml

cfg_path, level = sys.argv[1], int(sys.argv[2])
with open(cfg_path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

base = Path(cfg["trainer"]["save_dir"]) / f"level_{level}"
prefix = f'{cfg["trainer"]["experiment_name"]}_L{level}_'
candidates = [p for p in base.glob(f"{prefix}*") if p.is_dir()]
if not candidates:
    raise SystemExit(f"No experiment directory found under {base} with prefix {prefix}")
latest = max(candidates, key=lambda p: p.stat().st_mtime)
print(latest)
PY
)

  python eval_boundary_level.py \
    --config "$run_cfg" \
    --model_path "${exp_dir}/best.pt" \
    --split_file splits/mazurka_piece_split_v1.yaml \
    --level "${level}" \
    --split val \
    --head dist \
    --device cuda

  python eval_boundary_level.py \
    --config "$run_cfg" \
    --model_path "${exp_dir}/best.pt" \
    --split_file splits/mazurka_piece_split_v1.yaml \
    --level "${level}" \
    --split test \
    --head dist \
    --device cuda
done
