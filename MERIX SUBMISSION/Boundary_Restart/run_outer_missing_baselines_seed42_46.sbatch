#!/bin/bash
#SBATCH --job-name=br_outer_missing
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --time=72:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu3090
#SBATCH --qos=1gpu
#SBATCH --gres=gpu:1

set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$PROJECT_DIR"

VENV_PATH="${VENV_PATH:-$PROJECT_DIR/../MIREX_Model/.venv}"
if [ ! -f "$VENV_PATH/bin/activate" ]; then
  echo "Missing virtualenv activate script: $VENV_PATH/bin/activate" >&2
  exit 1
fi
source "$VENV_PATH/bin/activate"

SEEDS=(${SEEDS:-42 43 44 45 46})
SEED_SLUG="$(printf '%s' "${SEEDS[@]}")"

CONFIG="${CONFIG:-configs/salience_grouped3_hi8_score_only_xml_curated.yaml}"
OUTER_HELDOUT=(${OUTER_HELDOUT:-M06-1 M06-2 M06-3})
MAX_INNER_FOLDS="${MAX_INNER_FOLDS:-}"

NONSEQ_MODELS=(${NONSEQ_MODELS:-all_boundary periodic downbeat logreg_window7})
NONSEQ_OUTPUT_TEMPLATE="${NONSEQ_OUTPUT_TEMPLATE:-paper_outer_missing_baselines_seed%s}"

BILSTM_DEVICE="${BILSTM_DEVICE:-cuda}"
BILSTM_BATCH_SIZE="${BILSTM_BATCH_SIZE:-128}"
BILSTM_EPOCHS="${BILSTM_EPOCHS:-60}"
BILSTM_EARLY_STOP_PATIENCE="${BILSTM_EARLY_STOP_PATIENCE:-10}"
BILSTM_OUTPUT_TEMPLATE="${BILSTM_OUTPUT_TEMPLATE:-paper_outer_baselines_weighted_topdown_all_seed%s_bilstm}"

# Optional backfill if you later decide to extend existing retained baselines to new seeds.
EXTRA_EXISTING_MODELS="${EXTRA_EXISTING_MODELS:-}"
EXTRA_EXISTING_TARGET_DESIGN="${EXTRA_EXISTING_TARGET_DESIGN:-weighted_topdown}"
EXTRA_EXISTING_FEATURE_FAMILY="${EXTRA_EXISTING_FEATURE_FAMILY:-all}"
EXTRA_EXISTING_OUTPUT_TEMPLATE="${EXTRA_EXISTING_OUTPUT_TEMPLATE:-paper_outer_existing_backfill_seed%s}"

# Optional external Grouper results for later merge.
GROUPER_LEVELWISE_CSV="${GROUPER_LEVELWISE_CSV:-}"

echo "PROJECT_DIR=$PROJECT_DIR"
echo "VENV_PATH=$VENV_PATH"
echo "CONFIG=$CONFIG"
echo "OUTER_HELDOUT=${OUTER_HELDOUT[*]}"
echo "SEEDS=${SEEDS[*]}"
echo "NONSEQ_MODELS=${NONSEQ_MODELS[*]}"
echo "BILSTM_DEVICE=$BILSTM_DEVICE"

for SEED in "${SEEDS[@]}"; do
  echo
  echo "=== Seed ${SEED} ==="

  NONSEQ_OUT="reports/$(printf "$NONSEQ_OUTPUT_TEMPLATE" "$SEED")"
  CMD_NONSEQ=(
    python run_outer_score_baselines.py
    --config "$CONFIG"
    --models "${NONSEQ_MODELS[@]}"
    --outer_heldout_piece "${OUTER_HELDOUT[@]}"
    --target_design weighted_topdown
    --feature_family all
    --seed "$SEED"
    --output_dir "$NONSEQ_OUT"
  )
  if [ -n "$MAX_INNER_FOLDS" ]; then
    CMD_NONSEQ+=(--max_inner_folds "$MAX_INNER_FOLDS")
  fi
  printf 'Command: %q ' "${CMD_NONSEQ[@]}"
  echo
  "${CMD_NONSEQ[@]}"

  if [ -n "$EXTRA_EXISTING_MODELS" ]; then
    EXTRA_OUT="reports/$(printf "$EXTRA_EXISTING_OUTPUT_TEMPLATE" "$SEED")"
    read -r -a EXTRA_MODELS_ARR <<< "$EXTRA_EXISTING_MODELS"
    CMD_EXISTING=(
      python run_outer_score_baselines.py
      --config "$CONFIG"
      --models "${EXTRA_MODELS_ARR[@]}"
      --outer_heldout_piece "${OUTER_HELDOUT[@]}"
      --target_design "$EXTRA_EXISTING_TARGET_DESIGN"
      --feature_family "$EXTRA_EXISTING_FEATURE_FAMILY"
      --seed "$SEED"
      --output_dir "$EXTRA_OUT"
    )
    if [ -n "$MAX_INNER_FOLDS" ]; then
      CMD_EXISTING+=(--max_inner_folds "$MAX_INNER_FOLDS")
    fi
    printf 'Command: %q ' "${CMD_EXISTING[@]}"
    echo
    "${CMD_EXISTING[@]}"
  fi

  BILSTM_OUT="reports/$(printf "$BILSTM_OUTPUT_TEMPLATE" "$SEED")"
  CMD_BILSTM=(
    python run_outer_bilstm_baseline.py
    --config "$CONFIG"
    --outer_heldout_piece "${OUTER_HELDOUT[@]}"
    --seed "$SEED"
    --device "$BILSTM_DEVICE"
    --batch_size "$BILSTM_BATCH_SIZE"
    --epochs "$BILSTM_EPOCHS"
    --early_stop_patience "$BILSTM_EARLY_STOP_PATIENCE"
    --output_dir "$BILSTM_OUT"
    --reuse_existing
  )
  if [ -n "$MAX_INNER_FOLDS" ]; then
    CMD_BILSTM+=(--max_inner_folds "$MAX_INNER_FOLDS")
  fi
  printf 'Command: %q ' "${CMD_BILSTM[@]}"
  echo
  "${CMD_BILSTM[@]}"
done

SUMMARY_CMD=(
  python summarize_outer_missing_baselines.py
  --seeds "${SEEDS[@]}"
  --report_root reports
  --nonseq_dir_template "$NONSEQ_OUTPUT_TEMPLATE"
  --bilstm_dir_template "$BILSTM_OUTPUT_TEMPLATE"
  --output_prefix "paper_outer_missing_baseline_summary_seed${SEED_SLUG}"
)
if [ -n "$GROUPER_LEVELWISE_CSV" ]; then
  SUMMARY_CMD+=(--grouper_levelwise_csv "$GROUPER_LEVELWISE_CSV")
fi

echo
printf 'Command: %q ' "${SUMMARY_CMD[@]}"
echo
"${SUMMARY_CMD[@]}"
