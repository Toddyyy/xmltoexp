#!/bin/bash
#
# Interactive experiment runner - choose which experiments to run
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"; pwd)"
cd "$SCRIPT_DIR"

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

clear

echo -e "${BLUE}╔════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║${NC}   🔬 Local Experiment Runner${NC}"
echo -e "${BLUE}║${NC}   Interactive Configuration${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════╝${NC}"
echo ""

# Select seeds
echo -e "${YELLOW}1. Select Seeds:${NC}"
echo "   a) All seeds (42, 43, 44, 45, 46)  [default]"
echo "   b) Single seed"
echo "   c) Custom selection"
read -p "Choose [a/b/c]: " seed_choice

SEEDS=""
case $seed_choice in
  b)
    read -p "Enter seed number: " seed_num
    SEEDS="$seed_num"
    ;;
  c)
    read -p "Enter seeds (space-separated, e.g. '42 43 44'): " SEEDS
    ;;
  *)
    SEEDS="42 43 44 45 46"
    ;;
esac

echo -e "${GREEN}✓ Seeds: $SEEDS${NC}"
echo ""

# Select components
echo -e "${YELLOW}2. Select Components to Run:${NC}"
read -p "   Run TCN experiments? [Y/n]: " run_tcn
RUN_TCN=${run_tcn:-Y}
[[ $RUN_TCN =~ ^[Yy]$ ]] && RUN_TCN="1" || RUN_TCN="0"

read -p "   Run Baseline experiments? [Y/n]: " run_baselines
RUN_BASELINES=${run_baselines:-Y}
[[ $RUN_BASELINES =~ ^[Yy]$ ]] && RUN_BASELINES="1" || RUN_BASELINES="0"

read -p "   Run BiLSTM experiments? [Y/n]: " run_bilstm
RUN_BILSTM=${run_bilstm:-Y}
[[ $RUN_BILSTM =~ ^[Yy]$ ]] && RUN_BILSTM="1" || RUN_BILSTM="0"

read -p "   Run summary generation? [Y/n]: " run_summary
RUN_SUMMARY=${run_summary:-Y}
[[ $RUN_SUMMARY =~ ^[Yy]$ ]] && RUN_SUMMARY="1" || RUN_SUMMARY="0"

echo -e "${GREEN}✓ Configuration selected${NC}"
echo ""

# Select device
echo -e "${YELLOW}3. Select Device:${NC}"
echo "   a) CPU  (slower, good for testing)"
echo "   b) GPU (cuda)  [default if available]"
read -p "Choose [a/b]: " device_choice

DEVICE="cuda"
if [ "$device_choice" = "a" ]; then
  DEVICE="cpu"
fi
echo -e "${GREEN}✓ Device: $DEVICE${NC}"
echo ""

# Confirm
echo -e "${YELLOW}4. Review Configuration:${NC}"
echo "   Seeds: $SEEDS"
echo "   TCN: $([ $RUN_TCN = "1" ] && echo "✓ Yes" || echo "✗ No")"
echo "   Baselines: $([ $RUN_BASELINES = "1" ] && echo "✓ Yes" || echo "✗ No")"
echo "   BiLSTM: $([ $RUN_BILSTM = "1" ] && echo "✓ Yes" || echo "✗ No")"
echo "   Summary: $([ $RUN_SUMMARY = "1" ] && echo "✓ Yes" || echo "✗ No")"
echo "   Device: $DEVICE"
echo ""

read -p "Start experiments? [Y/n]: " confirm
if [[ ! $confirm =~ ^[Yy]$ ]]; then
  echo -e "${YELLOW}Cancelled.${NC}"
  exit 0
fi

echo ""
echo -e "${GREEN}Starting experiments...${NC}"
echo ""

# Run the main script
export SEEDS="$SEEDS"
export RUN_TCN="$RUN_TCN"
export RUN_BASELINES="$RUN_BASELINES"
export RUN_BILSTM="$RUN_BILSTM"
export RUN_SUMMARY="$RUN_SUMMARY"
export DEVICE="$DEVICE"

bash ./run_experiments_locally.sh
