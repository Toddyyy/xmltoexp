#!/bin/bash
#
# Quick test script for CPU - verify pipeline works with minimal time investment
# Usage: bash test_quick_cpu.sh
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"; pwd)"
cd "$SCRIPT_DIR"

# Color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

clear

echo -e "${BLUE}╔════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║${NC}   🚀 Quick CPU Test (Minimal Runtime)${NC}"
echo -e "${BLUE}║${NC}   Estimated time: 30-60 minutes${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════╝${NC}"
echo ""

echo -e "${YELLOW}This will test:${NC}"
echo "  ✓ TCN (5 targets) - reduced epochs"
echo "  ✓ Baselines (LogReg + LBDM only)"
echo "  ✗ BiLSTM (skipped - very slow on CPU)"
echo "  ✗ Summary (skipped - needs all seeds)"
echo ""

echo -e "${YELLOW}Configuration:${NC}"
echo "  • Seed: 42 (single seed only)"
echo "  • Epochs: 10 (reduced from 60)"
echo "  • Early stop patience: 10"
echo ""

read -p "Start quick test? [Y/n]: " confirm
if [[ ! $confirm =~ ^[Yy]$ ]] && [ -n "$confirm" ]; then
  echo "Cancelled."
  exit 0
fi

echo ""
echo -e "${GREEN}Starting quick test...${NC}"
echo ""

# Run with optimized settings
DEVICE="cpu" \
EPOCHS=10 \
SEEDS="42" \
RUN_BILSTM=0 \
RUN_SUMMARY=0 \
bash ./run_experiments_locally.sh

echo ""
echo -e "${GREEN}Quick test completed!${NC}"
echo ""
echo "Next steps:"
echo "  1. Check the results in 'reports/' directory"
echo "  2. If successful, run full experiments:"
echo "     bash run_experiments_locally.sh --seed 42  # or other seeds"
echo "  3. For background execution:"
echo "     nohup bash run_experiments_locally.sh > exp.log 2>&1 &"
