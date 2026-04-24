#!/bin/bash
#
# Clean up corrupted results from failed run
# This safely removes all results for specified seeds
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

echo -e "${BLUE}╔════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║${NC}   🧹 Cleanup Failed Run Results${NC}"
echo -e "${BLUE}╚════════════════════════════════════════╝${NC}"
echo ""

if [ $# -eq 0 ]; then
  echo "Usage: bash clean_reports.sh [SEED1 SEED2 ...]"
  echo ""
  echo "Examples:"
  echo "  bash clean_reports.sh 42          # Clean seed 42 results"
  echo "  bash clean_reports.sh 42 43 44    # Clean seeds 42, 43, 44"
  echo "  bash clean_reports.sh --all       # Remove all results (dangerous!)"
  echo ""
  echo "Found seeds in reports:"
  find reports -maxdepth 1 -type d -name "*seed*" 2>/dev/null | sed 's|.*seed||' | sort -u | grep -o '[0-9]*' | sort -u
  exit 0
fi

SEEDS_TO_CLEAN=()

if [ "$1" = "--all" ]; then
  echo -e "${RED}⚠️  WARNING: This will remove ALL results from reports/ directory${NC}"
  read -p "Type 'yes' to confirm: " confirm
  if [ "$confirm" != "yes" ]; then
    echo "Cancelled."
    exit 0
  fi
  rm -rf reports/*
  echo -e "${GREEN}✓ All reports cleaned${NC}"
  exit 0
fi

# Parse seeds from arguments
for seed in "$@"; do
  if [[ $seed =~ ^[0-9]+$ ]]; then
    SEEDS_TO_CLEAN+=("$seed")
  else
    echo -e "${RED}✗ Invalid seed: $seed${NC}"
    exit 1
  fi
done

echo "Seeds to clean: ${SEEDS_TO_CLEAN[*]}"
echo ""

# Find and list directories to be removed
DIRS_TO_REMOVE=()

for seed in "${SEEDS_TO_CLEAN[@]}"; do
  echo -e "${YELLOW}Looking for seed $seed results...${NC}"

  # Find all directories matching this seed
  while IFS= read -r dir; do
    if [ -n "$dir" ] && [ -d "$dir" ]; then
      DIRS_TO_REMOVE+=("$dir")
      echo "  ├─ $dir"
    fi
  done < <(find reports -maxdepth 2 -type d -name "*seed${seed}*" 2>/dev/null)

done

echo ""

if [ ${#DIRS_TO_REMOVE[@]} -eq 0 ]; then
  echo -e "${YELLOW}No results found for seeds: ${SEEDS_TO_CLEAN[*]}${NC}"
  exit 0
fi

echo -e "${YELLOW}Total directories to remove: ${#DIRS_TO_REMOVE[@]}${NC}"
echo ""

read -p "Confirm deletion? [y/N]: " confirm
if [[ ! $confirm =~ ^[Yy]$ ]]; then
  echo "Cancelled."
  exit 0
fi

echo ""
for dir in "${DIRS_TO_REMOVE[@]}"; do
  echo "Removing: $dir"
  rm -rf "$dir"
done

echo ""
echo -e "${GREEN}✓ Cleanup complete!${NC}"
echo ""

# Show remaining seeds
echo "Remaining seeds in reports:"
find reports -maxdepth 1 -type d -name "*seed*" 2>/dev/null | sed 's|.*seed||' | sort -u | grep -o '[0-9]*' | sort -u | tr '\n' ' '
echo ""
