#!/usr/bin/env bash
# Align every ATEPP MIDI to its score using the same pipeline as batch_pagodes/run_align.sh.
# Usage: bash align_atepp_all.sh [ATEPP_ROOT]

set -uo pipefail

ROOT="${1:-/Users/toddywang/Documents/VsCodeProjects/xmltoexp/ATEPP-1.2/ATEPP-1.2}"
SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROG_DIR="$SCRIPT_DIR/Programs"

require_prog() {
  local name="$1"
  if [[ ! -x "$PROG_DIR/$name" ]]; then
    echo "Missing $PROG_DIR/$name (run compile.sh first)" >&2
    exit 1
  fi
}

for p in midi2pianoroll MusicXMLToFmt3x MusicXMLToHMM ScorePerfmMatcher ErrorDetection RealignmentMOHMM; do
  require_prog "$p"
done

choose_score_file() {
  # Prefer *.musicxml / *.xml; otherwise unpack the first *.mxl into score_from_mxl.xml
  local piece_dir="$1"
  local xml
  xml=$(find "$piece_dir" -maxdepth 1 -type f \( -iname '*.musicxml' -o -iname '*.xml' \) | head -n1)
  if [[ -n "$xml" ]]; then
    printf '%s\n' "$xml"
    return 0
  fi

  local mxl
  mxl=$(find "$piece_dir" -maxdepth 1 -type f -iname '*.mxl' | head -n1)
  if [[ -n "$mxl" ]]; then
    local out="$piece_dir/score_from_mxl.xml"
    if [[ ! -f "$out" || "$mxl" -nt "$out" ]]; then
      echo "  unpacking $(basename "$mxl") -> $(basename "$out")"
      if ! unzip -p "$mxl" >"$out"; then
        echo "  failed to unzip $mxl" >&2
        return 1
      fi
    fi
    printf '%s\n' "$out"
    return 0
  fi
  return 1
}

process_piece() {
  local piece_dir="$1"
  echo "=== Piece: $piece_dir ==="

  local score_xml
  if ! score_xml=$(choose_score_file "$piece_dir"); then
    echo "  no score file found, skipping"
    return
  fi

  local score_fmt3x="$piece_dir/score_fmt3x.txt"
  local score_hmm="$piece_dir/score_hmm.txt"

  if [[ ! -f "$score_fmt3x" || "$score_xml" -nt "$score_fmt3x" ]]; then
    echo "  MusicXMLToFmt3x..."
    "$PROG_DIR/MusicXMLToFmt3x" "$score_xml" "$score_fmt3x" || { echo "  MusicXMLToFmt3x failed" >&2; return; }
  fi

  if [[ ! -f "$score_hmm" || "$score_xml" -nt "$score_hmm" ]]; then
    echo "  MusicXMLToHMM..."
    "$PROG_DIR/MusicXMLToHMM" "$score_xml" "$score_hmm" || { echo "  MusicXMLToHMM failed" >&2; return; }
  fi

  find "$piece_dir" -maxdepth 1 -type f -name '*.mid' -print0 | while IFS= read -r -d '' midi; do
    local base="${midi%.mid}"
    local spr="${base}_spr.txt"
    local pre="${base}_pre_match.txt"
    local err="${base}_err_match.txt"
    local match="${base}_match.txt"

    if [[ -f "$match" ]]; then
      echo "  $(basename "$midi"): match exists, skipping"
      continue
    fi

    echo "  aligning $(basename "$midi")"

    if ! "$PROG_DIR/midi2pianoroll" 0 "$base"; then
      echo "    midi2pianoroll failed" >&2
      continue
    fi
    if ! "$PROG_DIR/ScorePerfmMatcher" "$score_hmm" "$spr" "$pre" 1.0; then
      echo "    ScorePerfmMatcher failed" >&2
      continue
    fi
    if ! "$PROG_DIR/ErrorDetection" "$score_fmt3x" "$score_hmm" "$pre" "$err" 0; then
      echo "    ErrorDetection failed" >&2
      continue
    fi
    if ! "$PROG_DIR/RealignmentMOHMM" "$score_fmt3x" "$score_hmm" "$err" "$match" 0.3; then
      echo "    RealignmentMOHMM failed" >&2
      continue
    fi
  done
}

export -f choose_score_file process_piece
export PROG_DIR

find "$ROOT" -mindepth 2 -maxdepth 2 -type d -print0 | while IFS= read -r -d '' piece_dir; do
  process_piece "$piece_dir"
done
