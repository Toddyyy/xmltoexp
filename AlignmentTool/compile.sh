#! /bin/bash

ProgramFolder="./Programs"
CodeFolder="./Code"

mkdir -p "$ProgramFolder"

# Explicitly point clang++ to the SDK and libc++ headers to avoid "fstream not found" on fresh CLT installs.
SDK_PATH=$(xcrun --show-sdk-path 2>/dev/null)
STD_FLAGS="-std=c++11 -stdlib=libc++"
INCLUDE_FLAGS=""
if [ -n "$SDK_PATH" ]; then
  INCLUDE_FLAGS="-isysroot $SDK_PATH -I$SDK_PATH/usr/include/c++/v1"
fi

g++ -O2 $STD_FLAGS $INCLUDE_FLAGS "$CodeFolder/ErrorDetection_v190702.cpp"      -o "$ProgramFolder/ErrorDetection"
g++ -O2 $STD_FLAGS $INCLUDE_FLAGS "$CodeFolder/RealignmentMOHMM_v170427.cpp"    -o "$ProgramFolder/RealignmentMOHMM"
g++ -O2 $STD_FLAGS $INCLUDE_FLAGS "$CodeFolder/ScorePerfmMatcher_v170101_2.cpp" -o "$ProgramFolder/ScorePerfmMatcher"
g++ -O2 $STD_FLAGS $INCLUDE_FLAGS "$CodeFolder/midi2pianoroll_v170504.cpp"      -o "$ProgramFolder/midi2pianoroll"
g++ -O2 $STD_FLAGS $INCLUDE_FLAGS "$CodeFolder/MusicXMLToFmt3x_v170104.cpp"     -o "$ProgramFolder/MusicXMLToFmt3x"
g++ -O2 $STD_FLAGS $INCLUDE_FLAGS "$CodeFolder/MusicXMLToHMM_v170104.cpp"       -o "$ProgramFolder/MusicXMLToHMM"
g++ -O2 $STD_FLAGS $INCLUDE_FLAGS "$CodeFolder/SprToFmt3x_v170225.cpp"          -o "$ProgramFolder/SprToFmt3x"
g++ -O2 $STD_FLAGS $INCLUDE_FLAGS "$CodeFolder/Fmt3xToHmm_v170225.cpp"          -o "$ProgramFolder/Fmt3xToHmm"
g++ -O2 $STD_FLAGS $INCLUDE_FLAGS "$CodeFolder/MatchToCorresp_v170918.cpp"      -o "$ProgramFolder/MatchToCorresp"
