#!/usr/bin/env bash
set -euo pipefail
PROG=../Programs

unzip -p '/Users/toddywang/Documents/VsCodeProjects/xmltoexp/ATEPP-1.2/ATEPP-1.2/Claude_Debussy/Estampes,_L._100/1._Pagodes/Estampes_I_Pagodes_--_Debussy.mxl' > 'score.xml'
$PROG/MusicXMLToFmt3x 'score.xml' 'score_fmt3x.txt'
$PROG/MusicXMLToHMM 'score.xml' 'score_hmm.txt'

echo 'aligning 03237...'
cp '/Users/toddywang/Documents/VsCodeProjects/xmltoexp/ATEPP-1.2/ATEPP-1.2/Claude_Debussy/Estampes,_L._100/1._Pagodes/03237.mid' './03237.mid'
$PROG/midi2pianoroll 0 './03237'
$PROG/ScorePerfmMatcher 'score_hmm.txt' '03237_spr.txt' '03237_pre_match.txt' 1.0
$PROG/ErrorDetection 'score_fmt3x.txt' 'score_hmm.txt' '03237_pre_match.txt' '03237_err_match.txt' 0
$PROG/RealignmentMOHMM 'score_fmt3x.txt' 'score_hmm.txt' '03237_err_match.txt' '03237_match.txt' 0.3

echo 'aligning 03238...'
cp '/Users/toddywang/Documents/VsCodeProjects/xmltoexp/ATEPP-1.2/ATEPP-1.2/Claude_Debussy/Estampes,_L._100/1._Pagodes/03238.mid' './03238.mid'
$PROG/midi2pianoroll 0 './03238'
$PROG/ScorePerfmMatcher 'score_hmm.txt' '03238_spr.txt' '03238_pre_match.txt' 1.0
$PROG/ErrorDetection 'score_fmt3x.txt' 'score_hmm.txt' '03238_pre_match.txt' '03238_err_match.txt' 0
$PROG/RealignmentMOHMM 'score_fmt3x.txt' 'score_hmm.txt' '03238_err_match.txt' '03238_match.txt' 0.3

echo 'aligning 03239...'
cp '/Users/toddywang/Documents/VsCodeProjects/xmltoexp/ATEPP-1.2/ATEPP-1.2/Claude_Debussy/Estampes,_L._100/1._Pagodes/03239.mid' './03239.mid'
$PROG/midi2pianoroll 0 './03239'
$PROG/ScorePerfmMatcher 'score_hmm.txt' '03239_spr.txt' '03239_pre_match.txt' 1.0
$PROG/ErrorDetection 'score_fmt3x.txt' 'score_hmm.txt' '03239_pre_match.txt' '03239_err_match.txt' 0
$PROG/RealignmentMOHMM 'score_fmt3x.txt' 'score_hmm.txt' '03239_err_match.txt' '03239_match.txt' 0.3

echo 'aligning 03240...'
cp '/Users/toddywang/Documents/VsCodeProjects/xmltoexp/ATEPP-1.2/ATEPP-1.2/Claude_Debussy/Estampes,_L._100/1._Pagodes/03240.mid' './03240.mid'
$PROG/midi2pianoroll 0 './03240'
$PROG/ScorePerfmMatcher 'score_hmm.txt' '03240_spr.txt' '03240_pre_match.txt' 1.0
$PROG/ErrorDetection 'score_fmt3x.txt' 'score_hmm.txt' '03240_pre_match.txt' '03240_err_match.txt' 0
$PROG/RealignmentMOHMM 'score_fmt3x.txt' 'score_hmm.txt' '03240_err_match.txt' '03240_match.txt' 0.3

echo 'aligning 03241...'
cp '/Users/toddywang/Documents/VsCodeProjects/xmltoexp/ATEPP-1.2/ATEPP-1.2/Claude_Debussy/Estampes,_L._100/1._Pagodes/03241.mid' './03241.mid'
$PROG/midi2pianoroll 0 './03241'
$PROG/ScorePerfmMatcher 'score_hmm.txt' '03241_spr.txt' '03241_pre_match.txt' 1.0
$PROG/ErrorDetection 'score_fmt3x.txt' 'score_hmm.txt' '03241_pre_match.txt' '03241_err_match.txt' 0
$PROG/RealignmentMOHMM 'score_fmt3x.txt' 'score_hmm.txt' '03241_err_match.txt' '03241_match.txt' 0.3

echo DONE
