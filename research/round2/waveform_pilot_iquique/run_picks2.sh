#!/bin/zsh
# resume picker (per-day cache), 4 concurrent jobs with 1 torch thread each, niced
D=/Users/cnavarreteliz/Documents/Repos/earthquakes/research/round2/waveform_pilot_iquique
M=/Users/cnavarreteliz/Documents/Repos/earthquakes/research/data/round2_waveform_pilot_iquique/mseed
O=/Users/cnavarreteliz/Documents/Repos/earthquakes/research/data/round2_waveform_pilot_iquique/picks
export OMP_NUM_THREADS=1 PICK_THREADS=1
for w in T2014 C2013 C2017; do for s in PSGCX PB11 PATCX; do echo $w $s; done; done | \
  xargs -P 4 -n 2 zsh -c 'nice -n 10 /opt/miniconda3/bin/python3 -I '$D'/pick.py '$M' '$O' $0 $1 0.3 0.3 > '$D'/pick_$0_$1.log 2>&1'
echo ALLPICKSDONE
