#!/bin/zsh
# runs the picker: one process per (window, station), 1 torch thread each, niced; per-day cache allows restarts
D=/Users/cnavarreteliz/Documents/Repos/earthquakes/research/round2/waveform_pilot_iquique
M=/Users/cnavarreteliz/Documents/Repos/earthquakes/research/data/round2_waveform_pilot_iquique/mseed
O=/Users/cnavarreteliz/Documents/Repos/earthquakes/research/data/round2_waveform_pilot_iquique/picks
export OMP_NUM_THREADS=1 PICK_THREADS=1
for w in T2014 C2013 C2017; do for s in PSGCX PB11 PATCX; do
  nice -n 10 /opt/miniconda3/bin/python3 -I $D/pick.py $M $O $w $s 0.3 0.3 > $D/pick_${w}_${s}.log 2>&1 &
done; done
wait
echo ALLPICKSDONE
