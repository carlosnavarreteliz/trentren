#!/bin/zsh
# Sequential robustness runs (event-time-shift surrogates, smaller n). Args: none.
P=/opt/miniconda3/bin/python3
D=/Users/cnavarreteliz/Documents/Repos/earthquakes/research/data/teleseismic_triggering
C="$D/global_m65_1973.csv $D/global_m65_2000.csv"
cd /Users/cnavarreteliz/Documents/Repos/earthquakes/research/hypotheses/teleseismic_triggering
for t in y_all y_main; do
  $P -I run_primary.py ${=C} h1 $t 200 0 7 7 5.5 1000 logit
  $P -I run_primary.py ${=C} h1 $t 200 0 7 7 6.0 2000 logit
  $P -I run_primary.py ${=C} h1 $t 200 0 30 30 6.0 1000 logit
  $P -I run_primary.py ${=C} h1 $t 100 0 7 7 6.0 1000 gbm
  $P -I run_primary.py ${=C} h2diff $t 200 0 7 7 6.0 1000 logit
  $P -I run_primary.py ${=C} h1 $t 100 0 1 1 6.0 1000 logit
done
$P -I run_altsplit.py ${=C} 200
echo CHAIN_DONE
