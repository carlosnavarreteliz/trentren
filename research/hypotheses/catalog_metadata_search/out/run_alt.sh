#!/bin/zsh
D=/Users/cnavarreteliz/Documents/Repos/earthquakes/research/hypotheses/catalog_metadata_search
FP=/Users/cnavarreteliz/Documents/Repos/earthquakes/research/data/catalog_metadata_search/features_weekly.parquet
cd $D
for t in y_main y_all; do
  /opt/miniconda3/bin/python3 -I select_features.py $FP $t 2006 out > out/select_${t}_2006.log 2>&1
  /opt/miniconda3/bin/python3 -I final_alt.py $FP out $t 200 > out/final_alt2006_${t}.log 2>&1
done
echo ALTDONE >> out/final_alt2006_y_all.log
