#!/bin/bash
# usage: download.sh <stalist> <outdir>
while read s; do
  f="$2/$s.tenv3"
  [ -s "$f" ] && continue
  curl -s --compressed -m 120 -o "$f.part" "https://geodesy.unr.edu/gps_timeseries/IGS20/tenv3/IGS20/$s.tenv3" && mv "$f.part" "$f"
  sleep 0.3
done < "$1"
