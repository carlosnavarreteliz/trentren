# RESULTS: waveform_pilot_iquique (round 2, descriptive pilot)

**Verdict: `inconclusive_data`**, as preregistered for a descriptive pilot. No information gain (IG) vs ETAS was computed (see PREREGISTRATION.md for why).
In substance, it is a **null for the added value of the waveform micro-catalog**. The target period differs from the two controls on several statistics. But every difference comes from seismicity that the routine catalog (ComCat M>=4) already shows, and it starts in the same 10-day block. Once micro-events near catalogued M>=4 events are removed, nothing stands out.

## What was run (preregistered pipeline, unchanged)
- IPOC/CX (GEOFON) BH? 20 Hz at PSGCX, PB11 and PATCX. Three windows of 101 days each: T2014 (2013-12-22..2014-04-01 23:46), C2013 and C2017. Uptime was 100 % at all 3 stations in all windows. The download was 5.46 GB.
- Picker: SeisBench PhaseNetLight with `stead` weights. These are not the `iquique` weights, which were trained on the aftershocks. The run gave 196.7k picks in T2014, 162.1k in C2013 and 154.0k in C2017. The resumed run took about 5 s per station-day on 1 thread, roughly 1.5 CPU-h in total.
- Association: PyOcto with a 0D model, keeping events with >=3 stations. This gave 10,551, 7,361 and 7,200 events (T, C2013, C2017). In box R (lat -21..-18.5, lon -71.6..-69.8, depth <=60 km) the counts were 3,522, 576 and 737.
- Magnitude proxy: Wood-Anderson ML, with one offset calibrated on C2013 only. ComCat recall was 0.92, 0.86 and 0.98. Mc_common = 1.7.

## Preregistered statistics (PRE = before the analog of the M6.7 foreshock of 2014-03-16; R box; M>=1.7)
| Stat | T2014 | C2013 | C2017 | Preregistered rule |
|---|---|---|---|---|
| S1 rate (ev/day), **primary** | **3.22** | 1.50 | 2.11 | distinguishable (gap 1.11 > spread 0.61) |
| S1 vs 16 control 10-day blocks | z = 2.63, rank 0/16 | | | 3 of 8 T blocks > control max |
| S2 acceleration (last 30 d / earlier) | 0.64 (deceleration) | 0.87 | 0.85 | "distinguishable", but in the opposite direction to the precursor hypothesis |
| S3 Spearman(time, dist to 2014 hypocentre) | -0.40 | 0.02 | 0.03 | distinguishable (a hindsight statistic) |
| S3 shift in median distance (km) | -46 | -0.2 | +4.7 | distinguishable (hindsight) |
| S4 b (proxy) | 0.51 +/- 0.03 | 0.69 +/- 0.05 | 0.72 +/- 0.05 | distinguishable |
| S5 fraction in repeater-like families | 0.037 | 0.063 | 0.061 | distinguishable (T lower) |
| S6 tremor-like hours/day | 0 | 0 | 0 | no |
| S0 enhanced / ComCat (R) | 12.4 | 127 | 35.8 | no |
| Contrast T/C, enhanced S1 | 2.15 (vs C2013), 1.53 (vs C2017) | | | |
| Contrast T/C, routine ComCat (R) | 22.0 (vs C2013), 4.4 (vs C2017) | | | **ComCat contrast > enhanced contrast** |

At the FULL cutoff (including the M6.7 and its aftershocks) the S1 rate is 20.6/day vs 1.6 and 2.2. That is the foreshock sequence and has no in-advance value.

**Preregistered S0 criterion:** "If the enhanced T/C contrast is no larger than the ComCat contrast, the micro-catalog adds nothing beyond the routine catalog." The enhanced contrast is 1.5-2.2x and the ComCat contrast is 4.4-22x, so **the micro-catalog adds nothing**.

## Robustness and leakage (exploratory, run after the preregistered analysis)
- **R1, clustering in disguise.** Micro-events within 100 km and from -1 to +2 days of a ComCat M>=4 event were removed. The S1 PRE rates become T 1.32, C2013 1.45 and C2017 1.77 per day, so the target is the *lowest*. With a +7-day mask they are 0.79, 1.28 and 1.52. The excess rate is the swarm already visible in the routine catalog. The test is conservative against T, because T has more ComCat events and therefore more masked space-time.
- **R2, no lead time.** The first 10-day block where the enhanced rate exceeds the control maximum is 2014-01-01. That is also the first block where ComCat M>=4 in R exceeds the control maximum (8 events, the January 2014 swarm). The micro-catalog saw nothing before the routine catalog.
- **R3, the routine catalog alone already flags the period.** T had 21 ComCat M>=4 events in R before the M6.7. That exceeds every one of 677 85-day windows from 2001 to 2013 (maximum 16, median 2). Caveat: ComCat is not homogeneous over that period. The anomaly was visible without waveforms, so any value lies in what the routine catalog plus ETAS already capture (round 1: null).
- **R6, b-value.** After the same mask, b(T) rises to 0.78 (2 d mask) or 0.83 (7 d), against 0.67/0.66 (C2013) and 0.75/0.79 (C2017). The low b comes from the clustered M4-5 events of the swarm and is not a background change. The slope of ComCat magnitude on the proxy is 0.45-0.78, below 1, so the absolute b values are not physical. Only relative comparisons are valid.
- **R4 and R5, detection homogeneity.** The same 3 stations, all installed before 2009, were used with 100 % uptime. The median daily P picks per station are similar (PB11: 242, 210, 224). b(T) is stable with each window's own offset and with Mc = 1.7/2.0/2.3 (0.51/0.51/0.49).
- **Leakage audit.** The picker weights are not trained on Iquique 2014. No station was installed after 2013. Data stop at 23:46:00, before the M8.2 origin. The migration statistic (S3) uses the future hypocentre and is hindsight only. The C2013 control precedes the M8.2 by about 12 months and could be "pre-preparatory". C2017 contains late postseismic activity. The ComCat used for calibration and masking is the current, revised catalog.

## Is T distinguishable from the controls *in advance*?
On raw statistics, yes: rate, migration toward the future hypocentre and low b. Against the right reference, no. All of it is the January-March 2014 swarm, which the routine catalog already recorded (R1-R3), and the micro-catalog gives neither lead time nor a contrast larger than ComCat. The 30 days before the M6.7 are *decelerating* (S2 = 0.64). Two controls cannot estimate the false-alarm rate of swarms that are not followed by a mainshock, and northern Chile has had such swarms. The best significance attainable with this design is p ~ 1/3.

## What a Chile-wide study would need
- **Data:** about 6 MB per station-day at 20 Hz. 60 broadband stations (CX + C1 + C) x 15 years is about 330k station-days, or about 2 TB.
- **Compute:** at about 5-6 s per station-day on 1 CPU thread, that is about 500 CPU-h (a few GPU-hours).
- **Pipeline:** 3D or 1D velocity model association with GaMMA or PyOcto, a calibrated local magnitude, and checks of homogeneity against network densification. Densification is critical, because C1 grew strongly after 2012.
- **Statistics:** use a micro-catalog ETAS as the baseline and test against it, not against ComCat ETAS. There are only about 6-10 M>=7 targets since 2007, so power to beat ETAS at alpha 0.0083 would be very low. Realistically it would take a multi-region effort (Japan or Chile plus others) or targets of M>=6.

## Files
- Scripts: `download.py`, `pick.py`, `associate.py`, `measure.py`, `analyze.py`, `robustness.py`, `robust_b.py`, `run_picks2.sh`
- Outputs: `out/results.json`, `out/robustness.json`, `out/robust_b.json`, `out/daily_series.json`, `out/fig_windows.png`, `summary.json`
- Data: `research/data/round2_waveform_pilot_iquique/{mseed,picks,assoc,meas}`
