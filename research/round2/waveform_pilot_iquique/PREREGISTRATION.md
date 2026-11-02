# PREREGISTRATION: waveform_pilot_iquique (round 2)

Written 2026-10-17, **before** any pick, association or statistic was computed on the target or control
windows. The only data touched so far: one test station-day (PSGCX 2014-03-29) used to benchmark picker
speed and check the preprocessing (pick counts only, no catalog statistics).

## Question
Using a deep-learning picker on continuous waveforms (an "enhanced" micro-catalog), was the 4-month period
before the 2014-04-01 Iquique M8.2 distinguishable **in advance** from control periods of equal length at the
same stations, beyond control-to-control variability? This is a **descriptive pilot**; it also estimates what
a Chile-wide study would need.

## Why the harness test is not the primary test here
The harness evaluates a feature defined on the whole 1976-2026 weekly panel (train <2013, test 2013+). A waveform
catalog for 3 windows x ~100 days at 3 stations cannot populate that panel, so `H.evaluate`/`surrogate_test`
cannot be run meaningfully. IG vs ETAS is therefore **not computed** (reported as N/A). Per the verdict guide,
a descriptive pilot gets `inconclusive_data` regardless of how the comparison comes out.

## Data (fixed before looking)
- Network CX (IPOC), served by GEOFON FDSN (CX is not at IRIS/EarthScope; checked). Channels BH? (20 Hz, STS-2).
- Stations: PSGCX, PB11, PATCX (the three CX stations closest to the 2014 source; all operating since <=2008;
  no later-installed station such as PX02/PX03 or C1.* is used, to avoid network densification).
- Windows (101 days each, same calendar span, last day cut at 23:46:00 UTC):
  - T2014 target: 2013-12-22 .. 2014-04-01 23:46 (mainshock origin 23:46:47; no aftershocks).
  - C2013 control: 2012-12-22 .. 2013-04-01 23:46 (no M>=7 nearby in the following year? NO: Iquique follows
    ~12 months later; kept as the task suggests, flagged as possibly "pre-preparatory").
  - C2017 control: 2016-12-22 .. 2017-04-01 23:46 (no M>=7 in N Chile afterwards; 2.7 yr after Iquique, so it
    contains late postseismic activity).
- Window length was reduced from the suggested 4 months to 101 days only because 3 stations x 3 windows x 121 d at
  ~6.3 MB per station-day exceeds the 6 GB download cap. The cut keeps the January 2014 swarm.

## Pipeline (fixed)
1. Picker: SeisBench `PhaseNetLight`, pretrained `stead` weights (v2, peak norm). NOT the `iquique` weights:
   those were trained on Iquique 2014 aftershock waveforms (knowledge of later events).
   Preprocessing: bandpass 1-9 Hz at 20 Hz, resample x5 to 100 Hz, 30 s windows, stride 25 s, blinding 2.5 s.
   Thresholds P = 0.30, S = 0.30, peaks >= 1 s apart. Same settings in every window.
2. Association: PyOcto with a homogeneous (0D) model vp = 6.8 km/s, vs = 3.9 km/s, tolerance 1.5 s;
   volume lat -22.5..-17.5, lon -72.5..-68.0, depth 0..200 km; require >= 4 picks, >= 3 stations,
   >= 1 station with both P and S. Same in all windows.
3. Magnitude proxy: peak horizontal amplitude after Wood-Anderson simulation (STS-2 approximated as flat
   velocity response with the station sensitivity from metadata) and Hutton & Boore (1987):
   ML = log10(A_mm) + 1.11 log10(R) + 0.00189 R - 2.09, median over stations. One additive constant is
   calibrated against ComCat M>=4 events matched in C2013 **only**, then applied to all windows.
4. Station uptime: a day counts as "valid" if all 3 stations have >= 90 % usable windows. Rates are per valid day.

## Statistics (each computed for all windows)
Source region R: lat -21.0..-18.5, lon -71.6..-69.8, depth <= 60 km (interface + forearc around the 2014
rupture). Reference point for migration: the Iquique hypocenter (-19.610, -70.769), used for every window.
Common completeness Mc_common = max over windows of (max-curvature Mc + 0.2) for R events.
Two cutoffs: **PRE** = up to day-of-year analog of 2014-03-16 21:00 UTC (before the M6.7 foreshock; the honest
"in advance" test), and **FULL** = up to 04-01 23:46.

- S1 (PRIMARY): uptime-corrected rate of R events with M >= Mc_common, PRE cutoff.
- S2 acceleration: rate in the last 30 days before cutoff / rate in the earlier part of the window.
- S3 migration: Spearman rho between time and epicentral distance to the reference point (R events,
  M >= Mc_common); and median distance in the last 30 days minus earlier.
- S4 b-value: Aki-Utsu MLE above Mc_common (0.1 bins), bootstrap SE; whole window and last 30 days.
- S5 repeating-like families: R events with M >= Mc_common, PSGCX vertical 1-8 Hz, window P-1 s .. P+11 s,
  pairs with epicentral separation <= 30 km, linked if max normalized CC >= 0.90; report fraction of events in
  families and number of families (>= 2 members).
- S6 tremor-like hours: hours where the hourly median 2-8 Hz envelope exceeds 3x its window median at all three
  stations simultaneously and no associated event occurs in that hour; per valid day.
- S0 enhancement: enhanced events / ComCat events (same region/time) and the T/C ratio of rates in ComCat M>=4
  (the routine catalog) vs the enhanced catalog. If the enhanced T/C contrast is no larger than the ComCat contrast,
  the micro-catalog adds nothing beyond the routine catalog.

## Decision rule (descriptive)
For each statistic, the target is called "distinguishable from controls" only if its PRE value lies outside
[min(C2013, C2017)] .. [max(...)] **and** its distance to the nearest control exceeds |C2013 - C2017|
(control-to-control spread). As an additional dispersion check, the PRE-period S1 is compared with the
distribution of 10-day-block rates within the controls (z-score and empirical rank).
With two controls no p-value below 1/3 is attainable; nothing here can meet alpha = 0.0083.

## Leakage / bias checks to report
- No picker weights trained on 2014 Iquique data; no stations installed after 2013; no aftershocks of the M8.2.
- Detection-rate differences from station noise/uptime (report uptime and daily pick counts per station).
- The FULL-cutoff target contains the M6.7 foreshock and its aftershocks: an ETAS-normal sequence that is
  "clustering in disguise". The PRE cutoff is the honest test.
- The reference point (future hypocenter) is known only in hindsight; migration toward it is a hindsight
  statistic and is reported as such.

## Amendment (2026-10-17, before any statistic was computed)
The ML constant written above (-2.09) belongs to the IASPEI form with A = WA-simulated ground displacement in nm.
For A = Wood-Anderson trace amplitude in mm the equivalent constant is +0.591 (= Hutton & Boore 3.0 at 100 km).
This only shifts all magnitudes by a constant, which the C2013 offset calibration absorbs anyway.
