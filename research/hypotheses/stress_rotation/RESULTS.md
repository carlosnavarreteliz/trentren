# RESULTS: stress_rotation (GCMT focal-mechanism stress rotation, heterogeneity, thrust fraction, CLVD)

**Verdict: NULL.** Focal-mechanism stress-state features add no out-of-sample information beyond the
ETAS/clustering baseline. On the preregistered primary test the point estimate is negative
(IG = -0.051 nats/eq). The phase-randomization p is 0.93 (n=1000) and the mechanism-shuffle p is 0.89.
No variant comes near p < 0.05.

## Data
- GCMT NDK files: `jan76_dec20.ndk`, the monthly files 2021-01..2026-05 (final solutions) and `qcmt.ndk`
  (quick solutions, used only after 2026-05-31; just 1 regional event). All are cached in
  `research/data/stress_rotation/`.
- `parse_gcmt.py` turned these into 2,970 events at lat -48..-15 and lon -80..-64, 1976-2026.
  - Two malformed blocks in the 1976-2020 file are not in the regional set: one is outside the region and
    one has a fused field.
  - Event density rises strongly over time: about 25 per year before 1985 and about 100 per year after 2010.
- The features are built only from shallow events (centroid depth <= 70 km), in bins b±1, over a recent
  window of 730 d with a 2 d availability lag, compared against the long-term history before that window.
- The features follow PREREGISTRATION.md: `x_cons` (Kostrov consistency), `x_rot` (tensor angle between the
  recent and long-term summed mechanisms), `x_dthrust` (change in thrust fraction) and `x_dclvd` (change in
  mean |eps|).
- About 44-47 % of cells are NaN because they have fewer than 3 recent events. The harness imputes these
  with the train mean.

## Primary test (preregistered): weekly M>=6.0, y_main, logit, 4 features jointly
| quantity | value |
|---|---|
| N test cells positive | 72 |
| IG(full - base) | **-0.0508 nats/eq** |
| year-block bootstrap 95 % CI | [-0.124, +0.016] |
| IG(full - ETAS additive) | -0.059 |
| AUC base / full | 0.665 / 0.627 |
| phase_randomize p (n=1000) | **0.931** (null mean -0.015, q95 +0.020) |
| mechanism-shuffle p (n=200) | 0.886 (null mean -0.022, q95 +0.013) |
| bin_permute p (n=200) | 0.776 |
| Rhoades T (unreliable) | -1.54 |
| standardized coefs | x_cons -0.12, x_rot +0.04, x_dthrust -0.02, x_dclvd -0.20 |

The logit fitted on 1976-2012 learned that lower CLVD and lower consistency come before mainshocks. These
in-sample associations did not carry over to 2013-2026, and adding them makes the test likelihood worse.

## Secondary and robustness (phase_randomize, n=200 unless noted)
| variant | target | N | IG full-base (nats/eq) | 95 % CI (year boot) | phase p |
|---|---|---|---|---|---|
| weekly M6 (primary set) | y_all | 85 | -0.017 | [-0.045, +0.010] | 0.72 (mech-shuffle 0.70, bin_permute 0.54) |
| weekly M5.5 | y_main | 193 | -0.005 | [-0.015, +0.002] | 0.50 |
| weekly M5.5 | y_all | 228 | -0.0003 | [-0.003, +0.002] | 0.21 |
| 30-day M6 | y_main | 72 | -0.033 | [-0.092, +0.018] | 0.74 |
| 30-day M6 | y_all | 80 | -0.015 | [-0.042, +0.012] | 0.66 |
| weekly M6, recent window 365 d | y_main | 72 | -0.031 | [-0.096, +0.031] | 0.84 |
| weekly M6, recent window 365 d | y_all | 85 | -0.013 | [-0.049, +0.027] | 0.55 |
| weekly M6, only Mw>=5.5 (real-time quick-CMT set) | y_main | 72 | -0.052 | [-0.109, +0.020] | 0.83 |
| weekly M6, only Mw>=5.5 | y_all | 85 | -0.028 | [-0.055, +0.009] | 0.76 |
| weekly M6, lag 180 d | y_main | 72 | -0.063 | [-0.119, -0.010] | 0.93 |
| weekly M6, lag 180 d | y_all | 85 | -0.050 | [-0.094, -0.006] | 0.96 |
| weekly M6, GBM (n=100) | y_main | 72 | +0.0007 | [-0.064, +0.066] | 0.22 |
| weekly M6, GBM (n=100) | y_all | 85 | -0.002 | [-0.049, +0.040] | 0.31 |
| alt split: train 1976-2004, test 2005-2012 | y_main | 34 | -0.017 | [-0.157, +0.088] | 0.33 |
| alt split: train 1976-2004, test 2005-2012 | y_all | 58 | -0.028 | [-0.095, +0.035] | 0.58 |
| EXPLORATORY: + Michael stress-inversion misfit | y_main | 72 | -0.076 | [-0.152, -0.005] | 0.95 |
| EXPLORATORY: + Michael misfit | y_all | 85 | -0.022 | [-0.048, +0.005] | 0.69 |
| EXPLORATORY: misfit only | y_main | 72 | -0.046 | [-0.075, -0.018] | 0.97 |
| EXPLORATORY: misfit only | y_all | 85 | -0.006 | [-0.011, -0.001] | 0.59 |
| daily M6 (phase n=50; n=200 run aborted under machine load) | y_main | 72 | -0.041 | [-0.110, +0.022] | 0.94 |

Single features on weekly M6 (IG vs base, descriptive, no surrogates):

| feature | IG y_main | IG y_all |
|---|---|---|
| x_cons | -0.011 | -0.001 |
| x_rot | -0.006 | -0.000 |
| x_dthrust | +0.005 | -0.000 |
| x_dclvd | -0.031 | -0.015 |

None of the single features is distinguishable from zero.

## Leakage checks
- **Brute force.** 300 random (bin, t_start) rows were recomputed by brute force. There were 0 mismatches,
  and every event used has time < t_start - 2 d.
- **Real-time availability.** Final GCMT solutions appear months after the event. The Mw >= 5.5 variant
  (events that have quick CMTs within about a day) and the 180 d lag variant both give the same null.
  Leakage cannot have inflated anything: the result is null or negative throughout.
- **Network densification.** The GCMT detection threshold drops over time. All features are normalised
  (ratios, angles, anomalies relative to the long-term mean), and the mechanism-shuffle null keeps the
  timing, location and magnitude of every CMT event. It showed no mechanism-specific signal.
- **Features use only `bg`/raw events.** No harness `mainshock` flag and no target-derived quantity is used.

## Interpretation
GCMT samples the Chilean margin too sparsely: about 10 shallow events per 3° latitude in 2 years, and about
45 % of cells have fewer than 3. With that coverage, mechanism-based measures of stress-state rotation or
heterogeneity carry no detectable prospective information about where or when M>=5.5-6 earthquakes happen,
once clustering is accounted for. Retrospective case studies report stress or mechanism changes before
Iquique 2014 and Illapel 2015. Those changes do not turn into a forecast that beats ETAS when the features
are fixed in advance and evaluated out of sample. The test resolves about +0.05 nats/eq. A small effect
below that, or a signal visible only in a denser local catalog with M~3 mechanisms (for example CSN/IPOC
moment tensors), cannot be excluded.

## Files
- `parse_gcmt.py`, `features.py`, `check_features.py`, `run_primary.py`, `run_robust.py`
- `primary.json` / `primary.log`, `robust.json` (all robustness rows), `robust_main.log`, `robust_daily.log`

Reproduce:
```
cd research/hypotheses/stress_rotation
D=/Users/cnavarreteliz/Documents/Repos/earthquakes/research/data/stress_rotation
/opt/miniconda3/bin/python3 -I parse_gcmt.py $D $D/gcmt_chile.csv
/opt/miniconda3/bin/python3 -I run_primary.py $D/gcmt_chile.csv primary.json 1000 200
/opt/miniconda3/bin/python3 -I run_robust.py $D/gcmt_chile.csv robust.json 200   # daily row was run with: ... robust.json 50 daily_m6
```
