# RESULTS: gnss_transients (GNSS transient deformation as a precursor)

**Verdict: NULL.** Trenchward GNSS velocity anomalies (30- or 90-day hinge against each station's own
2-year trend) add no out-of-sample skill over the ETAS/clustering baseline for M>=6 or M>=5.5
earthquakes in Chile, for either target (y_all or declustered y_main). The primary test was IG = +0.0025
nats/eq with phase-randomized surrogate p = 0.15. That is far from both alpha = 0.05 and the Bonferroni
level of 0.0036. Every robustness variant is also null. The part of the feature that does carry
information (training-period AUC 0.69 on y_all) is postseismic afterslip from earlier large quakes.
That is clustering in disguise: it disappears once postseismic periods are masked, and ETAS already
captures it.

## Data
- Nevada Geodetic Laboratory, IGS20 daily final `tenv3` files for 339 stations (lat -46.5..-16.5,
  lon -76..-64, >=365 solutions), 252 MB, downloaded 2026-10-17 to `research/data/gnss_transients/tenv3/`.
  Also used: NGL `steps.txt` (equipment and earthquake steps).
- 2 to 11 stations were active before 2002, 46 in 2007, and more than 150 from 2010. Of these, 256 lie
  west of -68.5 deg and were used. Each bin (+/-1 bin) had about 1 contributing station in 2007-08 and
  6-12 from 2013 onward.

## Method (see PREREGISTRATION.md)
- Each station and weekly issue time t is fitted by OLS on the days in [t-L-760, t-L), with L = 2 d.
  The model has a trend, annual and semiannual terms, and Heaviside offsets at the NGL steps known
  before t. It also includes a hinge `max(0, tau - (t-L-30d))`, which measures the change of velocity
  over the last 30 days. The fit is applied to the horizontal motion projected on azimuth 78 deg, and
  outliers above 5 robust sigma are removed.
- z = -c/se(c), where positive means trenchward acceleration. The per-bin feature is the median z over
  stations in bins b-1..b+1, and 0 where no station exists.
- Primary test: weekly M>=6 panel restricted to t_start >= 2007 (train 2007-2012, test 2013-2026-10).
  Target y_main, logit model. Significance comes from phase-randomized surrogates of the per-bin
  feature grid, using the same phases in all bins and restoring the missing-data mask, with n = 200.
- **Amendment (bug fix, disclosed):** v1 created duplicate step columns when NGL listed several steps on
  one day, which gave NaN for 32 % of station-weeks. v2 fixes this; the spec is otherwise unchanged.
  The v1 primary result was IG = +0.0041 with p = 0.11. Its y_all result was IG = -0.026 with p = 0.69.
  Both versions are null.

## Results (test 2013-01-01 .. 2026-10-17; IG in nats per target event)

`IG` is the full model minus the ETAS-logit baseline, which is the primary metric. `vs ETAS` is the full
model minus the pure additive ETAS. The restricted (2007+) logit baseline is trained on only 6 years and
is itself about 0.07-0.11 nats/eq worse than pure ETAS on test. The full-panel rows avoid that handicap.

| Config | Target | IG | 95% year-block CI | vs ETAS | Surrogate p (phase-rand) | other p | AUC base -> full | n test |
|---|---|---|---|---|---|---|---|---|
| **PRIMARY** restricted2007, weekly M6, z30 | **y_main** | **+0.0025** | [-0.004, +0.008] | -0.112 | **0.154** | bin-perm 0.109, circ-shift 0.129 | 0.638 -> 0.637 | 72 |
| restricted2007, weekly M6, z30 | y_all | -0.0012 | [-0.016, +0.012] | -0.070 | 0.343 | bin-perm 0.436, circ 0.317 | 0.677 -> 0.675 | 85 |
| full panel 1976+, weekly M6, z30 | y_main | -0.0015 | [-0.007, +0.004] | -0.0095 | 0.465 | | 0.665 -> 0.665 | 72 |
| full panel 1976+, weekly M6, z30 | y_all | +0.0010 | [-0.007, +0.008] | +0.0004 | 0.248 | | 0.697 -> 0.698 | 85 |
| restricted, GBM, z30 | y_main | +0.012 | [-0.065, +0.094] | -0.46 | n/a | | 0.626 -> 0.625 | 72 |
| restricted, GBM, z30 | y_all | -0.006 | [-0.073, +0.071] | -0.23 | n/a | | 0.670 -> 0.674 | 85 |
| z90 (90-day hinge) | y_main | +0.0034 | [-0.001, +0.008] | -0.111 | 0.129 | | 0.638 -> 0.638 | 72 |
| z90 | y_all | +0.0002 | [-0.001, +0.001] | -0.069 | 0.238 | | 0.677 -> 0.677 | 85 |
| abs(z30) | y_main | -0.0048 | [-0.016, +0.005] | | 0.446 | | | 72 |
| abs(z30) | y_all | -0.0029 | [-0.010, +0.004] | | 0.416 | | | 85 |
| z30, postseismic masked | y_main | -0.0104 | [-0.033, +0.008] | | 0.535 | | 0.638 -> 0.634 | 72 |
| z30, postseismic masked | y_all | -0.0252 | [-0.082, +0.020] | | 0.535 | | 0.677 -> 0.671 | 85 |
| z30, lag 14 d (product latency) | y_main | +0.0053 | [-0.002, +0.012] | | 0.129 | | 0.638 -> 0.639 | 72 |
| z30, lag 14 d | y_all | -0.0029 | [-0.010, +0.005] | | 0.386 | | | 85 |
| weekly M>=5.5, z30 | y_main | -0.0274 | [-0.049, -0.008] | -0.074 | 0.901 | | 0.655 -> 0.651 | 193 |
| weekly M>=5.5, z30 | y_all | -0.0080 | [-0.017, +0.001] | -0.041 | 0.673 | | 0.693 -> 0.692 | 228 |
| 30-day step/horizon M6, z30 | y_main | +0.0023 | [-0.002, +0.006] | -0.184 | 0.198 | | 0.638 -> 0.638 | 72 |
| 30-day M6, z30 | y_all | -0.0144 | [-0.039, +0.017] | -0.059 | 0.594 | | 0.658 -> 0.653 | 80 |
| 30-day M6, z90 | y_main | +0.0074 | [-0.001, +0.016] | -0.179 | 0.149 | | 0.638 -> 0.639 | 72 |
| split 2016 (train 2007-15) | y_main | -0.0115 | [-0.049, +0.027] | -0.077 | 0.644 | | 0.642 -> 0.629 | — |
| split 2016 | y_all | -0.0060 | [-0.035, +0.022] | -0.064 | 0.584 | | 0.650 -> 0.641 | — |

The primary used n = 200 surrogates and the variants used n = 100. No p-value falls below 0.05
anywhere, so a Bonferroni-level (n = 1000) run was not needed.

### Stand-alone diagnostics (no baseline, covered bin-weeks only; `descriptive.log`)
| feature | split | y_main AUC | y_all AUC |
|---|---|---|---|
| z30 | train 2007-12 | 0.548 (22 pos) | **0.694** (46 pos) |
| z30 | test 2013-26 | 0.505 (72) | 0.540 (85) |
| z90 | test | 0.543 | 0.587 |
| z30, postseismic masked | test | 0.454 (25) | 0.495 (28) |

The training-period y_all AUC of 0.69 comes from Maule 2010 afterslip. In the year after the
mainshock, coastal stations accelerate trenchward exactly when aftershocks occur. This is the expected
postseismic confound. It vanishes when stations within 2 years of an M>=6.5 step are masked, and it
adds nothing over ETAS.

### Event-level look (`event_look.log`; weekly issue time just before each event, bin of the epicentre)
| event | z30 at last issue | percentile among all bin-weeks | comment |
|---|---|---|---|
| Tocopilla 2007 M7.7 | +2.19 | 0.87 | 2 stations |
| Maule 2010 M8.8 | +3.05 | 0.92 | one station (ANTC, z = 6.8); z30 was at the 5th percentile 4 weeks earlier |
| Iquique 2014 foreshock M6.7 (03-16) | +1.31 | 0.77 | z90 +2.3; mild rise from late January |
| Iquique 2014 M8.2 (04-01) | +2.69 | 0.90 | rise after 03-16 is foreshock-sequence afterslip, already visible to ETAS |
| Illapel 2015 M8.3 | +0.53 | 0.65 | no signal (18 stations) |
| Chiloe 2016 M7.6 | +3.40 | 0.93 | elevated for about 6 weeks (4 stations) |
| Valparaiso 2017 M6.9 | -1.77 | 0.24 | the 2-day precursor of Ruiz et al. 2017 is shorter than the lag + weekly step |
| Coquimbo 2019 M6.7 | -0.72 | 0.41 | none |

Some large events sit in the upper decile (Maule, Iquique, Chiloe). However, a bin is in the upper
decile 10 % of the time by construction, and the systematic test shows that this does not translate
into skill. Weekly daily-position GNSS at NGL noise levels does not resolve the Iquique precursor
reported by Ruiz et al. 2014 and Socquet et al. 2017 as a forecastable anomaly. The bins near Iquique
had z30 between -0.6 and +1.9 for the 10 weeks before the foreshock.

## Leakage checks
1. **Causality:** each station fit uses only days < t - L. Steps are used only when dated before t - L.
   The 30-day panel uses the latest weekly value <= t_start. No centered windows are used.
2. **Reprocessing:** the NGL final series is reprocessed, not what was available in real time. The
   L = 14 d variant (final-product latency) gives the same null.
3. **Network densification:** the station count is strongly tied to recent big quakes (Spearman
   rho(nsta, log days since M6 nearby) = -0.37), so stations do get installed after large earthquakes.
   For that reason station count is not a feature and missing values are set to 0. The feature itself
   is only weakly correlated with station count (rho = -0.12) and with the clustering features
   (|rho| < 0.05).
4. **Postseismic masquerade:** confirmed as the only source of stand-alone skill (see above). The
   masked variant is null or negative.

## Caveats
- Coverage before 2007 is too sparse to train on, so the primary train period is only 2007-2012
  (22 y_main and 46 y_all positive cells). With a 6-year training window, the restricted logit baseline
  is about 0.1 nats/eq below pure ETAS. The full-panel rows, which use the long baseline, are also null.
- The OLS se ignores flicker noise, so z is inflated by about 3x (sd about 3). This is uniform across
  stations and does not matter after standardization, but z is not a calibrated probability.
- Only the horizontal motion projected on the convergence direction was tested. Vertical motion,
  strain-rate tensors from station triangles, sub-daily or 5-minute kinematic solutions, and lags under
  2 days (the scale of the Valparaiso 2017 precursor) were not. With only a handful of precursory
  episodes in 14 years, any such signal would remain statistically untestable at this panel resolution.
- Harness: no bug found. The surrogate test needs the full panel order, and restricted panels were
  built with `t_start >= 2007` filtering, which preserves it.

## Files
- `PREREGISTRATION.md`, `build_features.py` (station anomalies), `evaluate_gnss.py` (harness tests),
  `event_look.py`, `descriptive.py`, `download.sh`
- `out/results_primary.json`, `out/results_all.json`, `out/preds_primary_*.parquet`, `out_all.log`,
  `event_look.log`, `descriptive.log`, `out_v1_buggy_steps/` (pre-fix run)
- Data: `research/data/gnss_transients/{tenv3/, steps.txt, chile_stations.csv, derived/station_anoms.parquet}`

## Reproduce
```
D=/Users/cnavarreteliz/Documents/Repos/earthquakes/research/data/gnss_transients
cd /Users/cnavarreteliz/Documents/Repos/earthquakes/research/hypotheses/gnss_transients
NPROC=2 /opt/miniconda3/bin/python3 -I build_features.py $D $D/derived            # ~15 min
/opt/miniconda3/bin/python3 -I evaluate_gnss.py $D/derived/station_anoms.parquet ./out 200 all   # ~20 min
```
