# RESULTS: space_weather (solar / geomagnetic activity as an earthquake precursor in Chile)

**Verdict: NULL.** Solar-wind proton density (the Marchitelli et al. 2020 claim), geomagnetic storms (Kp, Dst),
IMF Bz, F10.7 and sunspot number add no forecasting information beyond the ETAS/clustering baseline, either for
all M>=6 events or for declustered mainshocks. Marchitelli's correlation also fails to replicate in its own,
non-forecast form on the global USGS catalog, including inside their own 1996-2016 study window.

The preregistration is in `PREREGISTRATION.md`, written before any test-period evaluation.

## Data obtained (cached in `research/data/space_weather/`)
| source | content | coverage |
|---|---|---|
| NASA SPDF OMNI2 hourly (`omni2/omni2_YYYY.dat`, 1973-2026) | Np, V, Bz GSM, Dst (Kyoto via OMNI), Kp, F10.7 | Np/V 70-99 % of days in 1973-82, **27-53 % in 1983-94** (IMP-8 gaps), >=88 % from 1995 to 2026-10-12 |
| GFZ Potsdam `Kp_ap_Ap_SN_F107_since_1932.txt` | Kp, Ap, SN, F10.7 (primary source for these) | complete to 2026-10-16 |
| SILSO `SN_d_tot_V2.0.csv` | daily sunspot number (cross-check) | r = 0.99999 with GFZ SN |
| USGS ComCat global M>=5.6, 1990-2026 (`usgs_global/`) | 14,207 events, used only for the Marchitelli replication | |

Cross-checks: Kp(GFZ) vs Kp(OMNI) r = 0.9998, F10.7(GFZ) vs F10.7(OMNI) r = 0.98.
Dst was not downloaded separately from WDC Kyoto: OMNI's Dst column is the Kyoto product.

## Primary test (preregistered): Marchitelli H1, daily M>=6 panel, logit
Feature `x_np_lag1` = log10 of the daily mean proton density on the UTC day ending at t_start. Surrogates are
AAFT phase-randomized versions of the raw daily series, re-aggregated, n = 200.

| target | N test cells | IG vs base (nats/eq) | 95 % year-block CI | IG full vs ETAS | surrogate p | AUC base / full | Molchan ASS base / full | coef (std) |
|---|---|---|---|---|---|---|---|---|
| y_all | 92 | **+0.0000** | [-0.0010, +0.0008] | -0.0010 | **0.33** | 0.742 / 0.742 | 0.742 / 0.742 | -0.004 |
| y_main | 72 | **+0.0017** | [-0.0028, +0.0069] | +0.024 (ETAS is weak on y_main) | **0.19** | 0.679 / 0.681 | 0.679 / 0.680 | +0.025 |

On y_all the fitted coefficient is essentially zero and has the wrong sign for Marchitelli's claim.

## Marchitelli in its original form (`run_epoch.py`, null = circular shift of Np by at least 1 year, n = 400)
Global daily count of M>=5.6 on day D was compared with log Np on day D-1.

| window | days | events | Spearman rho | p | rate ratio after top-10 % Np days | p |
|---|---|---|---|---|---|---|
| 1996-2016 (their window) | 7671 | 8332 | +0.004 | 0.44 | 0.91 | 0.94 |
| 2017-2026 (out of sample) | 3577 | 3490 | -0.001 | 0.54 | 1.00 | 0.46 |
| 1996-2026 | 11248 | 11822 | +0.000 | 0.52 | 0.94 | 0.91 |

A lag scan from -5 to +5 days found |rho| <= 0.026 at every lag.

A superposed-epoch analysis for Chile looked at mean log Np on days D-1, D-2, D-3 and D-7 before Chilean M>=6
events, using all events and ZBZ mainshocks, over 1976-2026 and 2013-2026. Every z score lies between -1.2 and
+0.6, and every p is >= 0.29. Before M>=6 events the proton density is, if anything, slightly lower than average.

## Secondary: Family B, weekly M>=6 panel, each feature alone (logit, AAFT surrogates n = 200)
| feature | IG y_all | CI y_all | p y_all | IG y_main | CI y_main | p y_main | coef sign |
|---|---|---|---|---|---|---|---|
| x_np7 (log Np, 7 d) | +0.0043 | [-0.026, +0.031] | 0.14 | +0.0035 | [-0.024, +0.031] | 0.19 | + |
| **x_v7 (solar-wind speed)** | **+0.0134** | [-0.008, +0.033] | **0.040** | **+0.0119** | [-0.011, +0.031] | **0.050** | **-** |
| x_bzs7 (southward Bz) | +0.0004 | [-0.002, +0.003] | 0.40 | +0.0001 | [-0.004, +0.005] | 0.43 | ~0 |
| x_kpmax7 (max Kp) | +0.0042 | [-0.006, +0.018] | 0.16 | +0.0003 | [-0.000, +0.001] | 0.32 | - |
| x_dstmin7 (min Dst) | +0.0025 | [-0.001, +0.006] | 0.19 | -0.0012 | [-0.003, 0.000] | 0.47 | ~0 |
| x_storm7 (Dst <= -50 nT) | +0.0013 | [-0.021, +0.022] | 0.27 | +0.0036 | [-0.014, +0.020] | 0.17 | **-** (storms precede *fewer* quakes) |
| x_f107_d81 (detrended F10.7) | -0.0091 | [-0.018, +0.001] | 0.85 | -0.0128 | [-0.033, +0.009] | 0.87 | - |
| x_ssn_d81 (detrended sunspots) | -0.0095 | [-0.019, -0.000] | 0.85 | -0.0090 | [-0.022, +0.005] | 0.83 | - |
| joint (all 8) | -0.0019 | [-0.046, +0.035] | - | -0.0081 | [-0.061, +0.039] | - | |

The baseline over Poisson is +0.221 (y_all) and +0.006 (y_main). Every AUC change is <= 0.005.

### The one nominal blip: solar-wind speed `x_v7`
- Its surrogate p is 0.040 on y_all and 0.050 on y_main. These are **not** significant after the preregistered
  within-family correction (0.05/8 = 0.006) or the project Bonferroni (0.0036). With 16 Family-B tests, about
  0.8 false hits at p < 0.05 are expected by chance.
- The effect is tiny: about 0.1 SD lower V in the week before M6 cells (mean 415 vs 423 km/s in test) and
  +0.013 nats/eq.
- Follow-ups (`run_v7_check.py`):
  - End-matched circular-shift p = 0.025 (y_all) and 0.045 (y_main).
  - The causally detrended version (7-day V minus the trailing 81-day mean) gives IG +0.008, p = 0.055 / 0.065.
  - GBM gives IG +0.001 / -0.008.
  - No single test year carries it: the minimum drop-one-year IG is +0.009.
  - The independent alternate split (train 1976-2000, test 2001-2012) gives the same sign: IG +0.016 / +0.011,
    p = 0.059 / 0.069.
  - **It is absent at M>=5.5 (weekly IG -0.001 / -0.001, p = 0.75 / 0.58)**, where there are 2.7 times more
    test events and power is higher.
  - It is also absent on the 30-day panel: IG +0.002 / +0.005, p = 0.29 / 0.25.
- Reading: this is most likely a multiple-comparison fluctuation. A reasonable physical effect would not vanish
  at M5.5 while appearing at M6, and its sign (slower wind before quakes) matches no proposed mechanism. It is
  worth at most a note for a future, independently preregistered test.

## Robustness (`run_robust.py`; surrogate n = 100 where a p is shown)
| check | IG y_all | IG y_main | p (y_all / y_main) |
|---|---|---|---|
| primary feature, GBM (daily M6) | +0.004 [-0.038, +0.036] | +0.024 [-0.008, +0.066] | n/a (GBM is noisy; the base AUC differs) |
| primary, training restricted to the SOHO/ACE era 1996-2012 | -0.004 | +0.004 | - |
| primary, daily M>=5.5 | +0.0000 | -0.0004 | - |
| primary, alternate split (test 2001-2012) | **-0.033** [-0.062, -0.004] | -0.016 | - |
| weekly M5.5: np7 / kpmax7 / dstmin7 / storm7 / v7 | -0.000 / -0.004 / -0.003 / -0.000 / -0.001 | -0.001 / -0.001 / -0.002 / -0.000 / -0.001 | all p >= 0.43 |
| 30-day M6: np30 / kpmax30 / dstmin30 / v30 | -0.004 / +0.002 / +0.002 / +0.002 | +0.000 / -0.008 / -0.002 / +0.005 | all p >= 0.24 |
| alternate split (train 1976-2000, test 2001-2012), weekly M6: np7 / kpmax7 / dstmin7 / storm7 / v7 | -0.005 / +0.002 / -0.006 / -0.011 / +0.016 | -0.006 / -0.000 / -0.005 / -0.007 / +0.011 | all p >= 0.25 except v7 (0.059 / 0.069) |

## Leakage checks
- All measured series are aligned with `H.align_daily_series`. Only fully elapsed UTC days before t_start are used:
  lag 0 for the daily primary, which is legitimate because real-time ACE/DSCOVR and nowcast Kp exist, and a
  1-day safety lag for the weekly features.
- Detrending uses trailing windows only (`causal_diff`). There are no centered windows and no test-period
  z-scores; standardization happens inside the harness on train only.
- Space weather is physically exogenous to Chilean seismicity, so reprocessed or definitive products cannot
  encode the quakes. The only residual risk is availability timing, which matters only if a signal exists, and
  none does.
- Missing solar-wind data (22 % of train days, mostly 1983-94; 0.9 % of test days) are imputed with the train
  mean by the harness. Training only on the SOHO era (1996-2012) does not change the result.
- Surrogates are AAFT: the harness `ft_surrogate` followed by a rank remap to the original marginal, so the heavy
  tails of Dst and Kp storms are preserved. They are rebuilt from the raw daily series through the same
  aggregation. The end-matched circular shift was used as a second null for `x_v7`. CALIBRATION_V2 (partial)
  shows phase_randomize FPR of 7 % on y_all and 0 % on y_main for noise features.

## Harness bug found (worked around, not modified)
`harness._baseline` caches the baseline by `(id(panel), len(panel), target, model)`. If a temporary panel copy
(for example one with a redefined `split`) is garbage-collected, a later panel of the same length can get the same
`id()` and silently reuse the stale baseline. This happened in my first robustness run: the daily M5.5 panel
returned the M6 / train-1996 baseline, with N_test = 92 instead of 252. The workaround is
`H._BASE_CACHE.clear()` whenever the panel changes (`run_robust.py`). All primary and Family-B runs used one live
panel each and were unaffected. Suggested fix: key the cache on the panel's content hash or a uuid stored in
`panel.attrs`.

## Power statement
With 72-92 test cells at M>=6, the year-block CIs for the primary feature are about +/-0.001-0.005 nats/eq.
That is very tight because the fitted coefficient is about 0. An effect larger than about +0.01 nats/eq from
proton density can be excluded on this panel. The global replication, with 11.8k events, excludes any
correlation |rho| > ~0.02.

## Reproduce
```
cd /Users/cnavarreteliz/Documents/Repos/earthquakes/research/hypotheses/space_weather
D=/Users/cnavarreteliz/Documents/Repos/earthquakes/research/data/space_weather
PY="/opt/miniconda3/bin/python3 -I"
$PY build_daily.py $D
$PY run_primary.py $D/space_weather_daily.csv out/primary_yall.json y_all 200
$PY run_primary.py $D/space_weather_daily.csv out/primary_ymain.json y_main 200
$PY run_family_b.py $D/space_weather_daily.csv out/famB_yall.json y_all 200
$PY run_family_b.py $D/space_weather_daily.csv out/famB_ymain.json y_main 200
$PY run_epoch.py $D/space_weather_daily.csv $D/usgs_global/global_m56_1990_2026.csv out/epoch.json 2000
$PY run_robust.py $D/space_weather_daily.csv out/robust.json 100
$PY run_v7_check.py $D/space_weather_daily.csv out/v7_check.json 200
```
Each primary surrogate run took about 25-45 minutes on the shared, overloaded machine.
