# PREREGISTRATION: space_weather

Written 2026-10-17, **before** any evaluation on the test period (2013-01-01 .. 2026-10-17) and before any
space-weather data were downloaded or inspected.

## Hypothesis
Solar / geomagnetic activity modulates the occurrence of large earthquakes in Chile, beyond what the
clustering (ETAS) baseline explains.

- **H1 (Marchitelli et al. 2020, Sci. Rep. 10:11495).** Peaks in solar-wind proton density precede large
  (M >= 5.6) earthquakes by about one day. Translated to a causal forecast: a high proton density on the UTC day
  that ends at the issue time raises the probability of an M >= 6 earthquake in the next day.
- **H2 (geomagnetic storms / solar activity).** Recent geomagnetic storms (high Kp, strongly negative Dst),
  solar-wind speed, southward IMF Bz, F10.7 and sunspot number raise the probability of an M >= 6 earthquake in
  the next week.

Prior expectation: null. Each feature is a global series, the same for all 29 latitude bins.

## Data (all treated as untrusted and cached in research/data/space_weather/)
- NASA OMNI2 hourly (SPDF low_res_omni): proton density Np, flow speed V, IMF Bz (GSM), plus Kp, Dst, F10.7 and
  R as bundled by OMNI. These are aggregated to UTC daily means, with min Dst and max Kp per day.
- GFZ Potsdam Kp/ap/Ap/SN/F10.7 file (`Kp_ap_Ap_SN_F107_since_1932.txt`), used as the primary source for Kp, Ap,
  sunspot number and F10.7 and as a cross-check of the OMNI values.
- WDC Kyoto Dst, taken via OMNI. If the Kyoto service can be reached it is used as a cross-check only.
- SILSO daily total sunspot number, used as a cross-check of the GFZ SN.

## Features (strictly causal, all from `H.align_daily_series`)
**Family A (H1), daily panel `get_panel(step_days=1, horizon_days=1, target_mag=6.0)`:**
- `x_np_lag1` = log10(daily mean Np) on the UTC day [t_start - 1 d, t_start). That is `window_days=1, lag_days=0`:
  the day has fully elapsed at issue time. This matches Marchitelli's 1-day lead.

**Family B (H2), weekly panel `get_panel(7, 7, 6.0)`, `lag_days=1`, `window_days=7`:**
- `x_np7` = mean log10 Np.
- `x_v7` = mean V.
- `x_bzs7` = mean of max(-Bz, 0).
- `x_kpmax7` = max of daily max Kp.
- `x_dstmin7` = min of daily min Dst.
- `x_storm7` = 1 if any daily min Dst <= -50 nT.
- `x_f107_d27` = F10.7 81-day mean minus the previous 81-day mean (a causal difference, because F10.7 is slow).
- `x_ssn_d27` = the same differenced construction for sunspot number.
Each Family B feature is tested **individually**. The joint model with all of them is reported as exploratory.

## PRIMARY TEST (decided now)
- **Primary:** Family A, `x_np_lag1`, daily M >= 6.0 panel, logit model, target **y_all**. This is
  Marchitelli's own setting: all events, not declustered.
- **Co-primary:** the same feature and panel on target **y_main** (ZBZ mainshocks).
- **Statistic:** `ig_per_event` (nats per target event, full minus the ETAS/clustering logit baseline) on the test
  period, 2013-01-01 .. 2026-10-17.
- **Significance:** `H.surrogate_test(builder=..., mode="phase_randomize")` on the RAW daily log10 Np series,
  re-aggregated exactly like the real feature, with n = 200. If p < 0.02, n is raised to 1000 to resolve the
  Bonferroni threshold.
- **Alpha:** 0.05 nominal. Project-level Bonferroni is 0.05/14 = 0.0036.
- **Verdict rule:** H1 is "supported" only if IG > 0 versus the baseline AND surrogate p < 0.0036 AND the
  leakage checks pass. If 0.0036 <= p < 0.05 it is "weak". Anything else is null.

## Secondary / robustness (reported, not used to overturn the primary verdict)
1. Family B, each feature alone, weekly M >= 6.0, y_all and y_main, phase_randomize builder surrogates
   (n = 200). Within Family B, an additional correction of 8 features is applied (alpha 0.05/8 at nominal level).
2. Family A feature aggregated to the weekly panel (`x_np7`) and to the 30-day panel (`window_days=30`).
3. M >= 5.5 versions of the primary test (daily and weekly).
4. The end-matched circular_shift surrogate as a second null for the primary test.
5. An alternate split: train 1976-2000, test 2001-2012. The real test period is excluded, so this is an
   independent replication window.
6. A superposed-epoch check: mean log Np on the day before each Chilean M >= 6 event (all, and ZBZ mainshocks)
   compared with a circular-shift null of the daily series. This tests Marchitelli's claim in its original,
   non-forecast form.
7. GBM model for the primary feature.

## Leakage considerations (decided now)
- Space-weather series are physically exogenous to Chilean seismicity, so reprocessing cannot encode the quakes.
  The remaining risk is real-time availability. OMNI/definitive Dst and Kp are published with delay. Real-time
  equivalents (ACE/DSCOVR RTSW, nowcast Kp, quick-look Dst) exist, so values are treated as known at the end of
  the day for Family A, and a 1-day safety lag is used for Family B.
- No centered windows, no interpolation across t_start, no z-scoring with test-period statistics. The logit
  standardization inside the harness uses train-period statistics only.
