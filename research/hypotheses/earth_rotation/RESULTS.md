# RESULTS: earth_rotation (LOD deceleration, pole tide, Chandler wobble)

**Verdict: NULL.** Earth-rotation variables add no out-of-sample forecasting skill for Chilean M>=6 (or
M>=5.5) earthquakes beyond the ETAS/clustering baseline.
- **H1 (Bendick & Bilham LOD deceleration, 5-yr lag):** IG is about +0.003 nats/eq, which is noise
  (surrogate p = 0.24 on y_main and 0.14 on y_all).
- **H2 (pole tide):** in the training period there was a nominal preference for mainshocks during
  *rising* pole-tide radial displacement (event-level Schuster p = 0.009, in-sample). It did not
  replicate in 2013-2026. The test-period mean phase is about 130° away, and the fitted feature makes
  test forecasts *worse* (IG = -0.051 nats/eq, CI excluding 0). That degradation is still within the
  surrogate null (p = 0.91), so this is a non-replicating in-sample fluctuation, not an anti-signal.

Preregistration: `PREREGISTRATION.md`, written before any test-period evaluation.

## Data
- IERS EOP 20 C04 daily series, 1962-01-01 to 2026-09-17, from
  `hpiers.obspm.fr/iers/eop/eopc04/eopc04.1962-now`, cached at
  `research/data/earth_rotation/eopc04_1962-now.txt`.
- The data was obtained successfully and has no gaps.
- Features use only days D < t_start - 30 d, which matches the C04 publication lag.

## Features (all causal; `features.py`)
- **H1:** `x_lod_decel_lag5y` is the annual change in the trailing 365-d mean LOD, taken 5 years
  before (t_start - 30 d).
- **H2:** `x_pt` and `x_pt_rate`, per bin.
  - Polar motion is fitted by least squares on the last 6 years of data available at the issue time,
    with constant + trend + annual + Chandler terms.
  - The oscillatory part is extrapolated to the forecast-window midpoint and turned into IERS-2010
    pole-tide radial displacement (mm) and its rate (mm/yr) at each bin latitude and lon -71.5°.
  - Extrapolated vs actual pole tide correlates at r = 0.95. The rms prediction error is 1.0 mm on a
    3.3 mm signal.
- **Exploratory:**
  - `x_lod_decel_lag0`: the same as H1 with no lag.
  - `x_lod365_anom`: 365-d LOD minus its trailing 10-yr mean, i.e. a detrended level.
  - `x_cw_amp`: the Chandler amplitude from the causal fit.
  - An event-level pole-tide phase test.

## Primary results
Weekly panel, M>=6.0, logit, test 2013-01-01 to 2026-10. All surrogate tests use n=200.
- IG is (full - ETAS-logit baseline) in nats per target cell.
- "IG vs pure ETAS" is the full model vs the additive MLE ETAS.
- p is the phase-randomized surrogate p. H1 uses an FT surrogate of the raw daily LOD, rebuilt through
  the same pipeline; H2 uses the panel feature grid.

| Hypothesis | Target | N test | IG vs baseline | 95% CI (year bootstrap) | IG vs pure ETAS | Surrogate p (phase) | Surrogate p (circ. shift) | AUC base → full | Molchan ASS base → full |
|---|---|---|---|---|---|---|---|---|---|
| H1 LOD decel lag 5y | **y_main (primary)** | 72 | +0.0027 | [-0.007, +0.013] | -0.005 | **0.244** | 0.194 | 0.665 → 0.667 | 0.664 → 0.666 |
| H1 LOD decel lag 5y | y_all | 85 | +0.0022 | [-0.006, +0.011] | +0.002 | 0.144 | 0.075 | 0.697 → 0.699 | 0.696 → 0.698 |
| H2 pole tide (level + rate) | **y_main (primary)** | 72 | **-0.0506** | [-0.087, -0.009] | -0.059 | **0.906** | 0.905 | 0.665 → 0.640 | 0.664 → 0.639 |
| H2 pole tide (level + rate) | y_all | 85 | -0.0312 | [-0.064, +0.005] | -0.032 | 0.871 | 0.821 | 0.697 → 0.681 | 0.696 → 0.681 |

- None of the tests come close to p < 0.05, let alone the Bonferroni threshold of 0.0036 (or 0.0018
  within this agent). Since no p fell below 0.02, the preregistered n=1000 rerun was not triggered.
- Surrogate null distributions:
  - H1 y_main: mean -0.009, sd 0.023, 95th percentile +0.015.
  - H2 y_main: mean -0.016, sd 0.023, 95th percentile +0.011.
- The observed H2 degradation is therefore about 1.5 null standard deviations below the null mean.
  Adding two noise columns typically costs about 0.015 nats/eq through overfitting, and the pole tide
  costs more than that.

## Robustness (n=200 phase surrogates unless noted)

| Variant | Target | IG vs baseline | 95% CI | p |
|---|---|---|---|---|
| H1, GBM | y_main / y_all | +0.009 / -0.004 | wide, includes 0 | — |
| H2, GBM | y_main / y_all | -0.010 / -0.055 | includes 0 | — |
| H1, weekly M>=5.5 | y_main / y_all | -0.005 / -0.003 | includes 0 | 0.86 / 0.82 |
| H2, weekly M>=5.5 | y_main / y_all | -0.004 / -0.003 | includes 0 | 0.80 / 0.71 |
| H1, 30-day panel M>=6 | y_main / y_all | +0.002 / +0.001 | includes 0 | 0.24 / 0.19 |
| H2, 30-day panel M>=6 | y_main / y_all | -0.047 / -0.043 | [-0.08, -0.007] / [-0.08, 0.00] | 0.92 / 0.87 |
| H1, alt split (train ≤2005, test 2006-2026) | y_main / y_all | +0.0065 / +0.0035 | [+0.0003, +0.013] / [-0.0001, +0.007] | 0.085 / 0.065 |
| H2, alt split | y_main / y_all | -0.045 / -0.024 | [-0.079, -0.009] / [-0.051, +0.002] | 0.88 / 0.81 |
| EXP `x_lod_decel_lag0` | y_main / y_all | +0.003 / +0.000 | includes 0 | 0.25 / 0.28 |
| EXP `x_lod365_anom` (detrended level) | y_main / y_all | -0.005 / +0.004 | includes 0 | 0.55 / **0.030** |
| EXP `x_cw_amp` (Chandler amplitude) | y_main / y_all | -0.195 / -0.081 | excludes 0 (negative) | 0.985 / 1.00 |

- **H1 alternate split.** When 2006-2012 is moved into the test period, the 5-yr-lagged LOD
  deceleration gets a tiny positive IG of about +0.005 nats/eq. Its sign matches Bendick & Bilham's
  claim (more events about 5 yr after deceleration), but the surrogate p is 0.065-0.085, so it is not
  significant even uncorrected. With the preregistered split it vanishes (p = 0.24).
- **Exploratory `x_lod365_anom` on y_all.** Its p of 0.03 is one of about 10 exploratory tests. It has
  IG of +0.004 with a CI that includes 0, and nothing on y_main. Per the preregistration it cannot
  upgrade the verdict, and it is consistent with the multiple-testing expectation.
- **Chandler amplitude.** This slowly varying (decadal-envelope) series badly overfits the train
  period, a textbook case of a trending slow feature failing out of sample.

### Event-level pole-tide phase
This is exploratory and retrospective. It uses the actual daily pole tide with the IERS-2018 secular
pole, and phase = atan2(-rate/ω_Chandler, level). It covers declustered M>=6 mainshocks and uses
`schuster_vs_background` (Monte-Carlo, 5,000 draws).

| Period | N | Schuster-vs-background p | Mean phase |
|---|---|---|---|
| Train 1976-2012 | 146 | 0.009 | -90° (pole-tide radial displacement rising, crossing zero) |
| Test 2013-2026 | 72 | 0.16 | +137° |

The training-period clustering at -90° matches the positive logit coefficient on `x_pt_rate` (+0.22,
standardized). The test-period events cluster, weakly, at a phase 133° away. This is the signature of
an in-sample fluctuation: 1976-2012 contains only about 8-9 Chandler/annual beat cycles, and
mainshocks are not fully independent. The phase test is retrospective, so it never served as a
forecast feature.

## Leakage checks
1. **Publication lag.** All measured EOP values are used with a 30-day lag (day D < t_start - 30 d),
   matching the real C04 latency. The pole extrapolation to the window midpoint uses only those
   values.
2. **Reprocessing.** C04 is a final, reprocessed product. Rapid-service (Bulletin A) values differ by
   about 0.02 ms in LOD and about 0.1-0.3 mas in pole position. That is two or three orders of
   magnitude below the feature variability (LOD dLOD sd 0.24 ms; pole-tide oscillation about 100 mas).
   It could not create a spurious signal and, in any case, no signal was found.
3. **Causal windows.** All rolling means are trailing (pandas default, with no center). The mean pole
   comes from the causal LS trend. No whole-period z-scoring is done; the harness standardizes on train.
4. **Exception.** The event-level phase test uses a centered 6-yr rolling mean and the actual
   (non-extrapolated) pole tide. It is retrospective and exploratory, not a forecast, and is labeled
   as such.
5. **Trend control.** The non-stationary y_main rate (+32% in test) could favour slow features. That
   would bias toward false positives, and none appeared. The detrended LOD variant is also null.

## Caveats
- **Power.** With 72-85 positive test cells, a true effect of about +0.02 nats/eq or less cannot be
  ruled out. The H1 CI on y_main is [-0.007, +0.013], so a large effect (≥ 0.05 nats/eq) of
  LOD-lag-5 on Chilean M>=6 weekly forecasts is ruled out at this resolution.
- **Bendick & Bilham's own claim.** It concerns *global* M7+ rates on multi-year bins. A Chile-only
  M>=6 weekly test is a different, though relevant, application. Even so, the 30-day panel (closer to
  a slow modulation test) is also null.
- **Pole-tide stress proxy.** The proxy is the radial displacement / potential at a single longitude.
  It is not a fault-resolved Coulomb stress on the slab interface. A resolved shear or normal stress
  would be a linear combination of the same m1/m2 projections, so the 2-column logit (level + rate)
  already spans any fixed-phase preference. Coulomb stresses resolved per fault geometry with depth
  were not modeled.
- **Calibration.** `calibration/CALIBRATION_V2.md` did not yet exist when this was written. No result
  is near significance, so the verdict does not depend on it.
- **Harness.** No harness bug was found. `harness.py` was not modified.

## Files
- Scripts: `features.py`, `check_features.py`, `run_primary.py`, `run_robust.py`.
- Outputs: `primary_n200.json/.log` and `robust_n200.json/.log` in this folder.
- Reproduce:
  `cd research/hypotheses/earth_rotation && /opt/miniconda3/bin/python3 -I run_primary.py ../../data/earth_rotation/eopc04_1962-now.txt primary_n200.json 200 && /opt/miniconda3/bin/python3 -I run_robust.py ../../data/earth_rotation/eopc04_1962-now.txt robust_n200.json 200`
