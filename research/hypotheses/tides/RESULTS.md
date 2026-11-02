# RESULTS: tides (tidal stress modulation and tidal sensitivity)

**Verdict: NULL.** The preregistered primary feature, rolling tidal sensitivity `x_tsens`, makes the weekly M>=6
forecast slightly worse out of sample. On y_main IG = -0.025 nats/eq and tide-shift surrogate p = 0.98. The
event-level phase effect is a small, direction-consistent excess of events on the rising limb of the tidal Coulomb
stress, but it does not survive a null that preserves event clustering (tide-shift p = 0.07 test, 0.64 train).

Preregistration: `PREREGISTRATION.md`, written before any phase or test-period evaluation.

## Data and tidal model
- Ephemeris: JPL DE421 via skyfield, Moon and Sun positions in ITRS, hourly, cubic-splined to a 10-min grid
  from 1972 to 2028.
- Solid-earth tide: degree-2 potential with h2 = 0.6078 and l2 = 0.0847, surface plane stress, lambda = mu = 30 GPa.
  Checks: body-tide uplift range at Valparaiso is -0.14 to +0.21 m, areal strain is about 4e-8, and the Moon
  direction correlates with topocentric altitude at r = 0.99997.
- Ocean load: utide OLS harmonic fits on pre-2013 UHSLC hourly data for four gauges.

  | gauge | variance explained | M2 amplitude (m) |
  |---|---|---|
  | Arica | 97.1 % | 0.33 |
  | Antofagasta | 97.7 % | 0.37 |
  | Caldera | 98.3 % | 0.39 |
  | Valparaiso | 97.9 % | 0.44 |

  The load is applied as a uniaxial-strain vertical load.
- Receiver fault: strike 0, dip 20 E, rake 90, mu' = 0.4.
- Size of the dCFS terms (std): solid about 0.25 kPa, ocean about 1.6-2.1 kPa, total about 1.5-2.1 kPa. Ocean
  loading dominates.
- Phases follow Tanaka (2002): 0 at the dCFS maximum, +-180 at the minimum. There are about 3.9 extrema per day.

Scripts:
- `t01_ocean_tides.py`
- `t02_tidal_stress.py`
- `t03_phase_test.py`
- `t05_primary.py`
- `t06_robust.py`
- shared code: `tidelib.py`, `tfeatures.py`

Outputs are in `out/`.

## (a) Event-level phase test (declustered ZBZ mainshocks, M>=4.7, depth <= 70 km)

| set | period | n | p (vs random-time background) | mean phase | excess amplitude | p (tide-shift null, 1000 shifts) |
|---|---|---|---|---|---|---|
| mainshocks M4.7 | train 1973-2012 | 1771 | 0.043 | -71 deg | 0.042 | 0.64 |
| mainshocks M4.7 | **test 2013-2026 (primary a)** | 637 | **0.044** | -64 deg | 0.070 | **0.070** |
| mainshocks M4.7 | all | 2408 | 0.0032 | -71 deg | 0.048 | 0.33 |
| bg events M4.7 | test | 637 | 0.099 | -55 deg | 0.062 | 0.16 |
| all events (incl. aftershocks) M4.7 | all | 4905 | 0.95 | n/a | 0.003 | n/a |
| mainshocks M5.5 | test | 99 | 0.68 | -86 deg | 0.063 | n/a |
| mainshocks M6.0 | all | 134 | 0.33 | +112 deg | 0.091 | n/a |

- The direction is consistent between train and test. Events cluster at about -65 deg, the rising limb of the
  tidal Coulomb stress, roughly 1.5-2 h before the peak. That is not the encouraging-stress peak predicted by H-a.
  The fraction of events at positive tidal stress is 51-52 % against 50 % expected.
- The random-time background test gives p = 0.003 on all years, but this test treats events as independent.
  - The tide-shift null applies one global time shift to the tides, which keeps the residual clustering of the
    declustered events.
  - Under that null the effect is not significant: p = 0.33 for all years and p = 0.07 for the test period.
  - **The naive Schuster or background test is therefore anti-conservative here.**
- Ocean-only phases behave like the total. Solid-only phases give a different, inconsistent direction:
  +167 deg in train and -144 deg in test.
- Robustness:
  - The effect sits in the 15-50 km interface depth range (p = 0.008 naive) and is absent at depths < 15 km and
    50-70 km.
  - It appears both at night (p = 0.012) and during the day (p = 0.003). A 24-h time-of-day pattern exists
    (p = 0.027) but does not explain it.
- Verdict for (a): at most a weak, non-robust hint (test p = 0.044 naive and 0.07 under the clustering-preserving
  null). Not Bonferroni significant. It does not extend to M>=5.5 or M>=6 mainshocks.

## (b) PRIMARY: tidal sensitivity `x_tsens` in the harness

Panel: weekly M>=6.0, logit, test period 2013-01-01 to 2026-10.

| target | IG full vs base (nats/eq) | 95 % CI (year bootstrap) | IG full vs ETAS | tide-shift p | harness phase_randomize p | bin_permute p | AUC base / full | Molchan ASS base / full | n test cells |
|---|---|---|---|---|---|---|---|---|---|
| **y_main (primary)** | **-0.0255** | [-0.073, +0.017] | -0.033 | **0.984** (n=1000) | 0.970 (n=1000) | 0.980 (n=200) | 0.665 / 0.654 | 0.664 / 0.653 | 72 |
| y_all | -0.0228 | [-0.053, +0.002] | -0.023 | 0.856 (n=200) | 0.980 (n=200) | 0.955 (n=200) | 0.697 / 0.695 | 0.696 / 0.694 | 85 |

- The train-fitted coefficient is positive (+0.19 std units): higher sensitivity went with more mainshocks in
  1976-2012. That relation does not transfer to the test period.
- The leave-one-test-year-out IG is negative in all 14 cases (best case -0.012). The Rhoades T is -1.05 and is not
  relevant here.
- Descriptive event study for M>=7 mainshocks: the bin's x_tsens percentile just before each event averages 0.496
  (n = 22; 0.5 expected; SE about 0.06).
  - Iquique 2014: 0.68. Illapel 2015: 0.83. Maule 2010: 0.30. Tocopilla 2007: too few events.
  - There is no systematic precursory rise of tidal sensitivity.

### Secondary features and robustness (logit unless noted; IG full vs base, nats/eq)

| panel / target | x_tsens | x_tsens_lp (Tanaka -log p) | x_tsens365 | x_tamp (window-max tidal dCFS, Ide 2016) | all three |
|---|---|---|---|---|---|
| weekly M6 y_main | -0.026 | -0.005 | 0.000 | +0.001 [-0.010, +0.015] | -0.026 |
| weekly M6 y_all | -0.023 | -0.002 | -0.001 | -0.008 | -0.024 |
| weekly M5.5 y_main | -0.020 | -0.008 | +0.000 | -0.006 | -0.037 |
| weekly M5.5 y_all | -0.017 | -0.003 | -0.000 | +0.002 | -0.018 |
| 30-day M6 y_main | -0.023 | +0.001 | +0.000 | -0.001 | -0.028 |
| 30-day M6 y_all | -0.026 | -0.000 | -0.001 | +0.002 | -0.029 |
| weekly M6 y_main, GBM | +0.013 [-0.046, +0.065] | n/a | n/a | +0.003 [-0.061, +0.056] | n/a |

- Alternate split (train <= 2004, test 2005-2026, which includes Maule 2010): x_tsens gives IG = +0.002
  [-0.036, +0.044] on y_main (n = 106) and -0.007 [-0.034, +0.019] on y_all.
- Every effect is within about +-0.03 nats/eq of zero and none is significant. The tidal-amplitude (spring-tide)
  feature is essentially zero, at |IG| < 0.01.

## Leakage checks
- Tides are deterministic. Harmonic constants were fitted only on 1980-2012 gauge data, and the ephemeris is
  computed, not observed.
- x_tsens uses events with t < t_start only, and the causal `bg` flag, never `mainshock`.
- The background moments c0 and v0 come from random times in 1973-2012 only.
- x_tamp uses the forecast window. That is allowed because it is a deterministic tidal prediction.
- Nothing was tuned on the test period. All feature definitions, windows and the fault geometry were fixed in
  PREREGISTRATION.md before evaluation.
- Since the primary result is negative, no leakage can have inflated it.

## Caveats
- The ocean load is a crude approximation, and it dominates the stress:
  - nearest-gauge tide for each latitude band;
  - Valparaiso used for every bin south of -30 deg;
  - an infinite uniaxial load, which overstates the effect at depth and landward of the coast;
  - no proper loading Green's functions (SPOTL/GOTIC2) and no ocean-tide model offshore.
- The receiver geometry is a single generic megathrust plane. Outer-rise, crustal and intraslab events have other
  mechanisms, and no focal mechanisms (GCMT) were used. The depth <= 70 km filter keeps some non-interface events,
  which dilutes any real effect.
- M>=4.7 bg events give only about 10-35 events per 3-bin, 3-year window. That makes x_tsens noisy, and the noise
  is the main reason this test has low power. A lower magnitude threshold (M>=3.5-4.0, complete only after about
  1990) or a local CSN catalog might be more sensitive.
- No harness bug was found. The naive Schuster or random-background phase test is anti-conservative when the
  events are not independent, as the tide-shift null shows. Other agents running phase tests should use a
  clustering-preserving null.
