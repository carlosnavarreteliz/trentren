# PREREGISTRATION: earth_rotation (LOD deceleration, pole tide, Chandler wobble)

Written 2026-10-17, before any `evaluate()` call on the test period. No test-period targets have been inspected.

## Data
IERS EOP 20 C04 (Paris Observatory), daily at 0h UTC, 1962-01-01 to 2026-09-17:
`https://hpiers.obspm.fr/iers/eop/eopc04/eopc04.1962-now`, cached at
`research/data/earth_rotation/eopc04_1962-now.txt`. Columns used: x, y (arcsec), LOD (s).

**Availability lag.** The file ends 30 days before today, so C04 is published with about a 30-day lag.
Every measured value is used only when day D < t_start - 30 d (`lag_days=30`). C04 is a reprocessed,
final series. Real-time Bulletin A values differ from it by about 0.02 ms in LOD and about 0.1-0.3 mas
in pole position. Those differences are 1-3 orders of magnitude below the signals used here (LOD decadal
variations of about 1-3 ms; pole excursions of about 100-300 mas), so reprocessing leakage is negligible.
It will still be stated as a caveat.

## H1: LOD deceleration (Bendick & Bilham 2017)
Bendick and Bilham claim that global M7+ seismicity increases about 5 years after episodes of
Earth-rotation deceleration (rising LOD).
- `LOD365(D)` = trailing 365-day mean of LOD in ms, over the days ending at D (causal).
- `dLOD(D)` = LOD365(D) - LOD365(D - 365 d): the annual change, where positive means deceleration.
- **Primary feature:** `x_lod_decel_lag5y` = dLOD at (t_start - 30 d - 5 yr). It is a global series,
  broadcast to every bin. It enters as one column.

## H2: Pole-tide modulation
- The secular mean pole is the causal trailing 6-year mean of x and y.
- m1 = x - x_mean and m2 = -(y - y_mean), in arcsec.
- At each issue time, fit x(t) and y(t) separately by least squares on the last 6 years of available
  data (D < t_start - 30 d). The model is a constant, a linear trend, and annual (365.25 d) plus Chandler
  (433 d) sin/cos terms.
- Extrapolate the fit to the forecast-window midpoint, t_start + h/2. This is causal, because it uses
  only past data.
- The pole-tide radial displacement follows IERS Conventions 2010 §7.1.4:
  S_r = -33 · sin(2θ) · (m1 cos λ + m2 sin λ) mm, with θ the colatitude of the bin centre and
  λ = -71.5° E.
- **Primary features (2 columns, per bin):**
  - `x_pt` = S_r at the window midpoint.
  - `x_pt_rate` = dS_r/dt, in mm/yr, from the derivative of the harmonic fit.
- These two columns are in quadrature for a quasi-sinusoidal forcing, so the logit can express any
  preferred pole-tide phase. That makes this a phase-modulation test.

## Primary tests (decided now)
- **Panel:** weekly (step 7 d, horizon 7 d), target M >= 6.0, 1° latitude bins. Model: logit.
  Split: the harness default (train 1976-2012, test 2013-01-01 to 2026-10).
- **Primary target:** `y_main` (ZBZ mainshocks). Both hypotheses concern a modulation of independent
  nucleation rates, not of aftershocks. `y_all` is reported alongside it as the co-primary secondary.
- **Metric:** `ig_per_event` (full vs the ETAS/clustering logit baseline) on test.
- **Significance:** `H.surrogate_test` with `mode="phase_randomize"`.
  - H1 uses a builder that FT-surrogates the raw daily LOD series (increment surrogate if it is
    random-walk-like) and recomputes the features.
  - H2 phase-randomizes the panel feature grid, applying the same phases across bins and both columns.
  - Start with n=200. If p < 0.02, rerun with n=1000.
- **Decision thresholds:** project-wide Bonferroni alpha = 0.0036. I am testing 2 co-primary hypotheses,
  so I additionally report the within-agent corrected threshold of 0.0018.
  - "positive" requires IG > 0 vs baseline on test, p < 0.0036, and passing the leakage checks.
  - "weak_positive" requires p < 0.05 uncorrected.
  - Anything else is null.

## Secondary / robustness (reported regardless of outcome)
1. `y_all` for H1 and H2.
2. A circular-shift surrogate (end-matched) as a robustness check.
3. A weekly M>=5.5 panel.
4. A 30-day panel (step 30, horizon 30), M>=6.0.
5. An alternate split: train up to 2005-12-31 and test 2006-01-01 to 2026. This uses a modified copy of
   the panel `split` column; the panel's Poisson column is not refit, but it is shared by base and full.
6. GBM model.
7. H1 detrended or differenced variant: `x_lod_decel_lag0` (dLOD with no 5-yr lag) and `x_lod365_anom`
   (LOD365 minus its trailing 10-yr mean). These are exploratory.
8. Chandler amplitude `x_cw_amp` (amplitude of the 433-d term in the causal 6-yr fit). This is exploratory.
9. Event-level: the pole-tide phase (atan2 of x_pt_rate-normalized and x_pt) at declustered M>=6 mainshocks.
   Compared with background via `H.schuster_vs_background`, for train and test periods separately.
   This is exploratory.
10. Drop-one-test-year check on `preds` if anything is significant.

Exploratory results (items 7-9) cannot upgrade the verdict.
