# PREREGISTRATION: gnss_transients

Written 2026-10-17, before any evaluation on the test period (2013-01-01 onward). Station list and the
NGL station-coverage table were inspected; no feature or target from the test period had been
looked at.

## Hypothesis
Transient trenchward (seaward) acceleration of forearc GNSS stations, relative to each station's own
recent interseismic trend, indicates slow slip / precursory unlocking on the subduction interface and
raises the short-term probability of M>=6 earthquakes in that along-strike segment, **beyond what
catalog clustering (ETAS baseline) already explains**. Motivation: Ruiz et al. 2014 / Socquet et al.
2017 (Iquique 2014), Ruiz et al. 2017 (Valparaiso 2017), Mavrommatis et al. 2014 (Tohoku).

## Data
Nevada Geodetic Laboratory (NGL) daily final solutions, IGS20 frame, `tenv3` files,
https://geodesy.unr.edu/gps_timeseries/IGS20/tenv3/IGS20/<STA>.tenv3, all stations with lat -46.5..-16.5,
lon -76..-64 and >= 365 solutions (339 stations). NGL `steps.txt` (equipment changes, type 1, and
earthquake steps, type 2). Coverage: ~1-11 stations before 2002, 17 in 2005, 46 in 2007 (19 bins),
150+ from 2010.

## Feature (strictly causal)
For each station and each panel issue time t, with lag L = 2 days (only days D < t - L are used):
- Component: horizontal displacement projected on the landward convergence azimuth 78 deg
  (u = E sin78 + N cos78). Trenchward = -u.
- Window: days [t - L - 760, t - L). Fit by OLS: u = a + b*tau + annual + semiannual sinusoids
  + Heaviside offsets at every NGL step (type 1 or 2) dated inside the window (all known before t)
  + c * max(0, tau - tau_W), where tau_W = t - L - 30 d (hinge = change of velocity in the last 30 days).
- Requirements: >= 365 days of data before tau_W and >= 15 days inside the last 30 days; else missing.
  Daily outliers > 5 robust-sigma from a first fit are dropped and the fit is repeated once.
- Station anomaly z = -c / se(c) (positive = trenchward acceleration), clipped to [-10, 10].
- Stations used: lon <= -68.5 (forearc/arc; excludes the Argentine foreland).
- Per-bin feature `x_gtw_z30` = median z over stations whose latitude falls in the bin or the two
  adjacent 1-deg bins (bin +/- 1). Missing (no station) = 0 (neutral, "no evidence of a transient").

Secondary features (reported, not primary): `x_gtw_z90` (90-day hinge), `x_gtw_absz30` (median |z|),
`x_gtw_z30_clean` (same as primary but a station is excluded for 2 years after any NGL earthquake step
of M>=6.5, i.e. postseismic periods removed).

## Primary test (single, decided now)
- Panel: weekly issue times, 7-day horizon, target M>=6.0 (`H.get_panel(7, 7, 6.0)`), **restricted to
  t_start >= 2007-01-01** (the era with >= 46 stations covering >= 19 bins). Train = 2007-01-01..2012-12-31
  (harness purge), test = 2013-01-01..2026-10.
- Target: **y_main** (ZBZ-declustered mainshocks), model: logit (harness defaults).
- Statistic: `ig_per_event` (full minus ETAS-logit baseline, nats per target event).
- Significance: phase-randomized surrogates of the per-bin feature grid (same phases for all bins,
  applied only to the coverage era t >= 2007 through a builder), one-sided, n >= 200 (n = 1000 if
  p < 0.02). Alpha = 0.05 uncorrected; Bonferroni alpha = 0.05/14 = 0.0036.
- Positive verdict requires IG > 0, surrogate p < 0.0036, AND surviving the leakage checks below.

## Secondary / robustness (reported regardless of outcome)
- Same on y_all.
- Full panel (1976+) with missing = 0.
- bin_permute surrogate and end-matched circular_shift surrogate.
- M>=5.5 weekly panel; 30-day step/horizon panel (M>=6.0).
- Alternate split: train 2007-2015, test 2016+ (removes Iquique/Illapel from the test).
- Lag L = 14 days (latency of NGL final products) instead of 2.
- Secondary features listed above; GBM model.
- Event-level look at Iquique 2014, Illapel 2015, Chiloe 2016 and the 2017 Valparaiso sequence.

## Leakage checks
1. Reprocessing: NGL series are reprocessed final solutions, not the real-time product. Mitigate with
   the L=14 d variant; the feature uses only each station's own relative motion over 2 years.
2. Network densification after big quakes: station count is NOT a feature; missing = 0. Report the
   correlation of the feature with station count.
3. Postseismic transients: `x_gtw_z30_clean` variant.
4. Step catalog: only steps dated before t - L are used at issue time t.

## Amendment 1 (implementation bug fix, written AFTER the v1 primary result was seen)
The v1 implementation added one Heaviside column per NGL step line. NGL lists several earthquake steps on the
same day (e.g. 3 entries on 2015-09-16 for SANT), which produced identical columns, a singular normal matrix and
NaN anomalies for ~32 % of station-weeks, concentrated in the 2 years after big sequences. v2 keeps one column
per distinct step position and returns NaN only for ill-conditioned fits (cond > 1e8). Nothing else changed
(same feature definition, panel, target, test). The v1 primary result is reported alongside v2 in RESULTS.md
(v1: IG = +0.004 nats/eq, phase-randomize p = 0.11 on y_main) so that the fix cannot be used to fish for
significance; v2 is the faithful implementation of the spec above.
