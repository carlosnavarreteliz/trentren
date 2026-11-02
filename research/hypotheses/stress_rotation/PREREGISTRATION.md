# PREREGISTRATION: stress_rotation (focal-mechanism stress-state rotation and heterogeneity, Global CMT)

Written 2026-10-17, after downloading and parsing the GCMT catalog and **before any `evaluate()` call**.
No test-period targets have been inspected. The only things looked at so far are the GCMT event counts per
5-year block (feature data only).

## Hypothesis
Before large Chilean earthquakes, the local stress state, as sampled by the focal mechanisms of moderate
earthquakes, rotates away from its long-term orientation and/or becomes more heterogeneous. This would show
up as lower Kostrov moment-tensor consistency, a larger angle between the recent and long-term summed
mechanism, a change in the fraction of interface-like thrust mechanisms, and changes in non-double-couple
(CLVD) content. Claims of this kind exist for single events (for example b-value and stress rotations before
Tohoku and Iquique), but they have not been tested prospectively against an ETAS/clustering baseline.

## Data
- GCMT NDK files: `jan76_dec20.ndk` plus the monthly files 2021-01..2026-05 (final solutions), and
  `qcmt.ndk` (quick solutions) for events after 2026-05-31.
- Cached in `research/data/stress_rotation/` and parsed by `parse_gcmt.py` into `gcmt_chile.csv`.
- Region: centroid lat -48..-15 and lon -80..-64, which is wider than the harness region so that edge bins
  have neighbours.
- The parsed catalog has 2,970 regional events from 1976 to 2026.

## Features (per bin b, issue time t)
Event set S consists of GCMT events with centroid depth <= 70 km and centroid latitude in bins b-1..b+1.
The recent window R is `[t - 730 d - L, t - L)` with availability lag L = 2 d. The long-term reference LT is
every S event with time `< t - 730 d - L`, starting in 1976.

Each moment tensor is normalised to unit scalar moment (m = M / M0, with M0 = ||M||_F / sqrt(2)).
Mechanism classes follow Frohlich (1992): thrust if the T-axis plunge is > 50 deg, normal if the P-axis
plunge is > 60 deg. CLVD is measured as |eps| = |lambda_N| / max(|lambda_T|, |lambda_P|), computed from the
NDK eigenvalues.

| column | definition | NaN when |
|---|---|---|
| `x_cons` | Kostrov consistency in R: `||sum m|| / N`, between 0 and 1. A low value means a heterogeneous stress field. | N_R < 3 |
| `x_rot` | tensor angle in degrees, `arccos(<A,B>/(|A||B|))`, between the summed normalised tensor in R (A) and in LT (B) | N_R < 3 or N_LT < 5 |
| `x_dthrust` | thrust fraction in R minus thrust fraction in LT | N_R < 3 or N_LT < 5 |
| `x_dclvd` | mean abs(eps) in R minus mean abs(eps) in LT | N_R < 3 or N_LT < 5 |

The harness imputes NaN with the train mean for the logit.

Nothing uses events at or after t - L. All aggregates are cumulative sums indexed by event time and are
evaluated with searchsorted on `t - L`.

## Primary test (decided now)
- **Panel and model:** weekly panel (step 7 d, horizon 7 d), target M >= 6.0, logit. Train 1976-2012, test
  2013-01-01..2026-10.
- **Primary target: `y_main`.** A precursor of a stress-state change should predict mainshocks, not
  aftershocks. `y_all` is a co-reported secondary.
- **Metric:** `ig_per_event` of baseline + 4 features vs the ETAS/clustering logit baseline.
- **Primary significance:** `H.surrogate_test(feats, mode="phase_randomize", n=1000)`. For y_all the run is
  n=200, raised to n=1000 if p < 0.02.
- **Mechanism-shuffle surrogate (required for a positive claim).** Randomly permute the moment tensors (and
  their derived class and CLVD values) among GCMT events, keeping every event's time, location, depth and
  magnitude. Then rebuild the features, with n=200. This null keeps all information that comes from *when
  and where* CMT events happen, which is clustering, and destroys only the mechanism content. A positive
  claim needs mechanism-shuffle p < 0.05 as well.
- `bin_permute` (n=200) is reported as a further secondary.

## Decision rule
- **positive:** IG > 0 vs the baseline on test, phase_randomize p < 0.0036 (Bonferroni, 14 hypotheses),
  mechanism-shuffle p < 0.05, and the leakage checks pass.
- **weak_positive:** p < 0.05 uncorrected on the primary test, with IG > 0.
- **null** otherwise. If IG < 0 with a bootstrap CI excluding 0, the verdict is **negative**.

## Secondary and robustness (reported whatever the outcome)
1. `y_all`, weekly M6.
2. Weekly M5.5 (y_main and y_all).
3. 30-day panel (step 30, horizon 30) M6 (y_main and y_all).
4. Daily panel M6 y_main.
5. Recent window of 365 d instead of 730 d.
6. Only events with Mw >= 5.5. These are reliably available in real time from quick CMT, which addresses
   the concern that monthly final solutions are published months after the event. Also a 180 d lag version.
7. Alternate split: train 1976-2004, test 2005-2012. This uses only the pre-2013 period, so it is not tuned
   on the official test.
8. GBM model (nonlinear) on weekly M6 y_main and y_all.
9. Exploratory: stress-inversion misfit (a linearised Michael 1984 inversion over R with >= 6 events,
   mean slip-shear angle, best of the two nodal planes) as an extra 5th column. This is labelled
   exploratory and is not part of the primary test.

## Leakage checks planned
- Confirm that GCMT centroid times and eigenvalues come from the catalog, not from later reprocessing of
  the same event. This risk remains: final solutions are produced months later, which is addressed by the
  Mw >= 5.5 and 180 d lag variants.
- Assert that no feature uses events with time >= t - L, by recomputing a random sample of rows by brute
  force.
- CMT network densification (the GCMT threshold dropped after about 2004) changes N but not the
  normalised mechanism features. The mechanism-shuffle null controls for any count effect.
