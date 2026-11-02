# PREREGISTRATION: teleseismic_triggering

Written 2026-10-17, before downloading the global catalog and before any `evaluate()` call on the test period.

## Hypotheses
- **H1 (remote dynamic triggering).** Strong recent remote shaking raises the short-term probability of an M>=6 event in a Chilean latitude bin, beyond what local clustering (ETAS and the harness baseline) already explains. Remote shaking means the peak dynamic strain from surface waves of large distant earthquakes.
- **H2 (global synchrony).** A high recent global rate of large-earthquake moment release, or of M>=7 events around the Pacific rim, raises the probability of large events in Chile.

Prior expectation: null. Parsons & Velasco (2011) found no remote short-term triggering of M>=5 beyond about 1000 km. Michael (2011) and Shearer & Stark (2012) found that the global rate of great earthquakes is consistent with a Poisson process after declustering.

## Data
- USGS ComCat global query: `type=earthquake`, M>=6.5, 1973-01-01 to 2026-10-17. Saved under `research/data/teleseismic_triggering/`.
- **Remote** events are those whose epicentre lies more than **1000 km** from the reference point of every Chilean bin. Each bin's reference point is (bin latitude centre, the median longitude of M>=4.7 events in the bin before 2013). This excludes near-field and static-stress triggering, as well as southern Peru and Argentina events just outside the harness box.
- **Causality.** Only events with `time < t_start - 1 hour` are used. USGS publishes M>=6.5 events within about 20 to 60 minutes. The final catalog magnitude is used even though it may differ by about 0.1 from the real-time value. This is a minor leak and is noted as a caveat.

## Features (fixed before testing)
The dynamic strain proxy for remote event k at bin b follows the surface-wave magnitude scaling used by van der Elst & Brodsky (2010):
`log10 eps_kb = M_k - 1.66*log10(Delta_kb [deg]) - 4.0`. The constant only sets the units, which are roughly microstrain. Events deeper than 100 km are excluded from the strain proxy because they generate weak surface waves.

**H1, primary per-bin feature set (two columns):**
- `x_strain_max_7d`: log10(max eps over remote events in [t_start-7d, t_start)). It takes a floor value of -3 when there is no event.
- `x_strain_sum_30d`: log10(1e-3 + sum of eps over remote events in [t_start-30d, t_start)).

**H2, secondary global feature set (two columns, broadcast to all bins):**
- `x_gmoment_365d`: log10 of the sum of seismic moment (M0 = 10^(1.5M+9.1)) of remote M>=6.5 events in [t_start-365d, t_start). It is expressed as an anomaly relative to its own trailing 10-year mean, computed causally.
- `x_pacific_m7_365d`: the count of remote M>=7.0 events on the Pacific rim in [t_start-365d, t_start). The Pacific rim is defined as longitude >= 100E or <= -60W and latitude in -65..65, excluding the 1000 km Chile exclusion zone.

## Primary test
- **Panel:** weekly (step 7 d, horizon 7 d), target M>=6.0, 1-degree bins.
- **Model and metric:** `model="logit"`. The metric is `ig_per_event` (full minus the harness ETAS/clustering baseline).
- **Train/test split:** train 1976-2012, test 2013-01-01 to 2026-10.
- **Targets:** `y_all` (primary) and `y_main` (ZBZ mainshocks, co-primary).
- **Primary significance test (H1 and H2):** a circular time shift of the **raw global event catalog**. Event times are shifted by a random offset of at least 365 days, modulo the 1973-01-01 to 2026-10-17 span, and the features are recomputed through the `builder` argument of `surrogate_test`. This keeps the global catalog's own clustering, magnitudes and geometry but breaks its timing relative to Chile. Use n=1000 for the primary test.
- **Secondary significance tests:** the harness `phase_randomize` on the panel feature grid (n=200), and `bin_permute` for H1 (n=200).
- **Alpha:** 0.05 uncorrected. The Bonferroni threshold is 0.0036 (14 parallel hypotheses).
- **Verdict rule:**
  - **positive** requires IG > 0 on test, a primary surrogate p < 0.0036 and a phase_randomize p < 0.05 for the same target. It must also survive the leakage checks.
  - **weak_positive** requires a primary p < 0.05.
  - Otherwise the verdict is null or negative.
  - H1 is the main claim. H2 counts only as secondary.

## Robustness (reported regardless of outcome)
- M>=5.5 weekly panel (y_all and y_main).
- Daily panel (step 1, horizon 1), M>=6.0. Immediate dynamic triggering should be strongest here.
- 30-day panel (step 30, horizon 30), M>=6.0.
- Alternate split: train to 2004-12-31 and test 2005-2012, using the training years only. The harness split is fixed, so this is done with my own refit of logit on the same baseline columns plus the features.
- GBM model on the primary panel.
- Leave-one-test-year-out IG.
- Distance threshold of 2000 km instead of 1000 km.
- Differenced or detrended version of H2 (the slow series).
