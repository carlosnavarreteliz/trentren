# PREREGISTRATION: tides (tidal stress modulation and tidal sensitivity as a criticality indicator)

Written 2026-10-17, BEFORE computing any tidal phase at event times and before any `evaluate` call on the test period.
Data downloaded so far: UHSLC research-quality hourly sea level (Arica h083a, Antofagasta h080a, Caldera h088a,
Valparaiso h081a) and the JPL DE421 ephemeris. No event phases or features have been computed yet.

## Hypotheses
- **H-a (phase modulation).** Shallow Chilean earthquakes occur preferentially when the tidal Coulomb stress on the
  megathrust is encouraging (phase near 0), i.e. their tidal phases are non-uniform relative to the phase occupation of
  random times.
- **H-b (tidal sensitivity as criticality; Tanaka et al. 2002/2004, Ide et al. 2016).** In the years before a large
  earthquake, small-event timing in the surrounding region becomes more correlated with tidal stress. A rolling per-bin
  tidal-sensitivity index therefore carries information on upcoming M>=6 events that ETAS/clustering does not.
- **H-c (secondary, Ide et al. 2016).** Large events are more likely when the tidal stress amplitude in the forecast
  window is large (spring tides). Deterministic, known in advance.

## Tidal stress model (fixed in advance)
- Solid-earth tide: degree-2 potential of Moon and Sun from DE421 (skyfield, ITRS frame), Love numbers h2=0.6078,
  l2=0.0847, surface strain via the standard spherical formulas, plane-stress elasticity with lambda=mu=30 GPa
  (sigma_zz = sigma_xz = sigma_yz = 0).
- Ocean load: tide height eta predicted from harmonic constituents (utide, constituents fitted only on 1985-2012 hourly
  data) at the nearest open-coast gauge by latitude: Arica (lat > -21.06), Antofagasta (-21.06..-25.36),
  Caldera (-25.36..-30.05), Valparaiso (< -30.05; also used for the southern bins, a known approximation; Puerto Montt
  is excluded because it is an amplified inland-sea gauge). Load as a laterally extensive vertical load (uniaxial
  strain): sigma_zz = -rho g eta, sigma_EE = sigma_NN = -rho g eta * nu/(1-nu), rho=1025 kg/m3, nu=0.25.
- Receiver fault: generic Chilean megathrust, strike 0 deg (N-S), dip 20 deg to the east, rake 90 deg (thrust).
  Coulomb stress change dCFS = tau(slip direction) + mu' * sigma_n (tension positive), mu' = 0.4.
- Locations: one point per 1-degree bin (bin latitude center; longitude = median longitude of M>=4.7 events with
  depth <= 70 km in that bin, 1973-2012). Time grid 10 min, 1972-2028.
- Tidal phase (Tanaka 2002 definition): from the total dCFS series at the bin point, phase = 0 at a local maximum,
  +-180 deg at a local minimum, linear in time in between (positive going from max to min, negative from min to max).
- Background phase occupation: phases at 20,000 uniformly random times (1973-2026) at locations drawn from the event
  bins (matches spatial distribution).

## Test (a): phase test (event-level, not in the harness)
- Events: ZBZ declustered mainshocks (non-causal flag is fine for a phase test), M>=4.7, depth <= 70 km, in region.
- Statistic: `H.schuster_vs_background` (n_mc=5000), computed separately for train (1973-2012) and test (2013-2026).
- Primary for (a): test-period p-value; also the direction (mean phase) must be consistent with train (|mean phase|
  < 90 deg in both, i.e. near encouraging stress) for a claim.
- Secondary: M>=5.5 mainshocks; all `bg` events; solid-earth-only and ocean-only phases; fraction of events with
  dCFS>0 at the event time vs background.

## Test (b): harness evaluation -- PRIMARY TEST OF THIS FOLDER
- Feature `x_tsens` (per bin, strictly causal): for issue time t and bin b, take events with `bg == True`
  (causal ZBZ flag), M>=4.7, depth <= 70 km, in bins b-1..b+1, with time in [t-1095 d, t). With phases phi_i at the
  event times and background mean c0 and variance v0 of cos(phi) (from random times at the same bin points,
  computed once over 1973-2012):
  `x_tsens = sum_i (cos(phi_i) - c0) / sqrt(N * v0)`, NaN if N < 5 (imputed to the train mean by evaluate).
- Panel: weekly (step 7 d, horizon 7 d), target M>=6.0. Model: logit (harness default).
- **Primary target: `y_main`** (declustered mainshocks). `y_all` reported as co-result, not primary.
- **Primary significance: "tide-shift" surrogate**, n=1000: every event's phase is recomputed at t_i + D with one
  random continuous offset D ~ U(60, 3650) days (random sign; wrapped inside the grid), the same D for all events.
  This keeps event times, counts, clustering and the feature's autocorrelation exactly, and destroys only the
  alignment between events and tides. p = (1 + #{IG_null >= IG_obs}) / (n + 1).
- Also reported: harness `surrogate_test` phase_randomize (n=1000) and bin_permute (n=200), year block bootstrap CI,
  AUC and Molchan ASS.
- Decision: **positive** only if IG(full vs base) > 0 on test AND tide-shift p < 0.0036 (Bonferroni, 0.05/14)
  AND harness phase_randomize p < 0.05 AND leakage checks pass. weak_positive if p < 0.05 uncorrected. Else null.

## Secondary features / robustness (no claims without the primary)
- `x_tsens_lp`: non-directional Tanaka statistic, -log10 of the background-corrected Schuster p of the same event set.
- `x_tamp` (H-c): max of total dCFS at the bin point over the forecast window [t, t+h) (deterministic, known in
  advance), in kPa.
- Window 365 d instead of 1095 d; M>=5.5 weekly panel; 30-day panel (step 30, horizon 30); GBM; alternate split
  (train through 2004, test 2005-2026; Poisson background still from 1976-2012, which affects base and full equally).
- Leave-one-test-year-out on preds.

## Leakage notes (decided in advance)
- Tides are deterministic functions of time: predicting them for the forecast window is legitimate.
- Harmonic constants are fitted on pre-2013 data only.
- `x_tsens` uses only events with time < t_start and the causal `bg` flag; never `mainshock`.
- Network densification changes N (and hence noise) but not the expected phase of real events.
