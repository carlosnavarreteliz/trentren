# PREREGISTRATION: quiescence_migration (seismic quiescence, migration and inter-segment synchronization)

Written 2026-10-17, **before any `evaluate()` call and before looking at any test-period target or feature value.**
The only data used is the harness catalog (USGS ComCat, M>=4.7, ZBZ-declustered with the causal `bg` flag).
No new download is needed (the folder `research/data/quiescence_migration/` stays empty except for a feature cache).

## Hypothesis
Large Chilean earthquakes are preceded by intermediate-term changes in the space-time pattern of moderate
seismicity that a short-term clustering model (ETAS) does not encode:
1. **Quiescence** near the future hypocentre (RTL of Sobolev & Tyupkin 1997; Z-value of Habermann 1983/Wiemer & Wyss 1994).
2. **Mogi doughnut**: the ring around the future rupture stays active while the core goes quiet (Mogi 1969).
3. **Along-strike migration** of background seismicity toward the future hypocentre.
4. **Inter-segment synchronization**: the background-rate fluctuations of a segment become correlated with
   those of its neighbours (a "critical" or loading-coherent state), measured by rolling cross-correlation of
   latitude-bin rate series and the bin's eigenvector centrality in that correlation network.
5. **Stress shadow / stress loading** from past M>=7 ruptures: bins inside an old rupture are relaxed, bins just
   beyond its ends are loaded (long-term, beyond the 1-year aftershock regime that ETAS already covers).

## Inputs (all strictly causal)
- Events: `H.get_events(min_mag=4.7)` (ZBZ). Quiescence, doughnut, migration and synchronization use only
  **`bg == True` events with depth <= 70 km** (background, interface/upper-plate depth range), so that they measure
  the background rate, not aftershock clustering. `bg` depends only on earlier events (causal). The non-causal
  `mainshock` flag is never used in features.
- Stress shadow/loading uses all M>=7.0 events (any depth <= 70 km) since 1973, **older than 365 days** at t.
- Every quantity at issue time t uses only events with time < t. Any normalization (mean/std) is an
  **expanding, past-only** statistic of the feature's own history (values at issue times < t), with a 5-year burn-in
  (NaN before 1981, imputed with the train mean by the harness).
- Distances are along strike: r = 111 km x |lat_event - lat_bin_center| (floored at 10 km). Longitude is ignored,
  consistent with the harness's latitude-only spatial model.

## Features (literature defaults, fixed now, not tuned)
| column | definition |
|---|---|
| `x_rtl` | Sobolev RTL at the bin centre, r0 = 100 km, t0 = 1 yr, events within 2 r0 and 2 t0, l_i = 10^(0.5 M - 1.8) km. R = sum exp(-r/r0), T = sum exp(-dt/t0), L = sum (l/r). Each term is standardized by its own expanding past mean/std; RTL = R*T*L; feature = signed cube root. Negative = quiescence. |
| `x_zq` | Z-value quiescence, bins b-1..b+1: recent window W = 1 yr vs the preceding 10 yr (or since 1973). Z = (lam_bg - lam_w)/sqrt(lam_bg/T_bg + lam_w/T_w + 0.1). Positive = quiescence. |
| `x_doughnut` | over the last 5 yr: log((ring+1)/(core+1)) minus the same quantity over the preceding 15 yr scaled to 5 yr; core = bins b-1..b+1, ring = bins at distance 2..3. Positive = doughnut (ring active, core quiet). |
| `x_migr` | events within +-4 deg in the last 365 d (need >= 5): minus the OLS slope of along-strike distance |dlat| (deg) on time (yr). Positive = seismicity approaching the bin. 0 when too few events. |
| `x_sync` | 28-day counts of bg shallow events per bin (log1p), trailing 5-yr window (65 intervals, last interval ends at t): mean Pearson correlation of bin b with bins at distance 1..3. |
| `x_central` | eigenvector centrality of bin b in the same trailing correlation matrix (negative correlations set to 0, diagonal 0), scaled so the max is 1. |
| `x_shadow` | sum over M>=7 events with 1 yr < age: 10^(0.75(M-7)) exp(-age/10 yr) if the bin centre lies inside the rupture half-length (Wells & Coppersmith, as in the harness). |
| `x_load` | same, for bins within 150 km beyond the rupture ends. |

## Primary test (decided now)
- **Panel / model:** weekly panel (step 7 d, horizon 7 d), target **M >= 6.0**, logit (harness default). Train
  1976-2012, test 2013-01-01..2026-10.
- **Primary target: `y_main`** (a genuine precursor must predict mainshocks, which ETAS cannot). `y_all` is co-reported.
- **Primary statistic:** `ig_per_event` of baseline + all 8 features jointly (one block) vs the ETAS/clustering
  logit baseline.
- **Primary significance:** `H.surrogate_test(mode="phase_randomize")` on the 8-feature panel grid (same phases
  for every bin and feature), n = 1000 if the observed IG > 0 (n = 200 otherwise, since then p > 0.5 anyway).
- **Claim rule:** "positive" only if IG > 0 AND phase_randomize p < 0.0036 (Bonferroni, 14 hypotheses) AND the
  bin_permute surrogate p < 0.05 AND the leakage checks pass. "weak_positive" if p < 0.05 uncorrected.
  Otherwise null / negative.

## Secondary analyses (exploratory, reported with no multiplicity credit)
1. `y_all` on weekly M6; both targets on weekly M5.5.
2. Each family alone (8 single-feature or 2-feature runs), n = 200 surrogates.
3. GBM model (nonlinear) on the joint block.
4. Robustness: 30-day panel (s30_h30) M6; alternate split (train 1976-2005, test 2006-2012, i.e. a test window
   entirely inside the original training period, which includes Maule 2010); shallow targets are not
   available in the harness so are not run; M>=5.0 input events instead of M>=4.7 (completeness check).
5. **Training-only tuning:** a small grid per family (RTL r0 in {50,100} km x t0 in {0.5,1} yr; Z window in
   {0.5,1,2} yr; doughnut window in {3,5} yr; migration window in {180,365} d; sync window in {3,5} yr; shadow
   decay in {5,10,20} yr), selected by IG on an inner split (fit 1976-1999, validate 2000-2012, y_all + y_main
   summed). The selected set is then evaluated **once** on the 2013+ test. Reported as secondary.
6. Amplitude-adjusted (rank-remapped) phase surrogate and end-matched circular shift as robustness checks of the null.

## Leakage checks planned
- Assert that each feature at t is unchanged when all events at/after t are deleted (spot check on random t).
- No centered windows; all normalizations past-only.
- Catalog-density drift: repeat with M>=5.0 inputs (M>=5.0 is complete throughout).
- Look for the effect being driven by a single sequence (leave-one-year-out of the test IG via the year bootstrap).
