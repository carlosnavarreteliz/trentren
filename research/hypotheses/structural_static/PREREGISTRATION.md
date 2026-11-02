# Preregistration: structural_static

Written 2026-10-17, **before** any feature was evaluated on test data (2013-01-01 onward). Data downloaded so far:
Sandwell–Smith free-air gravity V33.1 and topography V27.1 (1′, topex.ucsd.edu), GlobSed v3 sediment thickness
(Straume et al. 2019, Caltech mirror), and the EarthByte 2020 seafloor-age grid. Slab2 could not be downloaded
(ScienceBase is behind a Cloudflare challenge), and Slab2 also contains seismicity up to 2017, which would leak.
Slab geometry is therefore estimated **from pre-2013 hypocenters only** (USGS catalog, 1973-2012).

## Hypothesis
Static structure of the margin and the incoming plate explains where M≥6 earthquakes occur, beyond what
smoothed past seismicity and ETAS clustering already explain. The structure includes trench-parallel gravity anomaly,
roughness of the incoming plate, trench sediment, and subducting ridges and fracture zones. If so, adding these
features should raise the test-period (2013–2026) log-likelihood of an along-strike forecast.

This is a "where", not a "when", hypothesis. The features do not change in time.

## Features
All features are computed on a 0.05° latitude profile and then averaged within each forecast bin. Trench position
is the deepest point of each 1′ latitude row between 76.5°W and 69.5°W, median-smoothed along strike. x is the
east-west distance from the trench in km, positive landward.

| id | feature | definition |
|---|---|---|
| F1 `x_tpga` | trench-parallel gravity anomaly | free-air gravity minus the margin-average trench-normal profile ḡ(x), averaged over x ∈ [20,120] km (Song & Simons 2003 style) |
| F2 `x_rough` | incoming-plate roughness | RMS of high-passed bathymetry (bathymetry minus a 50 km running mean along x) over x ∈ [−150,−30] km |
| F3 `x_logsed` | trench sediment | log10(1 + GlobSed thickness in m) averaged over x ∈ [−30,0] km |
| F4 `x_logdist_ridge` | distance to a subducting ridge or fracture zone | log10(0.1 + |lat − nearest feature-trench intersection|), using a list fixed from pre-2013 literature: Iquique Ridge 20.5°S, Taltal Ridge 25.5°S, Copiapó Ridge 27.3°S, Challenger FZ 30.3°S, Juan Fernández Ridge 32.7°S, Mocha FZ 38.3°S, Valdivia FZ 40.0°S, Chiloé FZ 42.3°S, Guafo FZ 44.5°S, Chile Rise 46.3°S |

**Secondary features, never part of the primary composite:**
- F5 `x_age`: plate age at x ∈ [−60,−10] km.
- F6 `x_slabdip`: slab dip from the slope of hypocenter depth against x for 1973–2012 M≥4.5 events at 20–200 km depth within ±0.5° of latitude.
- F7 `x_trenchdepth`.
- F8 `x_inplate_bathy`: mean incoming-plate bathymetry at x ∈ [−150,−30] km, with ridges showing as shallow.
- F9 `x_forearc_width`: distance from the trench to the coastline.

The primary composite is **F1 to F4** as four standardized columns added together to the harness logit.

## Primary test (single, preregistered)
- **Panel:** harness weekly panel (step 7 d, horizon 7 d), target M≥6.0, 1° bins, `model="logit"`.
- **Target:** `y_main` (declustered mainshocks). Clustering cannot explain where mainshocks happen, and the hypothesis is about long-term location.
- **Statistic:** `ig_per_event`, the information gain of baseline + F1–F4 over the harness baseline (ETAS + clustering + training-period smoothed seismicity rates), on test 2013-01-01..2026-10.
- **Significance:** the harness `phase_randomize` surrogate is meaningless here, because a constant-in-time series has only the DC component. The primary null is therefore a **spatial Fourier phase-randomized surrogate**:
  1. Take the 4 profiles on the 0.05° grid over 47°S–16°S.
  2. Apply `H.ft_surrogate` along latitude with the same phases for all 4 features, which preserves each spectrum, the spatial autocorrelation and the cross-correlations between features.
  3. Re-bin and re-evaluate.
  - n = 1000, one-sided p = (1+#null≥obs)/(n+1).
- **Decision:**
  - **positive:** IG > 0, and IG of the full model vs ETAS > 0, and p < 0.0036 (Bonferroni, 14 hypotheses), and the leakage checks pass.
  - **weak_positive:** p < 0.05.
  - **null:** otherwise.

## Secondary analyses (reported, not used for the verdict)
1. The same test on `y_all` (M6.0), and on M5.5 `y_main` / `y_all`.
2. A spatial circular-shift surrogate: all profiles shifted jointly along strike, with wrap and reflection, by at least 2°. Also a bin-permutation surrogate.
3. Each feature F1..F9 on its own, one at a time; this is exploratory and multiple-testing caveats apply.
4. A **0.25° along-strike Poisson count model**, which is closer to the long-term question:
   - Target: counts of M≥6 mainshocks and all M≥6 per 0.25° bin in the test period.
   - Baseline: log of Gaussian-smoothed 1976–2012 M≥4.7 rate. The bandwidth is chosen on training only: 1976–1994 rates predict 1995–2012 M≥6 counts.
   - Model: Poisson GLM fitted on 1976–2012 counts, with and without F1–F4.
   - Statistic: IG per test event.
   - Variant: spatial-block cross-validation, with coefficients fitted excluding a ±1.5° block around each predicted bin.
   - The same spatial FT surrogate is used.
5. Robustness:
   - A 30-day panel (s30_h30).
   - An alternate split year: train ≤2004, test 2005–2012. This runs in the count model, and in the harness panel with the background rates recomputed from the shortened training period.
   - GBM.
6. M≥7: as a descriptive check only, where the M≥7 epicenters of 2013–2026 fall in the rank order of each model's long-term rate. There are only a handful of events.

## Leakage checks planned
- Coseismic changes to gravity and bathymetry are microgal-scale, against anomalies of tens of mGal, so these grids are treated as static.
- No coupling or locking models are used, because those published after 2014 leak.
- Slab geometry comes from pre-2013 hypocenters only.
- The ridge/FZ list is fixed from pre-2013 literature, before looking at test data.
- Feature definitions are fixed here. If F1–F4 fail, no re-selection will be presented as confirmatory.
