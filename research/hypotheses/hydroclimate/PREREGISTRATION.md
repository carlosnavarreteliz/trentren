# PREREGISTRATION: hydroclimate (hydrological and climatic loading)

Written 2026-10-17. No `evaluate()` call on any test-period data has been made before this file was written.

## Hypothesis
Surface water loading changes the stress on the Chilean megathrust and slab, and so modulates the timing of M>=6
earthquakes along strike. That loading covers rain, groundwater and Andean snowpack, and it is modulated
seasonally and by ENSO/PDO. If so, a causal per-latitude-bin hydrological load anomaly and loading rate should
add information beyond clustering (ETAS). The prior is weak: the stresses are ~1-10 kPa (1 m water = 10 kPa),
comparable to tides, and published claims of seasonality in subduction seismicity are mixed.

## Data (free, no login)
- **Precipitation:** NOAA CPC Global Unified Gauge-Based Daily Precipitation, 0.5 deg, 1979-present
  (PSL OPeNDAP, `cpc_global_precip/precip.YYYY.nc`), subset to lat -46.5..-16.5, lon -76.5..-66.
- **Temperature:** NOAA CPC Global daily Tmax/Tmin, 0.5 deg, 1979-present (`cpc_global_temp`). Used only for
  the secondary snow model.
- **Elevation:** Open-Meteo elevation API (Copernicus DEM 90 m) at the 0.5 deg cell centres. It is used to
  split cells into Andes (centre elevation >= 1500 m) and lowland/coast.
- **ONI:** NOAA CPC ONI (`oni.ascii.txt`).
- **PDO:** NOAA NCEI ERSSTv5 PDO index.
- **GRACE/GRACE-FO TWS is NOT used.** It needs an Earthdata login, starts only in 2002 (10 training years) and
  has an 11-month 2017-2018 gap. This will be stated in the results.
- Open-Meteo ERA5 point series were considered and rejected. 50 years x ~58 points exceeds the free daily call
  budget (~1300 weighted calls per point), whereas CPC is gauge-based and has a near-real-time product.

## Causality rules
- Every feature at issue time t_start uses only days D < t_start - 1 day (lag 1 day). This matches
  `H.align_daily_series(lag_days=1)`.
- ONI and PDO for month m are assumed available only from the 10th of month m+2. This is conservative: ONI is a
  centred 3-month mean, so the value labelled for month m uses month m+1.
- Climatologies (bin x week-of-year means) and any standardization are fitted on 1979-2012 only.
- Precipitation cannot be revised because of an earthquake, so data revision is not a leakage channel. Gauge
  network density is unrelated to seismic networks.
- 1976-1978 has no CPC data, so those rows are NaN and the logit imputes the train mean. That is about 8% of the
  training rows.

## Features
Per bin b, P_land[b,d] is the mean daily precipitation (mm/day) over CPC land cells whose centre lies in the
latitude bin and at lon -76.5..-66. P_andes[b,d] is the same quantity over cells with elevation >= 1500 m; if a
bin has no such cell, its lowland mean is used.

### PRIMARY feature set (A), per bin, 2 columns
1. `x_load180_anom`: the exponentially decaying water load L[b,t] = sum over D < t-1 of P_land[b,D]
   * exp(-(t-1-D)/180 d) (mm). The anomaly is relative to the bin x ISO-week-of-year climatology of L fitted on
   train years 1979-2012.
2. `x_rate30_anom`: the 30-day precipitation sum over days [t-31, t-1), minus its bin x week-of-year train
   climatology (the loading rate).

### SECONDARY feature sets (reported, none of them primary)
- **B, seasonal phase (global):** sin and cos of 2*pi*doy/365.25 at the forecast-window midpoint. This is
  deterministic and known in advance. An event-level Schuster test of the day-of-year of declustered M>=6.0 and
  M>=5.5 mainshocks is run with `schuster_vs_background` against uniform random times, for 1976-2012 and
  2013-2026 separately.
- **C, ENSO/PDO (global):** the latest available ONI and PDO, plus their 12-month causal differences
  (`x_oni`, `x_pdo`, `x_doni12`, `x_dpdo12`).
- **D, Andean snow load (per bin):** a degree-day SWE model on the Andes cells. Snow accumulates when
  Tmean <= 0 C. Melt is 3 mm/C/day * max(Tmean, 0), capped at the available SWE. The features are the SWE
  anomaly (train week-of-year climatology) and the 30-day SWE change (`x_swe_anom`, `x_dswe30`).
- **E, Andes-only load (per bin):** as A but from P_andes, with tau = 365 d (`x_andesload365_anom`).

## Primary test
- **Panel and target:** the weekly harness panel (step 7 d, horizon 7 d), target M>=6.0, **target = y_main**
  (declustered ZBZ mainshocks). Clustering cannot predict mainshocks, so a real loading precursor should show
  up there.
- **Model:** logit (harness default). Train 1976-2012 (purged), test 2013-01-01..2026-10-17.
- **Statistic:** `ig_per_event` (full minus baseline).
- **Significance:** `H.surrogate_test(featsA, mode='phase_randomize', n=1000, target='y_main')`, one-sided.
  The multivariate FT surrogate keeps the spectrum, including the annual line with a random phase, and the
  cross-bin structure.
- **Decision:** the result is POSITIVE only if all three hold:
  - IG > 0, and IG(full vs ETAS) > IG(base vs ETAS);
  - p < 0.0036 (Bonferroni, 14 hypotheses);
  - the gain is not carried by a single test year (leave-one-year-out IG stays > 0).
  - WEAK_POSITIVE if p < 0.05 but not < 0.0036. Otherwise NULL.

## Secondary and robustness checks (all reported regardless of outcome)
- **Feature set A on other targets:** the same feature set A on y_all (M>=6 weekly) and on M>=5.5 weekly
  (y_all, y_main), each with phase_randomize n=200 (n=1000 if p<0.01).
- **Feature set A, other surrogate modes:** bin_permute (n=200). A circular shift by whole years plus a random
  offset in [-30, +30] days is applied to the raw daily precipitation, then the features are rebuilt (builder,
  n=200). The whole-year shift keeps the shared seasonal cycle aligned, so it tests the anomaly content
  specifically.
- **A causally differenced version of feature set A:** the 4-week change of each feature.
- **Feature sets B-E on M6 weekly:** y_main and y_all, logit, phase_randomize n=200.
- **Robustness:**
  - GBM model;
  - the 30-day panel (s30_h30);
  - an alternative split: train 1976-2005, test 2006-2026 (the Poisson background columns still come from
    the harness's pre-2013 fit, which is a caveat);
  - leave-one-test-year-out IG.

## Amendment 1 (implementation details, made before any test-period evaluation)
- **Elevation source:** Open-Meteo returned HTTP 429 (shared quota), so elevations come from OpenTopoData
  ETOPO1 at the 0.5 deg cell centres. The 1500 m Andes threshold is unchanged. Bins 0-8 (-46..-37) have at most
  2 cells at or above 1500 m; bin 0 has none and falls back to all land cells.
- **Reservoir spin-up:** reservoirs start at 0 on 1979-01-01. The tau=180 load and the SWE features are NaN
  before 1980-01-01, and the tau=365 load is NaN before 1981-01-01. NaN rows are imputed with the train mean
  inside the logit.
- **Land-cell domain:** "Land cells" include the Argentine and Bolivian parts of each latitude band within
  lon -76.5..-66. That is regional loading within ~300 km of the trench-parallel margin.
- **Results from the event-level Schuster seasonality test** (secondary set B) were seen before the panel
  evaluations. They do not change any feature definition.
