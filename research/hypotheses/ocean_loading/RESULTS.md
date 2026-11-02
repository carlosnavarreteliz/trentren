# RESULTS: ocean_loading (non-tidal sea level + barometric loading of the forearc)

**Verdict: NULL.** On the preregistered primary test, a causal ocean-bottom-pressure (OBP) anomaly adds no information beyond the ETAS/clustering baseline:
- target: weekly M>=6 mainshocks;
- IG = -0.0023 nats/eq;
- phase-randomized surrogate p = 0.41 (n = 1000).

The secondary analyses show a small nominal gain on y_all (all events), but it does not survive the checks: it shrinks under masking, is absent at other horizons and in the components, and its p-value is within the harness's measured false-positive rate.

## Data obtained
- **UHSLC hourly tide gauges** (fast-delivery files, full history): Arica 083, Antofagasta 080, Caldera 088, Valparaiso 081 and Puerto Montt 684.
  - Coverage from 1976 is 0.68-0.91.
  - FD ends 2026-08-31, and Puerto Montt ends 2024-04.
  - Iquique, Coquimbo, Talcahuano and Corral are **not in UHSLC**. IOC/SHOA real-time records start around 2008 and are too short for a 1976-2012 training period.
  - The RQDS hourly files were used for a cross-check. They end in 1998/1999 at Arica and Caldera and in 2018 elsewhere.
- **ERA5 daily mean surface pressure** (Open-Meteo archive, `models=era5`) at 29 offshore forearc points, one per 1° bin, 1974-2026.
- Cached under `research/data/ocean_loading/`.

## Features (see PREREGISTRATION.md)
- `L_b(D) = 0.1006 hPa/mm * eta_anom(D) + P_anom,b(D)`.
  - `eta_anom` is the causal anomaly (relative to the trailing 365 days) of the UTC-day mean sea level at the nearest station with data, within 6° of latitude.
  - `P_anom` is the same causal anomaly for ERA5 surface pressure.
- `x_dL7`: the mean of L over the last 7 days minus the mean of L over the 30 days before that.
- `x_L30`: the mean of L over the last 30 days.
- Only days D < t_start - 1 day are used.
- My vectorized window mean matches `H.align_daily_series` to 1e-13.
- NaN rate: 12% in train and 5% in test.
- Training-period correlation with y_main and y_all is between 0.002 and 0.009.

## Results (logit; IG in nats per target event versus the ETAS-logit baseline; test period 2013-01-01 to 2026-10)

| analysis | target | N test | IG full-base | IG full-ETAS | 95% CI (year bootstrap) | AUC base/full | ASS base/full | surrogate p |
|---|---|---|---|---|---|---|---|---|
| **PRIMARY** weekly M6 | **y_main** | 72 | **-0.0023** | -0.0103 | [-0.017, +0.017] | 0.665/0.662 | 0.664/0.661 | **0.407** (phase, n=1000); bin-perm 0.50; circ 0.49 |
| weekly M6 | y_all | 85 | +0.0120 | +0.0114 | [+0.002, +0.023] | 0.697/0.703 | 0.696/0.702 | 0.035 (phase, n=200); bin-perm 0.035; circ 0.050 |
| weekly M6, GBM | y_all / y_main | 85/72 | +0.0014 / +0.054 | - | - | - | - | not run (GBM baseline itself overfits) |
| weekly M5.5 | y_all | 228 | +0.0060 | +0.0071 | [-0.003, +0.015] | 0.697/0.699 | 0.694/0.697 | 0.020 |
| weekly M5.5 | y_main | 193 | +0.0046 | -0.0198 | [-0.003, +0.014] | 0.658/0.662 | 0.657/0.660 | 0.050 |
| sea level only, weekly M6 | y_all / y_main | 85/72 | -0.0174 / -0.0155 | - | - | - | - | 0.78 / 0.74 |
| pressure only, weekly M6 | y_all / y_main | 85/72 | -0.0456 / -0.0254 | - | - | - | - | 0.98 / 0.92 |
| masked after M>=7 (±3 bins, 400 d) | y_all / y_main | 85/72 | +0.0086 / -0.0017 | - | - | - | - | 0.095 / 0.44 |
| 30-day panel M6 | y_all / y_main | 80/72 | -0.0188 / -0.0196 | - | - | - | - | 0.74 / 0.72 |
| daily panel M6 | y_all / y_main | 92/72 | -0.0050 / -0.0231 | - | - | - | - | 0.41 / 0.76 (n=50) |
| alt split (train to 2004, test 2005-2026) | y_all / y_main | 143/106 | +0.0055 / +0.0060 | - | - | - | - | 0.090 / 0.114 |
| causal Godin-detided sea level (deviation from prereg) | y_all / y_main | 85/72 | +0.0137 / +0.0014 | - | - | - | - | 0.040 / 0.24 |
| RQDS source instead of FD | y_all / y_main | 85/72 | +0.0085 / -0.0018 | - | - | - | - | - |

Notes on the table:
- In the alt-split rows, the Poisson background columns in the panel were still fitted on data before 2013.
- FD and RQDS features correlate at 0.98 (`x_dL7`) and 0.94 (`x_L30`).

## Interpretation
- **Primary (preregistered):** a clean null. The IG is indistinguishable from 0. Its year-bootstrap CI is ±0.017 nats/eq, so any real effect on mainshock probability is smaller than about 0.02 nats/eq at weekly resolution. With 72 test cells, an effect of about 0.05 nats/eq would already be hard to detect.
- **Secondary y_all signal (p of 0.035 at M6 and 0.020 at M5.5):** I do not consider it evidence, for five reasons.
  1. It is not Bonferroni-significant. About 20 secondary tests were run here, and calibration v2 measures a phase_randomize false-positive rate of about 7% on y_all for pure noise (14% for bin_permute).
  2. It appears on y_all, where aftershock sequences dominate, but not on y_main. A physical loading trigger should show up on mainshocks.
  3. Masking periods after M>=7 shrinks it from +0.012 to +0.009 (p = 0.095). The masking removes coseismic and post-seismic tide-gauge steps, for example about +60 to +75 mm local relative steps at Antofagasta in 1995 and Valparaiso in 2010. Some of the signal is therefore probably post-seismic contamination.
  4. Neither physical component works alone. Sea level alone gives IG < 0 and pressure alone gives IG < 0, and the coefficient signs are unstable across panels. The 7-day change carries a positive coefficient (more loading means more events), which is the opposite of the unclamping hypothesis.
  5. It vanishes on the 30-day and daily panels and with GBM.
- **Magnitude:** even taken at face value, +0.012 nats/eq means target probabilities shift by about 1%. That has no practical forecasting value.

## Leakage checks
- Only hourly data averaged by UTC day (or a trailing Godin filter) were used, never UHSLC's centered filtered daily product.
- All anomalies use trailing windows.
- No missingness indicator was used, because gauge outages after tsunamis could leak.
- The M>=7 masking test is described above.
- FD and RQDS give consistent results.
- ERA5 is a reanalysis. Pressure is not affected by earthquakes, but its real-time latency is about 5 days. A strictly operational version would need a lag of 5 days or more, and with the result already null this does not change the verdict.

## Caveats
- Only 5 continental gauges are available, so bins are assigned the nearest gauge within 6°. Gauges are on the coast, not at the trench.
- The UTC-day mean leaves a few-cm aliased M2 residual. The Godin variant was run as a check and changes nothing material.
- Puerto Montt data end in 2024, and all FD data end on 2026-08-31, so the last weeks of features are mean-imputed.
- Calibration v2 was still incomplete when this was read.
- No bug was found in harness.py.

## Reproduce
```
cd research/hypotheses/ocean_loading
/opt/miniconda3/bin/python3 -I fetch_era5.py ../../data/ocean_loading/era5
# (UHSLC hourly: curl https://uhslc.soest.hawaii.edu/data/csv/fast/hourly/hNNN.csv and .../rqds/pacific/hourly/hNNNa.csv -> data/ocean_loading/hNNN_{fd,rq}.csv)
/opt/miniconda3/bin/python3 -I build_features.py ../../data/ocean_loading fd
/opt/miniconda3/bin/python3 -I build_features.py ../../data/ocean_loading rq
/opt/miniconda3/bin/python3 -I build_features.py ../../data/ocean_loading fd godin
/opt/miniconda3/bin/python3 -I analysis.py ../../data/ocean_loading primary 1000
/opt/miniconda3/bin/python3 -I analysis.py ../../data/ocean_loading secondary 200
/opt/miniconda3/bin/python3 -I analysis.py ../../data/ocean_loading robust 200
```
Outputs: `results_primary.json`, `results_secondary.json`, `results_robust.json`, plus the `*.log` files.
