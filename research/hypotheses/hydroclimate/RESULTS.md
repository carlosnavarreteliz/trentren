# RESULTS: hydroclimate (rain, snow, groundwater-proxy loading, seasonality, ENSO/PDO)

**Verdict: NULL.** Hydrological and climatic loading variables add no out-of-sample skill beyond clustering
(ETAS) for Chilean M>=6 earthquakes. The preregistered primary test fails. The test period runs 2013-01-01 to
2026-10-17, with training on 1976-2012 (precipitation from 1979).

## Data obtained
- **Precipitation:** NOAA CPC Global Unified Gauge-Based Daily Precipitation, 0.5 deg, 1979-01-01..2026-10-16.
  1403 land cells were used, lat -46..-17, lon -76.5..-66, via PSL OPeNDAP.
- **Temperature:** NOAA CPC daily Tmax/Tmin, same grid. It drives a degree-day snow model.
- **Elevation:** OpenTopoData ETOPO1, used to define Andes cells (>= 1500 m). Open-Meteo was rate-limited
  (HTTP 429, shared quota).
- **Climate indices:** NOAA CPC ONI and NOAA NCEI ERSSTv5 PDO.
- **GRACE/GRACE-FO TWS was NOT used.** It needs an Earthdata login and has only ~10 training years (2002+).
- **ERA5 via Open-Meteo was not used.** The free call budget is far too small for 50 years x 58 points.

All features are strictly causal (days <= t_start - 2 d; ONI/PDO as of the 10th of month m+2). Climatologies
are fitted on train only, and reservoir spin-up years are masked. Scripts: `download.py`, `download_aux.py`,
`features.py`, `run.py`. Logs: `log_*.txt`. Numbers: `results_*.json`.

## Primary test (preregistered)
Weekly panel, M>=6.0, target **y_main**, logit, feature set A per latitude bin:
- the 180-day exponentially decaying precipitation load anomaly;
- the 30-day precipitation (loading-rate) anomaly.

| metric | value |
|---|---|
| IG full vs base (nats/eq) | **+0.0051** |
| year-block bootstrap 95% CI | [-0.0026, +0.0116] |
| IG full vs ETAS / base vs ETAS | -0.0029 / -0.0080 |
| phase_randomize surrogate p (n=1000) | **0.227** (null mean -0.005, q95 +0.020) |
| AUC base -> full | 0.665 -> 0.667 |
| Molchan ASS base -> full | 0.664 -> 0.666 |
| Rhoades T (anti-conservative, not relied upon) | T = 1.49 |
| leave-one-year-out IG | +0.003 .. +0.007 |
| n test events (positive cells) | 72 |

The observed IG sits well inside the surrogate null, so this is a **NULL**.

## All results (logit unless noted)
"IG" is full minus base in nats per target event. "vs ETAS" compares the full model with the pure additive
ETAS model. "p" is the surrogate p-value: phase_randomize unless another mode is named.

| feature set | panel | target | IG | vs ETAS | p | AUC base->full | n ev |
|---|---|---|---|---|---|---|---|
| **A load+rate (PRIMARY)** | wk M6 | y_main | +0.0051 | -0.0029 | **0.227** (n=1000) | 0.665->0.667 | 72 |
| A, bin_permute surrogate | wk M6 | y_main | +0.0051 | | 0.085 | | 72 |
| A, whole-year+offset shift of raw precip | wk M6 | y_main | +0.0051 | | 0.259 | | 72 |
| A | wk M6 | y_all | +0.0120 | +0.0114 | **0.055** (n=1000) | 0.697->0.700 | 85 |
| A, bin_permute surrogate | wk M6 | y_all | +0.0120 | | 0.035 (anti-conservative mode, FPR ~14%) | | 85 |
| A, end-matched circular shift | wk M6 | y_all | +0.0120 | | 0.130 (n=500) | | 85 |
| A, whole-year+offset shift | wk M6 | y_all | +0.0120 | | 0.075 | | 85 |
| A, 4-week differenced | wk M6 | y_main | +0.0169 | +0.0089 | 0.040 | 0.665->0.676 | 72 |
| A, 4-week differenced | wk M6 | y_all | +0.0008 | +0.0003 | 0.289 | 0.697->0.702 | 85 |
| A, GBM | wk M6 | y_main / y_all | +0.0059 / -0.0093 | -0.075 / -0.078 | | 0.644->0.644 / 0.695->0.681 | 72/85 |
| A | wk M5.5 | y_main | -0.0120 | -0.036 | 0.945 | 0.658->0.654 | 193 |
| A | wk M5.5 | y_all | -0.0135 | -0.012 | 0.955 | 0.697->0.693 | 228 |
| A | 30-day M6 | y_main | -0.0028 | -0.013 | 0.398 | 0.664->0.665 | 72 |
| A | 30-day M6 | y_all | -0.0056 | -0.019 | 0.552 | 0.671->0.668 | 80 |
| A, alt split (train <2006, test 2006+) | wk M6 | y_main | -0.0026 | -0.043 | 0.393 | 0.649->0.649 | |
| A, alt split | wk M6 | y_all | -0.0190 | -0.060 | 0.707 | 0.711->0.705 | |
| B seasonal sin/cos | wk M6 | y_main | **-0.0533** | -0.061 | 0.821 | 0.665->0.635 | 72 |
| B seasonal sin/cos | wk M6 | y_all | -0.0503 | -0.051 | 0.861 | 0.697->0.672 | 85 |
| C ONI, PDO, 12-mo changes | wk M6 | y_main | -0.0104 | -0.018 | 0.438 | 0.665->0.674 | 72 |
| C | wk M6 | y_all | -0.0062 | -0.007 | 0.299 | 0.697->0.702 | 85 |
| C, 4-week differenced | wk M6 | y_main | +0.0122 | +0.004 | 0.065 | 0.665->0.671 | 72 |
| C, 4-week differenced | wk M6 | y_all | +0.0042 | +0.004 | 0.199 | 0.697->0.699 | 85 |
| D Andean snow (SWE anom, 30-d dSWE) | wk M6 | y_main | +0.0028 | -0.005 | 0.154 | 0.665->0.666 | 72 |
| D | wk M6 | y_all | +0.0019 | +0.001 | 0.239 | 0.697->0.698 | 85 |
| E Andes 365-d load | wk M6 | y_main | -0.0167 | -0.025 | 0.930 | 0.665->0.660 | 72 |
| E | wk M6 | y_all | +0.0054 | +0.005 | 0.159 | 0.697->0.698 | 85 |

### Event-level seasonality (Schuster test vs uniform-time background)
Declustered (ZBZ) mainshocks, phase = day of year:

| set | n | p | mean day of year |
|---|---|---|---|
| M>=6.0, 1976-2012 | 146 | 0.10 | 109 (April) |
| M>=6.0, 2013-2026 | 72 | 0.057 | 315 (November) |
| M>=6.0, all | 218 | 0.79 | |
| M>=5.5, 1976-2012 | 483 | 0.039 | 94 (April) |
| M>=5.5, 2013-2026 | 195 | 0.35 | 353 (December) |
| M>=5.5, all | 678 | 0.17 | |

The preferred season flips by about 7 months between train and test, and the full-period test is null. That is
why the seasonal panel feature (B) is clearly *harmful* out of sample (IG -0.05). There is no evidence of
seasonal modulation of Chilean M>=5.5 mainshocks.

## Interpretation
- **No primary signal.** The preregistered primary test (A on y_main) is null: p = 0.23 with n=1000.
- **A on y_all is borderline and does not replicate.** The weekly M6 y_all gain of +0.012 nats/eq
  (p = 0.055) is fragile:
  - it disappears or reverses on the M>=5.5 panel (-0.014, p = 0.96), the 30-day panel, the GBM model and the
    2006 split;
  - the end-matched circular-shift p is 0.13;
  - calibration shows phase_randomize on y_all has an FPR of ~7% at p<0.05.

  These are the signature of noise.
- **Nothing survives multiple comparisons.** ~25 secondary tests were run. The smallest p-values are 0.035
  (bin_permute, a mode the calibration shows to be anti-conservative) and 0.040 (differenced A on y_main).
  About 1-2 such values are expected by chance, none of them is anywhere near Bonferroni (0.0036), and none
  replicates across panels.
- **Effect sizes are tiny.** Every |IG| is <= 0.017 nats/eq, about one-tenth of the ETAS clustering gain on
  y_all. Physically this fits: the loads are ~kPa (1 m of water = 10 kPa), spread over hundreds of km at the
  surface, while the seismogenic megathrust lies at 10-50 km depth.

## Leakage checks
- **Precipitation, temperature and ENSO data:** physically independent of earthquakes, so there is no
  "revised because of the quake" channel. The CPC gauge network is unrelated to the seismic network.
- **Feature windows:** every feature uses days <= t_start - 2 d (verified in code: `_sample_at` and
  `_window_sum` use lag_days=1, as in `H.align_daily_series`). ONI uses a centred 3-month mean, so a
  conservative availability date of the 10th of month m+2 is applied. PDO uses the same rule.
- **Climatologies:** fitted only on train issue times (t+h <= 2013-01-01). For the 2006 split they are refitted
  on <2006.
- **Seasonal features (B):** deterministic, so using the forecast window is legitimate.
- **Targets and declustering flags:** only the harness targets are used, and no `mainshock` flags appear in any
  feature.
- **Unresolved data-provenance caveat:** CPC 1979-2005 is a retrospective gauge analysis, while 2006+ is the
  real-time product. This is a level and coverage change, not a seismic leak.

## Caveats
- **Spatial unit:** 1-degree latitude bins. Precipitation is averaged over all land in each band, including
  the Argentine and Bolivian sides. Finer spatial or depth-specific loading (e.g. shallow crustal or volcanic
  seismicity under the Andes) was not tested. The harness target is mostly interface and intraslab M>=6.
- **Snow model:** CPC 0.5 deg temperatures are too warm for the high Andes, so the degree-day SWE
  underestimates snowpack (bin max ~150 mm anomaly). Feature D is a weak proxy.
- **Power:** with 72-85 test events, an effect of ~0.01-0.02 nats/eq cannot be distinguished from noise.
  The result rules out a *useful* hydroclimate predictor, not an arbitrarily small modulation.
- **Precipitation start:** precipitation starts in 1979. Rows from 1976-1979 (1981 for E) are NaN and are
  imputed with the train mean (~8-11% of train rows).
- **Load:** the machine was heavily loaded (load avg ~40), so surrogate runs were slow, but all completed.

Reproduce:
```
D=/Users/cnavarreteliz/Documents/Repos/earthquakes/research/data/hydroclimate
S=/Users/cnavarreteliz/Documents/Repos/earthquakes/research/hypotheses/hydroclimate
/opt/miniconda3/bin/python3 -I $S/download.py $D && /opt/miniconda3/bin/python3 -I $S/download_aux.py $D
for st in schuster sanity; do /opt/miniconda3/bin/python3 -I $S/run.py $D $st; done
/opt/miniconda3/bin/python3 -I $S/run.py $D primary 1000
for st in yall1000 secondary robust; do /opt/miniconda3/bin/python3 -I $S/run.py $D $st; done
```
