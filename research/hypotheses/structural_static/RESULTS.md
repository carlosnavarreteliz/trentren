# Results: structural_static (static structural controls on WHERE M≥6 earthquakes occur)

**Verdict: NULL.** Static structure adds no information beyond smoothed past seismicity and ETAS. The structure
tested is the trench-parallel gravity anomaly, incoming-plate roughness, trench sediment, and distance to subducting
ridges or fracture zones, plus secondary features (plate age, slab dip, trench depth, incoming-plate bathymetry,
forearc width). The preregistered primary test gives a slightly **negative** information gain. No configuration
reaches Bonferroni significance, and the few nominal p<0.05 results are scattered, inconsistent in sign across
settings, and within the number expected by chance.

## Data obtained
All data was obtained and is cached in `research/data/structural_static/`.

- Sandwell–Smith free-air gravity V33.1 and topography V27.1, 1′ grid, from topex.ucsd.edu, covering 47°S–15°S and 80°W–66°W.
- GlobSed v3 sediment thickness (Straume et al. 2019), from the Caltech mirror of the retired NCEI page.
- EarthByte 2020 seafloor-age grid.
- Slab2 could not be fetched, because ScienceBase sits behind a Cloudflare challenge. It would also leak, since it includes seismicity up to 2017. Slab dip was therefore estimated from **1973–2012 hypocenters only**, using Theil–Sen fits of depth against distance from the trench.
- Profiles are on a 0.05° latitude grid in `profiles_0p05.csv`. The figure is `out/profiles_vs_seismicity.png`.

## Primary test (preregistered, see PREREGISTRATION.md)
Setup: weekly harness panel, M≥6.0, 1° bins, `y_main`, logit, with baseline + F1–F4 compared to the harness baseline.
The null is a spatial Fourier phase-randomized surrogate of the along-strike profiles, n=1000. The harness's own
time-domain surrogates are degenerate for constant-in-time features.

| metric | value |
|---|---|
| N test cells | 72 |
| IG (full − base) | **−0.019 nats/eq** |
| year-block 95 % CI | [−0.036, −0.004] |
| IG (full − ETAS) | −0.027 |
| spatial-FT surrogate p (n=1000) | **0.82**; null mean −0.008, q95 +0.012 |
| spatial circular-shift p (n=500) | 0.84 |
| bin-permutation p (n=200) | 0.84 |
| AUC base → full | 0.665 → 0.651 |
| standardized coefficients | tpga −0.079, logsed −0.133, rough +0.001, logdist_ridge +0.002 |

The sign of the TPGA coefficient matches Song & Simons (2003): more negative TPGA goes with more large events.
That relation fitted on 1976–2012 does not carry over to 2013–2026.

## Secondary and robustness results (all logit unless noted)

| test | target | N test | IG full−base (nats/eq) | surrogate p (spatial FT) |
|---|---|---|---|---|
| weekly M6 | y_all | 85 | −0.008 | 0.81 (shift 0.83, permute 0.73) |
| weekly M5.5 | y_main | 193 | −0.008 | 0.72 |
| weekly M5.5 | y_all | 228 | −0.003 | 0.44 |
| weekly M6, GBM | y_main | 72 | +0.024 | 0.25 (null mean +0.018) |
| weekly M6, GBM | y_all | 85 | +0.016 | 0.28 (null mean +0.011) |
| 30-day M6 | y_main | 72 | −0.005 | 0.55 |
| 30-day M6 | y_all | 80 | −0.000 | 0.42 |
| alternate split, train ≤2004 / test 2005–2012 (backgrounds recomputed) | y_main | 34 | +0.009 | 0.13 |
| alternate split, same | y_all | 58 | +0.007 | 0.23 |
| all 9 features together | y_main | 72 | −0.023 | 0.66 |

On the GBM rows: adding *any* static column improves the overfit GBM baseline. The surrogate null mean is positive,
so the apparent +0.024 is not structure.

**Single features, exploratory.** There were 18 tests, each with spatial FT n=200, IG on weekly M6.
- Only `x_rough` on y_all reached p<0.05 (IG +0.0075, p=0.025). That is 1 of 18, which is chance level: about 0.9 such results are expected.
- `x_slabdip` on y_main gave p=0.055 and `x_logdist_ridge` on y_main gave p=0.090.
- TPGA alone made forecasts worse: IG −0.015 on y_main and −0.008 on y_all.

**0.25° along-strike Poisson count model (`count_model.py`).**
- How the forecast is built:
  - The model is fitted on covariates from 1976–1994 against counts in 1995–2012.
  - The forecast uses covariates from 1976–2012 to predict counts in 2013–2026.
  - The kernel bandwidth is chosen on training data only.
  - A variant uses leave-3°-block-out spatial cross-validation.

| config | IG Poisson (nats/eq) | p | spatial-only IG | p |
|---|---|---|---|---|
| M6 mainshocks, no CV (N=72) | +0.015 | 0.13 | +0.013 | 0.14 |
| M6 mainshocks, CV3 | −0.016 | 0.16 | −0.015 | 0.15 |
| M6 all events | −0.28 | 0.98 | −0.20 | 0.98 |
| M5.5 mainshocks, no CV (N=195) | +0.027 | **0.010** | +0.023 | 0.015 |
| M5.5 mainshocks, CV3 | +0.015 | 0.050 | +0.010 | 0.085 |
| M5.5 all events | −0.12 | 0.87 | −0.09 | 0.87 |
| alternate split 2005–12, M6 mainshocks, no CV / CV3 | −0.016 / +0.015 | 0.09 / 0.015 | | |
| alternate split, M6 all, CV3 | +0.11 | 0.08 | +0.13 | 0.035 |
| alternate split, M5.5 mainshocks / all | ≤0 | 0.42–0.83 | | |

That gives 16 configurations × 2 metrics. A few p-values fall between 0.01 and 0.05, but their signs flip between the
CV and no-CV versions and between splits, and none is below 0.0036. This is consistent with noise.

**M≥7 test events (descriptive).** Adding structure *lowered* the percentile rank of the epicentral bin for 5 of the
6 M≥7 events in 2013–2026:
- Iquique 2014 M8.2: 0.61 → 0.52.
- Iquique aftershock 2014 M7.7: 0.70 → 0.37.
- Illapel 2015 M8.3: 0.95 → 0.84.
- 2024 M7.4: 0.98 → 0.96.

It raised the rank only for Chiloé 2016 (0.03 → 0.07).

## Leakage checks
- **Static grids.** Coseismic gravity changes are at the microgal level, compared with anomalies of tens of mGal. Seafloor uplift is meters, compared with relief of kilometers. Altimetry products released after 2013 therefore carry negligible event information.
- **Slab geometry.** Built from pre-2013 hypocenters only. Fixed USGS depths of 33 and 35 km were excluded. South of about 42°S the dip is interpolated because seismicity there is sparse.
- **Locking models.** None were used.
- **Ridge and fracture-zone list.** The latitudes were fixed in the preregistration from pre-2013 literature, but written from memory (±0.5°) by an analyst who knows where the 2014–2015 events occurred. That is a possible subjective leak, but it could only bias *toward* a positive result, and the result is null.
- **Model fitting.** Feature definitions, the primary target and the primary null were written down before any test evaluation. Nothing was tuned on test data.

## Caveats
- The test period has only 72 M6 mainshock cells. An effect of about 0.02 nats/eq is not resolvable, so this null rules out only a sizeable incremental gain.
- **Harness note (not a bug):** a feature that is constant in time makes `phase_randomize` and `circular_shift` degenerate. The surrogate equals the original, so p≈1. I used spatial surrogates implemented in `common.py`. `bin_permute` works, but it ignores spatial autocorrelation.
- The harness bins are latitude only, and the static features are trench-band averages.
- The baseline's `log_bg_target` and `p_poisson` are the training-period target rates, which are in-sample for the train fit. This makes it harder for any static feature to earn weight. The count model avoids that problem and is also null for M≥6.
- The question of how structure controls *rupture extent* of great earthquakes, for example ridges acting as barriers, is not testable with 6 M≥7 events in the test period. This is a test of incremental forecasting power over smoothed seismicity, not a test of the physical association.

## Reproduce
```
cd research/hypotheses/structural_static
D=../../data/structural_static
/opt/miniconda3/bin/python3 -I build_profiles.py $D $D/profiles_0p05.csv
/opt/miniconda3/bin/python3 -I analyze.py $D/profiles_0p05.csv out primary 1000      # also: secondary, single 200, robust 500
/opt/miniconda3/bin/python3 -I count_model.py $D/profiles_0p05.csv out/count_model.json 1000
/opt/miniconda3/bin/python3 -I plot_profiles.py $D/profiles_0p05.csv out/count_model.json out/profiles_vs_seismicity.png
```
Outputs are in `out/*.json` and the logs in `out_*.log`.
