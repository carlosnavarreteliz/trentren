# RESULTS: enhanced_catalog (round 2)

**Verdict: NULL.** Lowering the magnitude of completeness from about M4.7 (USGS) to ML 3.0 (IPOC) or ML 2.5
(Riedel-Hornig, an ML catalog) adds no predictive information about M>=5 shocks beyond the harness ETAS-logit
baseline. The descriptive foreshock analysis finds no excess of small events before mainshocks compared with
matched control windows.

## Data obtained
| Catalog | Events | Period | Footprint | Threshold used | How it was built |
|---|---|---|---|---|---|
| IPOC, Sippl et al. 2023 (GFZ, doi:10.5880/GFZ.4.1.2023.004) | 182,841 | 2007-2021 | 18-24.5°S | ML>=3.0 | Uniform automatic picking on IPOC, no templates |
| Riedel-Hornig et al. 2026 (ESSD; Zenodo 15284376) | 601,979 | 2017-2021 | 18-42°S | ML>=2.5 | EQTransformer pretrained on Italy (not retrained on Chile), PyOcto, NonLinLoc/HypoDD, station set held fixed |

The CSN catalog was not needed, since two high-resolution catalogs were obtained. Total download size is 123 MB.

## Primary test (preregistered and unchanged)
The primary test used the IPOC catalog at ML>=3, the block {x_act7, x_act30, x_dbpos}, the weekly panel with
M>=5 targets, footprint latitude bins 22-27, and test period 2013-01-01..2021-12-24. The observed IG was below
zero, so the preregistered n=200 surrogates applied.

| Target | n_test | IG (nats/eq) | 95% year-bootstrap CI | p phase (n=200) | p window (n=1000) | p bin_permute | p circ | AUC base/full |
|---|---|---|---|---|---|---|---|---|
| **y_main** | 153 | **-0.0164** | [-0.026, -0.009] | **0.62** | 0.60 | 0.67 | 0.82 | 0.503 / 0.489 |
| y_all | 184 | -0.0047 | [-0.012, +0.001] | 0.37 | 0.32 | 0.45 | 0.72 | 0.543 / 0.538 |

## Key contrast: does lowering Mc help? (secondary)
The table compares the same features computed from the same catalog at different magnitude thresholds. Riedel
p-values come from the windowed surrogate.

| Features from | y_main IG | p | y_all IG | p |
|---|---|---|---|---|
| IPOC ML>=4.5 (what a M~4.7 catalog sees) | -0.0163 | 0.48 | -0.0128 | 0.41 |
| IPOC ML>=3.0 (primary) | -0.0164 | 0.62 | -0.0047 | 0.37 |
| IPOC ML>=2.5 | -0.0146 | 0.66 | -0.0066 | 0.50 |
| Riedel ML>=4.5 (2019-07..2021 test) | -0.0226 | 0.63 | -0.0158 | 0.61 |
| Riedel ML>=2.5 (2019-07..2021 test, 101/108 events) | +0.0006 | 0.31 | +0.0055 | 0.19 |

Lowering the threshold moves IG from slightly negative to about zero. No version beats the baseline, and none
comes close to alpha = 0.0083.

## Other robustness checks (IPOC, y_main / y_all IG; p from phase surrogates)
- **Single features:**
  - x_act7: -0.012 / -0.002.
  - x_act30: -0.0005 / -0.001.
  - x_dbpos (b-positive change): +0.0019 (p 0.38) / +0.0014 (p 0.30).
- **Detrended** (feature minus its causal trailing 52-week mean): -0.035 / -0.018.
- **GBM:** -0.018 / -0.006.
- **Shallow events only (<=60 km):** -0.0004 (p 0.15) / -0.0017 (p 0.13).
- **Daily panel (x_act1 + x_act7):** -0.034 / -0.004.
- **Target M>=5.5:** +0.028 (p 0.07, n=44) / +0.005 (p 0.14).
- **Target M>=6.0:** -0.14 / -0.18 (n=17/22, underpowered).
- **Riedel at M>=5.5:** -0.028 / -0.012. The M>=5.5 hint therefore does not replicate in the independent catalog.
- **Year by year:** the y_main per-year ΔLL is negative in 8 of 9 test years. Leave-one-year-out IG stays between
  -0.014 and -0.019, so the result does not depend on 2014 (Iquique).

## Descriptive foreshock analysis (pilot; 227 USGS M>=5 mainshocks inside the IPOC footprint, 2008-2021)
Each mainshock is compared with 200 control times at the same epicentre. Small events are counted within 50 km.
Under the null, 5% of mainshocks would exceed their own control 95th percentile by chance.

| Subset | n | Exceed q95, 1 d | Exceed q95, 7 d | Exceed q95, 30 d | Binomial p, 7 d |
|---|---|---|---|---|---|
| All, ML>=3 | 227 | 2.6% | 4.8% | 4.8% | 0.59 |
| Shallow <=60 km | 49 | 6.1% | 8.2% | 6.1% | 0.23 |
| Intermediate depth >60 km | 178 | 1.7% | 3.9% | 4.5% | 0.79 |
| Quiet history (no M4.7 in the prior 30 d) | 134 | 0.7% | 2.2% | 4.5% | 0.97 |
| M>=6 | 19 | 10.5% | 5.3% | 5.3% | 0.62 |
| All, ML>=4.5 (contrast) | 227 | 0.9% | 2.2% | 3.1% | 0.99 |

The well-known Iquique 2014 precursory sequence is a single case that the ETAS baseline already captures as
clustering. Outside ongoing clustering ("quiet history"), there is no foreshock excess. Pre-mainshock b-positive
over 90 d could be computed for only a few events, so it is uninformative.

## Calibration and amendment (disclosed)
The enhanced features exist only during the catalog window, which is a small part of the 1976-2021 panel. The
harness-default phase_randomize therefore randomizes a mostly constant, imputed series.
- **Calibration with pure-noise features in the Riedel design (10 runs):** the default null gave p<=0.01 in
  2 of 10 runs, so it is anti-conservative.
- **Windowed surrogate:** `ec_features.window_builder` applies the harness FT surrogate only inside the catalog
  window. In the same 10 noise runs its minimum p was 0.03.
- **Consequence for Riedel:** the default-null p-values were 0.011 (y_main) and 0.005 (y_all) with IG ≈ 0. Under
  the windowed null they become 0.31 and 0.19. The default-null Riedel values are a calibration artefact and
  should not be read as evidence.
- **Effect on the primary result:** the primary test stays as preregistered, with the default null (p=0.62). The
  windowed null was added after the fact as a robustness check and gives the same conclusion (p=0.60).

## Leakage checks
- **Annual counts:**
  - ML>=3 is about 600 per year throughout 2007-2021, except 2014 (the Iquique sequence).
  - ML<2.5 drops by about 45% after 2017, because of network degradation, which is why the threshold was set at 3.0.
  - Riedel's yearly counts are flat.
- **Self-normalization:** every feature is normalized against its own trailing rate in the same bin.
- **Causality:** features recomputed from a catalog truncated at 2014-03-20 are identical (max diff 0).
- **Retrospective construction:** neither catalog uses templates or a detector tuned on the Iquique sequence.
  Riedel's picker was trained on Italian data.
- **Inputs excluded:** neither ML of the large events nor the mainshock flag is used as an input.
- **Bias direction:** because the result is null, any remaining leakage would only have favoured the features.

## Caveats
- The test periods are short, about 9 years for IPOC and 2.5 years for Riedel, and the footprints are small. IPOC
  covers 6 latitude bins, so the IPOC test has limited power at M>=6.
- The footprint baseline itself has almost no skill on y_main (AUC 0.50). The question is therefore whether small
  events add information, not whether they beat a strong model.
- Detection saturates during aftershock sequences, which under-counts small events right after large ones.

## Files
- Scripts: `ec_load.py`, `ec_features.py`, `run_primary.py`, `run_secondary.py`, `run_riedel.py`,
  `run_window_null.py`, `run_events.py`, `annual_counts.py`.
- Outputs:
  - `primary.json`/`.log`, `secondary.json`/`.log`, `riedel.log`, `window_null_*.json`.
  - `events_foreshock.json`, `events_foreshock_ML*.csv`, `fig_diagnostics.png`.
  - Predictions: `preds_primary_y_*.csv`.
