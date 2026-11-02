# RESULTS: global_transfer_gbm (round 2)

**Verdict: NULL.** A LightGBM trained on 16 subduction margins (strict pre-2013 cutoff everywhere) does not beat the
harness ETAS-logit baseline on Chile 2013-2026. Pooling the other margins does rescue a flexible model: trained on
Chile only, the same model overfits badly. The gain only brings it back up to the level of the simple logit.

The preregistration (`PREREGISTRATION.md`, written before any fit) was followed as written, with no amendments. A
previous agent ran the pipeline and was cut off before writing this file. This file reports its completed outputs
(`results/`, `logs/`) without recomputing them.

## Data and pipeline
* USGS ComCat M>=4.7, 1973-2026, 54 yearly FDSN chunks (`download_catalog.py`, `raw/`).
* 16 margins with hand-digitized trench polylines (`margins.py`, `figs/margins_map.png`), 1-degree along-strike bins,
  harness weekly panel (7 d step and horizon, target M>=6.0). The panel has 916,900 rows. Margin sizes are in
  `logs/build_panels.log`. Outside Chile there are about 4,300 M>=6 events, versus 194 positive Chile training cells.
* Features come from the harness' own feature functions. **Check:** the Chile panel rebuilt with this wrapper matches
  `H.get_panel()` exactly (max abs diff = 0.0, 76,850 rows, at both M6.0 and M5.5).

## Primary test (Chile 2013+, weekly M>=6.0; M1 pooled strict GBM vs harness ETAS-logit)
Significance uses the preregistered exact year-block sign-flip test (2^14 patterns), with alpha 0.0042 per target.

| target | N events | IG nats/eq | sign-flip p | year-bootstrap 95% CI | AUC base / M1 | Molchan ASS base / M1 |
|---|---|---|---|---|---|---|
| y_all | 85 | **+0.024** | 0.241 | [-0.039, +0.083] | 0.697 / 0.697 | 0.696 / 0.696 |
| y_main | 72 | **-0.009** | 0.631 | [-0.056, +0.039] | 0.665 / 0.646 | 0.664 / 0.645 |

## Secondary comparisons (Chile, M6.0)
| target | comparison | IG | p | CI |
|---|---|---|---|---|
| y_all | M1 pooled GBM vs M2 Chile-only GBM (key contrast) | +0.227 | 0.0021 | [+0.101, +0.361] |
| y_all | M2 Chile-only GBM vs base | -0.203 | 0.987 | [-0.360, -0.039] |
| y_all | M3 pooled logit (F_base) vs base | +0.031 | 0.065 | [-0.007, +0.074] |
| y_all | M4 non-strict pooled GBM vs base | +0.022 | 0.254 | [-0.037, +0.085] |
| y_all | M5 pooled GBM on F_base vs base | +0.010 | 0.402 | [-0.065, +0.082] |
| y_main | M1 vs M2 (key contrast) | +0.289 | 0.0001 | [+0.181, +0.428] |
| y_main | M2 vs base | -0.298 | 0.9998 | [-0.451, -0.171] |
| y_main | M3 pooled logit vs base | -0.006 | 0.673 | [-0.027, +0.023] |
| y_main | M4 non-strict vs base | -0.002 | 0.527 | [-0.045, +0.039] |
| y_main | M5 vs base | -0.025 | 0.749 | [-0.091, +0.041] |

The transfer contrast (M1 vs M2) is highly significant for both targets. It measures the cost of overfitting a GBM on
about 150-200 Chilean positives, not a gain over physics-based clustering. The extra features (b-value,
time since M7, 10-year counts) add nothing: M1 and M5 are within noise of each other. ETAS rate dominates the pooled
GBM's gain importance (44% for y_all, 37% for y_main). The pooled strict logit (M3) also fails to beat the base.

## Leave-one-margin-out (16 margins, test 2013+; IG vs the margin's own local harness-equivalent logit)
Full table: `logs/summarize.log`. Figure: `figs/per_margin_ig.png`. Event-weighted means:

| target | pooled GBM vs local logit | strict LOMO GBM (margin never seen) vs local logit | local GBM vs local logit | pooled GBM vs local GBM |
|---|---|---|---|---|
| y_all | +0.020 (10/16 margins >0) | +0.000 | -0.272 | +0.292 (16/16 >0) |
| y_main | +0.028 (11/16 >0) | +0.004 | -0.309 | +0.337 (16/16 >0) |

Transfer works everywhere in the sense that a model that never saw a margin forecasts it about as well as that
margin's own fitted logit. It does not, however, beat the local logit in a consistent way. The few nominal per-margin
wins (HEL +0.52 with p=0.026 on 16 events, JPN y_main +0.20 with p=0.025) come from small samples, sit among
32 margin x target tests, and are not significant after multiplicity correction. JPN y_all goes the other way
(-0.11, p=0.96).

## Robustness and leakage checks
* **Strict cutoff:** every margin is trained only on cells with t_start+7d <= 2013-01-01. The non-strict variant
  (M4, other margins including 2013+) is no better than the strict one, so post-2013 information from other margins
  does not help either.
* **Feature reproduction:** the Chile features match the harness exactly (diff 0). No margin ID, coordinates or
  calendar time is used. Static per-bin rates come from pre-2013 rows only.
* **Level shift:** calibration-in-the-large on test (sum p / observed) is 81.3/85 for M1 and 85.2/85 for the base
  (y_all), and 59.4/72 for M1 and 57.6/72 for the base (y_main). The oracle level-matched IG is +0.025 (y_all) and
  -0.014 (y_main), so a constant offset does not hide anything.
* **Drop-one-test-year:** the primary IG ranges over [+0.006, +0.050] for y_all and [-0.021, +0.006] for y_main.
  It is never clearly positive.
* **Calibration of the sign-flip test** (`results/calibration_signflip.json`): 400 noise-feature model pairs give a
  pooled FPR of 0.068 at p<0.05 and 0.005 at p<0.0042. For random-walk noise on y_main the FPR at 0.05 rises to 0.15.
  The test is roughly calibrated at the Bonferroni level, but nominal p values near 0.05 are not trustworthy.
* **M>=5.5 panel (preregistered robustness item 5):** M1 vs base gives +0.036 (p=0.028, N=228) for y_all and +0.027
  (p=0.046, N=193) for y_main. The pooled logit gives +0.024 and +0.025 (p=0.03 and 0.015). These are nominal only:
  they are far from alpha=0.0042, sit in the range where the calibration shows inflated FPR, and the gain is small.
  They do not change the verdict. A small benefit from pooling at lower thresholds remains possible.
* **Disclosure:** the primary test is the preregistered year-block sign-flip, not `H.surrogate_test`. The surrogate
  test evaluates a feature added to the harness logit, and M1 is a different model family with a different training
  set. Running it would need out-of-fold M1 predictions over the training period, which were not produced.

## Caveats
* ComCat magnitudes are heterogeneous (mb vs Mw, revised values), and M4.7 completeness differs by margin and era.
  This applies equally to the baseline.
* The trench traces are coarse (about 50 km), and the 300/700 km corridor is fixed.
* The test has only 85 (y_all) and 72 (y_main) positive cells. Effects of about ±0.05 nats/eq cannot be detected.
* The GBM hyperparameters were fixed a priori and never tuned. Another learner might differ, but the pooled logit
  also ties the baseline.

## Reproduce
```
R=/Users/cnavarreteliz/Documents/Repos/earthquakes/research; D=$R/data/round2_global_transfer_gbm; G=$R/round2/global_transfer_gbm
PY="nice -n 10 /opt/miniconda3/bin/python3 -I"
$PY $G/download_catalog.py $D/raw
$PY $G/build_panels.py $D/raw $D/panels 6.0 5.5
$PY $G/calibrate_signflip.py $G/results/calibration_signflip.json 100
$PY $G/run_models.py $D/panels $G/results 6.0
$PY $G/run_models.py $D/panels $G/results 6.0 --lomo --targets=y_all   # and --targets=y_main
$PY $G/run_models.py $D/panels $G/results 5.5 --targets=y_all          # and --targets=y_main
$PY $G/summarize.py $G/results $G
```
