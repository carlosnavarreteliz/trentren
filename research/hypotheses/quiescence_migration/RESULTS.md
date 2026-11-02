# RESULTS: quiescence_migration (RTL, Z-value quiescence, Mogi doughnut, migration, inter-bin synchronization, stress shadow)

**Verdict: NULL.** On the preregistered primary test (weekly panel, M>=6.0, `y_main`, all 8 features as one block, logit),
the features make the forecast slightly **worse** than the ETAS/clustering baseline: IG = -0.051 nats/eq, with a
phase-randomized surrogate p of 0.71. On `y_all` the IG is -0.013 (p = 0.47). No family helps on its own at M6.
One exploratory secondary result came in at p≈0.01 uncorrected: weekly M5.5 `y_all`, IG +0.020. It does not reach the
Bonferroni threshold. It does not appear on `y_main`, at M6, on the 30-day panel, or on the
independent 2006-2012 validation window. It looks like intermediate-term clustering (recent activation), not a
quiescence precursor.

## What was tested
The features are defined in `PREREGISTRATION.md` and implemented in `qm_features.py`. They are built from the harness
catalog only: M>=4.7, ZBZ background events with the causal `bg` flag, depth <= 70 km. Every feature at issue time t
uses only events before t.

| family | columns |
|---|---|
| RTL (Sobolev), r0=100 km, t0=1 yr | `x_rtl` |
| Z-value quiescence, 1 yr window vs the preceding 10 yr, bins +-1 | `x_zq` |
| Mogi doughnut, 5 yr ring (bins 2-3 away) vs core (bins +-1), relative to the preceding 15 yr | `x_doughnut` |
| along-strike migration toward the bin (slope of distance on time, last 365 d, +-4 deg) | `x_migr` |
| inter-bin synchronization: 5-yr rolling correlation of 28-day log-counts, mean with neighbours 1-3 bins away | `x_sync` |
| eigenvector centrality in that correlation network | `x_central` |
| stress shadow / loading from M>=7 ruptures older than 1 yr (inside / within 150 km beyond the ends, 10-yr decay) | `x_shadow`, `x_load` |

## Primary result (preregistered)
Weekly panel, M>=6.0, logit, train 1976-2012, test 2013-01-01 to 2026-10. Script `run_primary.py`, output `primary.json` and `primary.log`.

| target | N test cells | IG vs baseline (nats/eq) | IG vs ETAS | year-bootstrap 95% CI | phase_randomize p | bin_permute p | circ-shift (end-matched) p | AUC base / full |
|---|---|---|---|---|---|---|---|---|
| **y_main (primary)** | 72 | **-0.051** | -0.059 | [-0.118, -0.007] | **0.71** (n=200) | 0.68 | 0.71 | 0.665 / 0.642 |
| y_all | 85 | -0.013 | -0.013 | [-0.044, +0.013] | 0.47 (n=200) | 0.30 | — | 0.697 / 0.694 |

The preregistration set n=1000 surrogates only if the observed IG was positive. With a negative IG, n=200 is enough.
In the surrogate null, an 8-column block of autocorrelation-matched noise costs about -0.034 nats/eq on y_main. The real
features did no better than that noise.

## Secondary and robustness results (exploratory, no multiplicity credit)
Scripts `run_secondary.py` (families, variants, tune) and `run_m55_followup.py`. Outputs: `families.json`,
`variants.json`, `tune.json`, `m55_followup.json`.

| analysis | target | IG (nats/eq) | surrogate p (phase) |
|---|---|---|---|
| family RTL, weekly M6 | y_main / y_all | -0.022 / -0.009 | 0.88 / 0.79 |
| family Z-value | y_main / y_all | -0.000 / -0.001 | 0.38 / 0.48 |
| family doughnut | y_main / y_all | -0.009 / +0.003 | 0.76 / 0.10 |
| family migration | y_main / y_all | -0.004 / -0.005 | 0.62 / 0.71 |
| family sync + centrality | y_main / y_all | -0.007 / -0.006 | 0.55 / 0.61 |
| family shadow + load | y_main / y_all | -0.009 / +0.013 | 0.71 / 0.085 |
| GBM joint, weekly M6 | y_main / y_all | -0.056 / -0.019 | — |
| 30-day panel, M6 | y_main / y_all | -0.039 / -0.001 | 0.58 / 0.26 |
| inputs M>=5.0 (complete throughout), weekly M6 | y_main / y_all | -0.049 / -0.002 | 0.78 / 0.33 |
| alternate split (train 1976-2005, test 2006-2012, includes Maule 2010), M6 | y_main / y_all | +0.002 / +0.003 | 0.19 / 0.32 |
| params tuned on inner split only (fit 1976-1999, validate 2000-2012), tested once on 2013+ | y_main / y_all | -0.027 / +0.022 | 0.51 / 0.060 |
| weekly M5.5 | y_main | -0.003 | 0.43 |
| **weekly M5.5** | **y_all** | **+0.020** (CI [+0.002, +0.036]) | **0.013 (n=1000)**; bin_permute 0.010 (n=500); circ 0.045 |
| weekly M5.5, inputs M>=5.0 | y_all | +0.015 | 0.040 |
| weekly M5.5, alternate split 2006-2012 | y_all | -0.012 | 0.47 |
| weekly M5.5, GBM | y_all | +0.009 | — |

The tuning grid selected r0=100 km and t0=0.5 yr for RTL, a 0.5 yr Z window, a 5 yr doughnut window, a 180 d
migration window, a 5 yr sync window and a 5 yr shadow decay.

### The M5.5 `y_all` result
- It does not reach Bonferroni significance: p = 0.013 against the 0.0036 threshold. It is also one of about 25
  secondary tests in this folder.
- It is not a mainshock signal. On M5.5 `y_main` the IG is -0.003.
- The gain comes from **lowering** forecasts in cells that stayed empty (dLL +7.9 on negative cells). Forecasts for
  cells that did have events got slightly worse (-3.4).
- The largest coefficients point to *activation*, not quiescence. `x_zq` has a negative coefficient: a recent rate
  above the 10-yr rate means more events. `x_sync` and `x_load`/`x_shadow` also have negative coefficients.
- It is spread across years: leave-one-year-out IG ranges from +0.015 to +0.024.
- It reverses sign on the independent 2006-2012 window (-0.012).
- Reading: an intermediate-term (months to years) clustering or rate-change term that the short-memory baseline lacks
  adds a little calibration at M5.5. It is not evidence for quiescence, migration or synchronization as precursors.

## Post-hoc look at the large test-period mainshocks
`diagnose.py` and `diagnose.log` give each feature's value in the epicentral bin just before the event, as a
train-period percentile.
- **Illapel 2015 (M8.3)** looked like the textbook pattern: Z quiescence at the 91st percentile, doughnut at the 96th,
  and seismicity migrating toward the bin at the 87th.
- **Chiloé 2016 (M7.6)** showed Z quiescence at the 92nd percentile.
- **Iquique 2014 (M8.2)** showed the opposite: activation from its foreshock sequence, with Z at the 19th percentile
  and doughnut at the 10th.
- **2024 M7.4 (lat -23.1)** was unremarkable.

Two textbook-looking cases out of four is what selective case studies in the literature report. Scored over every
bin-week, though, the same patterns occur often without a following M6, so they carry no forecast skill.

## Leakage checks
- **Truncation test** (`check_features.py`, `check_features.log`): the catalog was cut at t* (1995-07-28, 2010-02-25,
  2014-03-20), re-declustered with the fixed eta0, and every feature rebuilt. All rows with t_start <= t* matched the
  full-catalog features to within 4e-10, with 0 NaN mismatches. No look-ahead.
- No centered windows. All standardizations (RTL) are expanding, past-only.
- The non-causal `mainshock` flag is never used in features.
- **Completeness drift:** rerunning with M>=5.0 inputs, which are complete throughout, gives the same conclusion.
- Remaining unavoidable issue: USGS ComCat magnitudes and locations are the current, revised values, not the values
  available in real time. For a null result this matters little.

## Caveats
- The spatial model is latitude-only (1-degree bins), as in the harness. RTL, Mogi and migration are normally computed
  in 2-D around a hypocentre with smaller radii and lower magnitude thresholds (M>=3). The USGS catalog is only
  complete to about M4.7 here, which limits sensitivity. A denser local catalog (CSN Chile, M>=3 since about 2000)
  could change this, but it gives a short training history.
- The stress-shadow feature is partly confounded on `y_main`. M6 events inside an old M8 rupture are often ZBZ-linked
  to that rupture and so are not labelled mainshocks.
- 72-85 positive test cells at M6: effects under about 0.05 nats/eq cannot be resolved.
- Harness: no bug found. As documented, `_BASE_CACHE` is keyed by `id(panel)`, so the alternate-split panels are
  separate copies and get their own baselines. Their `p_poisson` and `log_bg` columns were computed by the harness from
  1976-2012, which overlaps the 2006-2012 alternate test window. This affects baseline and full models equally.

## Reproduce
```
cd /Users/cnavarreteliz/Documents/Repos/earthquakes/research/hypotheses/quiescence_migration
/opt/miniconda3/bin/python3 -I check_features.py
/opt/miniconda3/bin/python3 -I run_primary.py primary.json 1000
/opt/miniconda3/bin/python3 -I run_secondary.py families families.json
/opt/miniconda3/bin/python3 -I run_secondary.py variants variants.json
/opt/miniconda3/bin/python3 -I run_secondary.py tune tune.json
/opt/miniconda3/bin/python3 -I run_m55_followup.py m55_followup.json
/opt/miniconda3/bin/python3 -I diagnose.py
```
