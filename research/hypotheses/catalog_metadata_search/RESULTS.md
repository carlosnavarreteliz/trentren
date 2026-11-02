# RESULTS: catalog_metadata_search

**Verdict: NULL.** I searched 164 eligible causal features built from the catalog and its metadata, plus 16 that
were excluded for missing data, 180 in total. Inside the training period, none survived FDR-controlled nested
temporal cross-validation. That holds for both targets and for both split years. The preregistered primary test is
therefore null by construction. Every exploratory one-shot test on the test period, including the top five features
picked without FDR, the family blocks and the "kitchen sink", **lost** information against the ETAS / clustering
baseline. No exploratory set came close to significance.

The catalog metadata (nst, gap, rms, errors, magType, contributing network) is dominated by changes in network and
reporting policy, and a large part of it does not even exist before 2000–2010.

All numbers come from `out/*.json`. The preregistration is `PREREGISTRATION.md`, written before any evaluate call.

## 1. What was done
- **Feature library:** `features.py`. It has 180 features: 162 per bin (in the bin ±1 bin and the bin ±3 bins) and
  18 for the whole region. The windows are 30, 90, 365 and 1095 days, always [t−W, t).
- **Families:**
  - inter-event-time CV and log statistics
  - magnitude entropy, b-value, mean and std of magnitude
  - depth mean, std and fractions
  - longitude and latitude dispersion
  - Benioff strain and moment sums
  - fraction of ZBZ background events (the causal `bg` flag)
  - quiescence and rate contrasts
  - Fano, Allan and aggregated-variance Hurst statistics of weekly counts
  - correlation dimension D2 of epicentres
  - hour-of-day resultant, night and weekend fractions
  - magType mix
  - ComCat metadata: rms, nst, gap, dmin, horizontal / depth / magnitude errors, magNst, the fraction located by the
    `guc` network, the fraction with nst missing
  - statistics of all magnitudes including the incomplete small events (counts, small/large ratio, md fraction)
- **Causality unit test** (`causality_check.py`): I recomputed 1,568 feature values at random issue times from a
  catalog truncated at t_start. The maximum relative difference was 5.6e-8 (float32 rounding), so there is no
  look-ahead.
- **Eligibility:** 16 features were dropped for being more than 95 % NaN in 1976–2012. These are `m_herr`, `m_merr`
  and `m_dmin` at all windows, and `a_*_90` std-type statistics in sparse bins. horizontalError, magError and dmin are
  essentially absent before 2010, so they cannot be learned from the training period at all.
- **Selection:** nested expanding-window CV inside training.
  - Validation blocks: 1995–2001, 2002–2007 and 2008–2012.
  - Model: harness logit, baseline + one feature.
  - Statistic: pooled validation IG, with a calendar-year block-bootstrap p-value.
  - Rule: BH-FDR q = 0.10, plus a positive gain in at least 2 of 3 folds.

## 2. Selection inside training (no test data touched)
| split | target | eligible | p_boot < 0.05 | BH q<0.10 passes | best pooled val IG (feature) | min p_boot |
|---|---|---|---|---|---|---|
| 2013 | y_main | 164 | 0 | **0** | +0.020 (`a_fmw_90`: Mw-type fraction) | 0.10 |
| 2013 | y_all | 164 | 2 | **0** | +0.014 (`a_meanlon_90`) | 0.036 |
| 2006 (R3) | y_main | 164 | – | **0** | `n3_a_logcnt_1095` | 0.014 |
| 2006 (R3) | y_all | 164 | – | **0** | `n3_a_logcnt_1095` | 0.014 |

- Only 25 of 164 features (y_main) and 27 of 164 (y_all) had any positive pooled validation IG. The median feature
  *lowered* validation IG, by −0.009 nats/eq.
- Under H0, about 8 features would reach p < 0.05; I observed 0 and 2.
- The top-ranked features change completely between the 2006 and 2013 split designs, which is the signature of noise.

## 3. Primary test (preregistered)
- **y_main (primary) and y_all (co-primary), weekly M≥6.0:** the selected set S is empty, so IG = 0 and the result is
  null by construction. There was no test-period evaluation to run.
- Surrogate p: not applicable.

## 4. Exploratory one-shot test-period results (weekly M≥6, logit, test 2013-01-01 → 2026-10)
IG is in nats per target event versus the ETAS/clustering logit baseline. The CI is a 95 % calendar-year block
bootstrap. p is the phase_randomize surrogate p, with n = 200 (n = 100 for the blocks).

| set | target | # cols | IG vs base | IG vs ETAS | 95 % CI | p (phase) | AUC base → full | Molchan ASS base → full |
|---|---|---|---|---|---|---|---|---|
| E1 top-5, no FDR | y_main | 5 | −0.042 | −0.050 | [−0.080, −0.006] | 0.72 | 0.665 → 0.654 | 0.664 → 0.654 |
| E1 top-5, no FDR | y_all | 5 | −0.142 | −0.142 | [−0.222, −0.048] | 1.00 | 0.697 → 0.650 | 0.696 → 0.650 |
| E4 best single (`a_fmw_90`) | y_main | 1 | −0.053 | −0.061 | [−0.106, −0.008] | 0.99 | 0.665 → 0.648 | |
| E4 best single (`a_meanlon_90`) | y_all | 1 | −0.022 | −0.023 | [−0.046, +0.005] | 0.96 | 0.697 → 0.694 | |
| E3 physical-candidate block | y_main | 85 | −0.248 | −0.256 | [−0.39, −0.10] | 0.11 | 0.665 → 0.633 | |
| E3 physical-candidate block | y_all | 85 | −0.190 | −0.190 | [−0.34, −0.02] | 0.09 | 0.697 → 0.664 | |
| E3 artifact-prone block | y_main | 79 | −0.761 | −0.769 | [−1.20, −0.39] | 0.73 | 0.665 → 0.605 | |
| E3 artifact-prone block | y_all | 79 | −0.624 | −0.625 | [−0.90, −0.41] | 0.70 | 0.697 → 0.638 | |
| E3 ComCat metadata only | y_main | 23 | −0.354 | −0.362 | [−0.54, −0.16] | 0.87 | 0.665 → 0.572 | |
| E3 ComCat metadata only | y_all | 23 | −0.338 | −0.338 | [−0.49, −0.17] | 0.93 | 0.697 → 0.617 | |
| E2 kitchen sink, logit | y_main | 164 | −4.29 | −4.30 | [−7.1, −2.3] | 0.96 | 0.665 → 0.631 | |
| E2 kitchen sink, logit | y_all | 164 | −3.25 | −3.25 | [−5.0, −2.1] | 0.91 | 0.697 → 0.663 | |
| E2 kitchen sink, GBM | y_main / y_all | 164 | −0.007 / −0.069 | | | | | |

Robustness for E1, the top-5 set:
- **y_main:** GBM +0.013; M≥5.5 weekly −0.009; 30-day panel −0.010. bin_permute p = 0.86, circular-shift p = 0.74.
- **y_all:** GBM −0.035; M≥5.5 −0.022; 30-day −0.229. bin_permute p = 1.00, circular-shift p = 0.99.
- **R3, alternate split** (selection redone inside 1976–2005, test 2006 → 2026): S is empty for both targets.
  - y_main: the top-5 set gives IG = −0.061, CI [−0.12, −0.00], p = 0.84.
  - y_all: the top-5 set gives IG = −0.292, CI [−0.43, −0.13], p = 1.00.

A note on the large blocks: their phase surrogates are also strongly negative (null mean −0.3 to −1.9), because
adding dozens of regressors overfits whatever is fed in. So "p = 0.09–0.11" for the physical block means "it loses
somewhat less than random noise of the same structure would". It does not mean the block shows a gain.

## 5. Artifact / leakage diagnostics on the top-ranked candidates (`out/diagnose_*.json`)
| feature | a-priori class | finding | classification |
|---|---|---|---|
| `a_fmw_90`, `a_fmb_90` (magType mix) | artifact-prone | +0.93 SD shift after 2000 and +1.27 SD after 2013 (USGS moved to routine Mww/Mwr). Partly tracks recent large events (ρ = 0.43 with ETAS). Test IG −0.05 / −0.03, detrended p = 0.84 / 0.81 | **artifact** (magnitude-reporting policy); fails out of sample |
| `m_gap_90` (azimuthal gap) | artifact-prone | 100 % missing before 2000; −0.71 SD shift after 2013 from network densification. Test IG −0.12 (y_all) | **artifact** (network geometry) |
| `a_meanlon_90` (mean longitude, trench distance) | physical | ρ = 0.26 with baseline; test IG −0.022 (y_all) and −0.022 (y_main); detrended p = 0.49 | physical, **no signal** |
| `n3_a_logcnt_1095` (3-yr count, ±3 bins) | physical | ρ = 0.84 with `cnt_nbr_365d`; test IG +0.006 / +0.002; detrended p = 0.59 | **clustering / background rate in disguise** |
| `sp_d2_1095` (epicentral correlation dimension) | physical | ρ = 0.22 with baseline; raw test IG +0.000 (y_all) / −0.011 (y_main) | physical; see the post-hoc note |

**Post-hoc observation, NOT a finding.** The causally detrended D2, `x_n1_sp_d2_1095` minus its trailing 3-year
mean, was the only variant with a positive sign:
- weekly M6 y_main: IG = +0.005 nats/eq, phase p = 0.021 (n = 1000), bin_permute p = 0.11, year-bootstrap CI
  [−0.005, +0.021];
- weekly M6 y_all: +0.003, p = 0.12;
- **M5.5:** the sign reverses (−0.0001 for y_main, −0.004 for y_all, p = 0.52 / 0.94).

This variant came out of the diagnostics, not the preregistered pipeline. It is one of 180+ features and several
variants, and it fails Bonferroni and the M5.5 replication, so I count it as noise. It is the only lead I would
consider preregistering on an independent catalog: does epicentral clustering geometry roughen before mainshocks?

## 6. Why the metadata cannot carry a forecast here
- **Coverage:**
  - nst exists only from 2000, gap from about 2002.
  - horizontalError, magError and dmin exist for M ≥ 4.7 only from about 2010.
  - Before 1985, rms and depthError are absent.
  - As a result, a model trained on 1976–2012 sees mostly imputed values.
- **Non-stationarity by policy:**
  - the switch to Mww/Mwr reporting (fmw +1.3 SD after 2013);
  - network densification after Maule 2010 and Iquique 2014 (gap −0.7 SD);
  - the `guc` (Universidad de Chile / CSN) contribution and the md magnitudes concentrated in 1995–2010.
  - These shifts are exactly why the metadata block extrapolates badly (IG −0.34 to −0.35).
- **Revision leakage:** ComCat stores the *latest revision* of each event. It does not store the values available in
  real time. Any metadata "signal" would therefore also be suspect for look-ahead. Real-time snapshots of the catalog
  were not available.

## 7. Caveats
- Power is low: the test period has 72 (y_main) and 85 (y_all) positive cells, so effects of about 0.02–0.05
  nats/eq cannot be resolved. A null here does not exclude small effects.
- Selection used a year-block bootstrap p-value. That is not the harness's phase surrogate, which would have been too
  costly for 164 × 3 × 2 fits. If anything, the bootstrap is anti-conservative, and nothing passed even so.
- The fold panels reuse the harness's 1976–2012 Poisson background. This is a mild in-sample effect inside training,
  shared by the base and full models.
- In R3 the ZBZ `bg` flag inside the features uses eta0 fitted on pre-2013 data. The effect is negligible.
- Small-event (set S) features are incomplete before about 1990 and detection-dominated by construction.
- No bug was found in harness.py. One quirk: the weekly grid does not land on 2013-01-01, so the first test t_start is
  2013-01-03.

## Files
- `PREREGISTRATION.md`, `features.py`, `build_features.py` (with `build.log`), `cmlib.py`, `select_features.py`,
  `final_test.py`, `final_alt.py`, `diagnose.py`, `causality_check.py`, `posthoc_d2.py`
- `out/selection_*.json`, `out/cv_summary_*.csv`, `out/final_*.json`, `out/final_alt2006_*.json`,
  `out/diagnose_*.json`, `out/posthoc_d2.json`
- The feature cache is at `research/data/catalog_metadata_search/features_weekly.parquet`.
