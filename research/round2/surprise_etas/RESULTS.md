# RESULTS — round2/surprise_etas: surprise relative to a fitted ETAS

**Verdict: NULL.** The preregistered composite surprise score adds no information over the harness ETAS-logit
baseline (y_main IG = −0.0018 nats/eq, phase_randomize p = 0.49, n = 1000). One component, the Anderson–Darling
statistic of ETAS transformed times (`x_ad`), shows a nominal post-hoc signal (p_phase ≈ 0.006 on both targets).
It was selected from about 22 component tests, its year-block bootstrap CI includes 0, it shrinks to about zero
when the pipeline is rebuilt with m0 = 5.0, and it gives IG ≈ 0 on the M≥5.5 panel. It is a lead to preregister
in a future round, not a finding.

(Resumed run: a previous agent fitted ETAS, built the features and ran the primary, variants, components and
follow-up. This pass checked the logs and outputs and wrote this report. The preregistration was not changed.)

## Setup (as preregistered, PREREGISTRATION.md)
- Catalog: harness ComCat M≥4.7, Chile box. 1-D (latitude) × time ETAS fitted by maximum likelihood on 1976–2012
  and then frozen. m0 = 4.7: μ0 = 0.216/d, K = 0.20, α = 2.00, c = 0.031 d, p = 1.056, d = 0.056°, γ = 0.68,
  branching ratio 0.98. m0 = 5.0 refit: branching ratio 0.68.
- Seven causal components for each bin and issue time (neighbourhood of ±1 bin): rr30/90/365 (log obs/ETAS-expected),
  ad (Anderson–Darling distance of the last 20 transformed intervals), gap (ETAS-expected count since the last event),
  lld30/90/365 (ETAS log-likelihood minus Poisson log-likelihood, per event), mag (GR magnitude surprise with the
  training b-value), spat (along-strike surprise under the ETAS kernel), depth (depth surprise against the
  training KDE for the bin).
- PRIMARY: x_surprise = mean_k(−ln p_k), with p_k taken from the training-period empirical CDFs. Weekly panel,
  M≥6, logit. Surrogate test: phase_randomize n = 1000. α = 0.0083.

## Primary result (test period 2013-01-01 to 2026-10)
| target | N_test | IG vs ETAS-logit (nats/eq) | year-block 95% CI | p phase (n=1000) | p bin_permute (500) | p circ_shift (200) | AUC base→full |
|---|---|---|---|---|---|---|---|
| **y_main** | 72 | **−0.0018** | [−0.0058, +0.0030] | **0.488** | 0.513 | 0.413 | 0.665→0.662 |
| y_all | 85 | +0.0009 | [−0.0021, +0.0036] | 0.233 | 0.277 | 0.189 | 0.697→0.697 |

## Preregistered robustness checks (composite)
| variant | y_main IG / p_phase | y_all IG / p_phase |
|---|---|---|
| causally detrended (260-week trailing mean, shift 1) | −0.0027 / 0.50 | −0.0008 / 0.45 |
| m0 = 5.0 pipeline | −0.0063 / 0.78 | −0.0008 / 0.49 |
| M≥5.5 panel | −0.0003 / 0.47 (N = 193) | −0.0002 / 0.39 (N = 228) |
| daily panel, M≥6 | −0.0011 / 0.40 | +0.0001 / 0.29 |
| GBM (composite) | +0.025 vs GBM baseline, p = 0.19 (n=100) | +0.025, p = 0.010 (n=100) |

The GBM row does not count as evidence. The GBM baseline is worse than Poisson on y_main (−0.067 nats/eq), and the
GBM model with the composite is still *worse* than the ETAS-logit baseline (ig_full_vs_etas = −0.056 y_main,
−0.043 y_all). The positive IG only shows that the surprise score partly repairs a weak GBM baseline.

## Components (secondary, logit, phase_randomize n=200; no multiplicity credit)
y_main IG (p): rr30 −0.008 (0.78), rr90 −0.000 (0.43), rr365 −0.000 (0.43), lld30 −0.006 (0.88), lld90 −0.001 (0.52),
lld365 +0.002 (0.27), **ad +0.011 (0.020)**, gap −0.012 (0.87), mag +0.001 (0.54), spat −0.001 (0.53), depth +0.001 (0.31),
all-components block −0.048 (0.74; CI excludes 0, which points to overfitting).
y_all: **ad +0.011 (0.010)**. Everything else has IG ≤ +0.001 or a negative IG. depth −0.031 (CI excludes 0), block −0.112.

### Post-hoc follow-up on x_ad (exploratory)
| variant | y_main IG, CI, p_phase | y_all IG, p_phase |
|---|---|---|
| x_ad, m0 = 4.7 (n=1000) | +0.0107 [−0.009, +0.038], p = 0.006 (bin_permute 0.053) | +0.0115, p = 0.006 (bin_permute 0.017) |
| x_ad causally detrended (n=200) | +0.0180 [+0.002, +0.038], p = 0.005 | +0.0122, p = 0.010 |
| x_ad, m0 = 5.0 pipeline (n=200) | +0.0025 [−0.018, +0.033], p = 0.10 | +0.0012, p = 0.22 |
| x_ad, M≥5.5 panel | −0.0011 | +0.0050 |
| drop one test year (y_main) | range +0.003 (without 2014) to +0.015 | — |

How to read this: x_ad was chosen as the best of about 11 components per target. A crude selection correction
(p × 11) gives about 0.07, which is not significant. The effect depends on the m0 = 4.7 catalog, where completeness
in the 1970s–90s is marginal and the ETAS fit is near-critical (branching 0.98, α ≈ 2.0). In the cleaner
m0 = 5.0 rebuild it nearly disappears. It does not generalise to M≥5.5 targets. A large share of it comes from 2014
(the Iquique foreshock sequence, which is a known and visible case). It is a candidate for a new preregistered
test in a later round, with the m0 = 5.0 version or an out-of-region replication as the primary.

## Leakage checks
- **Truncation test:** the catalog was cut at 2014-03-20, 2015-09-10 and 2020-06-01. Features recomputed with
  frozen parameters match the full-catalog features to 1e-10 (max abs diff 4.6e-11), with no NaN mismatches.
  The features are causal.
- **Level shift between train and test** (in train SD units): composite +0.24, depth +0.51, mag −0.37, lld ≈ +0.2,
  ad +0.18, rr ≈ 0. The composite shift is one reason the preregistered detrended variant exists. That variant is
  also null.
- **ETAS calibration over time** (observed / ETAS-expected counts): 0.96 (1976–89), 0.81 (1990–99), 1.13 (2000–12),
  1.01 (2013–19), 0.89 (2020–26). These swings of 10–20% reflect network and completeness changes. They are the
  reason we treat m0 = 4.7 rr- and ad-type features with suspicion.
- **Large test events:** before the Mw 8.2 Iquique 2014 event the composite score was at the 99.9th percentile of
  training values (rr90 99.7, ad 99.8). Before Illapel 2015 (33.6), Chiloé 2016 (68.8) and 2024 M7.4 (47.8) it was
  unremarkable. One visible foreshock sequence does not make a forecasting signal.
- Mainshock flags were never used as features. No external downloads were used.

## Files
PREREGISTRATION.md; surprise_lib.py (ETAS and features); fit_and_build.py; run_primary.py → primary.json/log;
run_secondary.py → components.json/log, variants.json/log; run_followup.py → followup.json/log; check_leakage.py →
leakage.json/log; diagnose.py → diagnose.log. Cached features and ETAS parameters are in
research/data/round2_surprise_etas/.
