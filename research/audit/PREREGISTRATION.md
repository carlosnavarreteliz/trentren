# Audit pre-registration (written 2026-10-17, before any audit test-period evaluation)

Object under audit: PI-ST-Fusion (M6) from src/models_and_evaluation.py, target y_m55_14d
("after each M>=4.0 event, an M>=5.5 within 14 d and 150 km").

## Primary test (decided before looking at any audit result)
- Split mandated by the research effort: fit 1976-01-01..2012-12-31 (inner split for M6's
  own calibration: train 1976-2008, val 2009-2012), test 2013-01-01..(catalog end - 30 d).
- Baseline: Reasenberg-Jones / ETAS-style clustering baseline with GENERIC published parameters
  (a=-1.67, b=0.91, p=1.08, c=0.05 d; Reasenberg & Jones 1989), summed over ALL M>=4 parents
  within 150 km in the past 365 d INCLUDING the triggering event itself, plus a static background
  rate of M>=5.5 within 150 km estimated on 1976-2012; mapped to probability with a 2-parameter
  logistic fitted on 1976-2012 only.
- Statistic: information gain per test event, IG = mean_i [LL_M6(i) - LL_RJ(i)] (nats/event),
  Bernoulli log-likelihood.
- Significance: block bootstrap by calendar year over the test period (2000 resamples),
  one-sided p = fraction of resamples with IG <= 0. Bonferroni alpha = 0.05/14 = 0.0036.
- Secondary (reported, not primary): ROC-AUC, same on the declustered target (future M>=5.5
  that is a Gardner-Knopoff mainshock), on the original 2018+ split, and the incremental value of
  the "physics" features over clustering-only features in a GBDT.
