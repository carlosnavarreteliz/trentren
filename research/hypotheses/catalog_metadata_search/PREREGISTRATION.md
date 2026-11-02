# PREREGISTRATION: catalog_metadata_search

Written 2026-10-17. At this point the feature library has been built (`features.py`, `build_features.py`, `build.log`)
and its NaN coverage has been inspected. No `evaluate()` call has been made on any panel, and no target value of the
test period (t_start >= 2013-01-01) has been looked at.

## Hypothesis
Somewhere in the ComCat catalog itself, in quantities that standard ETAS / clustering forecasts ignore, there is
predictive information about upcoming M>=6 earthquakes in Chile. The candidates are:
- event statistics: inter-event-time irregularity, magnitude-sequence entropy, b-value, depth and spatial dispersion,
  Hurst / Fano / Allan dispersion of counts, correlation dimension of epicentres, and the fraction of background events;
- catalog metadata: rms, nst, gap, dmin, location / depth / magnitude errors, magType mix, contributing network,
  fixed-depth flags, and the time-of-day / weekday distribution.

H0: given the ETAS/clustering logit baseline, none of these quantities improves the out-of-sample log-likelihood
more than its own autocorrelation-preserving surrogates do.

## Data
- `data/raw/chile_usgs_1900_2026.csv` (USGS ComCat). This is the current revised version, which is a known caveat:
  ComCat metadata are the latest revision, not the values available in real time.
- Set A is the harness events with M >= 4.7 since 1973, carrying the causal ZBZ `bg` flag. ComCat metadata are merged
  on `id`. The non-causal `mainshock` / `cluster_id` columns are dropped before any feature is computed.
- Set S is every magnitude since 1973. It is incomplete and detection-dominated.

## Feature library (180 columns; exact code in `features.py`)
- Every feature at issue time t uses only events with time in [t − W, t), with W ∈ {30, 90, 365, 1095} days.
- Features are computed on the weekly grid of the harness panel and aligned to other panels backward as-of.
- Prefixes:
  - `x_n1_*`: the bin ±1 bin. This scale has the full family set.
  - `x_n3_*`: the bin ±3 bins. Reduced set.
  - `x_gl_*`: the whole region. Reduced set, broadcast to all bins.
- Families, with their a-priori classification:
  - Physical candidates:
    - `a_meanmag`, `a_stdmag`, `a_bval` (Aki, Mc = 4.7), `a_magent`, `a_meandep`, `a_stddep`, `a_fshallow`, `a_fdeep`,
      `a_meanlon`, `a_stdlon`, `a_stdlat`
    - inter-event time: `a_cvdt`, `a_meanlogdt`, `a_stdlogdt`
    - `a_fbg`, `a_logbenioff`, `a_logmoment`, `a_logcnt`
    - temporal contrasts: `c_*_90v1095` and `c_zrate`
    - weekly-count dynamics: `w_*` (fano52, fano156, allan52, hurst156)
    - spatial correlation dimension: `sp_d2_1095`
  - Artifact-prone a priori (network, reporting or processing policy):
    - `m_*`: rms, nst, gap, dmin, herr, derr, merr, magnst, fraction of locations from network `guc`, fraction with nst missing
    - `s_*`: all-magnitude counts, the small/large ratio, md fraction, guc fraction, rms, night fraction, cvdt
    - `a_fmw` / `a_fmb` / `a_fml`: magType mix
    - `a_fixdep`: fixed depth 10 / 33 / 35 km
    - `a_night`, `a_weekend`, `a_hourR`: time-of-day and weekday detection effects
    - `c_rms`
- **Eligibility rule:** a feature is dropped if it is NaN in more than 95 % of training rows (t_start < 2013). From
  `build.log`, that drops `m_herr_*`, `m_merr_*` and `m_dmin_*`, which barely exist before 2013. Within a fold, a
  feature that is NaN in more than 95 % of the fold-train rows contributes ΔLL = 0 for that fold.
- **Preprocessing:** each feature is winsorized at the 0.5 / 99.5 percentiles of the respective training rows.
  Remaining NaN values are imputed by the harness with the train mean (logit).

## Feature selection: nested temporal CV strictly inside 1976–2012
- Panel: harness weekly panel (step 7 d, horizon 7 d, 1° bins). Primary target magnitude M >= 6.0.
- Model: harness logit (`evaluate`, model='logit'), baseline + one feature at a time. Run separately for y_all and y_main.
- Three expanding-window folds. The fold panel is the harness panel with `split` relabeled:
  - Train: t_start + 7 d <= V_start.
  - Validation: V_start <= t_start and t_start + 7 d <= V_end.
  - Every other row is `gap`.
- Validation blocks:
  - V1 = 1995-01-01 to 2002-01-01
  - V2 = 2002-01-01 to 2008-01-01
  - V3 = 2008-01-01 to 2013-01-01
- The Poisson background and `log_bg_*` columns are the harness's 1976–2012 values. This is a mild in-sample effect
  inside training, shared by base and full models, and it never touches the test period.
- For each feature and target:
  - ΔLL_v is computed for each fold v.
  - The pooled validation IG is ΣΔLL / Σ positive validation cells.
  - The p-value is a one-sided calendar-year block bootstrap (18 years, 4000 draws) of the pooled IG: P(IG <= 0).
- **Selection rule (per target):**
  1. Benjamini–Hochberg FDR at q = 0.10 over all eligible features.
  2. Pooled IG > 0, and ΔLL > 0 in at least 2 of 3 folds.
  3. Rank by pooled IG, then add features greedily, skipping any with |Spearman ρ| > 0.8 (on train rows) against a
     feature already chosen.
  4. Keep at most 10.
- The result is written to `selection_<target>.json` before any test evaluation.

## PRIMARY TEST (one shot on the test period)
- Final model: harness baseline + the selected set S_target, logit.
  - Fitted on train (1976–2012, purged) and evaluated on test (2013-01-01 → 2026-10), weekly panel, M >= 6.0.
  - Winsorization limits come from 1976–2012.
- **Primary target: `y_main`.** Co-primary: `y_all`. Each target uses its own selected set.
- Metric: `ig_per_event` (full vs the ETAS/clustering logit baseline).
- Significance:
  - `H.surrogate_test(mode="phase_randomize")` on the selected feature grid: same phases for all bins and columns.
  - n = 200, rerun with n = 1000 if p < 0.02.
  - Also reported: bin_permute and end-matched circular_shift (n = 200), and the year-block bootstrap CI.
- If S_target is empty, the primary result for that target is **null by construction** (IG = 0, no test run).
- **Decision:**
  - "positive" requires IG > 0 on test, phase_randomize p < 0.0036 (project Bonferroni), and passing the
    leakage / artifact checks below. I also report the stricter within-agent threshold of 0.0018 for 2 targets.
  - "weak_positive" requires p < 0.05 uncorrected.
  - Anything else is null or negative.
  - A hit made only of artifact-prone features cannot be called "positive" unless the artifact checks clear it.

## Secondary / exploratory (reported regardless; NOT used for the verdict)
- E1: the top-5 features by pooled validation IG with no FDR filter. This shows what undisciplined mining would claim.
- E2: the kitchen sink: all eligible features at once, logit (L2, C = 1) and GBM (harness LightGBM).
- E3: family blocks (physical vs artifact-prone groups), logit.
- E4: the single best feature per target, as measured in validation.

## Robustness (for the selected sets, or for E1 if S is empty)
- R1: the same features on the weekly M >= 5.5 panel (y_all, y_main).
- R2: the 30-day panel (step 30, horizon 30) at M >= 6.0.
- R3: an alternate split.
  - Re-run the entire selection inside 1976–2005, with folds V1 1990–1994, V2 1995–1999 and V3 2000–2005.
  - Then test once on 2006-01-01 → 2026-10.
  - The harness panel is rebuilt with `TEST_START = 2006-01-01`, so the Poisson background uses 1976–2005 only.
- R4: GBM model for the selected set.
- R5: drop each test year and check whether a single sequence carries the effect.

## Leakage / artifact checks for any hit
1. Plot and inspect the feature against year. Check for a step change at the 2000 metadata onset, at 2010
   (after Maule), or at 2014.
2. Does it survive on `y_main` (not just aftershock clustering)?
3. Correlation with the baseline clustering features on train rows (|ρ| > 0.7 means "clustering in disguise").
4. A causally detrended version: the feature minus its trailing 3-year mean, `.shift(1)`.
5. Physical vs artifact classification, using the a-priori list above.
