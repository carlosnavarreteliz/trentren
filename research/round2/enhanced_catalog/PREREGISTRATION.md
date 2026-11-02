# PREREGISTRATION: enhanced_catalog (round 2)

Written 2026-10-17, after downloading and inspecting the catalogs (event counts per year, magnitude
distributions, matching against USGS) and counting target cells per period, and BEFORE any feature was
evaluated against a target on any period.

## Question
Does lowering the magnitude of completeness (small events that USGS ComCat does not list) reveal predictive
information about M>=5 shocks that the M>=4.7 harness baseline (ETAS-logit) lacks?

## Data
1. **IPOC catalog**, Sippl, Schurr, Münchmeyer, Barrientos and Oncken (2023), GFZ Data Services,
   doi:10.5880/GFZ.4.1.2023.004. It has 182,841 events for 2007-01-01 to 2021-12-31, covering about 18-24.5°S
   and 72-67°W, with ML from Münchmeyer et al. (2020).
   - How it was built: automatic STA/LTA-type detection, picking and association on the permanent IPOC network
     plus auxiliary stations. The processing ran retrospectively (published 2018 and 2023), but it is uniform and
     uses no templates. No detector was tuned on the 2014 Iquique sequence.
   - Completeness: maximum curvature gives Mc of about 1.7-2.0. The annual count of ML>=3 is stable (about 600
     per year, except 2014), while counts below ML 2.5 fall after 2017. **Modeling threshold: ML>=3.0.** That is
     about 1.7 magnitude units below the harness M4.7, so roughly 50 times more events.
   - Catalog saturation: during aftershock sequences, small events are under-detected (for example, at most
     about 240 events per day after Iquique).
2. **Riedel-Hornig et al. (2026, ESSD 18:1-16)**, Zenodo doi:10.5281/zenodo.15284376. It has about 622k events
   for 2017-2021, covering 18-43°S, with Mc of about 2.5.
   - How it was built: an EQTransformer picker pretrained on INSTANCE (Italy) and not retrained on Chile,
     PyOcto association, NonLinLoc and HypoDD. The authors chose a station set intended to stay constant in time.
   - **Modeling threshold: ML>=2.5.**
3. Targets and baseline come from the harness, unchanged: USGS ComCat, ZBZ declustering, ETAS-logit baseline.

## Panel (harness machinery, restricted to the footprint)
- `H.get_panel(step_days=7, horizon_days=7, target_mag=5.0)`, restricted to the catalog footprint bins:
  - IPOC: bins 22-27 (-24..-18°).
  - Riedel: bins 3-27 (-43..-18°).
- Bin ids are relabelled 0..B-1 so the harness surrogate machinery works. Rows end at 2021-12-24, so every label
  window falls inside the catalog period.
- Target M>=5.0, which is complete in USGS throughout. At M>=6 the footprint test has only 17 y_main cells, so
  M>=6 is underpowered and is reported only as a secondary result.
- **IPOC split:**
  - The harness baseline is fitted on footprint rows from 1976 to 2012 (`split=='train'`, harness default).
  - The enhanced-catalog features exist from 2008-01-01, after a 1-yr warm-up for the trailing reference rates.
    Earlier rows are NaN, which the harness imputes with the train mean, i.e. neutral. Their coefficients are
    therefore learned only on 2008-2012.
  - **Test: 2013-01-01 to 2021-12-24.** This is the harness test start, and it includes Iquique 2014.
  - Counts at M>=5: 2008-2012 train 106 y_all and 100 y_main cells; test 184 y_all and 153 y_main cells.
- **Riedel split (justified alternative, since the catalog starts in 2017):**
  - Baseline and features are fitted on rows with t_start + 7 d <= 2019-07-01.
  - The features exist from 2017-07-01 (6-month warm-up), so their coefficients are learned on 2017-07 to 2019-06.
  - Test: 2019-07-01 to 2021-12-24, with 108 y_all and 101 y_main M>=5 cells.

## Features (all strictly causal: events with time < t_start; per latitude bin)
Built from enhanced-catalog events with ML >= Mthr (IPOC 3.0, Riedel 2.5). Any depth; ref = 365 d (IPOC) or 180 d (Riedel).
- `x_act7` = log((N[t-7d, t) + 0.5) / (7 * R + 0.5)), where R = N[t-7d-ref, t-7d) / ref counts events in the same
  bin. This is short-term activation of small events relative to the bin's own recent rate, the foreshock signal.
- `x_act30` = the same quantity with a 30-day window.
- `x_dbpos` = b-positive (van der Elst 2021; δ = 0.2) of the last 150 events in bins ±1, minus b-positive of the
  last 1500 events in bins ±1, both before t. This is a short-term b-value change. It is NaN if fewer than 150
  events are available.

The relative, self-normalized form (each bin against its own trailing rate) was chosen in advance to absorb slow
level shifts from network changes.

## PRIMARY test (one test; round-2 Bonferroni alpha = 0.05/6 = 0.0083)
- IPOC catalog, block {x_act7, x_act30, x_dbpos}, ML>=3.0.
- Weekly panel, target M>=5.0, **`y_main`**, logit.
- IG per event vs the ETAS-logit baseline on the 2013-01-01 to 2021-12-24 footprint test.
- `H.surrogate_test(mode='phase_randomize')`, n=1000 if the observed IG > 0, otherwise n=200.
- Verdict:
  - positive: IG>0 and p<0.0083, and the result survives the leakage checks below.
  - weak_positive: p<0.05.
  - null otherwise.

## Secondary (no multiplicity credit)
1. Same setup, `y_all`.
2. **Key contrast:** the same three features from the SAME IPOC catalog thresholded at ML>=4.5 (what an M~4.7
   catalog sees) and at ML>=2.5. If lowering Mc matters, IG(ML3.0) > IG(ML4.5).
3. Riedel-Hornig catalog, whole margin, ML>=2.5, `y_main` and `y_all`, with the phase_randomize surrogate.
4. Targets M>=5.5 and M>=6.0, weekly. GBM model. A daily panel (step 1, horizon 1) with `x_act1` (1 d vs ref)
   and `x_act7`.
5. bin_permute and end-matched circular_shift p for the primary block.
6. Detrended version: each feature minus its causal trailing 52-week mean (`.shift(1)`).
7. Leave-one-year-out IG and IG with 2014 removed.
8. Depth split: features from shallow (<=60 km) events only.

## Descriptive event analysis (pilot, not a forecast test)
- **Mainshocks:** USGS M>=5.0 mainshocks (ZBZ flag; non-causal, but used only as event times) inside the IPOC
  footprint (-24.5..-18.2°, -71.8..-67.2°), 2008-2021.
- **Counts:** IPOC ML>=3.0 events within 50 km epicentral distance in [t0-Δ, t0-60 s), for Δ = 1, 7 and 30 days.
- **Controls:** for each mainshock, 200 random times in 2008-2021 at the same epicentre, excluding times with a
  USGS M>=4.7 event within 50 km in [t, t+7 d).
- **Statistic:** the fraction of mainshocks whose count exceeds the 95th percentile of their own controls
  (5% expected under no precursor), with a binomial p.
- **"Quiet-history" subset:** mainshocks and controls with no USGS M>=4.7 within 100 km in the prior 30 days. This
  separates foreshocks from ongoing clustering.
- The analysis is split into shallow (<=60 km) and intermediate-depth events, and repeated at ML>=4.5 for the
  Mc contrast.
- b-positive in the 90 days before each mainshock vs controls is reported descriptively.

## Leakage checks to report
- Network changes and level shifts:
  - Annual counts per ML band.
  - Features are self-normalized.
  - The detrended version is reported (secondary 6).
  - The fixed-station Riedel catalog serves as a replication.
- Retrospective cataloguing: the catalog was built after the events, but with uniform automatic processing and
  no templates. Post-event temporary stations could raise detection after a mainshock. That can only affect
  *post*-mainshock features, which the baseline already treats as clustering.
- Magnitudes: ML of the large events is not used as an input. Features use counts and b-values of small events
  only.
- `mainshock` and `n_all` are not used as inputs.

### Implementation clarification (2026-10-17, before any target evaluation)
The first feature check (`check_features.py`) left `x_dbpos` NaN in 39% of the 2008+ rows. The cause: ML>=3
events in bins ±1 accumulate only about 300 per year, so a fixed 1500-event window takes about 5 years to fill.
The long reference window is therefore defined as **up to** the last 1500 events, with at least 300 required.
The short window stays fixed at 150 events. After the change, x_dbpos is NaN in 4% of 2008+ rows. No target was
looked at before or during this change. A causality test passed: recomputing the features from a catalog cut at
2014-03-20 gives identical values (max diff 0).
