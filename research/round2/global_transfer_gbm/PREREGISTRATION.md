# PREREGISTRATION: global_transfer_gbm (round 2)

Written 2026-10-17, before any evaluation on the Chile 2013+ test period and before any model was fitted.
The global catalog download was running while this was written; no feature or model had been computed.

## Question
Does a forecasting model trained on the clustering/seismicity features of **all major subduction margins**
(pooling ~15x more large earthquakes) beat the harness ETAS-logit baseline on Chile 2013-present, and does it beat a
Chile-only model with the same features (the transfer contrast)?

## Data
* USGS ComCat global catalog, M>=4.7, eventtype=earthquake, 1973-01-01 .. 2026-10-17, FDSN yearly chunks
  (`download_catalog.py` -> `research/data/round2_global_transfer_gbm/raw/`).
* Chile: exactly the harness catalog/box (lat -46..-17, lon -78..-66) and the harness events/declustering
  (`H.get_events(4.7)`), so the Chile targets and cells are identical to the harness panel. Chile-box events are
  removed from every other margin.

## Margins and bins
Trench polylines (hand-digitized, coarse, in `margins.py`) for: Peru-Ecuador-Colombia, Middle America (Mexico-Central
America), Cascadia, Alaska-Aleutians, Kamchatka-Kurils, Japan trench, Izu-Bonin-Mariana, Nankai-Ryukyu, Philippine
trench, Manila trench, Sunda (Andaman-Sumatra-Java), New Britain-Solomon, Vanuatu, Tonga-Kermadec-Hikurangi, Hellenic.
* An event belongs to a margin if its perpendicular distance to the polyline is within 300 km seaward / 700 km
  landward (landward side = side where the median M>=4.7 event deeper than 100 km lies), its projection falls inside
  the polyline, and it is not in the Chile box. Events in two corridors go to the nearest trench.
* Along-strike coordinate s (km along the polyline) / 111 is used as the "latitude" coordinate. Bins are 1 degree
  (111 km) along strike, exactly as the harness uses 1-degree latitude bins for Chile (Chile trench is ~N-S).
* Panel grid identical to the harness weekly panel: t_start = 1976-01-01 + 7k, horizon 7 d, target M>=6.0.

## Features (identical code for every margin)
Computed by calling the harness' own feature functions (`H._etas_features`, `H._count_features`,
`H._maxmag_features`, `H._days_since`, `H._smooth_bins`) on each margin's events (M>=4.7, from 1973), so the
features are the harness baseline by construction. Check: rebuilding Chile with my wrapper must reproduce the harness
panel columns (max abs diff ~0).
* F_base = the 21 harness baseline columns (12 etas_*, cnt_*, maxmag_*, log_days_since_m6_nbr, log_bg_m45,
  log_bg_target) + log_etas_rate (additive ETAS fitted by `H.fit_etas_additive` on the training rows of the model's
  own training set), with the harness transforms (log10 etas, log1p counts).
* F_ext = F_base + causal extras: cnt_nbr_7d, maxmag_nbr_7d, maxmag_nbr_3650d, cnt_nbr_3650d,
  log_days_since_m7_nbr, b-value (Aki-Utsu, Mc=4.7, last 100 M>=4.7 events within +-1 bins and 10 years before
  t_start; NaN if < 30 events).
* No margin identity, no latitude/longitude, no calendar time. Static per-bin rates (log_bg_*, p_poisson_*) use
  each margin's pre-2013 training rows only (as in the harness).
* Targets per margin: y_all = any M>=6.0 in the bin within [t_start, t_start+7d); y_main = same for ZBZ mainshocks
  (`H.decluster(cat, eta0=<harness Chile log10 eta0>)` run per margin; mainshock flag used ONLY for targets).

## Models (all fixed in advance; nothing tuned on test)
* **M0 reference**: harness baseline `H.evaluate(None, target, model="logit")` p_base on the Chile test cells.
* **M1 PRIMARY - pooled GBM**: LightGBM on F_ext, trained on rows with t_start+7d <= 2013-01-01 from ALL margins
  incl. Chile (STRICT pseudo-prospective cutoff for every margin). Params: n_estimators=400, learning_rate=0.02,
  num_leaves=15, min_child_samples=300, subsample=0.8 (freq 1), colsample_bytree=0.8, reg_lambda=1.0;
  predictions averaged over 5 seeds (0..4).
* **M2 Chile-only GBM**: same features, same params, same seeds, trained on Chile pre-2013 rows only (key contrast).
* M3 pooled logit on F_base (L2 C=1 standardized, `H.logit_newton`) - harness-equivalent pooled logit.
* M4 pooled GBM, non-strict: non-Chile margins all years (incl. 2013+) + Chile pre-2013 (secondary, labelled).
* M5 pooled GBM on F_base (is it the extras or the pooling?).
Separate fits for y_all and y_main.

## Primary test
For each target T in {y_all, y_main} (weekly, M>=6.0): IG = (LL_M1 - LL_M0) / N_test_events on the Chile test cells
(t_start >= 2013-01-01). Significance: **exact paired year-block sign-flip randomization test**: per-year sums
D_y of cell log-likelihood differences (14 calendar years 2013..2026), all 2^14 sign patterns, one-sided
p = #{sum(s_y D_y) >= sum(D_y)}/2^14. This replaces `H.surrogate_test`, which tests an *added feature* and does not
apply to a different model family/training set. Its minimum p (6e-5) can reach Bonferroni.
* alpha: round-2 Bonferroni slot 0.05/6 = 0.0083, split over the two targets -> **0.0042 per target**.
* Verdict: positive = IG > 0 and p < 0.0042 on at least one target and survives the leakage checks;
  weak_positive = IG > 0 and p < 0.05 on some target; null otherwise (negative if M1 is clearly worse).
* Calibration of the sign-flip test (run before reading the primary result): 100 pairs of harness models each
  augmented with an independent pure-noise feature (equal expected skill); report FPR at 0.05.

## Secondary / robustness (preregistered)
1. M1 vs M2 (does pooling help vs Chile-only, same features), sign-flip p.
2. Harness `H.block_bootstrap_ci` for every comparison; AUC and Molchan ASS for base and model.
3. Leave-one-margin-out: for each margin m with >= 10 positive test cells: (a) pooled-strict model incl. m's
   pre-2013 rows, (b) strict LOMO model without m, (c) m-only GBM, vs (d) the harness-equivalent local logit
   baseline fitted on m's pre-2013 rows (F_base). IG per event on m's 2013+ cells.
4. Leakage / level-shift checks: Chile feature reproduction; calibration-in-the-large (sum p / observed) on Chile
   train and test; "oracle level-matched" IG (each model shifted by one logit constant so sum p = observed on test;
   diagnostic only); drop-one-test-year; M5 vs M1.
5. M>=5.5 weekly panel (same pipeline).
