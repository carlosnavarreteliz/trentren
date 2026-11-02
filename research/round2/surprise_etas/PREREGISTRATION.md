# PREREGISTRATION — round2/surprise_etas

Written 2026-10-17, BEFORE any evaluation on the test period (2013-01-01 → 2026-10). No test-period
labels (`y_all`, `y_main`, `n_*`) have been looked at for this design.

## Question
Does the *surprise* of recent seismicity relative to a fitted clustering model (ETAS) carry information
about future M≥6 events, beyond the harness ETAS-logit baseline? Round 1 tested RTL / Z-value quiescence
relative to a flat rate (null). Here anomalies are measured **relative to a fitted space–time ETAS**, i.e.
"what clustering did not expect".

## Data
Only the harness catalog (`H.get_events(min_mag=4.7)`, USGS ComCat, region lat −46…−17, lon −78…−66,
events from 1973-01-01). No downloads. Depth from the catalog. Mainshock flags are never used for features.

## ETAS model (fit on training data only)
1-D space (latitude ≈ along-strike) × time ETAS, magnitude threshold m0 = 4.7:

λ(t, y) = μ0 · f_bg(y) + Σ_{t_j<t} K·e^{α(m_j−m0)} · g(t−t_j) · φ(y − y_j; σ_j)

* g(t) = (p−1)c^{p−1}(t+c)^{−p} (normalized Omori–Utsu), p>1.
* φ = Gaussian in latitude with σ_j = d·10^{γ(m_j−m0)/2} (degrees).
* f_bg(y) = Gaussian KDE (bandwidth 0.5°) of the latitudes of the training-period ZBZ `bg` (causal flag) events,
  normalized over [−46, −17].
* Free parameters (μ0, K, α, c, p, d, γ) by maximum likelihood on target events with
  1976-01-01 ≤ t < 2013-01-01 (events from 1973 act as history only). Integral term includes the
  fraction of each spatial kernel inside the region. Parameters are then FROZEN and applied causally to the
  whole period (train in-sample, test out-of-sample).
* Training b-value: Aki–Utsu MLE on 1976–2012 M≥4.7 (Δm = 0.1 correction).

## Causal surprise components (per bin b, issue time t; neighbourhood B = bins b−1..b+1; only events < t)
1. `rr90` — log((N_obs+0.5)/(Λ_ETAS+0.5)) over [t−90 d, t) in B (relative quiescence < 0 < relative activation).
   Secondary windows 30 d and 365 d.
2. `ad` — Anderson–Darling statistic of the last 20 transformed inter-event intervals in B
   (Ogata residual analysis: τ = ∫λ over B; u = 1−exp(−Δτ) should be U(0,1)).
3. `gap` — log Λ_B(t_last, t): ETAS-expected number of events in B since the last observed one (open interval).
4. `lld365` — per-event log-likelihood of ETAS minus that of a homogeneous Poisson with the same count, in
   [t−365 d, t) × B, divided by N (NaN if N<3). Low = ETAS explains recent events poorly. Secondary 30 d, 90 d.
5. `mag` — magnitude surprise of the last 20 events in B: z = (mean(m−m0+Δm/2)·β − 1)·√20 with β = b·ln10 from
   training (i.e. standardized GR log-probability deficit; > 0 = magnitudes larger than GR expects).
6. `spat` — along-strike spatial surprise of the last 20 events in B: mean of [−log f_i(y_i) − H_i], where f_i is
   the ETAS spatial density at t_i (normalized over the region) and H_i its entropy.
7. `depth` — depth surprise of the last 20 events in B: mean of [−log p_bin(z_i) − H_bin], with p_bin a
   training-period depth KDE (bandwidth 10 km) of the event's own bin.

## PRIMARY composite score (one score)
For each component k, F_k = empirical CDF over the training-period panel rows (pooled bins).
p_k = two-sided tail probability 2·min(F, 1−F) for rr90, lld365, mag, spat, depth; upper tail 1−F for ad and gap.
p_k clipped to [1/(n_train+1), 1]. **x_surprise = mean_k(−ln p_k)** over available components
(Fisher-type "total surprise": large = recent seismicity anomalous relative to ETAS in any direction).

## Primary test
* Panel: harness weekly, `step_days=7, horizon_days=7, target_mag=6.0`.
* Target: **y_main** (declustered mainshocks). Model: logit (harness default). Feature: `x_surprise` (per bin).
* Statistic: `ig_per_event` vs the harness ETAS-logit baseline.
* Significance: `H.surrogate_test(mode="phase_randomize", n=1000)`. Round-2 Bonferroni α = 0.05/6 = 0.0083.
* y_all with the same test is reported alongside (not the Bonferroni primary).

## Secondary / robustness (no multiplicity credit)
* bin_permute (n=500) and end-matched circular_shift (n=200) for the composite.
* Each component alone (signed, raw), all components as one block, windows 30/90/365, GBM.
* Causally detrended composite: x − trailing mean over the previous 260 weeks in the same bin (`shift(1)`).
* Whole pipeline rebuilt with m0 = 5.0 (complete throughout) to control catalog-completeness drift.
* M≥5.5 panel; daily panel; drop-one-test-year; year block bootstrap.
* Leakage: truncation test (catalog cut at t*, features recomputed with frozen parameters must match);
  train-vs-test level shift of the score.

## Verdict rule
positive: IG > 0 on y_main AND phase_randomize p < 0.0083 AND survives detrending / m0=5.0 / truncation.
weak_positive: p < 0.05 only. Otherwise null (negative if IG clearly < 0 with CI excluding 0).
