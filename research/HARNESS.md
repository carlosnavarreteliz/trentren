# Shared evaluation harness (`research/harness.py`)

> **CALIBRATED (2026-10-17).** The primary significance test is
> `surrogate_test(mode="phase_randomize", n>=200)` (the default). Do **not** use plain circular shift or
> the Rhoades T-test as evidence.
>
> Results on 120 pure-noise features (n=200 surrogates each): phase_randomize FPR at p<0.05 is
> y_all 5/60 = 8.3 %, y_main 0/60 = 0 %, pooled 5/120 = **4.2 %**. No feature reached p<0.0036.
> The test is calibrated for random walks, AR(0.98) and white noise alike (§9b).
>
> For slow or persistent series (LOD, sunspots, ENSO, sea level, GNSS trends, any AR(1) with phi > 0.9,
> anything random-walk-like):
> 1. Use `phase_randomize` as the primary test, preferably through `builder` on the RAW daily series.
> 2. ALSO report a causally differenced or detrended version of the feature, e.g.
>    x − trailing mean computed with `.shift(1)`, or a first difference.
> 3. Report the end-matched `circular_shift` p as a robustness check.

Use this file to test whether a candidate variable improves earthquake forecasts in Chile beyond what
clustering (ETAS) already explains. All hypothesis agents import the same harness, so their results
can be compared directly.

```python
import sys; sys.path.insert(0, "/Users/cnavarreteliz/Documents/Repos/earthquakes/research")
import harness as H
panel = H.get_panel(step_days=7, horizon_days=7, target_mag=6.0)   # cached parquet, ~1 s
res = H.evaluate(my_feats, target="y_all", model="logit", panel=panel, return_preds=True)
sur = H.surrogate_test(my_feats, n=200, mode="phase_randomize", target="y_all", panel=panel)
print(H.summarize(res, sur)); print(H.block_bootstrap_ci(res["preds"]))
```

A working template is in `research/hypotheses/_example/example_hypothesis.py`. Run it with
`/opt/miniconda3/bin/python3 -I example_hypothesis.py [daily.csv] [n_surr]`; it takes about 30 s with n=20.

---

## 1. Data, region, time conventions

| item | value |
|---|---|
| catalog | `data/raw/chile_usgs_1900_2026.csv` (USGS ComCat), `type=='earthquake'` |
| region | lat −46 … −17, lon −78 … −66 (includes the outer rise and the slab down to the back-arc) |
| history used for features | events from **1973-01-01** |
| first forecast issue time | **1976-01-01** |
| train | rows with `t_start + horizon <= 2013-01-01` (purged: no label window reaches into the test period) |
| test | rows with `t_start >= 2013-01-01` up to the catalog end (2026-10-17) |
| time zone | **UTC everywhere.** Timestamps are tz-naive UTC `datetime64[ns]`. If your data has a tz, run `tz_convert("UTC").tz_localize(None)` |
| spatial unit | 1° latitude bins (the margin runs N–S, so latitude ≈ along-strike). `bin_id` 0 = −46…−45, 28 = −18…−17. `H.lat_to_bin(lat)`, `H.bin_centers()` |

### Magnitude of completeness and modeling threshold
`H.completeness_by_year()` gives Mc per year from maximum curvature + 0.2. The results:
Mc ≈ 4.9–5.1 for 1973–1990, ≈ 3.5–4.0 for 1991–2008 (dense local reporting), and ≈ 4.4–4.7 for 2009–2026.
The ratio N(M≥4.5)/N(M≥5.0) climbs from 2.3 (1970s) to 4–5 (after 2013). That is too large to be
explained by b≈1 alone, so M4.5 is **not** homogeneous across the whole period. The ratio
N(4.7)/N(5.0) is 1.84–1.91 before 1990 and 2.2–2.3 after, which is only mildly inhomogeneous.
N(5.0)/N(5.5) has no trend, so M≥5.0 is complete throughout.

**Choice: `M_MODEL = 4.7`.** The baseline features are built from M≥4.7 events, and the declustering
runs on M≥4.7. This keeps enough events for short-term clustering, at the cost of about 20 % under-reporting
of M4.7–5.0 before 1990. Targets are M≥5.5 or M≥6.0, and both are complete for the whole period.
**Consequence for you:** any feature whose values trend over time (a station count, a long-term index)
can correlate spuriously with this rate drift. That is one more reason the circular-shift surrogate is
the primary test.

## 2. Declustering

`H.decluster(method="zbz"|"gk", min_mag=4.7)`. The default is **Zaliapin & Ben-Zion (2013)** nearest-neighbour
declustering: η = Δt[yr]·r[km]^1.6·10^(−m_parent), with r floored at 1 km. The threshold
log10 η0 = −5.36 comes from a 2-component Gaussian mixture fitted on pre-2013 events. Gardner–Knopoff
windows are available as `method="gk"`. GK uses a foreshock window of half the aftershock window and
processes events largest first.

Output columns:
- `bg` (CAUSAL): the event is not strongly linked to any **earlier** event. It depends only on the past and
  can be used to build features.
- `mainshock` (NON-CAUSAL): the event is the largest in its cluster. A later, bigger event turns an earlier
  "mainshock" into a foreshock (for example, the 2014-03-16 M6.7 before Iquique M8.2). **Use it only for targets.
  Never build features from it.**
- `cluster_id`, `parent_idx`, `log_eta`.

Counts for M≥4.7, 1973–2026: 8,087 events in 5,404 clusters. At M≥6.0 there are 349 events and
227 mainshocks (GK gives 216). At M≥5.5 there are 1,107 events and 717 mainshocks (GK gives 594).

`H.get_events(min_mag=4.7, declustered=False, start="1973-01-01", end=None)` returns the event list
(time, t_days, latitude, longitude, depth, mag, magType, id, bin_id, cluster_id, bg, mainshock).
`min_mag < 4.7` is allowed only with `declustered=False`, and that list is incomplete before about 1990.

## 3. The panel: `H.build_panel(step_days=7, horizon_days=7, target_mag=6.0, bin_deg=1.0)` / `H.get_panel(...)`

Each row is a pair (bin, t_start): a forecast issued at `t_start` for the window **[t_start, t_start+horizon)**.
Issue times are `1976-01-01 + k·step_days`. Use `get_panel`, which caches the result in memory and on disk
(`research/cache/panel_v1.0_s{step}_h{h}_m{M}_b{bin}.parquet`). Rows are ordered by (t_start, bin_id).
Any step/horizon/target combination works. The weekly panel builds in about 5 s and the daily one in about 15 s.

| column | meaning |
|---|---|
| `bin_id`, `lat_center`, `t_start`, `year`, `split` | keys; `split` ∈ {train, gap, test} (gap = the purged rows straddling 2013-01-01) |
| `y_all` | 1 if any M≥target event in the bin within the horizon |
| `y_main` | 1 if any declustered **mainshock** M≥target in the bin within the horizon |
| `n_all`, `n_main` | counts (reference only; **never a feature**) |
| `p_poisson_all`, `p_poisson_main` | Poisson smoothed-seismicity background: training-period frequency of positive cells per bin, Gaussian-smoothed across bins (σ=1 bin) |
| `etas_p{p}_c{c}_a{α}_s{s0}` (12) | Σ over past events (t_e < t_start) of 10^{α(M−4.7)} · ∫_{t}^{t+h}(s−t_e+c)^{−p}ds · (Gaussian latitude kernel integrated over the bin, σ² = s0² + (half rupture length)², with rupture length from Wells & Coppersmith). (p,c) ∈ {(1.08,0.02 d), (1.3,0.02 d), (0.9,0.5 d)}, α ∈ {0.8,1.1}, s0 ∈ {0.3°,1.0°} |
| `cnt_bin_{7,30,365}d`, `cnt_nbr_{30,365}d` | number of M≥4.7 events in the bin (or in bins ±1) in the last N days, before t_start |
| `maxmag_nbr_{30,365}d` | largest M in bins ±1 in the last N days (floor 4.2) |
| `log_days_since_m6_nbr` | log10(1 + days since the last M≥6 in bins ±1) |
| `log_bg_m45`, `log_bg_target` | log10 of the training-period M≥4.7 rate and target rate of the bin |

The baseline feature set is `H.baseline_feature_cols(panel)`. Any column that starts with `x_` is never
treated as a baseline feature.

## 4. Baselines (fitted on train only, inside `evaluate`)
1. **Poisson**: `p_poisson_*`, a time-independent smoothed background.
2. **ETAS (additive, MLE)**: rate = w0·r_bg + Σ_k w_k·etas_k with all w ≥ 0, and P = 1 − exp(−rate).
   The weights are fitted by Bernoulli maximum likelihood on train (`H.fit_etas_additive`). This is an
   ETAS-style conditional intensity with a fixed kernel library.
3. **Baseline used for comparisons** (`p_base`): a logistic regression (L2, C=1, standardized inputs) or a
   LightGBM model on all baseline features plus log10 of the ETAS rate. **This is the model your variable must beat.**

## 5. `H.evaluate(extra_features=None, target="y_all", model="logit", panel=None, extra_cols=None, return_preds=False, **panel_kw) -> dict`

- `extra_features`: a DataFrame with `t_start` (and optionally `bin_id`) plus numeric columns.
  - With `bin_id`: a per-bin feature. Without it: a global series that is broadcast to all bins.
  - The keys must be unique. Rows that are missing after the merge become NaN. For the logit model,
    NaN is imputed with the train mean; LightGBM handles NaN natively.
  - Name your columns `x_...`.
- The function fits *baseline* and *baseline + extra* on train with the same model class, then predicts test.
  Nothing is tuned on test: the hyperparameters are fixed in code.
- `model="logit"` is the primary choice. `"gbm"` uses LightGBM with 200 trees, learning rate 0.02,
  7 leaves, min_child_samples 300 and n_jobs 2. It is a secondary check for nonlinear effects and costs about 1 s per fit.

Returned keys:

| key | meaning |
|---|---|
| `ig_per_event` | **PRIMARY.** (LL_full − LL_base) / n_test_events, in nats per target event, Bernoulli log-likelihood on test |
| `ig_full_vs_etas`, `ig_full_vs_poisson`, `ig_base_vs_poisson`, `ig_etas_vs_poisson` | the same quantity against the other references |
| `base`, `full`, `etas`, `poisson` | dicts with `ll`, `brier`, `roc_auc`, `pr_auc`, `molchan_ass` (area skill score, uniform cell measure; 0.5 = random), `expected` (Σp), `observed` |
| `ttest_full_vs_base`, `ttest_base_vs_poisson` | paired T-test after Rhoades et al. (2011) with p used as the expected count: I_i = ln(pA_i/pB_i) − (ΣpA−ΣpB)/N. Returns `T`, `p_one_sided`, `IG`. **Anti-conservative under clustering**, so report it but do not rely on it |
| `n_test_events`, `n_train_events`, `n_test_cells` | event counts are **positive cells**, not raw earthquakes |
| `coef_extra` (logit, standardized) / `importance_extra` (gbm) | effect of your feature |
| `preds` (if `return_preds`) | per-test-cell `bin_id, t_start, year, y, p_base, p_full, p_poisson, p_etas` |

`H.block_bootstrap_ci(preds, a="p_full", b="p_base", n_boot=2000)` resamples calendar years and returns
the IG per event with a 95 % CI and `p_boot_le0`. `H.summarize(res, sur)` prints a one-line summary.

## 6. `H.surrogate_test(extra_features=None, builder=None, n=200, mode="phase_randomize", target="y_all", model="logit", panel=None, extra_cols=None, min_shift_days=365, seed=0, n_jobs=1, end_matched=True) -> dict`

This gives the significance test, and it is the **primary** one. Each surrogate repeats the full
fit-and-test pipeline with the feature–target alignment destroyed. Modes:

- `mode="phase_randomize"` (**default, primary**) uses multivariate Fourier phase-randomized surrogates
  (`H.ft_surrogate`; Theiler 1992, Prichard & Theiler 1994). The same random phases are applied to every
  bin and column, so the power spectrum (autocorrelation) and the cross-bin structure are preserved.
  Columns that behave like random walks (lag-1 autocorrelation > 0.99) are randomized on their increments
  and re-integrated. That keeps the drift and avoids any wrap-around jump.
- `mode="circular_shift"` rolls every feature column along the panel time axis by a random offset of
  at least 1 year from both ends. The same offset applies to all bins.
  - With `end_matched=True` (default), the straight line joining each series' first and last values is
    removed before rolling and added back afterwards. That avoids an artificial jump at the wrap point,
    and a pure linear trend earns no significance.
  - The plain `np.roll` version (`end_matched=False`) is **anti-conservative for slow or random-walk-like
    series**. In calibration it gave 4/20 false positives on `y_main` (§9b). Use circular_shift as a
    secondary check.
- `mode="bin_permute"` applies one random bin permutation to every time step. Use it for per-bin features.
- `builder` rebuilds the features from a surrogate of your **raw** series. Use it when the feature involves
  rolling, anomaly or other nonlinear operations:
  - phase_randomize: `builder(rng)`. Inside, call `H.ft_surrogate(raw_values, rng)` and then aggregate.
    `builder(None)` must return the real feature.
  - circular_shift: `builder(shift_days)`. Inside, call `H.circular_shift_series(raw_series, shift_days)`.
    `builder(0)` must return the real feature.

Returns `ig_obs`, `p_value` (one-sided: (1+#{null ≥ obs})/(n+1)), `null_mean`, `null_sd`, `null_q95`,
`significant_0.05`, `significant_bonferroni` (p < 0.05/14 = **0.0036**), and `null`.

**Recommended reporting:** report the phase_randomize p as the primary result. For per-bin features, also
report the bin_permute p. Report the circular_shift (end-matched) p as a robustness check.

- With n=200 the smallest p is 0.005, so it **cannot reach Bonferroni significance**. To claim Bonferroni
  significance, run n ≥ 1000. A logit run costs about 0.25 s per surrogate (weekly panel), so n=1000 takes
  about 4 minutes.
- `n_jobs` ≤ 2 (threads).

**Note:** the target is non-stationary. The y_main test rate is 32 % higher than the train rate
(test: 72 observed vs about 57 expected). Any slow, trending variable can look predictive simply because
its level drifts in the same direction. The surrogates above are built to control for this. Agents should
still report whether the gain persists after removing a causal trend from the feature.

## 7. Helpers for exogenous data (look-ahead safe)

- `H.align_daily_series(series, panel, window_days=30, agg="mean", lag_days=1, name=None) -> DataFrame[t_start, name]`
  aggregates a **measured** daily series. The window is the days D with
  `t_start − lag_days − window_days ≤ D < t_start − lag_days`. Day D covers 00:00–24:00 UTC.
  Example: t_start = 1976-01-01, window_days = 7, lag_days = 1 gives days 1975-12-24 … 1975-12-30.
  `agg` is one of mean, sum, max, min, std or last. Raise `lag_days` for products published late.
  Use the value **as it was available in real time** when that differs from the final or reprocessed
  version (for example, geomagnetic definitive vs. provisional indices, or GNSS reprocessing).
- `H.align_deterministic_series(series, panel, agg="mean")` is only for quantities **known in advance**,
  such as tidal stress from an ephemeris or lunar phase. It summarizes the forecast window
  [t_start, t_start + h) itself. Never use it for measured data.
- `H.past_event_count(events, panel, window_days, nbr=0, name)` counts events per bin in
  [t_start − window, t_start) and is strictly causal.
- To aggregate point data per bin, map latitude to bins with `H.lat_to_bin(lat)` and use only
  observation times < t_start. Then build a frame [bin_id, t_start, x_*].

Look-ahead traps. None of these are allowed:
- `rolling(center=True)`, `interpolate()` across t_start, or `bfill`.
- z-scores or detrending fitted on the whole period. Fit them on train only, or make them causal.
- Station or network metadata that changes *because of* an earthquake: densification after 2010 and 2014,
  or new GNSS sites installed after a big event.
- Using `mainshock` or `n_all` as inputs.
- Any feature built from events with time ≥ t_start.

## 8. Event-level tools (short-timescale hypotheses such as tides)
- `H.get_events(min_mag, declustered=True)` returns mainshocks only. The flag is non-causal, which is fine
  for a phase test.
- `H.schuster_test(phases_rad)` returns p = exp(−R²/N) and assumes uniform phase occupation.
- `H.schuster_vs_background(event_phases, background_phases, n_mc=5000)` is the Monte-Carlo version
  against the phase distribution at random times. **Use it for tides**, because tidal phase occupation is
  not uniform. Report results on declustered events, and declare the phase definition before looking at the data.

## 9. Calibration (run 2026-10-17; scripts and JSON in `research/calibration/`)

### 9a. Baseline metrics on test (2013-01-01 → 2026-10)
IG is in nats per target event. The year-bootstrap 95 % CI covers IG(base − Poisson) with 14 test years.

| panel | target | N test | IG ETAS vs Poisson | IG base(logit) vs Poisson | 95 % CI (year bootstrap) | AUC Poisson / base | Molchan ASS base |
|---|---|---|---|---|---|---|---|
| weekly, M≥6.0 | y_all | 85 | +0.222 | +0.221 | [−0.10, +0.54] | 0.699 / 0.697 | 0.696 |
| weekly, M≥6.0 | y_main | 72 | +0.014 | +0.006 | [−0.04, +0.06] | 0.678 / 0.665 | 0.664 |
| weekly, M≥5.5 | y_all | 228 | +0.185 | +0.186 | [−0.04, +0.40] | 0.672 / 0.697 | 0.694 |
| weekly, M≥5.5 | y_main | 193 | +0.002 | −0.022 | [−0.05, 0.00] | 0.657 / 0.658 | 0.657 |
| daily, M≥6.0 | y_all | 92 | +0.546 | +0.546 | [−0.05, +1.08] | 0.696 / 0.743 | 0.742 |
| daily, M≥6.0 | y_main | 72 | +0.022 | +0.045 | [+0.00, +0.09] | 0.677 / 0.678 | 0.678 |
| daily, M≥5.5 | y_all | 252 | +0.524 | +0.541 | [−0.02, +1.02] | 0.677 / 0.729 | 0.728 |
| 30-day, M≥6.0 | y_all | 80 | −0.040 | −0.053 | [−0.12, +0.01] | 0.698 / 0.671 | 0.668 |

See `calibration/baselines.json` for every number, including the GBM rows and Brier scores.

How to read this:
- At M≥6, clustering skill comes from the few cells that follow a large event. In the weekly M6 test,
  about 12 of the 85 positive cells (Iquique 2014, Illapel 2015, Chiloé 2016) have ETAS probabilities
  20–100× the background. On the remaining cells ETAS is about 0.85× Poisson.
- The ETAS gain therefore grows as the step gets shorter (daily +0.55 > weekly +0.22 > monthly ≈ 0),
  because at long steps the aftershocks fall inside the mainshock's own window.
- On `y_main`, ETAS ≈ Poisson, as expected: mainshocks are what clustering cannot predict.
  **A real new precursor would show up most clearly on `y_main`.**
- Test periods have few events (72–228 cells), and the year-bootstrap CIs are wide. A real effect of
  +0.05 nats/eq is hard to distinguish from noise. Say this explicitly when you report a null.

### 9b. Null calibration (pure-noise features, logit, weekly M6). FINAL, see `calibration/CALIBRATION_V2.md`

**v1, plain `np.roll` circular shift** (`calibration/v1_plain_roll/`). There were 20 noise features per
target: 7 global random walks, 7 global white-noise series, 3 per-bin white-noise series and 3 per-bin
random walks.

| target | FPR at p<0.05 | false positives among global random walks |
|---|---|---|
| y_all | 1/20 = 5 % | 0/7 |
| y_main | 4/20 = **20 %** | **3/7** (p = 0.03, 0.01, 0.01) |

The naive Rhoades T-test flagged 4/20 on y_main, and 3 of those were at p < 0.0036.

**Diagnosis.** Rolling a random walk creates a jump at the wrap point, which acts as a step regressor.
In about half the surrogates the logit fit uses that step to absorb the train-period rate drift and then
extrapolates badly to the test period. This pulls the null IG distribution down: for y_main random walks
the null mean was −0.02 to −0.05 nats/eq, while the observed IG was about 0. A pessimistic null makes
p-values too small. The effect is stronger on y_main because the mainshock rate is non-stationary
(the test rate is 32 % above train).

**Fix (v2).**
- `mode="phase_randomize"` is the new default (`H.ft_surrogate`). It is a multivariate Fourier
  phase-randomized surrogate that applies the same phases to all bins and columns.
  - Columns with lag-1 autocorrelation > 0.99 are randomized on their increments and re-integrated, which
    keeps the start value and drift and introduces no jump.
  - All other columns use classic levels-FT.
  - NaN positions are preserved.
- `circular_shift` is end-matched by default.
- The logit is now solved by exact Newton (`H.logit_newton`), warm-started from the baseline. It reaches
  the same optimum as sklearn and runs about 3–10× faster. Baseline metrics are unchanged to 3 decimals.

**v2 calibration** (`calibration/run_calibration_v2.py` → `v2_*.json`). There were 60 features per target:
20 global random walks, 20 global white-noise series, 10 global AR(1) series with phi = 0.98,
5 per-bin random walks and 5 per-bin white-noise series. Each used phase_randomize with n=200. Per-bin
features also got bin_permute with n=100. The y_main random walks also got an end-matched circular_shift
with n=100.

| test | target | rw_global | wn_global | ar98_global | rw_bin | wn_bin | **total** | p<0.0036 |
|---|---|---|---|---|---|---|---|---|
| phase_randomize (n=200) | y_all | 2/20 | 2/20 | 0/10 | 1/5 | 0/5 | **5/60 = 8.3 %** | 0 |
| phase_randomize (n=200) | y_main | 0/20 | 0/20 | 0/10 | 0/5 | 0/5 | **0/60 = 0 %** | 0 |
| phase_randomize, pooled | both | 2/40 | 2/40 | 0/20 | 1/10 | 0/10 | **5/120 = 4.2 %** | 0 |
| circular_shift end-matched (n=100) | y_main | 0/20 | | | | | 0/20 | 0 |
| bin_permute (n=100) | y_all / y_main | | | | 1/5 · 0/5 | 0/5 · 0/5 | 1/20 | 0 |

- 5/60 on y_all is within binomial noise of 5 %: P(X ≥ 5 | p = 0.05, n = 60) ≈ 0.18.
- 0/60 on y_main is slightly conservative: P(X = 0) ≈ 0.05.
- The mean p is 0.44 (y_all) and 0.48 (y_main), slightly below 0.5. The reason is that L2 shrinks useless
  features to IG ≈ 0, which produces ties near zero; this does not inflate the FPR at α = 0.05.
- **Never use the Rhoades T-test as evidence.** On random walks it is badly anti-conservative
  (T = +4.2 for pure noise).

### 9c. Controls (`calibration/controls.json`)
- **Positive control:** a deliberately leaky feature (y + N(0,1) noise) gives IG = +0.43 nats/eq,
  surrogate p = 0.0099 (the minimum with n=100) and T = 4.3. The harness detects real signal.
- **Clustering in disguise:** the count of M≥5 events within ±1 bin over the last 14 days has a naive
  test AUC of 0.57 on its own. Against the ETAS baseline it gives IG = +0.002 and p = 0.15, i.e. no gain.
  This is exactly the trap the harness is built to catch.

## 10. Reporting checklist for hypothesis agents
1. Write `PREREGISTRATION.md` (feature definition, panel, target, model and primary test) **before**
   running `evaluate` on test.
2. Report `ig_per_event` (vs the ETAS-logit baseline) for both `y_all` and `y_main`, the year-bootstrap CI,
   the surrogate p (n ≥ 200, or n ≥ 1000 if you claim Bonferroni significance), and AUC and Molchan ASS
   for base and full.
3. If the result is significant, run the GBM model, the M≥5.5 panel and another step/horizon as robustness
   checks. Also check that the effect is not carried by a single sequence: drop each test year in turn
   using `preds`.
4. A null is a valid result. Say plainly when data could not be obtained.
