# Audit of the earlier PI-ST-Fusion model (2026-10-17)

Scope: `src/physics_features.py`, `src/models_and_evaluation.py`, `predict_chile.py`, `results/*`, and the claims
in `ESTUDIO_PREDICCION_SISMICA_CHILE.md` / `README.md`. None of those files were modified. The audit scripts
are in `research/audit/` (`common.py`, `a1`–`a5`) and their outputs in `research/audit/a*_results*.json`. The
pre-registration was written before any test-period evaluation: `research/audit/PREREGISTRATION.md`.

## Bottom line

* The feature pipeline is **causal**. Recomputing all 87 features from a catalog cut just before Iquique 2014
  and Illapel 2015 gives exactly the stored values. No centered windows, no future events.
* The headline result reproduces exactly: Task 1 test ROC-AUC is 0.6537 for M6 and 0.6322 for M2.
* The task is **aftershock and clustering forecasting**. What was reported as "skill over ETAS" comes mostly
  from a weak ETAS baseline. Against a generic Reasenberg–Jones (1989) clustering baseline with 2 fitted
  parameters, the pre-registered primary test is a **null, and M6 is worse**.
* The "physics" variables (b-positive, slip deficit, locking, slab-pull ratios, migration, Zaliapin and others)
  add **no significant information** beyond clustering features in any setting tested.
* The study document overstates many claims. Its numbers do not match the code's output, and the live
  forecast is not calibrated: it sits **below plain climatology** in every zone.

## 1. Look-ahead leakage

| Item | Finding |
|---|---|
| Rolling-window features (counts, ETAS sums, b-values, Benioff, migration, Zaliapin) | Causal: they use `t_days[:i]` and `searchsorted(..., 'left')`. **a4: 0 differing values** across 87 columns × 562 events (Jan–Mar 2014, Jun–Sep 2015) after truncating the catalog. |
| Targets | Forward window from i+1. The last 30 days are excluded so every label is complete. OK. |
| `years_since_giant_rupture`, `tectonic_slip_deficit_m` | Causal. They combine a hard-coded list of pre-1961 giants with catalog M≥8 events that occur before each trigger. |
| `locking_chi` / `TRENCH_PROFILE` | Hand-typed values that the code says are "derived from Métois et al. 2016" and Slab2 (2018). Both were published after Iquique and Illapel, so this is **conceptual leakage**. The values are static and depend only on latitude, so they act at most as a spatial prior. Adding lat/lon/depth or the physics block to a clustering GBDT gives no gain (§5), so this leakage cannot be inflating the metrics materially. |
| Catalog revisions | The current, revised ComCat catalog is used everywhere, so the tests are pseudo-prospective. |
| Reporting non-stationarity | M≥4 events per year rise from about 150–250 (1976–91) to about 750–900 (2016–25). M<4 reporting is about 600–3000 per year in 1992–2008 and about 20–100 per year after 2009. As a result, `b_pos_subthresh_*`, the counts and the ETAS sums mean different things in training and in testing. |

## 2. Is the task aftershock forecasting?

Yes. After each M≥4 event, the label is whether an M≥5.5 occurs within 14 d and 150 km.
* With the 2018+ test period, event-indexed AUC is 0.65 on test but **0.92 on validation (2014–17)**. The
  validation period contains the Iquique and Illapel sequences. Scores are driven by which sequences fall in
  each period.
* With a 2013+ test period, AUC jumps to about 0.89 for every clustering model, again because of the 2014–15 sequences.
* The "pre-mainshock / declustered" subset is **not declustered**. It keeps 95% of test events and 91% of test
  positives, and it ignores the trigger's own magnitude.
* Under Gardner–Knopoff declustering, only 36–58% of positives remain mainshocks.
* `mag` ranks first in importance through the GBDT. About 22% of the published "importance" is the hand-set
  pathway weights × 600, so it is not a data-driven attribution. The "pre-mainshock" importance comes from the
  same LightGBM model (`lgb_pre = lgb_inter`). The "Pre-Mainshock Specialist" and the "Scaling-Law TPP" branch
  listed in M6's docstring do not exist in M6.

## 3. Hand-set weights (0.48/0.34/0.18 and the inner weights)

* No code fits them. The document's tables do not match `benchmark_metrics.json`. For example, Task 1 PR-AUC
  is 0.264 in the document vs 0.090 in the JSON, and Task 3 AUC is 0.7006 vs 0.691. The document's "Task 6"
  is also a different task from the code's. This means the 2018+ test was run and viewed several times while
  the code changed, i.e. the test set was reused in the process.
* The `predict_chile.py` alert thresholds were edited on disk during this audit (19:46 → 19:50) without any
  backtest.
* The numeric effect is small. Of 2000 random weight triples, 41% beat the chosen one on test and 62% beat it
  on validation (a1), and the AUC surface is flat (0.62–0.66).
* `map_score_to_calibrated_prob` overrides the fitted slope with 0.30 whenever it is below 0.05, so a score that
  points the wrong way can never show up as such.
* The monotone constraints set +1 on `b_pos_25` and `delta_b_pos` (higher b means higher hazard). The alert
  logic in `predict_chile.py` treats a drop in b as alarming. The two sign conventions contradict each other.

## 4. Calibration and live forecast

* At the event level, M6 over-forecasts on the 2018+ test: mean prediction 0.068 vs observed 0.045, and the top
  decile predicts 0.215 vs observed 0.099.
* On the 2013+ test it under-forecasts: mean 0.104 vs observed 0.156. The cluster de-weighting removes exactly
  the regime where events occur.
* The live forecast uses a **different score** from the tested M6: weights 0.45/0.20/0.15/0.20 instead of
  0.48/0.34/0.18, and z-scores computed over the whole 1900–2026 catalog. It then passes that score through
  M6's calibrator.
* An event-conditional probability ("within 150 km of this M4") is relabelled as a zone probability using
  `0.5·mean + 0.5·max` of the last 5 events. The 30-day, M5.0 and M6.0 values come from ad hoc
  power and Gutenberg–Richter (GR) scalings with b clipped to [0.72, 1.28]; 4 of 8 zones sit at the 1.28 clip.
* a5: every zone's live P(M≥5.5 in 30 d) is **below climatology**. Antofagasta is 7.2% vs 22–25%. Coquimbo is
  12.2% and flagged ALTA, yet its 2018–26 climatology is 21%.
* `predict_chile.py` prints a "Bouchon slab-pull" ratio from a column that does not exist, so it always shows 1.00×.
* The "CSEP Zero-Leakage" label is unjustified. The "CSEP IG" is a Bernoulli log-likelihood over overlapping,
  dependent event windows, not a CSEP test.
* The case studies (Iquique 2014 "3.6× two weeks before") fall in the validation and calibration period, so
  they are not out-of-sample.

## 5. How much real skill over ETAS remains (a2, a3)

The baseline is the generic Reasenberg–Jones model (a=−1.67, b=0.91, p=1.08, c=0.05). It sums over every M≥4
parent within 150 km in the past year, **including the trigger**, plus a static background, mapped to a
probability by a 2-parameter logistic fitted on the training period only. Significance comes from a block
bootstrap by year with 2000 resamples. The Bonferroni threshold is α=0.0036.

| Setting | M6 − RJ, IG (nats/event) [95% CI], p | ΔAUC (p) | Clust+physics GBDT − clust GBDT, IG (p) |
|---|---|---|---|
| **Primary: fit ≤2012, test 2013+, raw target** | **−0.085** [−0.18, +0.002], p=0.96 | −0.004 (0.72) | +0.0013 (0.24) |
| Same, declustered target | −0.012 [−0.029, +0.004], p=0.88 | −0.004 (0.58) | −0.041 (0.91) |
| Original split, test 2018+, raw | +0.0078 [+0.002, +0.015], p=0.0015 | +0.027 (0.08) | +0.0017 (0.16) |
| Same, declustered | +0.0030 [+0.001, +0.005], p<0.001 | +0.049 (0.006) | −0.0015 (0.88) |

* The original M2 "Classical-ETAS" leaves out the trigger's own contribution. Adding it is worth +0.0049
  nats/event (p=0.0055).
* On 2018+, RJ has a higher PR-AUC than M6 (0.098 vs 0.090).
* With only 9–14 yearly blocks, the CIs are approximate.

**Verdict.** The pre-registered primary test is a null: M6 does not beat a 2-parameter generic aftershock model
and is worse in log-likelihood on 2013+. On the 2018+ split, whose test period was reused during development,
M6 shows a small but significant gain of about 0.003–0.008 nats per event, but the gain reverses on 2013+. It is
not attributable to the new "physics" variables, which add nothing beyond clustering features in any setting.
Real, robust skill over ETAS-type clustering is **not demonstrated**. At best it is a few thousandths of a nat per
event and depends on the regime.
