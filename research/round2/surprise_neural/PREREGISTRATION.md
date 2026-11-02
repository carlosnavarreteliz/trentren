# PREREGISTRATION: surprise_neural (round 2)

Written 2026-10-17, BEFORE any evaluation on the test period (t_start >= 2013-01-01).
No `H.evaluate` / `H.surrogate_test` call has been made for this slug at the time of writing.

## Question
Does a small neural temporal point process (NTPP) of Chilean M>=4.7 seismicity, or "surprise" statistics
derived from it, or an autoencoder "normality" score of local activity, add information about upcoming
M>=6 earthquakes (especially declustered mainshocks) beyond the harness ETAS-logit baseline?

## Data
Only the harness catalog (`H.get_events(min_mag=4.7)`, USGS ComCat, region lat -46..-17, lon -78..-66,
from 1973-01-01). No downloads. The `mainshock` flag is never used as an input.

## Models
### M1. Neural TPP (RECAST-like, intensity-free; Shchur et al. 2020, Dascher-Cousineau et al. 2023)
- Single regional event sequence (all M>=4.7, all depths), time-ordered.
- Input per event i: log10 inter-event time (days, floor 1e-4), m-4.7, normalized latitude, longitude,
  log depth. GRU (1 layer, hidden H) -> state h_i.
- Given h_i, the next event has: inter-event time tau ~ log-normal mixture (K=8);
  magnitude excess m-4.7 ~ mixture of 3 exponentials; latitude ~ mixture of 8 Gaussians (normalized lat).
  Total NLL = time + magnitude + latitude NLL.
- Hyperparameters (H in {32, 64}, learning rate 3e-3, weight decay 1e-4, Adam, truncated BPTT chunks of 128
  events) and number of epochs are chosen ONLY on a selection split: fit 1973-2007, validate 2008-2012 (val NLL).
- **Cross-fitting** to avoid in-sample/out-of-sample level shifts in the train period: 4 time blocks
  (1973-1982, 1983-1992, 1993-2002, 2003-2012). For each block a model is trained (fixed selected epochs) on the
  other three blocks (state reset at block starts) and produces features/forecasts for issue times / events in
  the held-out block (GRU run causally over the full history up to t_start). A final model trained on
  1973-2012 produces all test-period (2013+) features. No data >= 2013 is used for any fitting.
- Forecast: for each weekly issue time t0 the network is rolled forward from its state at t0 (conditioning on
  the elapsed time since the last event) by Monte-Carlo simulation (300 paths, max 300 events per path in 7 d).
  The per-bin probability of >=1 event with M>=Mt in [t0, t0+7d) is computed with a Rao-Blackwellized
  estimator: P_b = 1 - mean_paths prod_j (1 - q_jb), q_jb = P(m_j>=Mt | h) * P(lat_j in bin b | h).

### M2. Denoising autoencoder of per-bin activity vectors
- Vector at (bin b, issue time t): log1p counts of M>=4.7 events in bins b-2..b+2 for each of the last 8 weeks
  (40 dims) plus log1p 52-week counts in the same 5 bins (5 dims); out-of-region bins padded with 0.
  Standardized with train-period statistics.
- MLP 45-32-4-32-45, Gaussian input noise 0.1, MSE. Epochs chosen on the same selection split
  (fit < 2008, validate 2008-2012); cross-fitted with the same 4 blocks; final model on all train rows for test.
- Anomaly score = log reconstruction MSE.

## Features (all strictly causal: built from events with time < t_start)
- `x_ntpp_logp`: log10 P_b from M1 (Mt = target magnitude of the panel).
- `x_resid_90`: Ogata-style residual of the NTPP in bins b-1..b+1 over [t-90d, t):
  log((N_obs + 0.5) / (Lambda_model + 0.5)), Lambda_model = compensator of M1 (time compensator x latitude
  mark probability of the 3 bins).
- `x_ae_score`: M2 anomaly score.
- Exploratory only (not in the primary test): `x_surp_mean_365` (mean per-event NLL of events in bins b-1..b+1
  in the last 365 d), `x_surp_mag_365`, `x_upd_norm_365` (mean ||h_i - h_{i-1}|| of those events = prediction-error
  embedding norm), `x_resid_30`, `x_resid_365`, `x_ae_score` variants.

## PRIMARY TEST (one test, counted once in the round-2 Bonferroni, alpha = 0.05/6 = 0.0083)
- Panel: weekly, step 7 d, horizon 7 d, target M>=6.0, 1-degree bins (harness default).
- Target: **y_main**.
- Model: harness logit; extra features = block {x_ntpp_logp, x_resid_90, x_ae_score}.
- Statistic: `ig_per_event` vs the ETAS-logit baseline.
- Significance: `H.surrogate_test(mode="phase_randomize", n=1000)` on the feature grid.
- Positive verdict requires IG > 0, p < 0.0083, and survival of the leakage checks below.

## Secondary / robustness (reported, not used for the verdict except as leakage checks)
1. Same block on y_all (phase_randomize n=1000).
2. Each feature alone, both targets (phase_randomize n=200), with bin_permute for per-bin features.
3. NTPP as a stand-alone forecast vs ETAS and vs the ETAS-logit baseline on test: IG per event with raw
   probabilities and after a 2-parameter logistic recalibration of log P_ntpp fitted on train (cross-fitted)
   rows; year-block bootstrap CI. AUC, Molchan ASS.
4. GBM model; M>=5.5 weekly panel; causally detrended versions (x minus trailing 52-week mean of the same bin,
   `.shift(1)`); end-matched circular_shift; drop-one-test-year.
5. Leakage checks: (a) cross-fit vs non-cross-fit level comparison of features between train and test;
   (b) causality unit test: recompute features at random issue times from a catalog truncated at t_start;
   (c) check whether any gain is concentrated in post-mainshock windows (clustering in disguise) by excluding
   cells with an M>=6 within +-1 bin in the previous 30 days.

## Caveats known in advance
- Test-period mainshock rate is ~32% above train: level shifts can masquerade as signal.
- Rate of M4.7-5.0 reporting changes over time (completeness); the NTPP learns from that inhomogeneous catalog.
- 72 y_main test events: an effect of +0.05 nats/eq is near the detection limit.
