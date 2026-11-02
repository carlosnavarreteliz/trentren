# PREREGISTRATION: global_transfer_neural

Written 2026-10-17, before any data download and before any test-period (2013+) evaluation.

## Question
Does a small neural sequence classifier, **pretrained on pre-2013 seismicity of many subduction margins**
(pooling dozens of M>=7 sequences) and then fine-tuned on Chile pre-2013, forecast M>=6 earthquakes in Chile
2013-present better than the harness ETAS-logit baseline? Does the global pretraining help relative to the same
network trained on Chile only?

## Data
- USGS ComCat FDSN, `eventtype=earthquake`, M>=4.7, 1973-01-01 to 2012-12-31 (global). **No post-2012 global
  data is downloaded**; pretraining therefore cannot see anything after the Chile train/test split.
- Margins (hand-drawn trench polylines, along-strike coordinate s, perpendicular distance d positive landward):
  Peru-Ecuador-Colombia, Central America-Mexico, Japan-Kuril-Kamchatka, Aleutians-Alaska, Izu-Bonin-Mariana,
  Ryukyu, Tonga-Kermadec, Sunda (Andaman-Sumatra-Java), Vanuatu, Philippine trench.
  Events kept if -200 km <= d <= 700 km (similar to the Chile box, which spans the outer rise to the back-arc);
  an event that falls in two margins goes to the one with smaller |d|. **Every event inside the harness Chile box
  (lat -46..-17, lon -78..-66) is removed from the global set.**
- Along-strike bins of 111.2 km (= 1 degree, like the Chile latitude bins).
- Chile: the harness panel (weekly, 7-day horizon, M>=6.0) and `H.get_events(4.7)`.

## Features (identical definitions for all margins)
- Tabular: the 22 harness baseline features, computed for every margin with the harness's own private functions
  (`_etas_features`, `_count_features`, `_maxmag_features`, `_days_since`) using the along-strike coordinate in
  place of latitude, plus `log_bg_m45` and `log_bg_target` from each margin's 1976-2012 rates; transformed with
  `H._transform_base`. For Chile the harness panel columns are used directly.
- Sequence: 52 weekly steps before t_start (windows [t-7(k+1), t-7k)), for bins b-2..b+2: log1p count M>=4.7,
  max magnitude above 4.2 (0 if none), count M>=6. 15 channels x 52 steps. All strictly before t_start.

## Model
- GRU (hidden 24) over the sequence + MLP over the tabular features -> 5 logits: y_all M>=5.5, y_main M>=5.5,
  y_all M>=6, y_main M>=6, y_all M>=7 (multi-task; y_main = ZBZ mainshock via `H.decluster` per margin, target only).
- **Pretraining**: all non-Chile margins, issue times 1976-01-01 ... with t_start+7d <= 2013-01-01. Early stopping on
  a validation set of randomly held-out (margin, 2-year block) units (20 %). Adam, lr 1e-3, batch 1024, <= 15 epochs.
  torch 2 threads.
- **Global-finetuned (GFT)**: pretrained weights, fine-tuned on Chile train rows (lr 3e-4).
- **Chile-only (CO)**: same architecture, random init, trained on Chile train rows only (lr 1e-3).
- **Global zero-shot (GZS)**: pretrained, no Chile data.
- Number of epochs for GFT (grid {1,2,4,8,16}) and CO (grid {4,8,16,32,64}) chosen by 5-fold time-blocked CV on
  Chile 1976-2012 (contiguous blocks, 26-week purge on each side of the held-out block), pooled OOF log-likelihood
  on the y_all-M6 + y_main-M6 heads. 3 seeds averaged in logit space.
- Chile-train-period predictions used for stacking are the out-of-fold predictions (so the harness logit does not
  see in-sample fits); Chile test predictions come from models fine-tuned on all of Chile 1976-2012 with the
  chosen epoch count.

## Comparison with the harness
- Feature handed to `H.evaluate`: `x_nn = logit(p)` of the relevant head (M6 y_all head for target y_all, M6 y_main
  head for target y_main), weekly panel, M>=6, `model="logit"`.
- **PRIMARY TEST (one test, counted in the round-2 Bonferroni, alpha = 0.05/6 = 0.0083):**
  GFT feature, target **y_main**, IG per event vs the ETAS-logit baseline, `surrogate_test(mode="phase_randomize",
  n=1000)`. Positive verdict requires IG > 0, p < 0.0083 and surviving the leakage checks.
- Key secondary (reported with the same machinery, nominal p only): GFT on y_all; CO and GZS on both targets
  (n=200-1000 surrogates); paired year-block bootstrap of GFT vs CO stand-alone log-likelihood (the direct
  "does pooling help" contrast); stand-alone NN log-likelihood vs base/ETAS/Poisson on test.
- Robustness (as HARNESS.md section 10): circular_shift end-matched p, `model="gbm"`, the M>=5.5 weekly panel
  (M5.5 heads), drop-one-test-year, year-block bootstrap CI.
- Descriptive only (too few events to test): M>=7 head on the Chile test (cells with an M>=7 event), versus a
  Gutenberg-Richter-scaled baseline probability.

## Leakage checks planned
- Global data restricted to < 2013; Chile box removed from the global set (assert zero overlap).
- Causality: recompute features for a sample of issue times from a catalog truncated at t_start.
- Fine-tuning and epoch selection use only Chile rows with t_start+7d <= 2013-01-01.
- Level shift: report mean of x_nn in train vs test and the IG with x_nn causally de-meaned
  (x_nn minus its own trailing 52-week mean for the same bin, `.shift(1)`).
- Clustering in disguise: the network sees only catalog seismicity, so any gain is by construction a better use of
  clustering information. That is still a forecasting gain, but it will be reported as such, not as a precursor.

## Amendment 1 (2026-10-17, still before any test-period evaluation; reason: CPU budget on a shared machine)
A smoke test (no test-period metrics computed; its output files were deleted unread) showed ~0.2 s per batch,
which would make the original plan take > 3 h. Changes:
- GRU hidden size 24 -> 16.
- Pretraining: each epoch uses all rows with any positive head plus a fresh random 35 % of the all-negative rows,
  with the sampled negatives weighted 1/0.35 in the loss (keeps calibration). Max 12 epochs, patience 3.
- Chile-only grid {4,8,16,32} epochs with lr 2e-3 (was {4,...,64} at 1e-3). GFT grid unchanged {1,2,4,8,16}, lr 3e-4.
- The training is split into stages (pretrain / CV per model and seed / final) so that each run is < 20 min.
Everything else (data, features, heads, stacking, primary test) is unchanged.

## Amendment 2 (2026-10-17, before any test-period evaluation; reason: machine load average ~48 on 8 cores)
Chile CV and final fits use 2 seeds {0, 1} instead of 3 (pretraining already finished with 3 seeds; seeds 0 and 1
of the pretrained network are used). Nothing else changes.
