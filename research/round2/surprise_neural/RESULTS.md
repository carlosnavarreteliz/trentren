# RESULTS: surprise_neural (round 2)

**Verdict: NULL.** The preregistered primary test was the block {x_ntpp_logp60, x_resid_90, x_ae_score} on y_main,
with the weekly M≥6 panel, the harness logit and phase_randomize surrogates (n = 1000).
- y_main: IG = **−0.0039 nats/eq**, surrogate **p = 0.45**, year-bootstrap 95 % CI [−0.016, +0.008].
- y_all: IG = −0.0010 nats/eq, p = 0.44.

On its own, the neural point process is a **worse** forecaster than the harness ETAS model on the test
period. None of the 10 neural "surprise" or anomaly features adds information beyond ETAS, on either
target or in any robustness variant.

The preregistration (`PREREGISTRATION.md`) was written before any test evaluation. Numbers come from `primary.json`,
`standalone.json`, `singles_prim.json`, `singles_expl.json`, `robust.json` and `causality_check.json`.

## 1. What was built
**M1, neural TPP.** A RECAST-like model in the intensity-free style (Shchur et al. 2020) with these parts:
- **Recurrent core:** a single GRU (H = 64) runs over the regional M≥4.7 sequence, 8,087 events from 1973 to 2026.
- **Inputs:** log inter-event time, magnitude, latitude, longitude and log depth.
- **Inter-event time:** a mixture of 8 log-normals.
- **Magnitude:** a mixture of 3 exponentials, with a defensive Gutenberg–Richter component.
- **Latitude:** a truncated mixture of 8 Gaussians, with a defensive histogram of training-period latitudes.

Training details:
- **Model selection:** fit on 1973–2007, validate on 2008–2012 (`select.log`). The configuration was chosen by
  mean validation NLL over 2 seeds: H = 64, 60 epochs, cosine learning rate and EMA weights.
- **Validation NLL against a stationary reference (Poisson + GR + latitude histogram):** time 1.49 → −0.07,
  latitude 1.08 → 0.47–1.18. Magnitude got slightly worse (0.036 → 0.06). The network learns clustering in time very
  well, but it does not improve magnitude prediction over GR.
- **Cross-fitting:** 4 blocks of 10 years before 2013 plus a final model trained on 1973–2012, with 3 seeds each
  (15 fits). Every feature value in the training period comes from a model that never saw that block, so there is no
  in-sample vs out-of-sample level shift.
- **Forecasts:** Monte-Carlo simulation of the network, 3 seeds × 300 paths per issue time. The per-bin
  P(M≥6 in 7 d) is Rao-Blackwellized. 248 of about 2.4 M paths were truncated at 300 events.

**M2, autoencoder.** A denoising autoencoder (45-32-4-32-45) of per-bin activity vectors:
- **Input vector:** 8 weekly counts plus a 52-week count, for bins b−2..b+2.
- **Anomaly score:** log reconstruction MSE. The model is cross-fitted the same way, with 3 seeds.

**Features** (all strictly causal):
- NTPP probability: `x_ntpp_logp60` / `x_ntpp_logp55`.
- Ogata-style residuals log((N_obs+0.5)/(Λ_model+0.5)) over bins b±1 for 30, 90 and 365 days: `x_resid_*`.
- The expected count `x_lam_90`.
- Mean per-event surprise (total, magnitude and time NLL) over 365 days: `x_surp_*`.
- Prediction-error embedding norm ‖h_i − h_{i−1}‖: `x_upd_norm_365`.
- The autoencoder score: `x_ae_score`.

## 2. Primary test (weekly, M≥6.0, logit, phase_randomize)
| target | N test | IG vs ETAS-logit base | 95 % CI (years) | p (phase, n=1000) | null q95 | AUC base→full | Molchan ASS base→full |
|---|---|---|---|---|---|---|---|
| **y_main** (primary) | 72 | **−0.0039** | [−0.016, +0.008] | **0.45** | +0.020 | 0.665 → 0.665 | 0.664 → 0.665 |
| y_all | 85 | −0.0010 | [−0.004, +0.003] | 0.44 | +0.018 | 0.697 → 0.697 | 0.696 → 0.696 |

The standardized coefficients on y_main were ntpp −0.16, resid_90 −0.02 and ae +0.01. The NTPP probability gets a
**negative** weight once ETAS is in the model.

## 3. The NTPP as a stand-alone forecast (test 2013–2026, weekly)
Recalibration is a 2-parameter logistic of log10 P fitted on training-period cells, which are cross-fitted.
IG is in nats per event; the CI is a year bootstrap.

| panel / target | IG NTPP(cal) vs ETAS | IG vs ETAS-logit base | IG vs Poisson | AUC NTPP / ETAS / Poisson |
|---|---|---|---|---|
| M6 y_all (85) | −0.226 [−0.51, +0.06] | −0.226 | −0.004 | 0.674 / 0.698 / 0.699 |
| M6 y_main (72) | −0.100 [−0.23, +0.01] | −0.092 | −0.086 | 0.645 / 0.677 / 0.678 |
| M5.5 y_all (228) | −0.120 [−0.30, +0.08] | −0.121 | **+0.065** [+0.02, +0.11] | 0.697 / 0.697 / 0.672 |
| M5.5 y_main (193) | −0.005 [−0.06, +0.04] | +0.019 [−0.03, +0.07] | −0.003 | 0.669 / 0.665 / 0.657 |

- Raw (uncalibrated) NTPP probabilities are worse still, at −0.2 to −0.3 nats/eq against ETAS.
- The recalibration slope is about 2, so the network's probabilities are too flat. It spreads aftershock
  probability over latitude more widely than the rupture-length kernel of the harness ETAS.
- At M5.5 it beats Poisson on y_all, so it does learn clustering, but it never beats ETAS.

## 4. Secondary and robustness results (all null)
- **Single features, n = 200 phase_randomize** (`singles_*.json`): there are 10 features × 2 targets.
  - Every IG lies between −0.008 and +0.001 nats/eq.
  - The smallest p is 0.35 (x_resid_90 on y_main, IG +0.0006, bin_permute p = 0.23).
  - No feature has p < 0.2.
- **GBM:** y_main −0.003, y_all −0.010.
- **M≥5.5 panel:** y_main −0.013 (p = 0.94; the CI excludes 0 on the negative side), y_all −0.008 (p = 0.81).
- **Causally detrended block** (x minus trailing 52-week mean, `.shift(1)`): y_main −0.007 (p = 0.58),
  y_all −0.017 (p = 0.81).
- **Other surrogates:** the end-matched circular_shift gives p = 0.36 / 0.35, and bin_permute gives p = 0.39 / 0.31.
- **Drop one test year:** IG on y_main stays in [−0.0065, +0.0002]. No single sequence drives anything.
- **Excluding cells with an M≥6 within ±1 bin in the previous 30 days:** y_main IG = −0.003 (69 events).

## 5. Leakage checks
- **Causality unit test** (`causality_check.json`): at 12 issue times (including 2014-04 Iquique and 2015-09 Illapel),
  GRU states, per-event NLL and the simulated forecasts were recomputed from a catalog truncated at t_start. The
  maximum absolute difference was 0 in every case.
- **Level shift between train and test** (cross-fitted features): ≤0.10 SD for the 3 primary features. The
  exploratory surprise features shift by −0.15 to −0.24 SD, and the update norm by +0.39 SD (the test period
  is more active). These shifts did not produce spurious gains, and the detrended versions are also null.
- **Data and catalog:** only the harness ComCat catalog was used, and `mainshock` never enters as an input. The
  revised catalog makes the test pseudo-prospective (the same caveat applies to all round-1 tests).

## 6. Decisions and deviations from the preregistration (all made before any test-period evaluation)
- **Stabilizing the network:** I added the defensive GR and latitude mixtures, the cosine learning rate and EMA,
  and changed the latitude-σ floor from 0.01 to 0.02. I also chose a 3-seed ensemble. The reason: in the first
  selection run, validation latitude NLL was unstable across epochs (0.3–1.9).
- **Simulation paths:** 3 × 300 = 900 per issue time instead of 300 (less Monte-Carlo noise).
- **Autoencoder epochs:** selected from {5, 15, 40} → 40, the edge of the grid. The validation MSE was still falling
  slowly (1.75 → 1.64 → 1.56).

## 7. Interpretation
Neural models make the same point as round 1, by a different route. A GRU point process learns temporal
clustering well (validation time NLL drops sharply), but it is spatially coarser than ETAS with a rupture-length
kernel, and it does not beat ETAS. "Surprise" under the network (Ogata residuals, per-event NLL, embedding update
norm) and the autoencoder "normality" score of local activity carry no detectable information about upcoming M≥6
mainshocks beyond the clustering features. With 72 y_main test events and a year-bootstrap CI half-width of about
0.012 nats/eq for the primary block, effects larger than about +0.01 nats/eq are unlikely. Smaller effects cannot be
excluded.

## Reproduce
```
cd research/round2/surprise_neural; export OMP_NUM_THREADS=2; D=../../data/round2_surprise_neural
python3 -I select_ntpp.py $D/models
python3 -I run_ntpp.py $D/models          # train, score, forecast (~15 min)
python3 -I build_features.py $D/models $D/feats_weekly.parquet
python3 -I causality_check.py $D/models $D/feats_weekly.parquet
python3 -I primary.py $D/feats_weekly.parquet 1000
python3 -I standalone.py $D/feats_weekly.parquet
python3 -I secondary.py $D/feats_weekly.parquet singles_prim 200   # also: singles_expl, robust
```
