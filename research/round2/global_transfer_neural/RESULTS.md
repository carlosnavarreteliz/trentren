# RESULTS: global_transfer_neural (round 2)

Neural sequence classifier (GRU over 52 weekly steps × 5 along-strike bins × 3 channels, plus an MLP over the
22 harness baseline features). It was pretrained on 10 non-Chile subduction margins (ComCat M≥4.7,
1973–2012, 507,590 bin-weeks, 1,955 M≥6 mainshock cells, 257 M≥7 cells), then fine-tuned on Chile 1976–2012
and tested on Chile 2013-01 → 2026-10 (weekly panel, M≥6, 7-day horizon). See `PREREGISTRATION.md`.
Amendments 1 and 2 (smaller GRU, negative subsampling, 2 Chile seeds) were made for CPU reasons before any
test evaluation. The run was cut off by a usage limit and then resumed: the stages finished in this order:
pretrain → CV (CO s0, s1; GFT s0, s1) → final → evaluation. Nothing was re-run after the test results were seen.

Models: **GFT** = global-pretrained + Chile fine-tuned (CV chose 1 epoch, so fine-tuning barely moves it);
**CO** = Chile-only, same architecture (CV chose 8 epochs); **GZS** = global zero-shot (no Chile data at all).

## Primary preregistered test (counted in the round-2 Bonferroni, α = 0.0083)

GFT logit as an extra feature in the harness ETAS-logit, target **y_main**, phase_randomize n = 1000:

| target | N test | IG vs ETAS-logit (nats/eq) | surrogate p (phase, n=1000) | null mean ± sd | AUC base → full | Molchan ASS base → full |
|---|---|---|---|---|---|---|
| **y_main** | 72 | **+0.0021** | **0.424** | +0.0019 ± 0.028 | 0.6645 → 0.6647 | 0.6640 → 0.6641 |
| y_all | 85 | +0.0006 | 0.519 | +0.0050 ± 0.023 | 0.6969 → 0.6972 | 0.6961 → 0.6964 |

**NULL.** The stacked gain is the same size as the gain from a phase-randomized copy of the feature.

## Secondary (stacked in the harness; nominal p only)

| model | target | IG stacked | phase p (n) | year-bootstrap 95 % CI | logit coef |
|---|---|---|---|---|---|
| GFT | y_main | +0.0021 | 0.424 (1000) | [0.000, 0.004] | +0.18 |
| CO | y_main | −0.0106 | 0.711 (200) | [−0.029, 0.005] | −0.73 |
| GZS | y_main | +0.0060 | 0.363 (200) | [−0.003, 0.018] | +0.61 |
| GFT | y_all | +0.0006 | 0.519 (1000) | [0.000, 0.001] | +0.03 |
| CO | y_all | −0.0242 | 0.935 (200) | [−0.061, 0.003] | −0.68 |
| GZS | y_all | +0.0108 | 0.303 (200) | [0.004, 0.020] | +0.42 |

Robustness for GFT:
- circular_shift (end-matched, n = 200): p = 0.35 on y_main and 0.34 on y_all.
- GBM model: IG +0.013 on y_main and +0.025 on y_all (point estimates, no surrogate).
- M≥5.5 panel: IG +0.0013 on y_main (phase p = 0.50) and −0.0005 on y_all (phase p = 0.77).
- Drop-one-test-year: IG stays between 0.0016 and 0.0026 on y_main, so no single year drives it.
- Causally de-meaned feature: IG +0.0049 on y_main and −0.0002 on y_all.

The year-bootstrap gives p_boot ≤ 0.02 for GFT stacked, while the surrogate p is 0.42. This is the known
anti-conservatism of the year bootstrap for tiny, systematically positive gains, which random features also
produce (null mean +0.002).

## Does pooling help? (preregistered secondary contrast)

Paired year-block bootstrap on test, GFT vs CO:
- **Stand-alone:** +0.057 nats/eq on y_main (CI [0.020, 0.092], p_boot 0.0005) and +0.074 on y_all
  (CI [0.012, 0.141], p_boot 0.0035).
- **Stacked:** +0.013 on y_main (p_boot 0.06) and +0.025 on y_all (p_boot 0.04).
- On the Chile pre-2013 OOF data, the GFT log-likelihood (all6 + main6 heads) is about 12 nats better than CO.

Within neural networks, then, pooling the global margins clearly helps. Chile alone is too small for a GRU,
and CO is worse than the ETAS-logit. GZS ≈ GFT, so the useful knowledge comes from the global data, and the
Chile fine-tuning adds nothing (CV picked 1 epoch).

## Stand-alone NN vs ETAS-logit base (preregistered as secondary, plus post-hoc decompositions)

The checks marked post-hoc (`extra_analysis.py`, `train_period_check.py`, `per_year_alone.py`) were written
after the test results were seen and are exploratory.

| model | target | IG raw vs base | IG level-matched vs base | p_boot (level-matched) | AUC nn / base | IG vs pure ETAS |
|---|---|---|---|---|---|---|
| GFT | y_main | +0.062 | +0.043 | 0.008 | 0.678 / 0.665 | +0.054 |
| GZS | y_main | +0.069 | +0.050 | 0.002 | 0.679 / 0.665 | +0.061 |
| CO | y_main | +0.005 | −0.002 | 0.55 | 0.662 / 0.665 | −0.003 |
| GFT | y_all | +0.030 | +0.032 | 0.025 | 0.705 / 0.697 | +0.030 |
| GZS | y_all | +0.036 | +0.037 | 0.043 | 0.709 / 0.697 | +0.036 |

- **Level-matched** means the NN logit is shifted so that its test-period sum equals the base sum. This
  removes the part of the gain that comes only from forecasting more events: the NN expects 65.5 y_main
  events, the base 57.6, and 72 were observed (the +32 % shift).
- About 70 % of the level-matched GFT y_main gain is preserved.
- The gain is spread across years: 11 of 14 years are positive, and the minimum drop-one-year IG is +0.026.
- About 75 % of the gain comes from cells outside the top ETAS decile, so it is not only aftershock cells.
- **Train period (post-hoc):** on Chile 1976–2012, with NN OOF vs base logit OOF (same 5 time blocks and
  26-week purge), the gain is +0.13 for GFT on y_main and +0.13 for GZS. GZS never saw Chile, so it is
  genuinely out-of-sample there.

Interpretation: as **stand-alone forecasts**, the globally trained networks rank and calibrate Chile M≥6
cells somewhat better than the 23-parameter ETAS-logit fitted on Chile alone. The harness stacking cannot
pass this on, for two reasons:
1. The stacked logit is trained with an **in-sample** base and **out-of-fold** NN predictions on 1976–2012,
   so the NN looks redundant and gets a small coefficient.
2. The NN and base logits are highly correlated (r = 0.96).

This is a hint, not evidence:
- It was not the primary test.
- It is one of several secondary comparisons (3 models × 2 targets).
- Its only significance measure is the year bootstrap, which is anti-conservative here, as the stacked case
  above shows.
- It is a gain from **using catalog clustering and rates better** (pooled across margins), not from a
  precursor.

A prospective, preregistered stand-alone test (GZS vs ETAS-logit, 2026+) would be the way to confirm it.

## M≥7 ("learn from the improbable by pooling"): descriptive only, 5 positive test cells

| model | LL (M7 head) | AUC | expected cells (observed 5) |
|---|---|---|---|
| base M6 prob × 10^(−1) (GR, b = 1) | **−42.3** | **0.774** | 8.7 |
| GFT | −47.9 | 0.767 | 13.3 |
| CO | −46.5 | 0.759 | 13.5 |
| GZS | −47.6 | 0.771 | 12.7 |

Pooling 257 global M≥7 cells **did not** improve the M≥7 forecast for Chile:
- All networks over-forecast M≥7 (about 13 cells expected vs 5 observed) and lose about 5 nats to the
  simple Gutenberg-Richter scaling.
- On the global validation set too, the GRU M7 head is −0.028 nats/eq worse than a logit
  (`work/global_val_logit.json`).
- The M≥7 rate per M≥6 cell is evidently not transferable across margins without recalibration.

## Leakage checks

- **Global data:** stops at 2012-12-31 17:43 (`causality_check.json`), and 0 global events fall inside the
  Chile box.
- **Causality:** sequence features were recomputed from catalogs truncated at t_start for 300 Chile test
  rows and 300 global rows, with a maximum absolute difference of 0.0.
- **Fine-tuning and epoch selection:** only Chile rows with t_start + 7 d ≤ 2013-01-01 were used. Train-row
  predictions given to the harness are out-of-fold.
- **Level shift:** the feature mean (logit) is −6.17 in train vs −6.09 in test for GFT y_main.
  - The de-meaned feature gives a stacked IG of +0.005 (y_main).
  - Level-matching the stand-alone forecast keeps +0.043.
- **Clustering in disguise:** by construction the networks see only seismicity, so any gain is a better use of
  clustering and rate information. About 25 % of the stand-alone gain sits in the top-decile ETAS cells.
- **Catalog revision and densification:** this uses the revised ComCat for all margins. M≥4.7 completeness
  changes over 1973–2012 differ by margin, which may affect the pretraining distribution, but not the test
  causality.

## Files

- `results/point.json`: stacked and stand-alone metrics, contrasts, drop-year analysis and M7 numbers.
- `results/sur_gft_main.json` and `results/sur_gft_all.json`: primary surrogates, n = 1000.
- `results/sur_others.json`: surrogates for the other models.
- `results/robust.json`: robustness checks.
- Post-hoc checks: `results/extra.json`, `results/train_period_check.json`, `results/per_year_alone.json`.
- `work/train.log`, `work/pretrain_log.json` and `work/train_results.json`: training logs.

**Verdict: NULL** on the preregistered primary test (y_main stacked IG +0.002, p = 0.42), with a noteworthy,
non-confirmatory secondary signal: globally pretrained networks are better stand-alone forecasters than both
the Chile-only network and the ETAS-logit (≈ +0.04–0.06 nats/eq on y_main).
