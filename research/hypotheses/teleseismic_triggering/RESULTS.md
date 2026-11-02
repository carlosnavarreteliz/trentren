# RESULTS: teleseismic_triggering (remote dynamic triggering and global great-earthquake synchrony)

## Verdict: NULL

**The preregistered hypothesis is not supported.** It said that strong remote shaking (H1), or high global or Pacific-rim activity (H2), *raises* the near-term probability of M>=6 earthquakes in Chile beyond what ETAS predicts.

- **H1 nominal result.** On the primary weekly M6 panel, the remote-strain features give a small, nominally significant information gain: +0.016 nats/eq on y_all (p=0.026) and +0.018 nats/eq on y_main (p=0.020). Both p-values come from n=1000 event-time-shift surrogates, and neither is close to the Bonferroni threshold of 0.0036.
- **H1 has the wrong sign.** Both coefficients are negative. Weeks *after* strong remote shaking have slightly *lower* Chilean M6 probability, by about 3% on positive cells. This is the opposite of dynamic triggering.
- **H1 does not replicate.**
  - The M>=5.5 panel, with 2.5 times more events, gives p=0.56 and 0.87.
  - The alternate 2005-2012 split gives p=0.17 and 0.60.
  - The 30-day panel is negative.
  - The GBM model is not significant.
- **No triggering in a model-free test.** A model-free superposed-epoch test finds no excess of Chilean events in the 1 to 5 days after strong remote waves. It covers 32 configurations and every p-value is at least 0.20.
- **H2 (global moment rate) on y_main** shows p=0.015 for the event-time shift and p=0.005 for phase randomization. About half of that gain is a level correction for the higher mainshock rate in the test period. The causally differenced version is null (p=0.59), and the alternate split is negative.

The verdict guide would label the raw primary p<0.05 as "weak_positive". I report NULL because the effect runs against the preregistered direction, is about 1/10 of the clustering gain, and fails every independent replication I ran. With 4 primary tests here and 14 hypotheses in the family, a p of about 0.02 is what chance produces.

## Data
- USGS ComCat global catalog, M>=6.5, 1973-01-18 to 2026-10-16: 2,317 events. The raw files are `research/data/teleseismic_triggering/global_m65_{1973,2000}.csv`, downloaded 2026-10-17.
- **Remote** means more than 1000 km from every Chilean bin's reference point (2,156 events). The reference point is the bin's latitude centre and the median longitude of pre-2013 events.
- **Strain proxy.** For each event and bin, log10 eps = M - 1.66 log10(Delta_deg) - 4. This is the surface-wave amplitude scaling (van der Elst & Brodsky 2010). Only events at depth <= 100 km are used.
- **Features** (computed in `tele_features.py`):
  - H1, per bin: `x_strain_max_7d`, `x_strain_sum_30d`.
  - H2, global: `x_gmoment_365d`, the 365-day remote moment sum as an anomaly from its trailing 10-year mean; and `x_pacific_m7_365d`.
  - Every feature uses only events with time < t_start - 1 h.

## Primary results (weekly panel, M>=6.0, logit, test 2013-01-01 to 2026-10)

| hyp | target | IG full-base, nats/eq [year-block 95% CI] | IG full vs ETAS | p event-time shift (n=1000) | p phase-rand (n=200) | p bin-permute (n=200) | AUC base/full | Molchan ASS base/full | N test cells | coefficients (std.) |
|---|---|---|---|---|---|---|---|---|---|---|
| H1 | y_all | **+0.0159** [-0.017, +0.043] | +0.015 | **0.026** | 0.040 | 0.109 | 0.697/0.706 | 0.696/0.706 | 85 | max7d -0.051, sum30d -0.090 |
| H1 | y_main | **+0.0179** [+0.003, +0.031] | +0.010 | **0.020** | 0.035 | 0.070 | 0.665/0.676 | 0.664/0.675 | 72 | max7d -0.082, sum30d -0.009 |
| H2 | y_all | +0.0016 [-0.038, +0.033] | +0.001 | 0.171 | 0.274 | - | 0.697/0.702 | 0.696/0.702 | 85 | gmoment -0.129, pac7 +0.007 |
| H2 | y_main | +0.0244 [-0.001, +0.044] | +0.016 | 0.015 | 0.005 | - | 0.665/0.672 | 0.664/0.671 | 72 | gmoment -0.068, pac7 -0.003 |

None of these results reaches Bonferroni significance (p < 0.0036). The baseline IG over Poisson is +0.221 (y_all) and +0.006 (y_main).

## Robustness (all preregistered)

Each row below uses event-time-shift surrogates.

| check | target | IG | p | comment |
|---|---|---|---|---|
| H1, M>=5.5 weekly | y_all | -0.0018 | 0.56 | null with 228 test events |
| H1, M>=5.5 weekly | y_main | -0.0054 | 0.87 | null with 193 test events |
| H1, daily (s1/h1) M6 | y_all | +0.0082 | 0.10 (n=100) | |
| H1, daily (s1/h1) M6 | y_main | +0.0174 | 0.03 (n=100) | same negative sign |
| H1, 30-day (s30/h30) | y_all | -0.0325 | 0.89 | |
| H1, 30-day (s30/h30) | y_main | -0.0141 | 0.75 | |
| H1, GBM weekly | y_all | +0.0050 | 0.26 (n=100) | |
| H1, GBM weekly | y_main | +0.0184 | 0.32 (n=100) | |
| H1, 2000 km exclusion | y_all | +0.0186 | 0.030 | nearly the same feature |
| H1, 2000 km exclusion | y_main | +0.0126 | 0.045 | nearly the same feature |
| H2, causally differenced (365 d) | y_all | -0.0046 | 0.48 | |
| H2, causally differenced (365 d) | y_main | -0.0062 | 0.59 | |
| Alternate split, train 1976-2004 / test 2005-2012, H1 | y_all | +0.0075 | 0.17 | 58 events |
| Alternate split, train 1976-2004 / test 2005-2012, H1 | y_main | -0.0111 | 0.60 | 34 events |
| Alternate split, train 1976-2004 / test 2005-2012, H2 | y_all | -0.0310 | 0.76 | |
| Alternate split, train 1976-2004 / test 2005-2012, H2 | y_main | -0.0212 | 0.69 | year CI entirely below 0 |

**Leave one test year out.** H1 IG stays positive whichever year is dropped: from +0.008 to +0.023 on y_all and from +0.015 to +0.021 on y_main. The small effect is diffuse and does not come from a single sequence.

## Diagnostics (exploratory, after the primary test)
1. **Sign.** The H1 gain comes from slightly *raising* the probability in globally quiet weeks: the median p_full/p_base is 1.03 to 1.07 on positive cells. The indicator coefficient of -0.10 per standard deviation (indicator sd about 0.5) implies odds about 0.82 times lower in weeks with a remote shallow M>=6.5 event than in weeks without one.
   - A single indicator, "any remote shallow M>=6.5 in the last 7 days", reproduces the y_main gain: IG +0.017 with coefficient -0.10.
   - In the training years the observed/expected ratio also falls with remote strain (1.31 with no remote event versus 0.72 in the top strain class, y_all). This is weak and non-monotonic.
   - No known physical mechanism predicts this. Remote shaking is physically expected to *raise* rates, if it has any effect at all.
2. **Global, not local.** Bin permutation is not significant (p of 0.11 and 0.07), so the bin-specific shaking geometry adds nothing. A global broadcast of the strain feature gives a smaller IG (+0.005 and -0.000).
3. **Not a level shift for H1.** The global remote-shallow M>=6.5 rate is 33.7/yr in 1976-2012 and 32.1/yr in 2013-2026. After rescaling both base and full models to the observed test total (an oracle, used only as a diagnostic), the H1 IG remains +0.016 on both targets.
4. **H2 y_main is about half level correction.**
   - x_gmoment_365d has a lower mean in the test period (-0.14) than in training (+0.06). After the great earthquakes of 2004-2011, the trailing-mean anomaly turned negative.
   - Its negative coefficient therefore raises test-period probabilities, which happens to match the 32% higher y_main rate in the test period.
   - An oracle rescaling of the base model alone gains +0.023. After level neutralization the H2 IG falls from +0.024 to +0.013.
   - The differenced feature is null, so this is the slow-series trap that HARNESS.md warns about.
5. **Superposed-epoch test (`superposed_epoch.py`, `results/superposed_epoch.json`).** For each remote shallow event with peak strain proxy >= 10^0.5 (118 events) or >= 10^1 (33 events), I compared Chilean event counts in the window (t, t+W] with the window (t-W, t].
   - Event sets: M>=4.7 all, M>=4.7 causal background, M>=5.5, and M>=4.0 since 1995.
   - Windows: W = 1 or 5 days. Periods: 1976-2026 and 2013-2026.
   - The null comes from 2,000 circular shifts of the trigger times.
   - All 32 p-values are at least 0.20. Example: for the 33 strongest triggers, M>=4.7, 1 day, there were 16 events after versus 12 before, null sd 12.4, p=0.25.
   - There is **no detectable immediate dynamic triggering** in the Chilean catalog at these magnitudes.

## Leakage checks
- **Causal cutoffs.**
  - Features use only events with origin time < t_start - 1 h. USGS publishes M>=6.5 events within about 20 to 60 minutes.
  - Bin reference longitudes use pre-2013 events only.
  - The H2 trailing mean is computed with a left-closed rolling window.
- **Magnitudes.** The catalog uses final (revised) magnitudes rather than real-time ones, a difference of about 0.1 to 0.2. This cannot create a timing signal, and the event-time-shift null keeps the same magnitudes.
- **No target leakage.** The Chilean events, which define the targets, are excluded from the remote set by the 1000 km rule. A 2000 km exclusion gives the same picture.
- **Surrogate choice.** The primary surrogate shifts the *raw global event times* and recomputes features. This preserves the global catalog's own clustering (aftershocks of remote mainshocks) and its geometry. Phase randomization and bin permutation agree in direction.

## Bottom line for the forecasting question
Remote large earthquakes, whether through shaking or through global "synchrony", do **not** provide usable predictive information for M>=6 earthquakes in Chile beyond ETAS-type clustering.
- The only nominal signal is tiny (0.016 nats/eq, about 1/14 of the clustering gain on y_all) and has the opposite sign to the physical hypothesis.
- It does not replicate at M>=5.5, in an earlier test window, or with a nonlinear model.
- This agrees with Parsons & Velasco (2011), who found no remote short-term triggering of M>=5, and with Michael (2011) and Shearer & Stark (2012), who found that the global rate of great earthquakes is Poisson after declustering.

## Reproduce

All runs use `/opt/miniconda3/bin/python3 -I` from this folder, with `D=research/data/teleseismic_triggering` and `C="$D/global_m65_1973.csv $D/global_m65_2000.csv"`.

```
python -I run_primary.py $C h1 y_all 1000 200
python -I run_primary.py $C h1 y_main 1000 200
python -I run_primary.py $C h2 y_all 1000 200
python -I run_primary.py $C h2 y_main 1000 200
./robust_chain.sh                       # M5.5, 2000 km, 30 d, GBM, h2diff, daily, altsplit
python -I superposed_epoch.py $C 2000
python -I diagnose.py $C
python -I level_check.py $C
python -I level_neutral.py $C
python -I h2_check.py $C
python -I summarize.py                  # table -> results/summary_table.md
```

## Caveats
- **Harness bugs.** None found. The harness passes `shift_days = k*step` to the builder in circular_shift mode, and I used it as a shift of the raw event times modulo the 1973-2026 span.
- **Strain proxy.** The proxy is a crude empirical scaling. It ignores radiation pattern, directivity, Love versus Rayleigh waves and the site, but it is monotone in the right physical quantities.
- **Not tested.** Triggering of small events (M<4) by remote waves was not tested. That effect is documented in some geothermal and volcanic areas, but the target here is M>=5.5 or 6.
- **Load and run time.** Machine load was about 25 to 45 throughout, so the robustness runs used n=100 to 200 surrogates.
