# RESULTS: vertical_coupling (intraslab / outer-rise seismicity as precursors of interface events)

**Verdict: NULL.**

The preregistered primary test reached a nominal p = 0.044. That would technically fall under the
`weak_positive` rule (p < 0.05 uncorrected), but I classify the result as null for these reasons:
1. The hypothesised mechanism, the Bouchon intermediate-depth increase, is null on its own, and its fitted
   coefficient has the wrong sign.
2. The small gain comes entirely from the outer-rise term, also with the sign opposite to the hypothesis.
3. About half of that gain is a level shift. It compensates for the known +32 % rise in the y_main rate in
   the test period and is not discrimination.
4. The gain disappears under every robustness check: causal differencing, background-only events, M ≥ 5.5,
   the 30-day panel, GBM and the 2005 split.
5. p = 0.044 is far above the Bonferroni threshold of 0.0036. Among about 14 parallel hypotheses, a result
   like this is expected by chance.

## What was tested (see PREREGISTRATION.md, written before any test-period evaluation)

**Event classes.** Every M ≥ 4.7 USGS event was assigned a class using these inputs:
- Slab geometry from **Slab1.0** (Hayes et al. 2012). It was built from pre-2012 data, so it is causal for
  the 2013 split.
- The trench trace from USGS 2017.

The classes are:
- **OR** (outer rise): seaward of the trench, z ≤ 60 km. 284 events.
- **IDS** (intermediate-depth intraslab): 70–300 km. 3,185 events.
- **INT** (interface): within ±20 km of the slab top, z ≤ 60 km. 3,496 events.
- **CRU** (upper-plate crustal): 510 events.

As a sanity check, GCMT mechanisms by class came out as expected. IDS events are mostly normal faulting
(709 normal vs 23 thrust), INT events mostly thrust (1,013 vs 68), and OR events mostly normal (76 vs 7).

**Features.** Per 1° bin, using bins ±1, and strictly from events before t:
- `x_ids_anom`: log ratio of the IDS count in the last 180 days to the count in the preceding 5 years.
- `x_or_anom`: the same ratio for OR events, with a 365-day short window and a 10-year long window.
- `x_vert_shift`: change in the log-odds share of IDS among IDS + INT events, 180 days vs the preceding
  5 years.

**Secondary feature.** `x_or_comp`, a GCMT outer-rise compression index:
- (thrust − normal)/(n+1) over 5 years in ±2 bins, with a 30-day lag.

**Primary test.** Weekly panel, M ≥ 6.0, **y_main**, logit, all three features together.
- Significance from phase-randomized surrogates, n = 1000.

## Primary results (test period 2013-01-01 to 2026-10; IG in nats per target event)

| target | IG full vs ETAS-logit base | IG full vs pure ETAS | year-bootstrap 95 % CI | phase_randomize p (n) | bin_permute p (n=500) | circ-shift end-matched p (n=500) | AUC base → full | Molchan ASS base → full | N test cells |
|---|---|---|---|---|---|---|---|---|---|
| **y_main (primary)** | **+0.021** | +0.013 | [−0.011, +0.050] | **0.044** (1000) | 0.034 | 0.078 | 0.665 → 0.673 | 0.664 → 0.672 | 72 |
| y_all | +0.005 | +0.005 | [−0.021, +0.030] | 0.233 (1000) | 0.096 | 0.250 | 0.697 → 0.704 | 0.696 → 0.703 | 85 |

Fitted standardized logit coefficients for y_main:
- x_ids_anom −0.030
- x_or_anom −0.122
- x_vert_shift −0.018

**All three are negative.** The Bouchon hypothesis predicts a positive coefficient for x_ids_anom.

The Rhoades T statistic was 1.27, with p = 0.10. That test is anti-conservative, so it is reported only.

Calibration note: in `calibration/CALIBRATION_V2.md` (still incomplete), phase_randomize on y_main
produced 0 false positives in 52 noise features at p < 0.05. The primary p is therefore probably not
inflated by mis-calibration. It is still only nominal.

## Secondary and robustness results (phase_randomize n = 200 unless noted)

| variant | y_main IG | y_main p | y_all IG | y_all p |
|---|---|---|---|---|
| x_ids_anom alone (pure Bouchon test) | −0.0004 | 0.42 | −0.001 | 0.52 |
| x_or_anom alone | +0.020 (coef −0.12) | 0.025 | +0.003 | 0.23 |
| x_vert_shift alone | +0.001 | 0.29 | +0.004 | 0.20 |
| F1 from `bg` (causal background) events only | +0.006 | 0.15 | +0.012 | 0.075 |
| F1 with 90-day short window | +0.025 | 0.040 | +0.009 | 0.16 |
| F1 causally differenced (x(t) − x(t−26 wk)) | −0.014 | 0.65 | −0.014 | 0.74 |
| GCMT outer-rise compression index | −0.035 | 1.00 | −0.006 | 0.85 |
| F1, GBM model (IG only) | +0.001 | n/a | +0.005 | n/a |
| F1, weekly M ≥ 5.5 (193 / 228 test cells) | +0.003 | 0.17 | −0.004 | 0.73 |
| F1, 30-day panel M ≥ 6.0 | +0.011 | 0.12 | −0.002 | 0.46 |
| F1, alternate split (train ≤ 2004, test 2005–2026) | +0.006 | 0.15 | −0.108 | 1.00 |

**Year-drop jackknife (primary, y_main).** IG stays between +0.011 and +0.028 whichever test year is
dropped, so no single sequence carries it. The effect is small and diffuse.

**Level vs discrimination (post-hoc, oracle diagnostic: `diag_level.py`).** I re-centred both models'
test-period intercepts so that the expected count equals the observed count, which uses test outcomes and
is for diagnosis only. After that, the primary y_main IG falls from +0.021 to +0.012, and x_or_anom alone
falls from +0.020 to +0.010. The reason:
- x_or_anom has a mean of −0.12 in train and −0.36 in test. Its 10-year reference window contains the
  2010–2011 Maule outer-rise aftershock surge, so the ratio stays depressed throughout the test period.
- With a negative coefficient, the depressed value raises the predicted rate. That happens to match the
  +32 % higher test-period mainshock rate: 57.6 expected under the base model, 60.8 under the full model,
  72 observed.

**Post-M7 check (post-hoc: `diag_outer_rise.py`).** The x_or_anom gain does not come from cells within
2 years after an M ≥ 7 in bins ±2. It is spread over ordinary cells. In the top quintile of x_or_anom
outside post-M7 periods, 4 events were observed against 6.85 expected under the base model.

**Event-level check (`event_level.py`, descriptive).** I looked at the 44 shallow (≤ 60 km) declustered
M ≥ 6.5 mainshocks from 1976–2026. For each one I took its bin's percentile of x_ids_anom in the week
before the event:
- The median percentile is **40**, where 50 is the no-effect value.
- 13.6 % of events were above the 80th percentile, against 20 % expected by chance.
- Iquique 2014 (M8.2) sat at the 89th percentile, consistent with Bouchon et al. for that one event.
- Maule 2010 (M8.8) sat at the 8th percentile, and Illapel 2015 (M8.3) at the 47th.

In the ComCat M ≥ 4.7 catalog there is no systematic increase in intermediate-depth activity before large
Chilean interface earthquakes.

## Leakage checks
- **Slab geometry.** Slab1.0 was published in 2012, so no test-period seismicity went into the class
  boundaries. The trench is bathymetric and static.
- **Feature windows.** All windows are [t − w, t) and use only events with time < t_start. There are no
  centered windows, and no normalization uses the whole period: the ratio features compare each window to
  its own past. The non-causal `mainshock` flag is never used in features; only the causal `bg` flag is
  used, in a variant.
- **GCMT.** The feature uses a 30-day publication lag. Mechanisms are from the final GCMT catalog, which
  is quick CMT revised.
- **Remaining issue 1: relocated hypocentres.** ComCat locations and depths are today's versions,
  including ISC-GEM and other relocations. A depth revision could move an event between classes. This is
  not obviously target-dependent, and the result is null anyway.
- **Remaining issue 2: catalog density.** Network densification after 2010 and 2014 raises INT counts at
  M4.7–5. The ratio features partly cancel this. The level-shift diagnosis above shows how slow
  catalog-density changes can masquerade as skill.

## Caveats
- The test period is small: 72 y_main cells and 85 y_all cells. An effect of about 0.02 nats per event
  cannot be confirmed or excluded at this sample size. The year-bootstrap CI is [−0.011, +0.050].
- **Magnitude threshold.** The data use M ≥ 4.7 only. Bouchon et al. used lower-magnitude local
  catalogs (M ≥ 3–4), so a weak precursor visible only in small events is not excluded. M < 4.7 is
  incomplete before about 1990 in this catalog.
- **Classification.** It is by epicentre and catalog depth only. Many older events have fixed depths of
  10, 33 or 35 km, which blurs the INT/CRU/OR distinction (IDS is robust).
- **Spatial unit.** 1° latitude bins with ±1 neighbours (about 330 km along strike). Bouchon's signal is
  defined beneath the rupture, which this approximates only roughly.
- **Grid edge.** The Slab1.0 grid starts at 45°S. Bin −46…−45 uses the nearest grid row.
- **Harness.** No harness bug was found.

## Files
All under `research/hypotheses/vertical_coupling/`.

Scripts:
- `vc_features.py`: classification and features
- `run_primary.py`, `run_secondary.py`
- `event_level.py`, `diag_outer_rise.py`, `diag_level.py`, `diag_classes.py`

Outputs in `out/`:
- `primary.json`, `secondary.json`, `diag_*.json`
- `event_level_m65.csv`
- `F1_weekly.parquet`, `preds_primary_*.parquet`
- logs

Data are in `research/data/vertical_coupling/`:
- `sam_slab1.0_augmented.grd`
- `trenches_usgs_2017.csv`
- GCMT ndk files
- `events_classified.parquet`

## Reproduce
```
cd /Users/cnavarreteliz/Documents/Repos/earthquakes/research/hypotheses/vertical_coupling
/opt/miniconda3/bin/python3 -I run_primary.py $PWD/out 1000 500      # ~45 min under load
/opt/miniconda3/bin/python3 -I run_secondary.py $PWD/out 200         # ~40 min under load
/opt/miniconda3/bin/python3 -I event_level.py $PWD/out
/opt/miniconda3/bin/python3 -I diag_level.py $PWD/out; /opt/miniconda3/bin/python3 -I diag_outer_rise.py $PWD/out
```
