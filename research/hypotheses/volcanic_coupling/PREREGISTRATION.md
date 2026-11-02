# PREREGISTRATION: volcanic_coupling

Written 2026-10-17, BEFORE any evaluation on the test period (2013-01-01 onward). The only things inspected
before writing this were (a) the GVP eruption list for the region and (b) yearly counts of shallow arc events
by magnitude threshold, both used to choose thresholds for completeness and data quality, never test targets.

## Hypothesis
Unrest in the Andean volcanic arc reflects stress changes in the overriding plate in the same along-strike
segment. Specifically, H1 says that shallow crustal seismicity near Holocene volcanoes and/or eruption
onsets in a latitude band raise the probability of a large (M>=6) earthquake in that band (mostly interface
events) over the following week, beyond what clustering (ETAS plus counts) already explains.

H0 says that, given the ETAS/clustering baseline, arc activity adds no information.

## Data
- Volcanoes come from the Smithsonian GVP VOTW Holocene Volcanoes layer (WFS download, 2026-10-17), lat -48..-15
  and lon -76..-62 (Chile, Argentina, Bolivia and southern Peru border).
- Eruptions come from the GVP VOTW Holocene Eruptions layer. Only `Activity_Type == "Confirmed Eruption"` is used, with
  a known start year, month and day and `StartDateDayUncertainty` <= 31 days (or missing).
- Arc seismicity comes from the harness catalog (`H.load_catalog`, USGS ComCat) with depth < 40 km, epicentre <= 50 km
  from the nearest Holocene volcano, and M >= 4.5.
  - M>=4.5 was chosen because M<4.5 arc counts are dominated by network changes. For example, M>=3 counts are
    50-190/yr in 1992-2008 and 1-30/yr after 2012.

## Features (per 1-degree latitude bin, strictly causal)
Each feature is aggregated over the bin plus its neighbours (bins b-1, b, b+1). Volcanoes and events just
outside -46..-17 still contribute to the edge bins.
1. `x_arc_365` = log1p(# arc events M>=4.5 with t_start-365 d <= time < t_start)
2. `x_arc_30`  = log1p(# arc events M>=4.5 with t_start-30 d  <= time < t_start)
3. `x_erupt_365` = # confirmed eruption onsets with "known-by" date in [t_start-365 d, t_start).
   - The known-by date is the onset date + day uncertainty + 1 day reporting lag. This is conservative and
     never earlier than the onset.

## Primary test (single, decided in advance)
- Panel: weekly (step 7 d, horizon 7 d), target M>=6.0, harness default.
- Target: **y_main** (ZBZ declustered mainshocks). That is where a non-clustering precursor must show.
- Model: logit (harness default). Features: all three jointly.
- Metric: `ig_per_event` (full minus ETAS-logit baseline) on test 2013-01-01..2026-10.
- Significance: `H.surrogate_test(mode="phase_randomize", n=1000)` on the 3-column per-bin feature grid,
  one-sided.
- alpha = 0.05 uncorrected, and Bonferroni alpha = 0.05/14 = 0.0036.
- Verdict rule:
  - "positive" requires IG > 0, p < 0.0036, and survival of the leakage checks.
  - "weak_positive" requires p < 0.05.
  - Otherwise the verdict is null (or negative if IG < 0 clearly).

## Secondary tests (reported, not used for the verdict unless the primary passes)
- The same analysis on y_all.
- bin_permute (n=500) and a raw-time circular shift of the arc-event and eruption times (builder, n=200).
- Each feature alone.
- GBM model.
- M>=5.5 weekly panel.
- 30-day panel (s30_h30).
- Alternate split: train 1976-2004 and pseudo-test 2005-2012. This lies inside the original training era, so it
  never touches 2013+.
- Year-dropping (leave-one-test-year-out) on preds, and the year block bootstrap.
- Reverse direction, context only: the rate of eruption onsets within ±2 bins in the 365 d after M>=7 events,
  compared with the long-run rate.

## Leakage notes (declared in advance)
- GVP is a curated database revised after the fact. Eruption onset dates are observational and do not depend on
  later earthquakes, but minor eruptions may have been added retrospectively. That would affect
  real-time availability, not the dates themselves.
- Arc catalog completeness changes with network densification (e.g. after 2010 and 2014). The M>=4.5 cut limits
  this, and the surrogates preserve the feature's own time structure.
- The `mainshock` flag is not used in features.
