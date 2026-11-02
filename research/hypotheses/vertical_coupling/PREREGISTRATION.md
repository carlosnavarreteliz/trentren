# PREREGISTRATION: vertical_coupling

Written 2026-10-17, before any `evaluate()` call on the test period, and before looking at any feature/target
relationship in 2013-2026.

## Hypothesis
Large interface (megathrust) events in Chile are preceded, over months, by changes in seismicity in the
*other* parts of the subduction system beneath/next to the same segment:

- H1 (Bouchon et al. 2016, Nat. Geosci.): intermediate-depth intraslab (IDS, 70-300 km) seismicity under a
  segment increases in the months before large interface events (slab pull / plunge).
- H2 (Christensen & Ruff 1988; Lay et al.): outer-rise seismicity seaward of the trench changes before large
  interface events (rate, and the tension -> compression switch late in the seismic cycle).
- H3 (depth-class migration): the share of IDS events relative to interface events in a segment departs from
  its own recent history before large events.

## Data
- USGS ComCat catalog already in the repo (`H.get_events(min_mag=4.7)`, all events, *not* the non-causal
  mainshock flag). Location and depth as in ComCat today (see leakage notes).
- Slab geometry: **Slab1.0** (Hayes et al. 2012) South America grid `sam_slab1.0_augmented.grd`
  (from github.com/usgs/slab2 TomoSlab_v1_2018/Slab_files). Slab1.0 was built from pre-2012 data, so the
  geometry does not use test-period seismicity. Slab depth for bin -46..-45 (outside the grid) is taken from
  the nearest grid row.
- Trench: `trenches_usgs_2017.csv` (usgs/slab2 misc/trenches), slab=='sam'. A static bathymetric feature.
- Secondary (H2 mechanism) only: GCMT focal mechanisms (1976-), if obtainable. Mechanisms are published within
  weeks; the feature uses a 30-day publication lag.

## Event classes (fixed now)
For each event with M >= 4.7 (harness `M_MODEL`), with z = ComCat depth (km), zs = Slab1.0 slab-top depth at
the epicentre (positive km), lon_tr(lat) = trench longitude interpolated at the event latitude:
- **OR (outer-rise)**: lon < lon_tr(lat) (seaward of the trench) and z <= 60.
- **IDS (intermediate-depth intraslab)**: landward of the trench and 70 <= z <= 300.
- **INT (interface)**: landward, z <= 60, zs defined and |z - zs| <= 20 km, or within 0.3 deg landward of the
  trench and z <= 40 (near-trench events where Slab1.0 is shallow/undefined).
- **CRU (crustal upper plate)**: landward, z <= 40 and z < zs - 20.
- other: everything else (not used).

## Features (per 1-degree latitude bin b, issue time t; only events with time < t; neighbourhood = bins b-1..b+1)
Long-term windows start at the catalog start (1973-01-01) when not enough history is available; if the
available long window is < 365 days the feature is NaN (imputed by the harness with the train mean).

Primary feature set F1 (3 columns, all jointly entered):
1. `x_ids_anom` = log((N_IDS[t-180d, t) + 0.5) / (N_IDS[t-180d-1825d, t-180d) * 180/1825 + 0.5)).
2. `x_or_anom`  = log((N_OR[t-365d, t) + 0.5) / (N_OR[t-365d-3650d, t-365d) * 365/3650 + 0.5)).
3. `x_vert_shift` = logit(f_s) - logit(f_l), f = (N_IDS + 0.5) / (N_IDS + N_INT + 1), short window 180 d,
   long window the preceding 1825 d.

Secondary (exploratory, reported but not the primary test):
- F1 computed from `bg` (causal ZBZ background) events only.
- F1 with short window 90 d.
- `x_or_comp` (GCMT): among outer-rise GCMT events in bins b-2..b+2 over the last 5 years (with 30-day lag),
  (n_thrust - n_normal) / (n_thrust + n_normal + 1). Mechanism class by plunge of P/T axes (Frohlich):
  thrust if T-axis plunge > 50 deg, normal if P-axis plunge > 50 deg.

## Primary test (single, fixed)
- Panel: weekly (step 7 d, horizon 7 d), target M >= 6.0, **target y_main** (declustered mainshocks), model
  logit, feature set F1.
- Statistic: `ig_per_event` (full vs ETAS-logit baseline) on test 2013-01-01..2026-10.
- Significance: `surrogate_test(F1, mode='phase_randomize', n=1000, target='y_main')`; Bonferroni alpha
  0.05/14 = 0.0036. Also bin_permute (n=500) and end-matched circular_shift (n=500) as secondary.
- y_all reported with the same machinery (co-primary for reporting, not for the verdict).

## Secondary / robustness (reported regardless)
- M >= 5.5 weekly (y_all, y_main); 30-day panel (s30_h30) M >= 6.0; GBM model; alternate split year
  (train <= 2004, test 2005-2026, done by re-labelling the panel `split` column; note the Poisson background
  columns still use 1976-2012 rates); year-drop jackknife on test preds; causally differenced features.
- Per-feature single-column runs (x_ids_anom alone = the pure Bouchon test).

## Verdict rule
positive: IG > 0 on y_main primary AND phase_randomize p < 0.0036 AND passes leakage checks.
weak_positive: p < 0.05 but not Bonferroni. Otherwise null/negative.

## Leakage notes (anticipated)
- ComCat hypocentres for old events have been revised over time (e.g. ISC-GEM relocations). Revision does not
  depend on later earthquakes in a way that encodes the target, but depth revisions after big sequences
  (denser local networks) could change the class mix. Checked by restricting to train-only diagnostics and
  by the bg-only variant.
- Network densification after 2010/2014 changes M4.7-5 completeness; the ratio features compare each window to
  its own preceding 5-10 years, which partly cancels slow completeness drift.
