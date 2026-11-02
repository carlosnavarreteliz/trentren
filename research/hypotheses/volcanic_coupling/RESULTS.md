# RESULTS: volcanic_coupling (Andean arc unrest and eruptions → large Chilean earthquakes)

**Verdict: NULL.** Volcanic-arc activity (shallow crustal seismicity near Holocene volcanoes and GVP eruption
onsets in the same along-strike band) adds no forecasting information beyond the ETAS/clustering baseline.
This holds for mainshocks and for all events, at M≥6.0 and M≥5.5, on weekly and 30-day panels, and in the
alternate 2005–2012 split. The preregistered primary test gives IG = −0.0009 nats/eq (phase-randomize
surrogate p = 0.32, n = 1000).

The preregistration is in `PREREGISTRATION.md` and was written before any test-period evaluation.

## Data
- **Smithsonian GVP VOTW** Holocene Volcanoes and Holocene Eruptions, WFS download 2026-10-17, cached in
  `research/data/volcanic_coupling/`.
  - 110 volcanoes, lat −48..−15 and lon −76..−62.
  - 113 confirmed eruptions since 1960 with a day-precision onset (day uncertainty ≤ 31 d): 71 in 1976–2012 and 24 in 2013+.
- **Arc seismicity** from the harness USGS catalog: depth < 40 km, ≤ 50 km from a Holocene volcano, M ≥ 4.5.
  - 224 events, 37 of them in 2013+.
  - M<4.5 was excluded a priori: arc M≥3 counts swing from 50–190/yr (1992–2008, local contributions) to 1–30/yr after 2012.

## Features
All features are per bin, aggregated over bins b−1..b+1, and strictly causal. Code is in `volc_features.py`.
- `x_arc_365` and `x_arc_30` are log1p counts of arc events in the past 365 and 30 days.
- `x_erupt_365` is the number of eruption onsets in the past 365 days. Each onset is dated at its
  "known-by" time = onset + day uncertainty + 1 d.
- Spot check: the Chaitén 2008-05-02 onset and its M4.5+ pre-eruptive swarm first appear at the 2008-05-08 issue
  time, with nothing earlier.

## Primary result (preregistered: weekly M≥6.0, y_main, logit, 3 features jointly, phase_randomize n=1000)

| target | N test cells | IG full−base (nats/eq) | IG full−ETAS | year-bootstrap 95 % CI (vs base) | p phase_randomize | p bin_permute | p raw-time circular shift | AUC base→full | Molchan ASS base→full |
|---|---|---|---|---|---|---|---|---|---|
| **y_main** (primary) | 72 | **−0.0009** | −0.0089 | [−0.035, +0.026] | **0.32** (n=1000) | 0.45 (n=300) | 0.42 (n=150) | 0.665→0.665 | 0.664→0.664 |
| y_all | 85 | +0.0019 | +0.0014 | [−0.015, +0.015] | 0.25 (n=300) | 0.25 (n=200) | 0.25 (n=100) | 0.697→0.700 | 0.696→0.699 |

- Standardized logit coefficients on y_main: arc_365 −0.11, arc_30 +0.08, erupt_365 +0.16. They are small and
  their signs are inconsistent.
- Leave-one-test-year-out IG on y_main ranges from −0.010 to +0.006, so no single year carries an effect, because
  there is none.
- Nothing is significant at α = 0.05, let alone at the Bonferroni α = 0.0036.

## Robustness
All rows use logit unless marked, with phase_randomize surrogates (n=200 for logit, 100 for GBM).

| panel | target | features | IG full−base | IG full−ETAS | p | AUC base→full | N |
|---|---|---|---|---|---|---|---|
| weekly M6 | y_main | arc_365 only | +0.0064 | −0.002 | 0.12 | 0.665→0.666 | 72 |
| weekly M6 | y_main | arc_30 only | −0.0037 | −0.012 | 0.66 | 0.665→0.663 | 72 |
| weekly M6 | y_main | erupt_365 only | −0.0091 | −0.017 | 0.78 | 0.665→0.662 | 72 |
| weekly M6 | y_all | arc_365 only | −0.0010 | −0.002 | 0.55 | 0.697→0.696 | 85 |
| weekly M6 | y_all | arc_30 only | +0.0038 | +0.003 | 0.15 | 0.697→0.698 | 85 |
| weekly M6 | y_all | erupt_365 only | −0.0012 | −0.002 | 0.63 | 0.697→0.699 | 85 |
| weekly M6, GBM | y_main | all 3 | +0.0147 | −0.066 | 0.33 | 0.644→0.651 | 72 |
| weekly M6, GBM | y_all | all 3 | +0.0108 | −0.058 | 0.27 | 0.695→0.695 | 85 |
| weekly M5.5 | y_main | all 3 | −0.0014 | −0.026 | 0.44 | 0.658→0.657 | 193 |
| weekly M5.5 | y_all | all 3 | +0.0025 | +0.004 | 0.21 | 0.697→0.696 | 228 |
| 30-day M6 | y_main | all 3 | +0.0033 | −0.007 | 0.26 | 0.664→0.665 | 72 |
| 30-day M6 | y_all | all 3 | +0.0093 | −0.004 | 0.12 | 0.671→0.671 | 80 |
| alt split (train 1976–2004, test 2005–2012) | y_main | all 3 | −0.0072 | −0.138 | 0.47 | 0.633→0.632 | 34 |
| alt split | y_all | all 3 | −0.0057 | −0.078 | 0.47 | 0.737→0.738 | 58 |

None of the 16 configurations reaches p < 0.05. The smallest p is 0.12.

## Reverse direction (context only, not a forecast test)
This counts eruption onsets within ±2 bins in the 365 days after an M≥7 event, using only the first M≥7 of each
sequence. Later events within 365 days and ±2 bins are dropped, which leaves 20 sequences.
- **8 eruptions were observed against 4.6 expected** from the 1976–2026 rate (Poisson p = 0.09).
- The counts are driven by Maule 2010 (2 × Planchón-Peteroa), the 2012 M7.1 event (2 × Copahue), and Lascar after
  the northern M7 events of 1987, 1993 and 2000.
- This is consistent with the published weak post-megathrust triggering of southern Andean volcanoes, but it is not
  significant here. It does not translate into the forward direction.
- Without de-duplication the count is 18 vs 9.0 (p = 0.005). That figure is inflated because overlapping sequences
  count the same eruptions several times.

## Leakage checks and caveats
- **Causality.** Eruption features use onset + uncertainty + 1 d, arc counts use only events with time < t_start, and
  no `mainshock` flag is used. The Chaitén spot check confirms there is no early appearance.
- **GVP is retrospectively curated.** Small eruptions might have been added after the fact, so real-time
  availability is slightly overstated. This could only inflate a positive result, and the result is null anyway.
- **Network changes.** Arc completeness changes over time, which is why M≥4.5 was used. The test period has only
  37 arc events and 24 eruptions, so a small real effect (< ~0.03 nats/eq) cannot be excluded: the y_main CI is
  [−0.035, +0.026].
- **Spatial model.** It is latitude-only. Arc events and eruptions are attributed to whole 1° bands (±1 bin),
  ignoring the 150–300 km across-strike distance to the interface. A finer, stress-transfer-based geometry was not
  tested.
- **Harness/runtime.** No harness bug was found. The machine was heavily oversubscribed (load average about 39 on 8 CPUs),
  so the primary run was executed with `OMP_NUM_THREADS=1`. Results do not depend on this.

## Reproduce
```
cd research/hypotheses/volcanic_coupling
OMP_NUM_THREADS=1 /opt/miniconda3/bin/python3 -I run_primary.py ../../data/volcanic_coupling primary.json
OMP_NUM_THREADS=1 /opt/miniconda3/bin/python3 -I run_robust.py  ../../data/volcanic_coupling robust.json 200
```
Outputs are `primary.json`, `primary.log`, `robust.json` and `robust.log`.
