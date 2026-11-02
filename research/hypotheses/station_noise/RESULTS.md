# RESULTS: station_noise (MUSTANG ambient-noise anomalies as a crustal-state sensor)

**Verdict: NULL.** Daily seismic-noise anomalies at Chilean broadband stations do not improve
M>=6 (or M>=5.5) forecasts beyond the clustering/ETAS baseline. In every configuration the added
features lower test log-likelihood slightly, or leave it unchanged, and none comes close to
significance (primary surrogate p = 0.42, n = 1000).

The preregistration is in `PREREGISTRATION.md`. It was written before any target evaluation, with one
amendment, also made before evaluation: the robust scale is IQR/1.349 because MUSTANG modes are integer dB.

## Data obtained
- **Source.** IRIS/EarthScope MUSTANG `noise-mode-timeseries`: the daily mode of the hourly PSD PDF at
  8 to 11 fixed frequencies on the BHZ channel. It was fetched for every BHZ station in lat -46..-16.5 and
  lon -78..-64 that MUSTANG covers: 94 NSLC targets with data, 2.6 M rows.
- **Raw files.** Saved in `research/data/station_noise/raw/`, with the long table in `noise_mode_long.parquet`
  and the causal z-scores in `site_z.parquet`.
- **Pre-2013 coverage (sparse).**
  - IU.LVC: -22.6, 1997-2018.
  - IU.LCO: -29.0, 1999-.
  - G.PEL: -33.1, 1996-2008 and 2011-.
  - GT.PLCA: -40.7, 1995-.
  - G.COYC: -45.6, 2011-.
  - C.GO01-07: 2012-.

  Only 14.5% of train panel cells have a feature value; the remaining cells get the train-mean imputation.
- **Test coverage.** 79% of test cells (primary set), or 97% when the C1 network is added (dense set).
- **CX/IPOC is not available.** The IPOC network (CX, PB01-PB16), which is the densest long record in the
  north, is archived at GFZ, not IRIS, so MUSTANG does not cover it.
- **C network gaps.** CLCH, HUNY, TLL and others have no MUSTANG metrics.

## Features (strictly causal)
- **Daily bands.** HF is the mean of the 1.04 and 3.2 Hz modes, LP the mean of the 0.0053 and 0.0096 Hz
  modes (100-190 s, tilt/pressure), and MS the mean of the 0.1 and 0.31 Hz modes (ocean microseism,
  used as a negative control).
- **Anomaly z.** z = (P(D) - median over D-364..D-30) / max(IQR/1.349, 1 dB). The baseline never crosses a
  channel-epoch or location-code change, and z is clipped to ±6.
- **Panel feature.** Each site's mean z over the 14 days [t-15, t-1) is averaged over the sites within 1.5°
  of the bin centre, giving x_hf, x_lp and x_ms.

## Primary test (preregistered): weekly M>=6.0, y_main, logit, x_hf + x_lp, phase_randomize n=1000

| target | IG full vs base (nats/eq) | IG full vs ETAS | surrogate p (phase, n) | bin_permute p | circ-shift p | AUC base / full | year-bootstrap 95% CI | n test cells |
|---|---|---|---|---|---|---|---|---|
| **y_main (primary)** | **-0.013** | -0.021 | **0.42** (1000) | 0.41 | 0.77 | 0.665 / 0.658 | [-0.031, +0.005] | 72 |
| y_all | -0.020 | -0.020 | 0.51 (1000) | 0.58 | 0.95 | 0.697 / 0.695 | [-0.039, -0.001] | 85 |

The GBM model gives IG -0.024 (y_main) and -0.013 (y_all). The Bonferroni threshold is 0.0036, and no
test comes near it.

## Secondary and robustness (logit, phase_randomize n=200 unless stated)

| configuration | target | IG vs base | IG vs ETAS | surrogate p | AUC base / full | n test cells |
|---|---|---|---|---|---|---|
| x_hf alone | y_main / y_all | -0.010 / -0.021 | -0.018 / -0.022 | 0.64 / 0.78 | 0.665/0.660 · 0.697/0.695 | 72 / 85 |
| x_lp alone | y_main / y_all | +0.0002 / -0.0002 | -0.008 / -0.001 | 0.27 / 0.30 | unchanged | 72 / 85 |
| x_ms (negative control) | y_main / y_all | -0.019 / -0.006 | -0.027 / -0.006 | 0.73 / 0.61 | 0.665/0.654 · 0.697/0.695 | 72 / 85 |
| M>=5.5 weekly | y_main / y_all | -0.028 / -0.009 | -0.053 / -0.008 | 0.82 / 0.62 | 0.658/0.650 · 0.697/0.694 | 193 / 228 |
| dense set (+C1, test only) | y_main / y_all | -0.008 / -0.019 | -0.016 / -0.020 | 0.38 / 0.56 | | 72 / 85 |
| 30-day panel (30 d window) | y_main / y_all | +0.004 / +0.004 | -0.006 / -0.009 | 0.24 / 0.11 | 0.664/0.666 · 0.671/0.676 | 72 / 80 |
| split 2017 (train <=2016) | y_main / y_all | -0.010 / -0.001 | -0.031 / -0.001 | 0.39 / 0.22 | | 49 / 51 |
| issue times from 1996 only | y_main / y_all | -0.018 / -0.039 | -0.028 / -0.049 | 0.37 / 0.68 | | 72 / 85 |
| lag 3 days | y_main / y_all | -0.027 / -0.038 | -0.035 / -0.038 | 0.58 / 0.72 | | 72 / 85 |
| quiet cells only (no M>=6 within 30 d in ±1 bin) | y_main / y_all | -0.017 / -0.009 | | boot CI [-0.035, 0.000] / [-0.027, +0.008] | | 69 / 75 |

## What the data do show (descriptive, not evidence)
- **HF noise is a seismicity detector, which is clustering in disguise.**
  - Alone, x_hf separates y_all cells: AUC 0.62 in train and 0.57 in test.
  - It does not separate mainshock cells in test (AUC 0.49).
  - GO01's HF z averaged +2.5 in the 30 days before Iquique 2014. That is the recorded foreshock
    sequence, including the M6.7 of 16 March, which the catalog-based baseline already uses.
  - Against the baseline, the information is redundant, so the extra parameters only add variance.
- **The only striking anomaly is an instrument artifact.**
  - GO04 (-30.2°) had an LP z of +6 (saturated) for a month before Illapel 2015 (M8.3).
  - The raw data show an abrupt 35 dB step on 2015-08-12 to a flat -140 dB level, which is a classic
    sensor fault (mass off-centre or a stuck level).
  - GO04 shows equal excursions in 2012, 2017, 2018, 2019, 2020, 2022, 2023 and 2025 with no great
    earthquake, and has |z|>=5 on 14.5% of its days.
  - A person scanning plots after the fact could mistake this for a precursor. The prospective test
    does not support it.
- **The negative control behaves as a null should.** x_ms (ocean microseism) scores the same as the
  "precursor" bands.

## Leakage checks
- Every feature uses only days before t_start - 1, with a lag-1 and a lag-3 variant. The baseline window ends
  30 days before D, the windows are trailing only, and nothing is centred, interpolated or back-filled.
- The anomaly is normalised per station by its own causal baseline. Station presence, which grew denser
  after the 2010 and 2014 quakes, is not a feature: missing values are imputed to the train mean, about 0.
- **Reprocessing leak.** MUSTANG values were recomputed in 2020 or later (lddate), so they are not what
  was available in real time. Response-corrected dB could carry later metadata fixes. Because the result
  is null, this could only have inflated a positive.
- The C1 dense set exists only in the test period. Its logit coefficients are the same as the primary set's
  (fitted on the same train rows), so the C1 data only change the test inputs.

## Caveats
- **Weak training signal.** The training signal is thin: 27 y_main and 32 y_all positive train cells
  have a feature value. With 72-85 test cells, a true effect below about +0.03 nats/eq could go undetected.
- **Data choices.** Only the vertical channel and the MUSTANG daily PDF mode were used. Horizontal tilt
  channels, hourly PSDs and IPOC (CX, at GFZ) were not tested.
- **Surrogate null.** The phase-randomize null mean is negative (about -0.03). With an observed IG that is
  itself negative, this does not change the verdict.
- **No harness bug was found.**
- **Machine load.** The machine was heavily loaded (load average about 40), but this affected only run time.

## Reproduce
```
cd research/hypotheses/station_noise
P=/opt/miniconda3/bin/python3; D=/Users/cnavarreteliz/Documents/Repos/earthquakes/research/data/station_noise
curl -sL "https://service.iris.edu/fdsnws/station/1/query?minlat=-46&maxlat=-16.5&minlon=-78&maxlon=-64&cha=BHZ,HHZ&level=channel&format=text" -o $D/channels_region.txt
$P -I download.py $D $D/channels_region.txt
$P -I parse.py $D/raw $D/noise_mode_long.parquet
$P -I build.py
$P -I run_primary.py $D/site_z.parquet primary.json 1000 200
$P -I run_robust.py  $D/site_z.parquet robust.json 200
$P -I descriptive.py $D/site_z.parquet
```
