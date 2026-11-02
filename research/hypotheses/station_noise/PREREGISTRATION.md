# PREREGISTRATION: station_noise

Written 2026-10-17, after downloading the data and checking which stations and days exist (coverage
tables only), and BEFORE any feature was evaluated against any target (train or test).

## Hypothesis
Daily ambient seismic noise levels at broadband stations carry information about the state of the
crust that the M>=4.7 catalog does not. Two physical channels:
1. **HF excess (1-3 Hz)**: rising high-frequency noise on the vertical channel reflects microseismicity
   below catalog completeness, micro-cracking or fluid activity near the station. If it is precursory,
   it should raise the probability of an M>=6 in nearby latitude bins in the next week.
2. **LP drift (100-190 s)**: long-period vertical noise is dominated by tilt and pressure effects;
   aseismic deformation (slow slip) could show up as anomalous long-period noise.

## Data
IRIS/EarthScope MUSTANG `noise-mode-timeseries` (daily mode of the hourly PSD PDF, dB rel. 1 (m/s^2)^2/Hz),
vertical channel BHZ, all stations in lat -46..-16.5, lon -78..-64 for networks IU, G, GT, C, C1 that
MUSTANG covers. Network CX (IPOC) is not at IRIS, so it is not available here.
Coverage (days with data): IU.LVC 1996-2018, IU.LCO 1998-2026, G.PEL 1995-2008 and 2011-2026,
GT.PLCA 1995-2026, G.COYC 2011-2026, C.GO01-GO07 2012-2026, C1.* 2013-2026 (test only).

## Feature construction (strictly causal)
- Bands per day D and NSLC stream: HF = mean of the modes at 1.037 Hz and 3.2 Hz; LP = mean of the modes
  at 0.00526 Hz and 0.00964 Hz. Values outside [-200, -60] dB (HF) or [-200, -80] dB (LP) are dropped.
- Anomaly z(D) = (P(D) - median(P over [D-365, D-30))) / (1.4826 * MAD over the same window); at least
  60 valid baseline days required; the baseline never crosses a channel-epoch boundary from the FDSN
  station metadata (instrument change) or a change of location code; z clipped to [-6, 6].
- One stream per site per day, priority loc 00 > '--' > 10 > 60 (no averaging of co-located sensors).
- Per-bin feature at issue time t: mean over sites with |lat_site - lat_center| <= 1.5 deg of the site's
  mean z over days D in [t-15, t-1) (14 days, 1-day lag), requiring >= 5 valid days. NaN where no site
  qualifies (harness imputes the train mean for the logit).
- Primary features: `x_hf` and `x_lp` (both in one model).
- Primary station set: every non-C1 site (IU, G, GT, C). The dense C1 set is a SECONDARY variant only,
  because C1 exists only in the test period.
- Negative control (not part of the primary): `x_ms`, the same anomaly for the ocean-microseism band
  (0.1 and 0.308 Hz), which is driven by ocean storms and should carry no precursory signal.

**Amendment (made while coding the features, still before any evaluation against targets):** MUSTANG
reports the modes as integer dB, so a MAD can be 0. The robust scale is therefore IQR/1.349 over the
same baseline window (equal to 1.4826*MAD for Gaussian data), floored at 1 dB. The baseline window is
implemented as days D-364..D-30 inclusive.

## Primary test
- Panel: weekly issue times, 7-day horizon, target M>=6.0, harness split (train 1976-2012 purged, test
  2013-01-01 .. 2026-10-17). Model: logit. Target: **y_main** (declustered mainshocks).
- Statistic: `ig_per_event` (full minus baseline, nats per target event).
- Significance: `surrogate_test(mode="phase_randomize")` on the panel features, n=1000.
- Claim: positive only if IG > 0 AND p < 0.0036 (Bonferroni for 14 hypotheses) AND IG vs ETAS > 0.

## Secondary (reported, not used for the claim)
- y_all on the same panel; M>=5.5 weekly panel (y_all, y_main); bin_permute and end-matched
  circular_shift surrogates; GBM model; year block-bootstrap CI.
- Each feature alone (x_hf, x_lp); negative control x_ms; dense C1 station set.
- Robustness: 30-day panel (step 30, horizon 30); alternate split year 2017 (train <= 2016, test 2017+)
  and a rolling origin restricted to 1996+ issue times (years with station data).
- Leakage checks: lag 1 vs 3 days; dropping cells with an M>=6 within 30 days in +-1 bin (to see whether
  any gain is aftershock noise from the ongoing sequence).
