# PREREGISTRATION: ocean_loading (written 2026-10-17, before any evaluation on test data)

## Hypothesis
Non-tidal changes in the load on the forearc/trench change the normal stress on the Chilean megathrust. These include storm surge, coastally trapped waves, ENSO-driven sea-level anomalies and barometric pressure. Rapid unloading (falling ocean-bottom pressure) unclamps the fault and should slightly raise the short-term probability of large earthquakes, and the opposite should hold for loading. If this is real, a causal ocean-bottom-pressure (OBP) anomaly feature should add information beyond the ETAS/clustering baseline.

## Data
- **Sea level:** UHSLC hourly tide-gauge data, using the fast-delivery files (`data/csv/fast/hourly/hNNN.csv`, which hold the full history). These are closer to the real-time product than the RQDS.
  - UHSLC has only 5 continental Chilean stations: Arica 083 (-18.47), Antofagasta 080 (-23.65), Caldera 088 (-27.07), Valparaiso 081 (-33.03) and Puerto Montt 684 (-41.49). Iquique, Coquimbo, Talcahuano and Corral are not in UHSLC.
  - Island stations (Juan Fernandez, San Felix) are excluded because they are not on the forearc.
  - The RQDS hourly files are used only for a cross-check.
- **Pressure:** ERA5 daily mean surface pressure from the Open-Meteo archive (`models=era5`), at one offshore forearc point per 1° bin. The point sits at the bin's latitude centre, 0.75° west of an approximate coastline.

## Feature construction (strictly causal)
1. **Daily sea level:** the UTC-day mean of the hourly values, kept only when at least 20 of 24 hours are valid. Days that deviate by more than 500 mm from the causal median of the previous 30 days are set to NaN.
2. **Anomalies:**
   - Sea level: `eta_a,s(D) = eta_s(D) - mean(eta_s over days D-365..D-1)`, requiring at least 180 valid days.
   - Pressure: `P_a,b(D) = P_b(D) - mean(P_b over D-365..D-1)`.
3. **Sea level for each bin:** on day D, the anomaly from the station nearest in latitude that has a valid value that day, if it lies within 6° of latitude. Otherwise NaN.
4. **Ocean-bottom-pressure anomaly** in hPa: `L_b(D) = 0.1006 * eta_a(D)[mm] + P_a,b(D)`. This uses rho = 1025 kg/m3 and g = 9.81.
5. **Panel features.** These are aligned with `H.align_daily_series` and `lag_days=1`, so only days D < t_start - 1 day are used:
   - `x_dL7`: the mean of L over the last 7 days minus the mean of L over the 30 days before those, i.e. the rapid loading change. Negative means unloading.
   - `x_L30`: the mean of L over the last 30 days, i.e. the level of the loading anomaly.
   - Missing values are imputed with the train mean by the harness, and no missingness indicator is used. A missingness indicator could encode gauge outages caused by earthquakes.

## Primary test (one test, fixed now)
- Panel: weekly (step 7 d, horizon 7 d), target M >= 6.0, **target `y_main`** (ZBZ mainshocks), model logit.
- Features: `[x_dL7, x_L30]` per bin.
- Statistic: `ig_per_event` (full minus the ETAS-logit baseline) on the test period, 2013-01-01 to 2026-10.
- Significance: `H.surrogate_test(mode="phase_randomize", n=1000)` on the panel feature grid, which uses the same phases in all bins. The p-value is one-sided.
- Verdict rule:
  - **positive** if IG > 0, p < 0.0036 (Bonferroni over 14 hypotheses), and the effect survives the leakage checks below;
  - **weak_positive** if p < 0.05;
  - otherwise **null**, or **negative** if IG is clearly < 0.

## Secondary and robustness analyses (reported, not used for the verdict)
- Weekly M6 `y_all`, and weekly M5.5 `y_all` and `y_main` (phase_randomize, n=200).
- `bin_permute` and end-matched `circular_shift` surrogates for the primary.
- The GBM model.
- A 30-day panel (s30_h30) and the daily panel (s1_h1) at M6.
- Component split: sea-level-only versus pressure-only.
- Alternate split year: train on 1976-2004 and test on 2005-2026, if the harness allows it via a modified `split` column.
- Year-block bootstrap CI; dropping each test year in turn.
- **Leakage checks:**
  - Mask the features (set them to NaN, which is imputed with the mean) for any bin-time where an M >= 7 occurred within ±3 bins in the previous 400 days. This uses the past catalog only and removes coseismic steps and post-seismic tide-gauge signals.
  - Compare FD against RQDS sea-level series.
  - Confirm that no centered filters are used. The hourly data are averaged by UTC day, and UHSLC's filtered daily product is not used.
