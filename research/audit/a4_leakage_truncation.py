"""A4: empirical look-ahead test. Recompute ALL features with the catalog truncated just before the
2014-04-01 Iquique Mw8.2 mainshock (and 2015-09-16 Illapel Mw8.3), and compare with the stored features for
the preceding months. Any difference means a feature used data from after the truncation time.
Also tabulates catalog reporting non-stationarity (sub-threshold M<4 events per year)."""
import json
import sys
import numpy as np
import pandas as pd
ROOT = "/Users/cnavarreteliz/Documents/Repos/earthquakes"
sys.path.insert(0, f"{ROOT}/src")
import physics_features as pf  # read-only import

raw = sys.argv[1] if len(sys.argv) > 1 else f"{ROOT}/data/raw/chile_usgs_1900_2026.csv"
stored = pd.read_csv(f"{ROOT}/data/processed/chile_features_m40.csv")
stored["time"] = pd.to_datetime(stored["time"], utc=True, format="ISO8601")
out = {}
df_full, df_cat = pf.load_full_and_homogeneous_catalogs(raw, mc_cutoff=4.0)

# reporting non-stationarity
yrs = df_full["time"].dt.year
sub = df_full[df_full["mag"] < 4.0]
out["subthreshold_M<4_events_per_year"] = {int(k): int(v) for k, v in sub.groupby(sub["time"].dt.year).size().items() if k >= 1976}
out["M>=4_events_per_year"] = {int(k): int(v) for k, v in df_cat.groupby(df_cat["time"].dt.year).size().items() if k >= 1976}
st = stored[stored["year_frac"] >= 1976]
out["frac_b_pos_subthresh_defaulted_to_1_by_year"] = {
    int(k): round(float(v), 3) for k, v in (st["b_pos_subthresh_40"] == 1.0).groupby(st["time"].dt.year).mean().items()}

for label, cut, start in [("iquique_2014", "2014-04-01T23:40:00Z", "2014-01-01"),
                          ("illapel_2015", "2015-09-16T22:50:00Z", "2015-06-01")]:
    T = pd.Timestamp(cut)
    f_cut = df_full[df_full["time"] < T].copy().reset_index(drop=True)
    c_cut = df_cat[df_cat["time"] < T].copy().reset_index(drop=True)
    feats = pf.extract_spatiotemporal_features(f_cut, c_cut)
    win = feats[feats["time"] >= pd.Timestamp(start, tz="UTC")]
    ref = stored.set_index("id").loc[win["id"].values]
    fcols = [c for c in feats.columns if c in ref.columns and not c.startswith("y_") and
             c not in ("max_mag_14d", "max_mag_30d") and pd.api.types.is_numeric_dtype(feats[c])]
    diffs = {}
    for c in fcols:
        a, b = win[c].values.astype(float), ref[c].values.astype(float)
        nd = int(np.sum(~np.isclose(a, b, rtol=1e-9, atol=1e-9, equal_nan=True)))
        if nd:
            diffs[c] = nd
    out[label] = {"n_events_compared": int(len(win)), "n_feature_cols": len(fcols), "cols_with_differences": diffs}
    print(label, out[label], flush=True)

json.dump(out, open(f"{ROOT}/research/audit/a4_results.json", "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if "per_year" in k or "defaulted" in k}))
