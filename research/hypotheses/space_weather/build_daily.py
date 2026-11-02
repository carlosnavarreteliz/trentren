"""Parse downloaded space-weather files into one daily UTC table.

usage: python -I build_daily.py <data_dir>
reads  <data_dir>/omni2/omni2_YYYY.dat, <data_dir>/Kp_ap_Ap_SN_F107_since_1932.txt, <data_dir>/SN_d_tot_V2.0.csv
writes <data_dir>/space_weather_daily.csv  (date, np, v, bz, bzs, kp_max_omni, dst_min, dst_mean, kp_max, ap, ssn, f107,
                                            ssn_silso, n_np_hours)
"""
import sys
import glob
import numpy as np
import pandas as pd

d = sys.argv[1]

# ---------------- OMNI2 hourly -----------------------------------------------------------------
frames = []
for f in sorted(glob.glob(f"{d}/omni2/omni2_*.dat")):
    a = np.loadtxt(f)
    if a.ndim != 2 or a.shape[1] < 51:
        print("skip", f, a.shape)
        continue
    frames.append(a)
A = np.vstack(frames)
yr, doy, hr = A[:, 0].astype(int), A[:, 1].astype(int), A[:, 2].astype(int)
t = pd.to_datetime(yr.astype(str), format="%Y") + pd.to_timedelta(doy - 1, "D") + pd.to_timedelta(hr, "h")


def col(i, fill):
    x = A[:, i - 1].astype(float)
    x[x >= fill] = np.nan
    return x


h = pd.DataFrame({
    "bz": col(17, 999.9),
    "np": col(24, 999.9),
    "v": col(25, 9999.0),
    "kp10": col(39, 99),
    "r_omni": col(40, 999),
    "dst": col(41, 99999),
    "f107_omni": col(51, 999.9),
}, index=t)
h.loc[h.np <= 0, "np"] = np.nan
day = h.index.floor("D")
g = h.groupby(day)
daily = pd.DataFrame({
    "np": g["np"].mean(),
    "n_np_hours": g["np"].count(),
    "v": g["v"].mean(),
    "bz": g["bz"].mean(),
    "bzs": g["bz"].apply(lambda s: np.nanmean(np.maximum(-s, 0)) if s.notna().any() else np.nan),
    "kp_max_omni": g["kp10"].max() / 10.0,
    "dst_min": g["dst"].min(),
    "dst_mean": g["dst"].mean(),
    "r_omni": g["r_omni"].mean(),
    "f107_omni": g["f107_omni"].mean(),
})
# require >= 12 hourly values for a daily solar-wind mean
for c in ("np", "v", "bz", "bzs"):
    daily.loc[g[c if c != "bzs" else "bz"].count() < 12, c] = np.nan

# ---------------- GFZ Kp / Ap / SN / F10.7 ------------------------------------------------------
rows = []
with open(f"{d}/Kp_ap_Ap_SN_F107_since_1932.txt") as fh:
    for line in fh:
        if line.startswith("#") or not line.strip():
            continue
        p = line.split()
        rows.append([int(p[0]), int(p[1]), int(p[2])] + [float(x) for x in p[7:15]] + [float(p[23]), float(p[24]),
                                                                                         float(p[25])])
k = pd.DataFrame(rows, columns=["y", "m", "d"] + [f"kp{i}" for i in range(1, 9)] + ["ap", "ssn", "f107"])
k.index = pd.to_datetime(dict(year=k.y, month=k.m, day=k.d))
kp = k[[f"kp{i}" for i in range(1, 9)]].where(lambda x: x >= 0)
gfz = pd.DataFrame({"kp_max": kp.max(1), "ap": k.ap.where(k.ap >= 0), "ssn": k.ssn.where(k.ssn >= 0),
                    "f107": k.f107.where(k.f107 > 0)})

# ---------------- SILSO daily total sunspot number ----------------------------------------------
s = pd.read_csv(f"{d}/SN_d_tot_V2.0.csv", sep=";", header=None)
s.index = pd.to_datetime(dict(year=s[0], month=s[1], day=s[2]))
silso = pd.DataFrame({"ssn_silso": s[4].where(s[4] >= 0)})

out = daily.join(gfz, how="outer").join(silso, how="left")
out = out.loc["1970-01-01":"2026-10-17"]
out.index.name = "date"
out.to_csv(f"{d}/space_weather_daily.csv")
print(out.describe().T[["count", "mean", "min", "max"]])
cov = out[["np", "v", "bz", "kp_max", "dst_min", "ssn", "f107"]].notna().groupby(out.index.year).mean()
print(cov.round(2).to_string())
print("corr kp_max(GFZ) vs kp_max(OMNI):", out[["kp_max", "kp_max_omni"]].corr().iloc[0, 1])
print("corr ssn(GFZ) vs ssn(SILSO):", out[["ssn", "ssn_silso"]].corr().iloc[0, 1])
print("corr f107(GFZ) vs f107(OMNI):", out[["f107", "f107_omni"]].corr().iloc[0, 1])
