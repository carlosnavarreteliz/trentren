"""Build along-strike static structural profiles (0.05 deg latitude) for the Chile margin.

Run: /opt/miniconda3/bin/python3 -I build_profiles.py <data_dir> <out_csv>
data_dir must contain grav_*.xyz, topo_*.xyz (topex.ucsd.edu 1' grids), GlobSed-v3.nc, age.2020.1.GTS2012.6m.nc.
Only static grids + pre-2013 catalog hypocenters are used (no test-period information).
"""
import sys, glob
from pathlib import Path
sys.path.insert(0, "/Users/cnavarreteliz/Documents/Repos/earthquakes/research")
import numpy as np, pandas as pd, h5py
from scipy.ndimage import median_filter, uniform_filter1d
import harness as H

data_dir = Path(sys.argv[1]); out_csv = Path(sys.argv[2])
LAT0, LAT1, DL = -47.0, -16.0, 0.05
prof_lat = np.round(np.arange(LAT0 + DL / 2, LAT1, DL), 4)
XG = np.arange(-200, 252, 2.0)  # km from trench, + landward


def load_xyz(pattern):
    parts = []
    for f in sorted(glob.glob(str(data_dir / pattern))):
        d = pd.read_csv(f, sep=r"\s+", header=None, names=["lon", "lat", "z"], dtype=float)
        parts.append(d)
    d = pd.concat(parts).dropna()
    d["lon"] = np.where(d.lon > 180, d.lon - 360, d.lon)
    d["lat"] = d.lat.round(4); d["lon"] = d.lon.round(4)
    d = d.drop_duplicates(["lat", "lon"])
    g = d.pivot(index="lat", columns="lon", values="z").sort_index()
    return g.index.to_numpy(), g.columns.to_numpy(), g.to_numpy()


def load_nc(fn):
    h = h5py.File(fn, "r")
    lon, lat, z = h["lon"][:], h["lat"][:], h["z"][:].astype(float)
    lon = np.where(lon > 180, lon - 360, lon)
    o = np.argsort(lon); lon = lon[o]; z = z[:, o]
    if lat[0] > lat[-1]:
        lat = lat[::-1]; z = z[::-1]
    return lat, lon, z


def sample_grid(lat, lon, z, qlat, qlon):
    """nearest-neighbour sampling of a regular grid"""
    i = np.clip(np.searchsorted(lat, qlat), 1, len(lat) - 1); i -= (qlat - lat[i - 1]) < (lat[i] - qlat)
    j = np.clip(np.searchsorted(lon, qlon), 1, len(lon) - 1); j -= (qlon - lon[j - 1]) < (lon[j] - qlon)
    return z[i, j]


tlat, tlon, topo = load_xyz("topo_*.xyz")
glat, glon, grav = load_xyz("grav_*.xyz")
print("topo", topo.shape, "grav", grav.shape)

# --- trench axis per topo row ---------------------------------------------------------------------------
band = (tlon >= -76.8) & (tlon <= -70.0)
tr_lon = np.full(len(tlat), np.nan); tr_dep = np.full(len(tlat), np.nan)
for r in range(len(tlat)):
    row = topo[r, band]
    if np.isfinite(row).sum() < 10:
        continue
    k = np.nanargmin(row); tr_lon[r] = tlon[band][k]; tr_dep[r] = row[k]
# along-strike median smoothing (~0.5 deg) of the trench longitude to remove spurious picks
nwin = max(3, int(round(0.5 / np.median(np.diff(tlat)))) | 1)
tr_lon_s = median_filter(tr_lon, size=nwin, mode="nearest")
trench = pd.DataFrame({"lat": tlat, "tr_lon": tr_lon_s, "tr_dep_raw": tr_dep})

def x_km(lat, lon, trlon):
    return (lon - trlon) * 111.32 * np.cos(np.deg2rad(lat))

# --- row-wise profiles on x grid ------------------------------------------------------------------------
def rows_on_x(lat, lon, Z):
    trl = np.interp(lat, tlat, tr_lon_s)
    out = np.full((len(lat), len(XG)), np.nan)
    for r in range(len(lat)):
        x = x_km(lat[r], lon, trl[r])
        ok = np.isfinite(Z[r])
        if ok.sum() > 10:
            out[r] = np.interp(XG, x[ok], Z[r, ok], left=np.nan, right=np.nan)
    return out

T_x = rows_on_x(tlat, tlon, topo)
G_x = rows_on_x(glat, glon, grav)

inreg = (glat >= -46) & (glat <= -17)
gbar = np.nanmean(G_x[inreg], axis=0)          # margin-average trench-normal gravity profile
tpga_rows = np.nanmean((G_x - gbar)[:, (XG >= 20) & (XG <= 120)], axis=1)

hp = T_x - uniform_filter1d(np.nan_to_num(T_x, nan=np.nanmean(T_x)), size=25, axis=1)  # 50 km running mean
sea = (XG >= -150) & (XG <= -30)
rough_rows = np.sqrt(np.nanmean(hp[:, sea] ** 2, axis=1))
inbathy_rows = np.nanmean(T_x[:, sea], axis=1)
# forearc width: first x>0 where topo >= 0
fw_rows = np.full(len(tlat), np.nan)
for r in range(len(tlat)):
    land = np.where((XG > 0) & (T_x[r] >= 0))[0]
    if len(land):
        fw_rows[r] = XG[land[0]]

# --- sediment, age ---------------------------------------------------------------------------------------
slat, slon, sed = load_nc(data_dir / "GlobSed-v3.nc")
alat, alon, age = load_nc(data_dir / "age.2020.1.GTS2012.6m.nc")
def band_mean_grid(lat_g, lon_g, Z, x0, x1, n=16):
    out = np.full(len(prof_lat), np.nan)
    trl = np.interp(prof_lat, tlat, tr_lon_s)
    for i, la in enumerate(prof_lat):
        xs = np.linspace(x0, x1, n)
        lo = trl[i] + xs / (111.32 * np.cos(np.deg2rad(la)))
        v = sample_grid(lat_g, lon_g, Z, np.full(n, la), lo)
        out[i] = np.nanmean(np.where(np.isfinite(v), v, np.nan)) if np.isfinite(v).any() else np.nan
    return out
sed_p = band_mean_grid(slat, slon, sed, -30, 0)
age_p = band_mean_grid(alat, alon, age, -60, -10)

# --- slab dip from PRE-2013 hypocenters only -------------------------------------------------------------
cat = H.load_catalog(min_mag=4.5)
cat = cat[(cat.time < H.TEST_START) & (cat.depth >= 20) & (cat.depth <= 200) & ~cat.depth.isin([33.0, 35.0])]
cat_trl = np.interp(cat.latitude.to_numpy(), tlat, tr_lon_s)
cat_x = x_km(cat.latitude.to_numpy(), cat.longitude.to_numpy(), cat_trl)
dip_p = np.full(len(prof_lat), np.nan)
for i, la in enumerate(prof_lat):
    m = (np.abs(cat.latitude.to_numpy() - la) <= 0.5) & (cat_x > 0) & (cat_x < 600)
    if m.sum() >= 15:
        from scipy.stats import theilslopes
        s = theilslopes(cat.depth.to_numpy()[m], cat_x[m])[0]
        dip_p[i] = np.degrees(np.arctan(s))

# --- ridges / fracture zones (fixed from pre-2013 literature, see PREREGISTRATION.md) -------------------
RIDGES = [-20.5, -25.5, -27.3, -30.3, -32.7, -38.3, -40.0, -42.3, -44.5, -46.3]
dist_p = np.min(np.abs(prof_lat[:, None] - np.array(RIDGES)[None, :]), axis=1)

# --- aggregate native rows to 0.05 deg profile ----------------------------------------------------------
def to_prof(lat, v):
    idx = np.floor((lat - LAT0) / DL).astype(int)
    s = pd.Series(v).groupby(idx).mean()
    out = np.full(len(prof_lat), np.nan)
    ok = (s.index >= 0) & (s.index < len(prof_lat))
    out[s.index[ok]] = s.values[ok]
    return out

prof = pd.DataFrame({"lat": prof_lat,
                     "trench_lon": to_prof(tlat, tr_lon_s),
                     "x_tpga": to_prof(glat, tpga_rows),
                     "x_rough": to_prof(tlat, rough_rows),
                     "x_logsed": np.log10(1 + np.clip(sed_p, 0, None)),
                     "x_logdist_ridge": np.log10(0.1 + dist_p),
                     "x_age": age_p,
                     "x_slabdip": dip_p,
                     "x_trenchdepth": to_prof(tlat, tr_dep),
                     "x_inplate_bathy": to_prof(tlat, inbathy_rows),
                     "x_forearc_width": to_prof(tlat, fw_rows)})
# fill isolated gaps along strike (static spatial profile; no time dimension so no look-ahead issue)
for c in prof.columns[1:]:
    prof[c] = prof[c].interpolate(limit_direction="both")
out_csv.parent.mkdir(parents=True, exist_ok=True)
prof.to_csv(out_csv, index=False)
print(prof.describe().T.round(3))
print(prof.iloc[::20].round(2).to_string())
