"""Solid-earth (degree-2, Moon+Sun, DE421) + ocean-load tidal Coulomb stress on a generic Chilean megathrust
receiver (strike 0, dip 20 E, rake 90, mu'=0.4) at one point per 1-degree latitude bin, 10-min grid 1972-2028.
Saves tidal extrema (for Tanaka phase) per bin and component, and daily max/min of total dCFS.
Usage: python -I t02_tidal_stress.py <data_dir>   (needs t01 outputs in <data_dir>/derived)"""
import sys, json, time
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, "/Users/cnavarreteliz/Documents/Repos/earthquakes/research")
import harness as H
from skyfield.api import Loader
from skyfield.framelib import itrs
from scipy.interpolate import CubicSpline

D = Path(sys.argv[1]); OUT = D / "derived"
grid = np.load(OUT / "grid.npy")                       # days since 1970-01-01 UTC, 10 min
GM = {"moon": 4.9028e12, "sun": 1.32712440018e20}
A_E, G0 = 6.371e6, 9.81
H2, L2 = 0.6078, 0.0847
MU = LAM = 30e9
LAMP = 2 * LAM * MU / (LAM + 2 * MU)                   # plane-stress lambda'
RHO_W, NU = 1025.0, 0.25
DIP, MUF = np.deg2rad(20.0), 0.4
sd, cd = np.sin(DIP), np.cos(DIP)

# --- 1. body positions in ITRS (hourly via skyfield, cubic spline to 10 min) ------------------------------
t0 = time.time()
L = Loader(str(D)); eph = L("de421.bsp"); ts = L.timescale(builtin=True)
earth = eph["earth"]
hrs = np.arange(grid[0] - 1, grid[-1] + 1.05, 1.0 / 24)
pos = {}
for b, key in [("moon", "moon"), ("sun", "sun")]:
    xyz = []
    for k in range(0, len(hrs), 100000):
        t = ts.utc(1970, 1, 1 + hrs[k:k + 100000])
        xyz.append((eph[key] - earth).at(t).frame_xyz(itrs).m.T)
    xyz = np.vstack(xyz)
    pos[b] = CubicSpline(hrs, xyz, axis=0)(grid)
    print(b, "positions", round(time.time() - t0), "s; mean dist km", np.linalg.norm(pos[b], axis=1).mean() / 1e3,
          flush=True)
bodies = {}
for b in pos:
    d = np.linalg.norm(pos[b], axis=1)
    bodies[b] = (pos[b] / d[:, None], GM[b] * A_E ** 2 / d ** 3)
del pos

# --- 2. bin points --------------------------------------------------------------------------------------
ev = H.get_events(min_mag=4.7)
ev = ev[(ev.depth <= 70) & (ev.time < H.TEST_START)]
centers = H.bin_centers()
lons = np.array([np.median(ev.longitude[ev.bin_id == b]) if (ev.bin_id == b).sum() > 5 else -71.5
                 for b in range(len(centers))])
GAUGE_ORDER = [("arica", -21.06), ("antofagasta", -25.36), ("caldera", -30.05), ("valparaiso", -99)]
def gauge_for(lat):
    for g, lo in GAUGE_ORDER:
        if lat > lo:
            return g
ocean = {g: np.load(OUT / f"ocean_{g}.npy").astype(np.float64) for g, _ in GAUGE_ORDER}


def extrema(x):
    dx = np.diff(x)
    s = np.sign(dx)
    # carry forward zero slopes
    nz = s != 0
    idx = np.where(nz, np.arange(len(s)), 0)
    np.maximum.accumulate(idx, out=idx)
    s = s[idx]
    ch = np.where(s[1:] != s[:-1])[0] + 1      # index in x of turning point
    is_max = s[ch - 1] > 0
    # parabolic refinement of the extremum time
    y0, y1, y2 = x[ch - 1], x[ch], x[np.minimum(ch + 1, len(x) - 1)]
    den = y0 - 2 * y1 + y2
    off = np.where(np.abs(den) > 0, 0.5 * (y0 - y2) / np.where(den == 0, 1, den), 0.0)
    off = np.clip(off, -0.5, 0.5)
    tt = grid[ch] + off * (grid[1] - grid[0])
    return tt, is_max


res = {"lons": lons, "lats": centers}
day0 = np.floor(grid[0]); di = (np.floor(grid) - day0).astype(int); nday = di.max() + 1
stats = {}
for bi, (lat, lon) in enumerate(zip(centers, lons)):
    phi, lam = np.deg2rad(lat), np.deg2rad(lon)
    n = np.array([np.cos(phi) * np.cos(lam), np.cos(phi) * np.sin(lam), np.sin(phi)])
    eE = np.array([-np.sin(lam), np.cos(lam), 0.0])
    eN = np.array([-np.sin(phi) * np.cos(lam), -np.sin(phi) * np.sin(lam), np.cos(phi)])
    eEE = np.zeros(len(grid)); eNN = np.zeros(len(grid))
    for b, (s, K) in bodies.items():
        c = s @ n; sE = s @ eE; sN = s @ eN
        W = K * (1.5 * c ** 2 - 0.5)
        eEE += (H2 * W + 3 * K * L2 * (sE ** 2 - c ** 2)) / (G0 * A_E)
        eNN += (H2 * W + 3 * K * L2 * (sN ** 2 - c ** 2)) / (G0 * A_E)
    sEE = 2 * MU * eEE + LAMP * (eEE + eNN)
    cfs_solid = sd * cd * (0.0 - sEE) + MUF * (sEE * sd ** 2)                     # sigma_UU = 0 at surface
    P = RHO_W * G0 * ocean[gauge_for(lat)]
    sUU, sEEo = -P, -P * NU / (1 - NU)
    cfs_ocean = sd * cd * (sUU - sEEo) + MUF * (sEEo * sd ** 2 + sUU * cd ** 2)
    tot = cfs_solid + cfs_ocean
    for comp, x in [("total", tot), ("solid", cfs_solid), ("ocean", cfs_ocean)]:
        tt, im = extrema(x)
        res[f"ext_t_{comp}_{bi}"] = tt; res[f"ext_max_{comp}_{bi}"] = im
    dmax = np.full(nday, -np.inf); dmin = np.full(nday, np.inf)
    np.maximum.at(dmax, di, tot); np.minimum.at(dmin, di, tot)
    res[f"dmax_{bi}"] = dmax.astype(np.float32); res[f"dmin_{bi}"] = dmin.astype(np.float32)
    stats[bi] = dict(lat=float(lat), lon=float(lon), gauge=gauge_for(lat), solid_std_kPa=float(cfs_solid.std() / 1e3),
                     ocean_std_kPa=float(cfs_ocean.std() / 1e3), total_std_kPa=float(tot.std() / 1e3),
                     n_ext=int(len(res[f"ext_t_total_{bi}"])))
    print(bi, stats[bi], round(time.time() - t0), flush=True)
res["day0"] = day0
np.savez_compressed(OUT / "tidal_extrema.npz", **res)
json.dump(stats, open(OUT / "tidal_stress_summary.json", "w"), indent=1)
