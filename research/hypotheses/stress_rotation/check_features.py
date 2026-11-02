"""Sanity / leakage checks of the features (no test-target inspection).
Usage: python3 -I check_features.py <gcmt_chile.csv>"""
import sys
sys.path.insert(0, "/Users/cnavarreteliz/Documents/Repos/earthquakes/research")
sys.path.insert(0, "/Users/cnavarreteliz/Documents/Repos/earthquakes/research/hypotheses/stress_rotation")
import numpy as np, pandas as pd
import harness as H
import features as F

cmt = F.load_cmt(sys.argv[1])
panel = H.get_panel(step_days=7, horizon_days=7, target_mag=6.0)
f = F.build(cmt, panel)
print("rows", len(f), "NaN frac:", f[F.COLS].isna().mean().round(3).to_dict())
print(f[F.COLS].describe().round(3).to_string())

# brute force recompute 300 random rows
rng = np.random.default_rng(0)
sh = cmt[cmt.depth <= 70]
bad = 0
for i in rng.choice(len(f), 300, replace=False):
    r = f.iloc[i]
    te = r.t_start - pd.Timedelta(days=2); tb = te - pd.Timedelta(days=730)
    e = sh[(sh.bin_id >= r.bin_id - 1) & (sh.bin_id <= r.bin_id + 1)]
    R = e[(e.time >= tb) & (e.time < te)]; LT = e[e.time < tb]
    assert (R.time < r.t_start).all()
    if len(R) >= 3:
        A = R[["n_rr", "n_tt", "n_pp", "n_rt", "n_rp", "n_tp"]].sum().to_numpy()
        cons = np.sqrt((A ** 2 * F.W2).sum()) / (len(R) * np.sqrt(2))
        bad += not np.isclose(cons, r.x_cons)
        if len(LT) >= 5:
            bad += not np.isclose(R.thrust.mean() - LT.thrust.mean(), r.x_dthrust)
    else:
        bad += not np.isnan(r.x_cons)
print("brute-force mismatches:", bad)

# Michael inversion synthetic check: slip parallel to shear of a known stress -> misfit ~ 0
S = np.array([[1.0, 0.2, 0.1], [0.2, -0.3, 0.05], [0.1, 0.05, -0.7]])
rows = []
for _ in range(30):
    s, d = rng.uniform(0, 360), rng.uniform(20, 80)
    n, _u = F._plane_vectors(np.array([s]), np.array([d]), np.array([0.0]))
    tr = n @ S; tau = tr - (tr * n).sum(1, keepdims=True) * n; tau /= np.linalg.norm(tau)
    # rake of tau on the plane
    st = np.array([np.cos(np.radians(s)), np.sin(np.radians(s)), 0.0])
    dd = np.cross(n[0], st)  # down-dip-ish direction
    rk = np.degrees(np.arctan2(-(tau[0] @ dd) * np.sign(dd[2]) if dd[2] != 0 else 0, tau[0] @ st))
    for rr in (rk, -rk):
        n2, u2 = F._plane_vectors(np.array([s]), np.array([d]), np.array([rr]))
        if np.degrees(np.arccos(np.clip((u2[0] * tau[0]).sum(), -1, 1))) < 1:
            rows.append(dict(strike1=s, dip1=d, rake1=rr, strike2=s, dip2=d, rake2=rr)); break
ev = pd.DataFrame(rows)
print("synthetic events", len(ev), "misfit (deg, expect ~0):", F.misfit_window(ev))
ev2 = ev.copy(); ev2["rake1"] = rng.uniform(-180, 180, len(ev)); ev2["rake2"] = ev2.rake1
print("random rakes misfit (expect large):", F.misfit_window(ev2))
