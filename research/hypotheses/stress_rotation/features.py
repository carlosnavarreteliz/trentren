"""Causal focal-mechanism features per (bin, t_start) from the parsed GCMT catalog. See PREREGISTRATION.md."""
import sys
sys.path.insert(0, "/Users/cnavarreteliz/Documents/Repos/earthquakes/research")
import numpy as np
import pandas as pd
import harness as H

COLS = ["x_cons", "x_rot", "x_dthrust", "x_dclvd"]
W2 = np.array([1, 1, 1, 2, 2, 2], float)   # Frobenius weights for (rr,tt,pp,rt,rp,tp)


def load_cmt(path):
    c = pd.read_csv(path, parse_dates=["time"])
    M = c[["mrr", "mtt", "mpp", "mrt", "mrp", "mtp"]].to_numpy(float)
    nrm = np.sqrt((M ** 2 * W2).sum(1))
    c[["n_rr", "n_tt", "n_pp", "n_rt", "n_rp", "n_tp"]] = M / (nrm / np.sqrt(2))[:, None]   # unit scalar moment
    c["thrust"] = (c.plT > 50).astype(float)
    c["normal"] = (c.plP > 60).astype(float)
    c["clvd"] = np.abs(c.eN) / np.maximum(np.abs(c.eT), np.abs(c.eP))
    c["bin_id"] = H.lat_to_bin(c.lat.to_numpy())   # may be <0 or >28 outside the harness region
    return c


MECH = ["n_rr", "n_tt", "n_pp", "n_rt", "n_rp", "n_tp", "thrust", "normal", "clvd",
        "strike1", "dip1", "rake1", "strike2", "dip2", "rake2"]


def shuffle_mechanisms(c, rng, maxdepth=70.0):
    """Permute mechanism content among events of the shallow set (keeps time/location/depth/mag)."""
    c = c.copy()
    idx = np.where(c.depth.to_numpy() <= maxdepth)[0]
    perm = rng.permutation(idx)
    c.loc[c.index[idx], MECH] = c.loc[c.index[perm], MECH].to_numpy()
    return c


def build(cmt, panel, W=730.0, L=2.0, mmin=None, maxdepth=70.0, nbr=1, n_min_r=3, n_min_lt=5):
    """Return DataFrame [bin_id, t_start] + COLS. Only events with time < t_start - L are used."""
    c = cmt[cmt.depth <= maxdepth]
    if mmin is not None:
        c = c[c.mw >= mmin]
    ts = np.sort(panel.t_start.unique())
    tq_end = (ts - np.timedelta64(int(L * 86400), "s"))
    tq_beg = (tq_end - np.timedelta64(int(W * 86400), "s"))
    nb = int(panel.bin_id.max()) + 1
    out = []
    for b in range(nb):
        e = c[(c.bin_id >= b - nbr) & (c.bin_id <= b + nbr)].sort_values("time")
        t = e.time.to_numpy()
        V = np.column_stack([e[["n_rr", "n_tt", "n_pp", "n_rt", "n_rp", "n_tp"]].to_numpy(float),
                             np.ones(len(e)), e.thrust.to_numpy(float), e.clvd.to_numpy(float)])
        cs = np.vstack([np.zeros((1, V.shape[1])), np.cumsum(V, 0)])
        i1 = np.searchsorted(t, tq_end, "left")    # events with time < t_start - L
        i0 = np.searchsorted(t, tq_beg, "left")    # events with time < t_start - L - W
        R = cs[i1] - cs[i0]
        LT = cs[i0]
        nR, nL = R[:, 6], LT[:, 6]
        A, B = R[:, :6], LT[:, :6]
        nA = np.sqrt((A ** 2 * W2).sum(1)); nB = np.sqrt((B ** 2 * W2).sum(1))
        with np.errstate(invalid="ignore", divide="ignore"):
            cons = nA / (nR * np.sqrt(2))
            cosang = (A * B * W2).sum(1) / (nA * nB)
            rot = np.degrees(np.arccos(np.clip(cosang, -1, 1)))
            dthr = R[:, 7] / nR - LT[:, 7] / nL
            dclvd = R[:, 8] / nR - LT[:, 8] / nL
        okR = nR >= n_min_r
        okL = okR & (nL >= n_min_lt)
        f = pd.DataFrame({"bin_id": b, "t_start": ts,
                          "x_cons": np.where(okR, cons, np.nan),
                          "x_rot": np.where(okL, rot, np.nan),
                          "x_dthrust": np.where(okL, dthr, np.nan),
                          "x_dclvd": np.where(okL, dclvd, np.nan)})
        out.append(f)
    return pd.concat(out, ignore_index=True)


# ---------------- exploratory: linearised Michael (1984) stress inversion misfit ----------------
def _plane_vectors(strike, dip, rake):
    s, d, r = np.radians(strike), np.radians(dip), np.radians(rake)
    # NED coordinates (Aki & Richards)
    n = np.stack([-np.sin(d) * np.sin(s), np.sin(d) * np.cos(s), -np.cos(d)], -1)
    u = np.stack([np.cos(r) * np.cos(s) + np.sin(r) * np.cos(d) * np.sin(s),
                  np.cos(r) * np.sin(s) - np.sin(r) * np.cos(d) * np.cos(s),
                  -np.sin(r) * np.sin(d)], -1)
    return n, u


def _michael_rows(n, u):
    # deviatoric stress with s33 = -(s11+s22); unknowns (s11,s12,s13,s22,s23); shear direction ~ slip (|tau|=1)
    n1, n2, n3 = n[:, 0], n[:, 1], n[:, 2]
    Arows = np.zeros((len(n), 3, 5))
    Arows[:, 0] = np.stack([n1 - n1 ** 3 + n1 * n3 ** 2, n2 - 2 * n1 ** 2 * n2, n3 - 2 * n1 ** 2 * n3,
                            -n1 * n2 ** 2 + n1 * n3 ** 2, -2 * n1 * n2 * n3], -1)
    Arows[:, 1] = np.stack([-n1 ** 2 * n2 + n2 * n3 ** 2, n1 - 2 * n1 * n2 ** 2, -2 * n1 * n2 * n3,
                            n2 - n2 ** 3 + n2 * n3 ** 2, n3 - 2 * n2 ** 2 * n3], -1)
    Arows[:, 2] = np.stack([-n1 ** 2 * n3 - n3 + n3 ** 3, -2 * n1 * n2 * n3, n1 - 2 * n1 * n3 ** 2,
                            -n2 ** 2 * n3 - n3 + n3 ** 3, n2 - 2 * n2 * n3 ** 2], -1)
    return Arows, u


def misfit_window(ev):
    """Mean angle (deg) between slip and resolved shear, choosing the better nodal plane per event."""
    n1, u1 = _plane_vectors(ev.strike1.to_numpy(), ev.dip1.to_numpy(), ev.rake1.to_numpy())
    n2, u2 = _plane_vectors(ev.strike2.to_numpy(), ev.dip2.to_numpy(), ev.rake2.to_numpy())
    best = None
    sel = np.zeros(len(ev), bool)  # start: plane 1
    for _ in range(3):
        n = np.where(sel[:, None], n2, n1); u = np.where(sel[:, None], u2, u1)
        A, uu = _michael_rows(n, u)
        s, *_ = np.linalg.lstsq(A.reshape(-1, 5), uu.reshape(-1), rcond=None)
        S = np.array([[s[0], s[1], s[2]], [s[1], s[3], s[4]], [s[2], s[4], -s[0] - s[3]]])
        angs = []
        for nn, u_ in ((n1, u1), (n2, u2)):
            tr = nn @ S
            tau = tr - (tr * nn).sum(1, keepdims=True) * nn
            tau /= np.linalg.norm(tau, axis=1, keepdims=True) + 1e-12
            angs.append(np.degrees(np.arccos(np.clip((tau * u_).sum(1), -1, 1))))
        sel = angs[1] < angs[0]
        best = np.minimum(angs[0], angs[1]).mean()
    return best


def build_misfit(cmt, panel, W=730.0, L=2.0, maxdepth=70.0, nbr=1, n_min=6, step_weeks=4):
    """Exploratory 5th column; recomputed every `step_weeks` issue times and carried forward (causal)."""
    c = cmt[cmt.depth <= maxdepth]
    ts = np.sort(panel.t_start.unique())
    out = []
    for b in range(int(panel.bin_id.max()) + 1):
        e = c[(c.bin_id >= b - nbr) & (c.bin_id <= b + nbr)].sort_values("time").reset_index(drop=True)
        t = e.time.to_numpy()
        vals = np.full(len(ts), np.nan)
        last = np.nan
        for k, tq in enumerate(ts):
            if k % step_weeks == 0:
                te = tq - np.timedelta64(int(L * 86400), "s")
                i1 = np.searchsorted(t, te, "left"); i0 = np.searchsorted(t, te - np.timedelta64(int(W * 86400), "s"), "left")
                last = misfit_window(e.iloc[i0:i1]) if i1 - i0 >= n_min else np.nan
            vals[k] = last
        out.append(pd.DataFrame({"bin_id": b, "t_start": ts, "x_misfit": vals}))
    return pd.concat(out, ignore_index=True)
