"""Independent causality check of the sequence inputs: for random (issue time, bin) samples, recompute the 52 weekly
windows from the raw event list truncated at t_start (time < t_start) and compare with what the network is fed.
Also checks that no global event is >= 2013-01-01 or inside the Chile box.
Usage: python3 -I causality_check.py WORK_DIR
"""
import sys, os, json
sys.path.insert(0, "/Users/cnavarreteliz/Documents/Repos/earthquakes/research")
import numpy as np, pandas as pd
import harness as H

WORK = sys.argv[1]
SEQ, NBR = 52, 2
rng = np.random.default_rng(7)
out = {}


def recompute(ev_t, ev_b, ev_m, t_issue, b, nb):
    keep = ev_t < t_issue                       # truncated catalog
    t, bb, m = ev_t[keep], ev_b[keep], ev_m[keep]
    S = np.zeros((SEQ, 2 * NBR + 1, 3), np.float32)
    for k in range(SEQ):
        lo, hi = t_issue - 7 * (SEQ - k), t_issue - 7 * (SEQ - 1 - k)
        for jb, db in enumerate(range(-NBR, NBR + 1)):
            if not (0 <= b + db < nb):
                continue
            sel = (t >= lo) & (t < hi) & (bb == b + db)
            S[k, jb, 0] = np.log1p(sel.sum())
            S[k, jb, 1] = max(0.0, (m[sel] - 4.2).max()) if sel.any() else 0.0
            S[k, jb, 2] = np.log1p((m[sel] >= 6).sum())
    return S


# Chile
c = np.load(os.path.join(WORK, "chile_samples.npz"))
evc = H.get_events(min_mag=4.7)
tc = H.to_days(pd.Series(c["t_start"]).drop_duplicates().sort_values())
K, W = c["key"], c["W"]
test_rows = np.where(c["split"] == "test")[0]
maxdiff = 0.0
for r in rng.choice(test_rows, 300, replace=False):
    ti = tc[K[r, 1]]
    S = recompute(evc.t_days.to_numpy(), evc.bin_id.to_numpy(), evc.mag.to_numpy(), ti, K[r, 2], 29)
    G = W[K[r, 1]:K[r, 1] + SEQ, K[r, 3] - NBR:K[r, 3] + NBR + 1]
    maxdiff = max(maxdiff, float(np.abs(S - G).max()))
out["chile_seq_maxabsdiff_300_test_rows"] = maxdiff

# global
g = np.load(os.path.join(WORK, "global_samples.npz"))
ev = pd.read_parquet(os.path.join(WORK, "global_events.parquet"))
out["global_max_time"] = str(ev.time.max())
out["global_in_chile_box"] = int((ev.latitude.between(H.LAT_MIN, H.LAT_MAX) & ev.longitude.between(H.LON_MIN, H.LON_MAX)).sum())
tg = H.to_days(pd.Series(g["t_starts"]))
Kg, Wg = g["key"], g["W"]
meta = json.load(open(os.path.join(WORK, "build_meta.json")))
names = [k for k in meta if not k.startswith("_")]
maxdiff = 0.0
for r in rng.choice(len(Kg), 300, replace=False):
    mk = Kg[r, 0]; nb = meta[names[mk]]["n_bins"]
    e = ev[ev.margin == mk]
    bins = np.clip((e.s_km.to_numpy() // 111.2).astype(int), 0, nb - 1)
    S = recompute(e.t_days.to_numpy(), bins, e.mag.to_numpy(), tg[Kg[r, 1]], Kg[r, 2], nb)
    G = Wg[Kg[r, 1]:Kg[r, 1] + SEQ, Kg[r, 3] - NBR:Kg[r, 3] + NBR + 1]
    maxdiff = max(maxdiff, float(np.abs(S - G).max()))
out["global_seq_maxabsdiff_300_rows"] = maxdiff
print(json.dumps(out, indent=1))
json.dump(out, open(os.path.join(WORK, "causality_check.json"), "w"), indent=1)
