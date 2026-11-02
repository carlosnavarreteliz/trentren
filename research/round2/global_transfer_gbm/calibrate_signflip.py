"""Calibration of the exact year-block sign-flip test with models of equal expected skill.
Pairs (A, B) = harness baseline + independent pure-noise feature each; also A vs harness baseline.
Usage: python3 -I calibrate_signflip.py <out_json> [n_pairs]"""
import sys, os, json
sys.path.insert(0, "/Users/cnavarreteliz/Documents/Repos/earthquakes/research")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import harness as H
from common import signflip_p

out = sys.argv[1]
NP = int(sys.argv[2]) if len(sys.argv) > 2 else 100
panel = H.get_panel(7, 7, 6.0)
T, B = panel.t_start.nunique(), panel.bin_id.nunique()
rng = np.random.default_rng(123)


def noise(kind):
    if kind == "wn_bin":
        v = rng.normal(size=(T, B))
    elif kind == "rw_global":
        v = np.repeat(np.cumsum(rng.normal(size=T))[:, None], B, 1)
    elif kind == "ar98_global":
        x = np.zeros(T)
        e = rng.normal(size=T)
        for i in range(1, T):
            x[i] = 0.98 * x[i - 1] + e[i]
        v = np.repeat(x[:, None], B, 1)
    return pd.DataFrame({"bin_id": panel.bin_id.to_numpy(), "t_start": panel.t_start.to_numpy(), "x_n": v.ravel()})


rows = []
kinds = ["wn_bin", "rw_global", "ar98_global"]
for tgt in ["y_all", "y_main"]:
    for k in range(NP):
        kind = kinds[k % 3]
        ra = H.evaluate(noise(kind), target=tgt, panel=panel, return_preds=True)
        rb = H.evaluate(noise(kind), target=tgt, panel=panel, return_preds=True)
        pa, pb = ra["preds"], rb["preds"]
        p_ab, ig_ab, _ = signflip_p(pa.y.to_numpy(), pa.p_full.to_numpy(), pb.p_full.to_numpy(), pa.year.to_numpy())
        p_ab2, ig_ab2, _ = signflip_p(pa.y.to_numpy(), pb.p_full.to_numpy(), pa.p_full.to_numpy(), pa.year.to_numpy())
        p_a0, ig_a0, _ = signflip_p(pa.y.to_numpy(), pa.p_full.to_numpy(), pa.p_base.to_numpy(), pa.year.to_numpy())
        rows.append(dict(target=tgt, kind=kind, p_ab=p_ab, p_ba=p_ab2, ig_ab=ig_ab, p_a_vs_base=p_a0, ig_a_vs_base=ig_a0))
df = pd.DataFrame(rows)
summ = {}
for (tgt, kind), g in df.groupby(["target", "kind"]):
    pp = np.r_[g.p_ab.to_numpy(), g.p_ba.to_numpy()]
    summ[f"{tgt}/{kind}"] = {"n_tests_AB_both_directions": int(len(pp)), "fpr05_AB": float(np.mean(pp < 0.05)),
                             "fpr0042_AB": float(np.mean(pp < 0.0042)),
                             "fpr05_A_vs_base": float(np.mean(g.p_a_vs_base < 0.05)), "n_A_vs_base": int(len(g))}
pp = np.r_[df.p_ab.to_numpy(), df.p_ba.to_numpy()]
summ["pooled"] = {"n": int(len(pp)), "fpr05_AB": float(np.mean(pp < 0.05)), "fpr0042_AB": float(np.mean(pp < 0.0042)),
                  "fpr05_A_vs_base": float(np.mean(df.p_a_vs_base < 0.05))}
print(json.dumps(summ, indent=1))
json.dump({"summary": summ, "rows": df.round(5).to_dict("records")}, open(out, "w"), indent=1)
