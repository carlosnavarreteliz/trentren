"""summarize_v2: aggregate all noise-calibration JSONs (v1 plain roll, end-matched shift, v2 phase_randomize) -> CALIBRATION_V2.md (FPR at p<0.05 by mode x target x feature kind)."""
import json, glob, os
import numpy as np
D = "/Users/cnavarreteliz/Documents/Repos/earthquakes/research/calibration/"
rows = []
def load(pat, src):
    for f in sorted(glob.glob(D + pat)):
        for r in json.load(open(f)):
            r = dict(r); r["src"] = src; rows.append(r)
load("v1_plain_roll/noise_*.json", "v1")       # p_shift = plain np.roll circular shift
load("noise_yall_*.json", "v1b")              # p_shift = end-matched circular shift (y_all, n=100)
load("v2_*.json", "v2")                       # p_pr = phase_randomize n=200; p_shift = end-matched shift (y_main, k<20)
tests = []  # (mode, target, kind, p)
for r in rows:
    if r["src"] == "v1":
        tests.append(("circular_shift_plain", r["target"], r["kind"], r["p_shift"]))
    elif "p_shift" in r:
        tests.append(("circular_shift_endmatched", r["target"], r["kind"], r["p_shift"]))
    if "p_pr" in r:
        tests.append(("phase_randomize", r["target"], r["kind"], r["p_pr"]))
    if "p_binperm" in r:
        tests.append(("bin_permute", r["target"], r["kind"], r["p_binperm"]))
    if "p_T" in r:
        tests.append(("rhoades_T_naive", r["target"], r["kind"], r["p_T"]))
import pandas as pd
t = pd.DataFrame(tests, columns=["mode", "target", "kind", "p"])
t["slow"] = t.kind.isin(["rw_global", "ar98_global", "rw_bin"])
def agg(g):
    return pd.Series({"n": len(g), "fp_0.05": int((g.p < 0.05).sum()), "FPR_0.05": round(float((g.p < 0.05).mean()), 3),
                      "fp_0.0036": int((g.p < 0.0036).sum()), "mean_p": round(float(g.p.mean()), 3)})
import subprocess
nrun = int(subprocess.run("ps aux | grep -c '[r]un_calibration'", shell=True, capture_output=True, text=True).stdout.strip() or 0)
out = ["# Harness calibration: pure-noise false-positive rates\n", f"**Status: {'INCOMPLETE - %d calibration processes still running' % nrun if nrun else 'final (all runs finished)'}.** Rows loaded: {len(rows)}.\n",
       "Feature kinds: rw = random walk, ar98 = AR(1) phi=0.98, wn = white noise; _global = one series broadcast to all bins, _bin = independent per bin.",
       "Expected under a calibrated test: FPR ~ 0.05 and mean p ~ 0.5. Binomial 95% range for n=20: 0-3 hits; n=40: 0-5 hits.\n",
       "## By mode x target\n", t.groupby(["mode", "target"]).apply(agg).to_markdown(), "",
       "## By mode x target x slow/fast features (slow = rw_*, ar98_*)\n", t.groupby(["mode", "target", "slow"]).apply(agg).to_markdown(), "",
       "## By mode x target x kind\n", t.groupby(["mode", "target", "kind"]).apply(agg).to_markdown(), ""]
open(D + "CALIBRATION_V2.md", "w").write("\n".join(out))
print("\n".join(out))
