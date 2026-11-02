"""Diagnostic figure: structural profiles vs train/test M>=6 mainshock counts per 0.25-deg bin.
Run: /opt/miniconda3/bin/python3 -I plot_profiles.py <profiles.csv> <count_model.json> <out.png>"""
import sys, json
sys.path.insert(0, "/Users/cnavarreteliz/Documents/Repos/earthquakes/research")
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import harness as H

prof = pd.read_csv(sys.argv[1]); cm = json.load(open(sys.argv[2])); out = sys.argv[3]
ev = H.get_events(min_mag=6.0)
cols = ["x_tpga", "x_rough", "x_logsed", "x_logdist_ridge", "x_age", "x_slabdip", "x_forearc_width"]
fig, ax = plt.subplots(1, len(cols) + 1, figsize=(18, 8), sharey=True)
pb = cm["profile_bins_0p25"]
c = np.array(pb["centers"])
tr = ev[(ev.time < "2013-01-01") & (ev.time >= "1976-01-01") & ev.mainshock]
te = ev[(ev.time >= "2013-01-01") & ev.mainshock]
ax[0].hist(tr.latitude, bins=np.arange(-46, -16.9, 0.25), orientation="horizontal", color="0.6", label="1976-2012")
ax[0].hist(te.latitude, bins=np.arange(-46, -16.9, 0.25), orientation="horizontal", histtype="step", color="C3",
           lw=2, label="2013-2026")
ax[0].plot(np.array(pb["mu_base"]), c, "k-", lw=1, label="base forecast")
ax[0].plot(np.array(pb["mu_full"]), c, "C0-", lw=1, label="+structure")
ax[0].set_xlabel("M>=6 mainshocks / 0.25 deg"); ax[0].legend(fontsize=7); ax[0].set_ylabel("latitude")
for e in ev[(ev.mag >= 7.0) & (ev.time >= "1976-01-01")].itertuples():
    ax[0].axhline(e.latitude, color="C3" if e.time >= pd.Timestamp("2013-01-01") else "0.3", lw=0.6, ls=":")
for a, col in zip(ax[1:], cols):
    a.plot(prof[col], prof.lat, "k-", lw=1); a.set_xlabel(col)
for a in ax:
    a.set_ylim(-46, -17)
plt.tight_layout(); plt.savefig(out, dpi=110)
print("saved", out)
