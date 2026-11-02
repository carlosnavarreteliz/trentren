"""C1: ONI/PDO como covariables sobre ETAS (panel diario). Vease PREREG."""
import sys
sys.path.insert(0, "research")
import numpy as np
import pandas as pd
import harness as H


def load_oni():
    rows = []
    for ln in open("research/round9_climate/data/oni_ncep.txt"):
        p = ln.split()
        if len(p) == 4 and p[1].isdigit():
            seas, yr, tot, anom = p
            mid = {"DJF": 1, "JFM": 2, "FMA": 3, "MAM": 4, "AMJ": 5, "MJJ": 6,
                   "JJA": 7, "JAS": 8, "ASO": 9, "SON": 10, "OND": 11, "NDJ": 12}[seas]
            y = int(yr) - (1 if seas == "DJF" else 0)
            rows.append({"date": pd.Timestamp(y, mid if seas != "NDJ" or y else 12, 1), "oni": float(anom)})
            if seas == "NDJ":
                rows[-1]["date"] = pd.Timestamp(y, 12, 1)
    s = pd.DataFrame(rows).drop_duplicates("date").set_index("date").sort_index()
    daily = s.resample("D").ffill()
    return daily.oni


def load_pdo():
    rows = []
    hdr = None
    for ln in open("research/round9_climate/data/pdo_ncei.dat"):
        p = ln.split()
        if len(p) == 13 and p[0].isdigit():
            for mi in range(12):
                try:
                    v = float(p[1 + mi])
                except ValueError:
                    continue
                rows.append({"date": pd.Timestamp(int(p[0]), mi + 1, 1), "pdo": v})
    s = pd.DataFrame(rows).set_index("date").sort_index()
    return s.resample("D").ffill().pdo


def main():
    panel = H.get_panel(step_days=1, horizon_days=1, target_mag=5.0)
    oni = load_oni()
    pdo = load_pdo()
    t = pd.DatetimeIndex(panel.t_start.unique()).sort_values()
    df = pd.DataFrame({"t_start": t})
    df["x_oni"] = pd.Series(oni).reindex(t).ffill().to_numpy()
    df["x_pdo"] = pd.Series(pdo).reindex(t).ffill().to_numpy()
    doy = t.dayofyear.to_numpy()
    df["x_oni_sin"] = df.x_oni * np.sin(2 * np.pi * doy / 365.25)
    # lageo causal 1 dia: el feature del dia D usa indice hasta D-1 (mensual, ok)
    df[["x_oni", "x_pdo", "x_oni_sin"]] = df[["x_oni", "x_pdo", "x_oni_sin"]].shift(1)
    print(df.head(3).to_string(), "\nNaNs:", int(df.isna().sum().sum()))
    for tgt in ("y_all", "y_main"):
        res = H.evaluate(df, target=tgt, model="logit", panel=panel, return_preds=True)
        sur = H.surrogate_test(df, n=200, mode="phase_randomize", target=tgt, panel=panel)
        print(f"--- {tgt}: {H.summarize(res, sur)}")
        print("   CI:", H.block_bootstrap_ci(res["preds"]))


if __name__ == "__main__":
    sys.exit(main())
