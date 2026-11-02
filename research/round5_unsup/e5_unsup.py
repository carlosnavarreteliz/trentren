"""E5: anomalias multivariadas exogenas antes de mainshocks M>=6. Vease PREREG."""
import sys
sys.path.insert(0, "research")
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import harness as H

RNG = np.random.default_rng(41)
DD = "research/data"


def load_matrix():
    days = pd.DataFrame({"date": pd.date_range("1990-01-01", "2026-10-17")})
    X = days.copy()
    # 1) clima espacial
    sw = pd.read_csv(f"{DD}/space_weather/space_weather_daily.csv", parse_dates=["date"])
    sw = sw.rename(columns={"f107_omni": "f107"})
    X = X.merge(sw[["date", "kp_max", "dst_min", "f107", "np", "v", "bz"]], on="date", how="left")
    # 2) EOP: year mo da hr MJD x y UT1... (21 cols)
    e = pd.read_csv(f"{DD}/earth_rotation/eopc04_1962-now.txt", sep=r"\s+", comment="#", header=None)
    e["date"] = pd.to_datetime(dict(year=e[0], month=e[1], day=e[2]))
    e = e.rename(columns={5: "px", 6: "py", 7: "ut1"})
    e = e.sort_values("date")
    e["lod"] = -e.ut1.diff() * 1000.0  # ms/dia, lageado abajo por shift global
    X = X.merge(e[["date", "px", "py", "ut1", "lod"]], on="date", how="left")
    # 3) carga oceanica: media diaria sobre columnas
    o = pd.read_parquet(f"{DD}/ocean_loading/eta_anom_fd.parquet")
    num = o.select_dtypes("number")
    oday = pd.DataFrame({"date": pd.date_range("1970-01-01", periods=len(o)),
                         "oc": num.mean(axis=1).to_numpy(),
                         "oc_abs": num.abs().mean(axis=1).to_numpy()})
    X = X.merge(oday, on="date", how="left")
    # 4) hidroclima
    h = np.load(f"{DD}/hydroclimate/bin_daily_v1.npz", allow_pickle=True)
    hd = pd.DataFrame({"date": pd.to_datetime(h["days"]),
                       "P_land": h["P_land"].mean(axis=1),
                       "P_andes": h["P_andes"].mean(axis=1),
                       "SWE": h["SWE_andes"].mean(axis=1)}).sort_values("date")
    for c in ["P_land", "P_andes", "SWE"]:
        m = hd[c].rolling(30, min_periods=10).mean().shift(1)
        s = hd[c].rolling(30, min_periods=10).std().shift(1)
        hd[c + "_a"] = (hd[c] - m) / s.replace(0, np.nan)
    X = X.merge(hd, on="date", how="left")
    # 5) ruido estaciones
    n = pd.read_parquet(f"{DD}/station_noise/noise_mode_long.parquet", columns=["date", "db", "target"])
    n["date"] = pd.to_datetime(n["date"]).dt.floor("D")
    nd = n.groupby("date").agg(nz_db=("db", "median"), nz_n=("target", "nunique")).reset_index()
    X = X.merge(nd, on="date", how="left")
    # 6) GNSS: media|z30| + n estaciones por dia de emision
    g = pd.read_parquet(f"{DD}/gnss_transients/derived/station_anoms.parquet",
                        columns=["sta", "issue_mjd", "z30"])
    g["date"] = pd.to_datetime(g.issue_mjd - 40587, origin="1970-01-01", unit="D").dt.floor("D")
    gd = g.dropna(subset=["z30"]).groupby("date").agg(
        gnss_z=("z30", lambda s: np.abs(s).mean()), gnss_n=("sta", "nunique")).reset_index()
    X = X.merge(gd, on="date", how="left")
    X = X.sort_values("date").reset_index(drop=True)
    X = X.loc[:, ~X.columns.duplicated()]
    feat = [c for c in X.columns if c != "date"]
    # causalidad: todo lageado 1 dia (dato del dia D usa mediciones <= D-1)
    X[feat] = X[feat].shift(1)
    return X, feat


def main():
    X, feat = load_matrix()
    print(f"matriz: {X.shape} feats={len(feat)} rango={X.date.min().date()}->{X.date.max().date()}")
    tr = X.date < "2013-01-01"
    med = X.loc[tr, feat].median()
    Xs = X[feat].fillna(med)
    print("NaN tras imputar:", int(Xs.isna().sum().sum()))
    sc = StandardScaler().fit(Xs.loc[tr].to_numpy())
    Z = sc.transform(Xs.to_numpy())
    Ztr = Z[tr.to_numpy()]
    iso = IsolationForest(n_estimators=200, contamination=0.05, random_state=41).fit(Ztr)
    X["s_if"] = -iso.score_samples(Z)  # mayor = mas anomalo
    pc = PCA(n_components=0.95, random_state=41).fit(Ztr)
    X["s_pca"] = ((Z - pc.inverse_transform(pc.transform(Z))) ** 2).sum(axis=1)
    print(f"PCA comps={pc.n_components_}")
    # casos: mainshocks M>=6 ZBZ 2013+
    ev = H.get_events(min_mag=6.0, declustered=True)
    cases = ev[(ev.time >= "2013-01-01") & (ev.time < "2026-09-17")].reset_index(drop=True)
    print(f"mainshocks M>=6 test: {len(cases)}")
    big6 = H.get_events(min_mag=6.0, declustered=False).time.to_numpy()
    sday = dict(zip(X.date.dt.date, X.s_if))
    pday = dict(zip(X.date.dt.date, X.s_pca))

    def winscore(day, sdict=sday):
        w = [day - pd.Timedelta(days=i) for i in range(1, 31)]
        v = [sdict.get(d) for d in w]
        return float(np.nanmean(v))

    rows = []
    for _, c in cases.iterrows():
        cb = H.lat_to_bin(np.array([c.latitude]))[0]
        grow = [("case", winscore(c.time.date()))]
        # controles: mismo bin, mes ±15d, año distinto, sin M6 ±90d
        cand = []
        for y in range(2013, 2027):
            if y == c.time.year:
                continue
            for dd in pd.date_range(f"{y}-{c.time.month:02d}-01", periods=31, freq="D"):
                if abs((dd.month - c.time.month)) > 0 or abs(dd.day - c.time.day) > 15:
                    continue
                d = dd.date()
                if d not in sday or d < pd.Timestamp("2013-01-01").date() or d > pd.Timestamp("2026-09-17").date():
                    continue
                if ((big6 > np.datetime64(d - pd.Timedelta(days=90))) &
                        (big6 < np.datetime64(d + pd.Timedelta(days=90)))).any():
                    continue
                cand.append(d)
        cand = RNG.permutation(cand)
        # mismo bin lat: controles heredan bin del caso por diseño (score global; bin solo etiqueta)
        for d in cand[:3]:
            grow.append(("ctrl", winscore(d)))
        if len(grow) == 4:
            rows.append((grow[0][1], [g[1] for g in grow[1:]]))
    print(f"casos con 3 controles: {len(rows)}/{len(cases)}")
    ca = np.array([r[0] for r in rows])
    ct = np.array([r[1] for r in rows])
    stat = ca.mean() - ct.mean()
    print(f"media casos={ca.mean():.4f} controles={ct.mean():.4f} diff={stat:+.4f} (PRIMARIO)")
    cnt = 0
    B = 10000
    for _ in range(B):
        k = RNG.integers(0, 4, size=len(rows))  # rol de "caso" aleatorio intra-set
        a = np.array([([r[0]] + r[1])[k[i]] for i, r in enumerate(rows)])
        b = np.array([[v for j, v in enumerate([r[0]] + r[1]) if j != k[i]] for i, r in enumerate(rows)])
        if a.mean() - b.mean() >= stat:
            cnt += 1
    print(f"p permutacion = {(1 + cnt) / (B + 1):.4f}")
    print("veredicto:", "NULO" if (1 + cnt) / (B + 1) >= 0.05 else "SEÑAL")
    # secundario (nominal): PCA-error, dias pre-mainshock vs resto (test global,
    # sin aparear; aproximado y declarado como tal)
    pv = X[X.date >= "2013-01-01"]
    case_days = set()
    for _, c in cases.iterrows():
        for i in range(1, 31):
            case_days.add((c.time - pd.Timedelta(days=i)).date())
    m1 = pv[pv.date.dt.date.isin(case_days)].s_pca.to_numpy()
    m0 = pv[~pv.date.dt.date.isin(case_days)].s_pca.to_numpy()
    d_obs = m1.mean() - m0.mean()
    pool = np.concatenate([m1, m0])
    n1 = len(m1)
    rng2 = np.random.default_rng(42)
    cnt2 = 0
    for _ in range(2000):
        prm = rng2.permutation(pool)
        if prm[:n1].mean() - prm[n1:].mean() >= d_obs:
            cnt2 += 1
    print(f"[PCA sec] pre-mainshock={m1.mean():.4f} resto={m0.mean():.4f} diff={d_obs:+.4f} p~{(1 + cnt2) / 2001:.4f}")
    # descriptivo: top dias anomalos en test
    Xt = X[X.date >= "2013-01-01"].nlargest(10, "s_if")[["date", "s_if"]]
    print("top-10 dias anomalos test:"); print(Xt.to_string(index=False))


if __name__ == "__main__":
    sys.exit(main())
