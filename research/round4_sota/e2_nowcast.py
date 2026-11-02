"""E2: nowcasting de Rundle para Chile (round4_sota). Vease PREREG_E2_NOWCAST.md."""
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

RNG = np.random.default_rng(11)


def main():
    u = pd.read_csv("data/raw/chile_usgs_1900_2026.csv", usecols=["time", "latitude", "longitude", "mag", "type"])
    u = u[u["type"] == "earthquake"]
    u["time"] = pd.to_datetime(u["time"], format="ISO8601", utc=True).dt.tz_convert(None)
    u = u[u.latitude.between(-46, -17) & u.longitude.between(-78, -66)].dropna(subset=["mag"])
    u = u.sort_values("time").reset_index(drop=True)
    BIG, TDAYS = 6.5, 90  # enmienda registrada: label original M6.0/180d era 95% positivo
    small = u[(u.mag >= 4.7) & (u.time >= "1990-01-01")].reset_index(drop=True)
    big_times = small[small.mag >= BIG].time.to_numpy()
    print(f"pequeños M>=4.7: {len(small)} | grandes M>={BIG}: {len(big_times)}")
    # ciclos completados en train (<=2012): conteo de pequenos entre grandes
    bidx = small[small.mag >= BIG].index.to_numpy()
    cyc = np.diff(bidx)  # nº pequenos entre grandes consecutivos (incluye el grande final)
    cyc_tr = cyc[small.time.to_numpy()[bidx[1:]] < np.datetime64("2013-01-01")]
    print(f"ciclos train: {len(cyc_tr)} mediana={np.median(cyc_tr):.0f}")
    # EPS en cada evento pequeno de test: n actual desde ultimo grande -> CDF train
    te = small[small.time >= "2013-01-01"].reset_index(drop=True)
    last_big = np.searchsorted(small.time.to_numpy(), te.time.to_numpy())  # posicion aprox
    # n actual: eventos desde el ultimo M>=6 anterior (excluyendolo)
    n_cur, since = [], []
    for t in te.time:
        pb = small[(small.time < t) & (small.mag >= BIG)]
        t_last = pb.time.max() if len(pb) else pd.Timestamp("1990-01-01")
        n_cur.append(int(((small.time >= t_last) & (small.time < t) & (small.mag >= 4.7)).sum()))
        since.append((t - t_last).total_seconds() / 86400.0)
    te["n_cur"] = n_cur
    te["since"] = since
    te["eps"] = np.searchsorted(np.sort(cyc_tr), te.n_cur.to_numpy(), side="right") / len(cyc_tr)
    # label: M>=6 en prox 180d (excluir ultimos 180d del catalogo)
    end = small.time.max() - pd.Timedelta(days=TDAYS)
    te = te[te.time <= end].reset_index(drop=True)
    bt = pd.DatetimeIndex(big_times)
    te["y"] = [int(((bt > t) & (bt <= t + pd.Timedelta(days=TDAYS))).any())
               for t in te.time]
    print(f"test: n={len(te)} pos={te.y.sum()} rate={te.y.mean():.3f}")
    auc = roc_auc_score(te.y, te.eps)
    auc_s = roc_auc_score(te.y, te.since)
    print(f"AUC EPS={auc:.3f} | AUC dias-desde-ultimo={auc_s:.3f} | corr(EPS,since)={te[['eps','since']].corr().iloc[0,1]:.3f}")
    cnt = sum(roc_auc_score(RNG.permutation(te.y.to_numpy()), te.eps.to_numpy()) >= auc
              for _ in range(5000))
    p = (1 + cnt) / 5001
    print(f"p permutacion = {p:.4f}")
    print("veredicto:", "NULO" if (p >= 0.05 or auc <= auc_s) else "SEÑAL (EPS supera tiempo)")


if __name__ == "__main__":
    sys.exit(main())
