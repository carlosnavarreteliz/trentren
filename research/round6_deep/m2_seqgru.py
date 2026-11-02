"""M2: SeqGRU escalar ML>=4 -> ML>=5/7d sobre secuencias Riedel. Vease PREREG."""
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score

CLEAN = "research/data/round2_enhanced_catalog/riedel/primary_clean.parquet"
RNG = np.random.default_rng(5)
R_KM = 6371.0
TYPES = ["Interplate", "Intraslab", "Outerrise", "Crustal", "NC"]
SEQ = 32


def hav_km(la1, lo1, la2, lo2):
    la1, lo1, la2, lo2 = map(np.radians, (la1, lo1, la2, lo2))
    h = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 2 * R_KM * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


def build(d):
    tm = d.tmin.to_numpy()
    la, lo, mg = (d[c].to_numpy() for c in ["Lat", "Lon", "Ml"])
    dp = d.Dep.to_numpy()
    ty = d.Type.to_numpy()
    trig = d[(d.Ml >= 4.0) & (d.time >= "2017-06-01") & (d.time < "2021-11-15")].reset_index()
    X, Hd, Y, Tm, SM = [], [], [], [], []
    for _, x in trig.iterrows():
        i0 = int(x["index"])
        lo90 = np.searchsorted(tm, x.tmin - 90 * 1440, side="left")
        pre = np.arange(lo90, i0)
        dd = hav_km(x.Lat, x.Lon, la[pre], lo[pre]) if len(pre) else np.array([])
        near_all = pre[(dd <= 150) & (mg[pre] >= 2.5)]
        dt_all = x.tmin - tm[near_all]
        SM.append([x.Ml, float(np.log1p(np.sum(dt_all <= 1440))),
                   float(np.log1p(np.sum(dt_all <= 7 * 1440)))])
        near = near_all
        if len(near) < 5:
            continue
        near = near[-SEQ:]
        prev_t = tm[near]
        dt_min = np.diff(np.r_[prev_t[0] - 1440.0, prev_t])
        seq = np.zeros((SEQ, 9), float)
        ti = {t: i for i, t in enumerate(TYPES)}
        for k, j in enumerate(near):
            seq[SEQ - len(near) + k] = [
                np.log10(dt_min[k] + 1), hav_km(x.Lat, x.Lon, la[j], lo[j]) / 150.0,
                (mg[j] - 2.5) / 2.0, np.clip(dp[j] / 150.0, 0, 2),
                *[1.0 if ty[j] == t else 0.0 for t in TYPES]]
        hi = np.searchsorted(tm, x.tmin + 7 * 1440, side="right")
        fut = np.arange(i0 + 1, hi)
        y = 0
        if len(fut):
            cand = fut[mg[fut] >= 5.0]
            if len(cand):
                y = int((hav_km(x.Lat, x.Lon, la[cand], lo[cand]) <= 150).any())
        # resumen baseline (E1): totales 24h/7d ML>=2.5 en 150km
        w7 = near
        dt7 = x.tmin - tm[w7]
        X.append(seq)
        Hd.append([(x.Ml - 4.0) / 1.5, np.clip(x.Dep / 150.0, 0, 2)])
        Y.append(y)
        Tm.append(x.time)
    return np.array(X), np.array(Hd), np.array(Y), np.array(Tm), np.array(SM)


def ll_bern(y, p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return y * np.log(p) + (1 - y) * np.log1p(-p)


def main():
    import torch
    import torch.nn as nn
    torch.manual_seed(5)
    d = pd.read_parquet(CLEAN)
    d["time"] = pd.to_datetime(d["time"])
    print("construyendo secuencias...", flush=True)
    X, Hd, Y, Tm, SM = build(d)
    Tm = pd.to_datetime(Tm).to_numpy()
    print(f"triggers con seq: {len(Y)} pos={Y.sum()} rate={Y.mean():.3f}")
    te_m = Tm >= np.datetime64("2020-01-01")
    tr_m = ~te_m
    Xtr, Xte, Htr, Hte = X[tr_m], X[te_m], Hd[tr_m], Hd[te_m]
    ytr, yte = Y[tr_m], Y[te_m]
    # val = ult. 15% temporal del train
    nval = int(0.15 * len(ytr))
    Xva, Hva, yva = Xtr[-nval:], Htr[-nval:], ytr[-nval:]
    Xtr, Htr, ytr = Xtr[:-nval], Htr[:-nval], ytr[:-nval]
    mu, sd = Xtr.reshape(-1, 9).mean(0), Xtr.reshape(-1, 9).std(0) + 1e-9

    def ten(x, h):
        return (torch.tensor((x - mu) / sd, dtype=torch.float32),
                torch.tensor(h, dtype=torch.float32))

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.gru = nn.GRU(9, 64, num_layers=2, batch_first=True, dropout=0.2)
            self.fc = nn.Sequential(nn.Linear(64 + 2, 32), nn.ReLU(), nn.Dropout(0.2),
                                    nn.Linear(32, 1))
        def forward(self, x, h):
            _, hn = self.gru(x)
            return self.fc(torch.cat([hn[-1], h], dim=1)).squeeze(-1)

    net = Net()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    pw = torch.tensor((len(ytr) - ytr.sum()) / max(ytr.sum(), 1))
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
    xtr, htr = ten(Xtr, Htr)
    xva, hva = ten(Xva, Hva)
    ytr_t = torch.tensor(ytr, dtype=torch.float32)
    yva_t = torch.tensor(yva, dtype=torch.float32)
    best, bad, best_state = 1e9, 0, None
    n = len(ytr)
    for ep in range(15):
        net.train()
        perm = torch.randperm(n)
        for i in range(0, n, 256):
            idx = perm[i:i + 256]
            opt.zero_grad()
            out = net(xtr[idx], htr[idx])
            loss = lossf(out, ytr_t[idx])
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            vl = lossf(net(xva, hva), yva_t).item()
        if vl < best - 1e-4:
            best, bad, best_state = vl, 0, {k: v.clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= 3:
                break
    net.load_state_dict(best_state)
    net.eval()
    xte, hte = ten(Xte, Hte)
    with torch.no_grad():
        p_gru = torch.sigmoid(net(xte, hte)).numpy()
    # baseline preregistrado (E1): logit sobre [mag trigger, tot24, tot7]
    SMtr = SM[tr_m][:len(ytr)]
    SMte = SM[te_m]
    scb = StandardScaler().fit(SMtr)
    clf = LogisticRegression(C=1.0).fit(scb.transform(SMtr), ytr)
    p_base = clf.predict_proba(scb.transform(SMte))[:, 1]
    ig = (ll_bern(yte, p_gru) - ll_bern(yte, p_base)).sum() / yte.sum()
    print(f"test n={len(yte)} pos={yte.sum()}")
    print(f"IG GRU vs logit-resumen = {ig:+.4f} (PRIMARIA)")
    print(f"AUC gru={roc_auc_score(yte, p_gru):.3f} base={roc_auc_score(yte, p_base):.3f}")
    print(f"PR-AUC gru={average_precision_score(yte, p_gru):.3f} base={yte.mean():.3f}")
    mo = pd.to_datetime(Tm[te_m]).to_period("M").astype(str).to_numpy()
    um = np.unique(mo)
    dll = ll_bern(yte, p_gru) - ll_bern(yte, p_base)
    boots = []
    for _ in range(2000):
        ms = RNG.choice(um, size=len(um), replace=True)
        m = np.isin(mo, ms)
        if yte[m].sum() > 0:
            boots.append(dll[m].sum() / yte[m].sum())
    boots = np.array(boots)
    lo, hi = np.percentile(boots, 2.5), np.percentile(boots, 97.5)
    print(f"IC95% mensual: [{lo:+.4f}, {hi:+.4f}]")
    cnt = 0
    for _ in range(5000):
        yp = RNG.permutation(yte)
        if yp.sum() == 0:
            continue
        cnt += ((ll_bern(yp, p_gru) - ll_bern(yp, p_base)).sum() / yp.sum()) >= ig
    p = (1 + cnt) / 5001
    print(f"p permutacion = {p:.4f}")
    print("veredicto:", "NULO" if (lo <= 0 or p >= 0.05) else "SEÑAL (control adverso requerido)")
    np.savez("research/round6_deep/m2_preds.npz", p_gru=p_gru, p_base=p_base, y=yte, t=Tm[te_m])


if __name__ == "__main__":
    sys.exit(main())
