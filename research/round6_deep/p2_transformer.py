"""P2: Transformer encoder sobre secuencias Riedel (mismo protocolo M2)."""
import sys
sys.path.insert(0, "research/round6_deep")
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score
from m2_seqgru import build, ll_bern

CLEAN = "research/data/round2_enhanced_catalog/riedel/primary_clean.parquet"
RNG = np.random.default_rng(5)


def main():
    import torch
    import torch.nn as nn
    torch.manual_seed(9)
    d = pd.read_parquet(CLEAN)
    d["time"] = pd.to_datetime(d["time"])
    print("construyendo secuencias...", flush=True)
    X, Hd, Y, Tm, SM = build(d)
    Tm = pd.to_datetime(Tm).to_numpy()
    print(f"triggers: {len(Y)} pos={Y.sum()}")
    te_m = Tm >= np.datetime64("2020-01-01")
    Xtr, Xte, Htr, Hte = X[~te_m], X[te_m], Hd[~te_m], Hd[te_m]
    ytr, yte = Y[~te_m], Y[te_m]
    nval = int(0.15 * len(ytr))
    Xva, Hva, yva = Xtr[-nval:], Htr[-nval:], ytr[-nval:]
    Xtr, Htr, ytr = Xtr[:-nval], Htr[:-nval], ytr[:-nval]
    mu, sd = Xtr.reshape(-1, 9).mean(0), Xtr.reshape(-1, 9).std(0) + 1e-9

    def ten(x, h):
        return (torch.tensor((x - mu) / sd, dtype=torch.float32),
                torch.tensor(h, dtype=torch.float32),
                torch.tensor((x[:, :, 0] != 0), dtype=torch.bool))

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Linear(9, 64)
            enc = nn.TransformerEncoderLayer(64, 4, 128, dropout=0.2, batch_first=True)
            self.enc = nn.TransformerEncoder(enc, 2)
            self.fc = nn.Sequential(nn.Linear(64 + 2, 32), nn.ReLU(), nn.Dropout(0.2),
                                    nn.Linear(32, 1))
        def forward(self, x, h, real):
            L = x.shape[1]
            pos = torch.zeros(L, 64)
            pe = torch.arange(L).float().unsqueeze(1)
            div = torch.exp(torch.arange(0, 64, 2).float() * (-np.log(10000.0) / 64))
            pos[:, 0::2] = torch.sin(pe * div)
            pos[:, 1::2] = torch.cos(pe * div)
            z = self.enc(self.emb(x) + pos, src_key_padding_mask=~real)
            w = real.float().unsqueeze(-1)
            z = (z * w).sum(1) / w.sum(1).clamp(min=1)
            return self.fc(torch.cat([z, h], dim=1)).squeeze(-1)

    net = Net()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    pw = torch.tensor((len(ytr) - ytr.sum()) / max(ytr.sum(), 1))
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
    xtr = ten(Xtr, Htr)
    xva = ten(Xva, Hva)
    xte = ten(Xte, Hte)
    ytr_t = torch.tensor(ytr, dtype=torch.float32)
    yva_t = torch.tensor(yva, dtype=torch.float32)
    best, bad, state = 1e9, 0, None
    for ep in range(15):
        net.train()
        perm = torch.randperm(len(ytr))
        for i in range(0, len(ytr), 256):
            idx = perm[i:i + 256]
            opt.zero_grad()
            out = net(xtr[0][idx], xtr[1][idx], xtr[2][idx])
            loss = lossf(out, ytr_t[idx])
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            vl = lossf(net(*xva), yva_t).item()
        if vl < best - 1e-4:
            best, bad, state = vl, 0, {k: v.clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= 3:
                break
    net.load_state_dict(state)
    net.eval()
    with torch.no_grad():
        p_tf = torch.sigmoid(net(*xte)).numpy()
    SMtr = SM[~te_m][:len(ytr)]
    scb = StandardScaler().fit(SMtr)
    clf = LogisticRegression(C=1.0).fit(scb.transform(SMtr), ytr)
    p_base = clf.predict_proba(scb.transform(SM[te_m]))[:, 1]
    ig = (ll_bern(yte, p_tf) - ll_bern(yte, p_base)).sum() / yte.sum()
    print(f"test n={len(yte)} pos={yte.sum()}")
    print(f"IG Transformer vs logit = {ig:+.4f} (PRIMARIA)")
    print(f"AUC tf={roc_auc_score(yte, p_tf):.3f} base={roc_auc_score(yte, p_base):.3f} "
          f"(ref GRU M2: 0.618 / 0.663)")
    print(f"PR-AUC tf={average_precision_score(yte, p_tf):.3f} base={yte.mean():.3f}")
    mo = pd.to_datetime(Tm[te_m]).to_period("M").astype(str).to_numpy()
    um = np.unique(mo)
    dll = ll_bern(yte, p_tf) - ll_bern(yte, p_base)
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
        cnt += ((ll_bern(yp, p_tf) - ll_bern(yp, p_base)).sum() / yp.sum()) >= ig
    print(f"p permutacion = {(1 + cnt) / 5001:.4f}")
    print("veredicto:", "NULO" if (lo <= 0) else "SEÑAL")


if __name__ == "__main__":
    sys.exit(main())
