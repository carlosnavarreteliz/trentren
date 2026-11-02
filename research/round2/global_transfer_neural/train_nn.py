"""Pretrain (global margins, <2013), fine-tune (Chile <2013) and Chile-only GRU+MLP forecasters.

Usage: python3 -I train_nn.py WORK_DIR STAGE
  STAGE = pretrain | cv_gft_<seed> | cv_co_<seed> | final
Writes WORK_DIR/nn_preds.parquet with columns bin_id, t_start, split and, for model in {gft, co, gzs} and head in
{all55, main55, all6, main6, all7}, the logit `{model}_{head}` (train rows of gft/co = out-of-fold predictions).
"""
import sys, os, time, json
import numpy as np, pandas as pd
import torch, torch.nn as nn

torch.set_num_threads(2)
WORK = sys.argv[1]
STAGE = sys.argv[2]
QUICK = False
HEADS = ["all55", "main55", "all6", "main6", "all7"]
HEAD_W = torch.tensor([0.5, 0.5, 1.0, 1.0, 0.5])
SEQ, NBR = 52, 2
SEEDS = [0, 1, 2] if STAGE == "pretrain" else [0, 1]
GRID_GFT = [1, 2, 4, 8, 16]
GRID_CO = [4, 8, 16, 32]
MAX_PRE_EPOCHS = 12
NEG_KEEP = 0.35
BS = 1024
LOG = open(os.path.join(WORK, "train.log"), "a")


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True); LOG.write(s + "\n"); LOG.flush()


g = np.load(os.path.join(WORK, "global_samples.npz"))
c = np.load(os.path.join(WORK, "chile_samples.npz"))
Xg, Yg, Kg, Wg = g["X"], g["Y"].astype(np.float32), g["key"], g["W"]
Xc, Yc, Kc, Wc = c["X"], c["Y"].astype(np.float32), c["key"], c["W"]
split_c = c["split"]
Xg = np.nan_to_num(Xg); Xc = np.nan_to_num(Xc)
# channel scaling for sequences (fixed constants, not fitted)
CH_SCALE = np.array([1.0, 0.5, 1.0], np.float32)
Wg = Wg * CH_SCALE; Wc = Wc * CH_SCALE

mu_g, sd_g = Xg.mean(0), Xg.std(0) + 1e-6
tr_c = split_c == "train"
mu_c, sd_c = Xc[tr_c].mean(0), Xc[tr_c].std(0) + 1e-6
DT = np.arange(SEQ)[None, :, None]
DB = np.arange(-NBR, NBR + 1)[None, None, :]


def gather(W, K, idx):
    t = K[idx, 1][:, None, None] + DT          # rows t .. t+51 of the extended grid (last = window ending t_start)
    b = K[idx, 3][:, None, None] + DB
    s = W[t, b]                                 # [n, 52, 5, 3]
    return s.reshape(len(idx), SEQ, -1)


class Net(nn.Module):
    def __init__(self, n_tab, ch=15, hid=16):
        super().__init__()
        self.gru = nn.GRU(ch, hid, batch_first=True)
        self.tab = nn.Sequential(nn.Linear(n_tab, 32), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(hid + 32, 32), nn.ReLU(), nn.Dropout(0.1), nn.Linear(32, len(HEADS)))

    def forward(self, seq, tab):
        _, h = self.gru(seq)
        return self.head(torch.cat([h[-1], self.tab(tab)], 1))


def batches(n, rng, bs=BS, shuffle=True):
    o = rng.permutation(n) if shuffle else np.arange(n)
    for s in range(0, n, bs):
        yield o[s:s + bs]


def tensors(X, W, K, mu, sd, idx):
    return (torch.from_numpy(gather(W, K, idx)), torch.from_numpy(((X[idx] - mu) / sd).astype(np.float32)))


def predict(net, X, W, K, Y, mu, sd, idx):
    net.eval(); out = []
    with torch.no_grad():
        for s in range(0, len(idx), 4096):
            ii = idx[s:s + 4096]
            out.append(net(*tensors(X, W, K, mu, sd, ii)).numpy())
    net.train()
    return np.concatenate(out) if out else np.zeros((0, len(HEADS)))


def bce_ll(logit, y):  # per-head summed Bernoulli log-likelihood
    return -(np.logaddexp(0, logit) - y * logit).sum(0)


def train(net, X, W, K, Y, mu, sd, idx, epochs, lr, seed, checkpoints=None, eval_fn=None, wd=1e-4):
    rng = np.random.default_rng(seed)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    lossf = nn.BCEWithLogitsLoss(reduction="none")
    out = {}
    for ep in range(1, epochs + 1):
        for bi in batches(len(idx), rng):
            ii = idx[bi]
            seq, tab = tensors(X, W, K, mu, sd, ii)
            loss = (lossf(net(seq, tab), torch.from_numpy(Y[ii])) * HEAD_W).sum(1).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        if checkpoints and ep in checkpoints:
            out[ep] = eval_fn(net)
    return out


def init_bias(net, Y):
    p = Y.mean(0).clip(1e-5, 0.5)
    with torch.no_grad():
        net.head[-1].bias.copy_(torch.from_numpy(np.log(p / (1 - p)).astype(np.float32)))


t0 = time.time()
idx_tr_c = np.where(tr_c)[0]
tw = Kc[idx_tr_c, 1]
f_edges = np.quantile(tw, np.linspace(0, 1, 6))
fold = np.clip(np.searchsorted(f_edges, tw, side="right") - 1, 0, 4)
PURGE = 26
CFG = {"gft": dict(grid=GRID_GFT, lr=3e-4, mu=mu_g, sd=sd_g), "co": dict(grid=GRID_CO, lr=2e-3, mu=mu_c, sd=sd_c)}
PRE_FN = os.path.join(WORK, "pretrained.pt")


def make_net(kind, seed):
    torch.manual_seed(1000 + seed)
    net = Net(Xc.shape[1])
    if kind in ("gft", "gzs"):
        net.load_state_dict(torch.load(PRE_FN)[seed])
    else:
        init_bias(net, Yc[idx_tr_c])
    return net


# ------------------------------------------------------------------ 1. pretraining on the global margins
if STAGE == "pretrain":
    rng0 = np.random.default_rng(12345)
    year = (pd.to_datetime(g["t_starts"]).year.values)[Kg[:, 1]]
    unit = Kg[:, 0] * 100 + (year - 1976) // 2
    units = np.unique(unit)
    val_units = rng0.choice(units, size=int(0.2 * len(units)), replace=False)
    is_val = np.isin(unit, val_units)
    idx_tr_g, idx_va_g = np.where(~is_val)[0], np.where(is_val)[0]
    pos_tr = idx_tr_g[Yg[idx_tr_g].sum(1) > 0]
    neg_tr = idx_tr_g[Yg[idx_tr_g].sum(1) == 0]
    log(f"global rows train {len(idx_tr_g)} (pos rows {len(pos_tr)}) val {len(idx_va_g)}; "
        f"pos train {Yg[idx_tr_g].sum(0)}; val units {len(val_units)}/{len(units)}")
    pre_states, pre_log = {}, {}
    lossf = nn.BCEWithLogitsLoss(reduction="none")
    for seed in SEEDS:
        torch.manual_seed(seed)
        net = Net(Xg.shape[1]); init_bias(net, Yg[idx_tr_g])
        rng = np.random.default_rng(seed)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
        best, best_state, bad, hist = -np.inf, None, 0, []
        for ep in range(1, MAX_PRE_EPOCHS + 1):
            ep_idx = np.r_[pos_tr, neg_tr[rng.random(len(neg_tr)) < NEG_KEEP]]
            wrow = np.where(Yg[ep_idx].sum(1) > 0, 1.0, 1.0 / NEG_KEEP).astype(np.float32)
            for bi in batches(len(ep_idx), rng):
                ii = ep_idx[bi]
                seq, tab = tensors(Xg, Wg, Kg, mu_g, sd_g, ii)
                l = (lossf(net(seq, tab), torch.from_numpy(Yg[ii])) * HEAD_W).sum(1)
                loss = (l * torch.from_numpy(wrow[bi])).mean()
                opt.zero_grad(); loss.backward(); opt.step()
            lv = bce_ll(predict(net, Xg, Wg, Kg, Yg, mu_g, sd_g, idx_va_g), Yg[idx_va_g])
            score = float((lv * HEAD_W.numpy()).sum())
            hist.append(dict(ep=ep, val_ll=lv.round(2).tolist(), score=score))
            log(f"pretrain seed {seed} ep {ep} val LL/event/head {np.round(lv / Yg[idx_va_g].sum(0), 3)} "
                f"score {score:.1f} {time.time()-t0:.0f}s")
            if score > best:
                best, bad = score, 0
                best_state = {k: v.clone() for k, v in net.state_dict().items()}
            else:
                bad += 1
                if bad >= 3:
                    break
        pre_states[seed] = best_state
        pre_log[seed] = hist
    torch.save(pre_states, PRE_FN)
    # reference: val LL of a per-head constant-rate model, to see that pretraining learned something
    p = Yg[idx_tr_g].mean(0)
    const = (Yg[idx_va_g] * np.log(p) + (1 - Yg[idx_va_g]) * np.log1p(-p)).sum(0)
    json.dump(dict(hist=pre_log, const_val_ll=const.round(2).tolist(), heads=HEADS),
              open(os.path.join(WORK, "pretrain_log.json"), "w"), indent=1)

# ------------------------------------------------------------------ 2. Chile CV (epoch count; OOF predictions)
elif STAGE.startswith("cv_"):
    _, kind, seed = STAGE.split("_"); seed = int(seed)
    cfg = CFG[kind]; grid = cfg["grid"]
    oof_k = {e: np.zeros((len(idx_tr_c), len(HEADS)), np.float32) for e in grid}
    for f in range(5):
        va = fold == f
        lo, hi = tw[va].min(), tw[va].max()
        trm = (~va) & ((tw < lo - PURGE) | (tw > hi + PURGE))
        net = make_net(kind, seed)
        ii_va = idx_tr_c[va]
        ev = lambda n: predict(n, Xc, Wc, Kc, Yc, cfg["mu"], cfg["sd"], ii_va)
        outs = train(net, Xc, Wc, Kc, Yc, cfg["mu"], cfg["sd"], idx_tr_c[trm], max(grid), cfg["lr"],
                     seed=100 * seed + f, checkpoints=set(grid), eval_fn=ev)
        for e in grid:
            oof_k[e][va] = outs[e]
        log(f"{kind} seed {seed} fold {f} done {time.time()-t0:.0f}s")
    np.savez(os.path.join(WORK, f"oof_{kind}_{seed}.npz"), **{f"e{e}": oof_k[e] for e in grid})

# ------------------------------------------------------------------ 3. epoch choice + final fits + zero-shot
elif STAGE == "final":
    results, final = {}, {}
    all_idx = np.arange(len(Xc))
    for kind, cfg in CFG.items():
        grid = cfg["grid"]
        oofs = [np.load(os.path.join(WORK, f"oof_{kind}_{s}.npz")) for s in SEEDS]
        oof_k = {e: np.mean([o[f"e{e}"] for o in oofs], 0) for e in grid}
        sc = {e: bce_ll(oof_k[e], Yc[idx_tr_c]) for e in grid}
        tot = {e: float(sc[e][2] + sc[e][3]) for e in grid}
        best_e = max(tot, key=tot.get)
        results[kind] = dict(oof_ll_by_epoch={e: np.round(sc[e], 2).tolist() for e in grid}, score=tot,
                             best_epochs=best_e)
        log(kind, "OOF LL (all6+main6) by epochs:", tot, "-> best", best_e)
        acc = np.zeros((len(Xc), len(HEADS)))
        for seed in SEEDS:
            net = make_net(kind, seed)
            train(net, Xc, Wc, Kc, Yc, cfg["mu"], cfg["sd"], idx_tr_c, best_e, cfg["lr"], seed=777 + seed)
            acc += predict(net, Xc, Wc, Kc, Yc, cfg["mu"], cfg["sd"], all_idx) / len(SEEDS)
        acc_full_fit_train = acc[idx_tr_c].copy()
        acc[idx_tr_c] = oof_k[best_e]  # train rows -> out-of-fold predictions for stacking
        final[kind] = acc
        results[kind]["insample_minus_oof_ll_train"] = (bce_ll(acc_full_fit_train, Yc[idx_tr_c])
                                                         - sc[best_e]).round(2).tolist()
        log(f"{kind} final done {time.time()-t0:.0f}s")
    acc = np.zeros((len(Xc), len(HEADS)))
    for seed in SEEDS:
        acc += predict(make_net("gzs", seed), Xc, Wc, Kc, Yc, mu_g, sd_g, all_idx) / len(SEEDS)
    final["gzs"] = acc
    results["gzs"] = dict(train_ll=bce_ll(acc[idx_tr_c], Yc[idx_tr_c]).round(2).tolist())
    df = pd.DataFrame({"bin_id": c["bin_id"], "t_start": c["t_start"], "split": split_c})
    for kind, A in final.items():
        for j, h in enumerate(HEADS):
            df[f"{kind}_{h}"] = A[:, j]
    df.to_parquet(os.path.join(WORK, "nn_preds.parquet"), index=False)
    json.dump(results, open(os.path.join(WORK, "train_results.json"), "w"), indent=1, default=str)
    log(f"final done {time.time()-t0:.0f}s")
