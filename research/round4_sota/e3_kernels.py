"""E3: kernels ETAS por profundidad + memoria corta vs baseline harness."""
import sys
sys.path.insert(0, "research")
import numpy as np
import pandas as pd
import harness as H

RNG = np.random.default_rng(21)


def depth_omori(t_issue, ev, edges, h, p, c, a, s0):
    te = ev.t_days.to_numpy()
    me = ev.mag.to_numpy()
    lat = ev.latitude.to_numpy()
    nb = len(edges) - 1
    w = 10 ** (a * (me - H.M_MODEL))
    Rmat = w[:, None] * H._spatial_weights(lat, me, edges, s0)
    acc = np.zeros((len(t_issue), nb))
    chunk = max(1, int(2e7 // max(1, len(te))))
    for s in range(0, len(t_issue), chunk):
        tt = t_issue[s:s + chunk]
        dt = tt[:, None] - te[None, :]
        K = np.where(dt > 0, H._omori_integral(np.maximum(dt, 1e-9), h, p, c), 0.0)
        acc[s:s + chunk] = K @ Rmat
    return acc


def main():
    panel = H.get_panel(step_days=1, horizon_days=1, target_mag=5.0)
    print(f"panel: {panel.shape} test_pos_yall={int(panel[panel.split=='test'].y_all.sum())} "
          f"test_pos_ymain={int(panel[panel.split=='test'].y_main.sum())}")
    ev = H.get_events(min_mag=H.M_MODEL)
    ev = ev.assign(bin_id=H.lat_to_bin(ev.latitude.to_numpy(), 1.0))
    edges = np.arange(H.LAT_MIN, H.LAT_MAX + 1e-9, 1.0)
    t_starts = np.sort(panel.t_start.unique())
    ti = H.to_days(t_starts)
    nb = len(edges) - 1
    feats = {}
    for name, sub in [("iface", ev[ev.depth <= 60]), ("intra", ev[ev.depth > 60])]:
        feats[f"x_om_{name}_fast"] = depth_omori(ti, sub, edges, 1.0, 1.3, 0.02, 1.1, 0.3)
        feats[f"x_om_{name}_slow"] = depth_omori(ti, sub, edges, 1.0, 0.9, 0.5, 1.1, 1.0)
    feats["x_cnt_nbr_1d"] = H._count_features(ti, ev, nb, [1], 1, "x_cnt_nbr")["x_cnt_nbr_1d"]
    feats["x_cnt_nbr_3d"] = H._count_features(ti, ev, nb, [3], 1, "x_cnt_nbr")["x_cnt_nbr_3d"]
    T, B = np.meshgrid(np.arange(len(ti)), np.arange(nb), indexing="ij")
    X = pd.DataFrame({"bin_id": B.ravel().astype(np.int16), "t_start": t_starts[T.ravel()]})
    for k, v in feats.items():
        X[k] = np.log1p(v.ravel()) if k.startswith("x_cnt") else np.log10(v.ravel() + 1e-3)
    print("features:", [c for c in X.columns if c.startswith("x_")])
    for tgt in ["y_all", "y_main"]:
        res = H.evaluate(X, target=tgt, model="logit", panel=panel, return_preds=True)
        sur = H.surrogate_test(X, n=200, mode="phase_randomize", target=tgt, panel=panel)
        print(f"--- {tgt}: {H.summarize(res, sur)}")
        print("   CI:", H.block_bootstrap_ci(res["preds"]))


if __name__ == "__main__":
    sys.exit(main())
