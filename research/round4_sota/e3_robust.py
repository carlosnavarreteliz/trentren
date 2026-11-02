"""E3 robustez: panel M5.5, leave-one-year-out, ablacion. Reusa logica de e3_kernels."""
import sys
sys.path.insert(0, "research")
sys.path.insert(0, "research/round4_sota")
import numpy as np
import pandas as pd
import harness as H
from e3_kernels import depth_omori


def build_X():
    panel = H.get_panel(step_days=1, horizon_days=1, target_mag=5.0)
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
    return X


def main():
    X = build_X()
    # 1) panel M5.5 (mismo X: horizonte 1d identico)
    p55 = H.get_panel(step_days=1, horizon_days=1, target_mag=5.5)
    res55 = H.evaluate(X, target="y_all", model="logit", panel=p55, return_preds=True)
    sur55 = H.surrogate_test(X, n=200, mode="phase_randomize", target="y_all", panel=p55)
    print(f"M5.5 y_all: {H.summarize(res55, sur55)}")
    print("   CI:", H.block_bootstrap_ci(res55["preds"]))
    # 2) LOOYO en M5.0
    p50 = H.get_panel(step_days=1, horizon_days=1, target_mag=5.0)
    res = H.evaluate(X, target="y_all", model="logit", panel=p50, return_preds=True)
    pr = res["preds"]
    print("LOOYO (IG sin cada año):")
    for y in sorted(pr.year.unique()):
        q = pr[pr.year != y]
        if q.y.sum() == 0:
            continue
        ll = lambda a: (q.y * np.log(np.clip(q[a], 1e-9, 1 - 1e-9)) +
                        (1 - q.y) * np.log1p(-np.clip(q[a], 1e-9, 1 - 1e-9))).sum() / q.y.sum()
        print(f"  sin {y}: IG={ll('p_full') - ll('p_base'):+.4f} (pos restantes={int(q.y.sum())})")
    # 3) ablacion: profundidad vs conteos cortos
    om = [c for c in X.columns if c.startswith("x_om")]
    ct = [c for c in X.columns if c.startswith("x_cnt")]
    for name, cols in [("solo-profundidad", om), ("solo-conteos1-3d", ct)]:
        r = H.evaluate(X[["bin_id", "t_start"] + cols], target="y_all", model="logit", panel=p50)
        print(f"ablacion {name}: IG={r['ig_per_event']:+.4f}")
    print("coefs extra (full):", res.get("coef_extra"))


if __name__ == "__main__":
    sys.exit(main())
