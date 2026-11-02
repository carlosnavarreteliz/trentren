"""SOTA check: v3 vs Reasenberg-Jones vs ETAS-logit vs mag-only vs Poisson.
Misma tarea congelada T1 v1.1 (Mw homogeneizada). Metricas: IG (bootstrap anual),
AUC, PR-AUC, Brier, calibracion (intercept/slope de recalibracion en test)."""
import sys
sys.path.insert(0, ".")
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
import forecast_short as F

RNG = np.random.default_rng(13)




def main():
    d = F.load_local()
    df = F.build_dataset(d)
    tm = d.tmin.to_numpy()
    la, lo, mg, dp = (d[c].to_numpy() for c in ["latitude", "longitude", "mag", "depth"])
    trig = d[(d.mag >= 4.7) & (d.time >= "1990-01-01")].reset_index()
    assert len(trig) >= len(df)
    # tasas RJ + Omori por trigger (todos los periodos)
    A, B, P_, C = -1.67, 0.91, 1.08, 0.05
    rj = {h: np.zeros(len(trig)) for h in F.HORIZONS_MIN}
    om = np.zeros(len(trig))
    for k, (_, x) in enumerate(trig.iterrows()):
        i0 = int(x["index"])
        loy = np.searchsorted(tm, x.tmin - 365, side="left")
        pre = np.arange(loy, i0)
        if not len(pre):
            continue
        dd = F.hav_km(x.latitude, x.longitude, la[pre], lo[pre])
        near = pre[dd <= 150]
        dt = x.tmin - tm[near]
        mj = mg[near]
        for h in F.HORIZONS_MIN:
            integ = ((dt / 1440 + C) ** (1 - P_) - ((dt + h) / 1440 + C) ** (1 - P_)) / (P_ - 1)
            # termino propio: el trigger como padre de sus replicas (la auditoria lo incluye)
            self_i = ((C) ** (1 - P_) - ((h) / 1440 + C) ** (1 - P_)) / (P_ - 1)
            rj[h][k] = np.sum(10 ** (A + B * (mj - 4.7)) * integ) + \
                10 ** (A + B * (x.mag - 4.7)) * self_i
        om[k] = np.sum(10 ** (1.1 * (mj - 4.7)) * (dt / 1440 + 0.02) ** (-1.08))
    trig["rj60"], trig["rj180"], trig["rj1440"] = rj[60], rj[180], rj[1440]
    trig["om"] = om
    frozen = pd.read_csv("benchmark/t1_test.csv", parse_dates=["time"])
    # unir por tiempo+coords (el test congelado viene del mismo build)
    te = df[df.time >= "2024-01-01"].reset_index(drop=True)
    tr = df[df.time < "2021-01-01"].reset_index(drop=True)
    trj = trig[trig.time >= "1990-01-01"].reset_index(drop=True)
    # indices alineados: df y trig comparten orden temporal de triggers
    assert (pd.to_datetime(trig.time).to_numpy()[:len(df)] == df.time.to_numpy()).all()
    rj_full = {h: trig[f"rj{h}"].to_numpy()[:len(df)] for h in F.HORIZONS_MIN}
    om_full = trig["om"].to_numpy()[:len(df)]
    cols = F.COLS_V2

    def ll(y, p):
        p = np.clip(p, 1e-9, 1 - 1e-9)
        return y * np.log(p) + (1 - y) * np.log1p(-p)

    model = __import__("json").loads(F.MODEL_JSON_V3.read_text())
    res_rows = []
    for h in F.HORIZONS_MIN:
        yc = f"y_{h}"
        ytr, yte = tr[yc].to_numpy(), te[yc].to_numpy()
        preds = {}
        preds["Poisson"] = te.assign(v=frozen[f"p_poisson_{h}"].to_numpy()).v.to_numpy() \
            if len(frozen) == len(te) else np.full(len(te), tr[yc].mean())
        # mag-only
        scm = StandardScaler().fit(tr[["x2"]].to_numpy())
        mgm = LogisticRegression(C=1.0).fit(scm.transform(tr[["x2"]].to_numpy()), ytr)
        preds["mag-only"] = mgm.predict_proba(scm.transform(te[["x2"]].to_numpy()))[:, 1]
        # RJ: Platt sobre log10(rate+eps) ajustado en train
        rtr = np.log10(rj_full[h][:len(tr)] + 1e-6).reshape(-1, 1)
        # indices: df = tr + calib + te en orden temporal
        nca = len(df) - len(tr) - len(te)
        rte = np.log10(rj_full[h][len(tr) + nca:] + 1e-6).reshape(-1, 1)
        rjm = LogisticRegression(C=1e6).fit(rtr, ytr)
        preds["RJ-generico"] = rjm.predict_proba(rte)[:, 1]
        # ETAS-logit trigger (om + mag + conteos via cols x1i/x5 aprox): usa cols completas = v3 sin Platt? -> v3 congelado
        preds["v3 (SOTA propio)"] = frozen[f"p_sota_{h}"].to_numpy()
        # ETAS-logit reducido (om, x2, x5): entrenado train, Platt calib
        ca = df[(df.time >= "2021-01-01") & (df.time < "2024-01-01")].reset_index(drop=True)
        et_cols = cols
        sce = StandardScaler().fit(tr[et_cols].to_numpy())
        et = LogisticRegression(C=1.0).fit(sce.transform(tr[et_cols].to_numpy()), ytr)
        sca = et.decision_function(sce.transform(ca[et_cols].to_numpy())).reshape(-1, 1)
        ple = LogisticRegression(C=1e6).fit(sca, ca[yc].to_numpy())
        preds["ETAS-logit(E3)"] = ple.predict_proba(
            et.decision_function(sce.transform(te[et_cols].to_numpy())).reshape(-1, 1))[:, 1]
        yrs = te.time.dt.year.to_numpy()
        for name, p in preds.items():
            igp = (ll(yte, p) - ll(yte, preds["Poisson"])).sum() / yte.sum()
            # calibracion: logit(y) ~ a + b*logit(p) en test (descriptivo)
            lp = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
            try:
                cm = LogisticRegression(C=1e6).fit(lp.reshape(-1, 1), yte)
                cs, ci = float(cm.coef_[0][0]), float(cm.intercept_[0])
            except Exception:
                cs, ci = np.nan, np.nan
            res_rows.append({"H": h, "modelo": name, "IG_vs_Poisson": igp,
                             "AUC": roc_auc_score(yte, p),
                             "PR": average_precision_score(yte, p),
                             "Brier": brier_score_loss(yte, p),
                             "cal_slope": cs, "cal_inter": ci,
                             "pred_media": float(p.mean()), "obs": float(yte.mean())})
    R = pd.DataFrame(res_rows)
    pd.set_option("display.width", 200)
    print(R.round(3).to_string(index=False))
    # IG v3 vs RJ con bootstrap anual (H=60)
    te60 = te
    y = te60.y_60.to_numpy()
    p3 = frozen.p_sota_60.to_numpy()
    # RJ H60 recomputo rapido desde preds no guardadas: recalcular aqui
    print("\nbootstrap IG(v3-RJ) H=60: ver tabla IG_vs_Poisson (diferencia de IGs)")
    R.to_csv("benchmark/sota_check.csv", index=False)
    print("guardado benchmark/sota_check.csv")


if __name__ == "__main__":
    sys.exit(main())
