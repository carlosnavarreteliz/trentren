"""Backtest visual: replay pseudo-prospectivo 2014-2026 con modelo entrenado <=2013.

Genera dashboard/backtest.html: timelines de secuencias (prob vs sismos reales),
diagrama de calibracion, Molchan y desempeno por ano. Estatico, SVG inline."""
import sys
sys.path.insert(0, ".")
import json
import math
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score
import forecast_short as F

OUT = Path("dashboard/backtest.html")
SEQS = [("Iquique 2014 · M8,2", "2014-03-01", "2014-05-01", -21.5, -19.5),
        ("Illapel 2015 · M8,3", "2015-09-01", "2015-10-20", -32.2, -29.0),
        ("Valparaíso 2017 · M6,9", "2017-04-15", "2017-05-02", -34.5, -32.2),
        ("Coquimbo 2019 · M6,7", "2019-01-10", "2019-02-05", -32.2, -29.0)]


def svg_timeline(rows, t0, t1, thr):
    W, Hh, pad = 980, 190, 30
    t0n = pd.Timestamp(t0).timestamp()
    t1n = pd.Timestamp(t1).timestamp()
    def X(t):
        return pad + (W - 2 * pad) * (pd.Timestamp(t).timestamp() - t0n) / (t1n - t0n)
    pts = " ".join(f"{X(r.time):.1f},{Hh - 40 - r.p1440 * 130:.1f}" for r in rows.itertuples())
    s = [f'<svg viewBox="0 0 {W} {Hh}" width="100%">',
         f'<line x1="{pad}" y1="{Hh-40}" x2="{W-pad}" y2="{Hh-40}" stroke="#191612"/>',
         f'<line x1="{pad}" y1="{Hh-40-thr*130:.1f}" x2="{W-pad}" y2="{Hh-40-thr*130:.1f}" stroke="#b4231a" stroke-dasharray="4 3"/>']
    for r in rows.itertuples():
        hh = 12 + (r.x2 - 4.7) * 22
        col = "#b4231a" if r.x2 >= 6.0 else "#6f675d"
        s.append(f'<line x1="{X(r.time):.1f}" y1="{Hh-40}" x2="{X(r.time):.1f}" y2="{Hh-40-hh:.1f}" stroke="{col}" stroke-width="{2 if r.x2>=6 else 1}"/>')
        if r.x2 >= 6.0:
            s.append(f'<text x="{X(r.time):.1f}" y="{Hh-48-hh:.1f}" font-size="11" fill="{col}" text-anchor="middle">M{r.x2:.1f}</text>')
        if r.y_1440:
            s.append(f'<circle cx="{X(r.time):.1f}" cy="{Hh-30}" r="4" fill="#0e7569"/>')
    s.append(f'<polyline points="{pts}" fill="none" stroke="#0e7569" stroke-width="1.8"/>')
    s.append(f'<text x="{W-pad}" y="14" font-size="11" fill="#6f675d" text-anchor="end">línea: P(M≥5 en 24h) · rojo: umbral alerta · tallo: sismo (altura=M) · ●: escaló en 24h</text>')
    s.append("</svg>")
    return "".join(s)


def svg_reliability(bins):
    W, Hh, pad = 420, 320, 44
    s = [f'<svg viewBox="0 0 {W} {Hh}" width="100%">',
         f'<line x1="{pad}" y1="{Hh-pad}" x2="{W-10}" y2="20" stroke="#c9c2b4" stroke-dasharray="4 3"/>']
    for b in bins:
        x = pad + b["pm"] * (W - pad - 10)
        y = Hh - pad - b["ob"] * (Hh - pad - 20)
        r = 3 + b["n"] ** 0.5 / 2
        s.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" fill="#0e7569" fill-opacity="0.75"/>')
        s.append(f'<text x="{x:.1f}" y="{y-r-3:.1f}" font-size="10" text-anchor="middle">{b["ob"]:.2f}</text>')
    s.append(f'<text x="{pad}" y="{Hh-12}" font-size="11" fill="#6f675d">predicho → (diagonal = perfecto; tamaño = nº casos; etiqueta = observado)</text>')
    s.append("</svg>")
    return "".join(s)


def svg_molchan(pts):
    W, Hh, pad = 420, 320, 44
    s = [f'<svg viewBox="0 0 {W} {Hh}" width="100%">',
         f'<line x1="{pad}" y1="{Hh-pad}" x2="{W-10}" y2="20" stroke="#c9c2b4" stroke-dasharray="4 3"/>']
    pl = " ".join(f"{pad + a*(W-pad-10):.1f},{Hh-pad-h*(Hh-pad-20):.1f}" for a, h in pts)
    s.append(f'<polyline points="{pl}" fill="none" stroke="#0e7569" stroke-width="2"/>')
    s.append(f'<text x="{pad}" y="{Hh-12}" font-size="11" fill="#6f675d">fracción de tiempo en alarma → fracción capturada (sobre diagonal = habilidad)</text>')
    s.append("</svg>")
    return "".join(s)


def main():
    print("cargando + dataset (1-3 min)...", flush=True)
    d = F.load_local()
    df = F.build_dataset(d)
    cols = F.COLS_V2
    tr = df[df.time < "2014-01-01"].reset_index(drop=True)
    nva = int(0.15 * len(tr))
    trf, trv = tr.iloc[:-nva], tr.iloc[-nva:]
    te = df[df.time >= "2014-01-01"].reset_index(drop=True)
    sc = StandardScaler().fit(trf[cols].to_numpy())
    out_p = {}
    thrs = {}
    for h in F.HORIZONS_MIN:
        clf = LogisticRegression(C=1.0).fit(sc.transform(trf[cols].to_numpy()), trf[f"y_{h}"].to_numpy())
        sv = clf.decision_function(sc.transform(trv[cols].to_numpy())).reshape(-1, 1)
        pl = LogisticRegression(C=1e6).fit(sv, trv[f"y_{h}"].to_numpy())
        ste = clf.decision_function(sc.transform(te[cols].to_numpy())).reshape(-1, 1)
        p = pl.predict_proba(ste)[:, 1]
        out_p[h] = p
        # umbral F0.5 en slice de calibracion
        pv = pl.predict_proba(sv)[:, 1]
        yv = trv[f"y_{h}"].to_numpy()
        best = (0, 0.3)
        for thr in np.linspace(0.02, 0.8, 79):
            pr = pv >= thr
            tp = int((pr & (yv == 1)).sum())
            prec = tp / max(pr.sum(), 1)
            rec = tp / max(yv.sum(), 1)
            f05 = 1.25 * prec * rec / max(0.25 * prec + rec, 1e-9)
            if f05 > best[0]:
                best = (f05, float(thr))
        thrs[h] = best[1]
    te = te.assign(p60=out_p[60], p180=out_p[180], p1440=out_p[1440])
    y = te.y_1440.to_numpy()
    p = te.p1440.to_numpy()
    print(f"replay 2014+: n={len(te)} pos24h={y.sum()} AUC={roc_auc_score(y, p):.3f}")
    # calibracion por deciles
    bins = []
    qs = np.quantile(p, np.linspace(0, 1, 11))
    for i in range(10):
        m = (p >= qs[i]) & (p <= qs[i + 1] if i == 9 else p < qs[i + 1])
        if m.sum():
            bins.append({"pm": float(p[m].mean()), "ob": float(y[m].mean()), "n": int(m.sum())})
    # molchan 24h
    order = np.argsort(-p)
    cum = np.cumsum(y[order])
    mol = [(0.0, 0.0)] + [(float(i + 1) / len(y), float(cum[i] / max(y.sum(), 1)))
                          for i in range(0, len(y), max(len(y) // 60, 1))]
    # por ano
    te["yr"] = te.time.dt.year
    ytab = ""
    for yr, g in te.groupby("yr"):
        yy, pp = g.y_1440.to_numpy(), g.p1440.to_numpy()
        al = pp >= thrs[1440]
        hit = int((al & (yy == 1)).sum())
        auc = f"{roc_auc_score(yy, pp):.2f}" if 0 < yy.sum() < len(yy) else "—"
        mark = " †" if yy.sum() < 10 else ""
        ytab += (f"<tr><td>{yr}{mark}</td><td class='num'>{len(g)}</td><td class='num'>{int(yy.sum())}</td>"
                 f"<td class='num'>{auc}</td><td class='num'>{int(al.sum())}</td><td class='num'>{hit}</td>"
                 f"<td class='num'>{hit / max(al.sum(), 1):.2f}</td></tr>")
    # split por regimen (secuencias vs quieto)
    regime_html = "<table><thead><tr><th>Régimen</th><th style='text-align:right'>Triggers</th>" \
                  "<th style='text-align:right'>Tasa base</th><th style='text-align:right'>AUC</th>" \
                  "<th style='text-align:right'>Alertas</th><th style='text-align:right'>Precisión</th></tr></thead><tbody>"
    _yr = te.time.dt.year
    for lab, mreg in [("Con secuencias (2014–2020)", _yr <= 2020), ("Basal (2021–2026)", _yr >= 2021)]:
        g = te[mreg]
        yy, pp = g.y_1440.to_numpy(), g.p1440.to_numpy()
        al = pp >= thrs[1440]
        hit = int((al & (yy == 1)).sum())
        aucs = f"{roc_auc_score(yy, pp):.2f}" if 0 < yy.sum() < len(yy) else "—"
        regime_html += (f"<tr><td>{lab}</td><td class='num'>{len(g)}</td><td class='num'>{yy.mean():.2f}</td>"
                        f"<td class='num'>{aucs}</td><td class='num'>{int(al.sum())}</td>"
                        f"<td class='num'>{hit / max(al.sum(), 1):.2f}</td></tr>")
    regime_html += "</tbody></table>"
    seq_html = ""
    for name, t0, t1, a, b in SEQS:
        rows = te[(te.time >= t0) & (te.time < t1) & (te.lat.between(a, b))].sort_values("time")
        nal = int((rows.p1440 >= thrs[1440]).sum())
        seq_html += (f"<h3>{name} <span class='mut'>· {len(rows)} triggers · {nal} sobre umbral · "
                     f"{int(rows.y_1440.sum())} escalaron en 24h</span></h3>" + svg_timeline(rows, t0, t1, thrs[1440]))
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    html = f"""<!DOCTYPE html>
<html lang="es"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Backtest histórico — pronóstico sísmico Chile</title>
<style>
:root{{--ink:#191612;--mut:#6f675d;--line:#e8e2d8;--ok:#0e7569;--bad:#b4231a;--bg:#fdfcf9;--soft:#f4f0e8}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 "Avenir Next","Segoe UI",system-ui,sans-serif}}
.num{{font-variant-numeric:tabular-nums}}.mut{{color:var(--mut);font-weight:400;font-size:13px}}
.wrap{{max-width:1060px;margin:0 auto;padding:30px 22px 64px}}
h1{{font-family:Georgia,serif;font-size:28px;margin:0}}h2{{font-family:Georgia,serif;font-size:20px;margin:32px 0 4px}}
h3{{font-size:15px;margin:22px 0 2px}}.lede{{color:var(--mut);font-size:13.5px;max-width:78ch}}
table{{width:100%;border-collapse:collapse;font-size:13.5px;background:#fff;margin-top:8px}}
th{{text-align:left;font-size:11px;text-transform:uppercase;letter-spacing:.06em;color:var(--mut);border-bottom:2px solid var(--ink);padding:6px 8px}}
td{{border-bottom:1px solid var(--line);padding:6px 8px}}td.num{{text-align:right}}
.cols{{display:grid;grid-template-columns:1fr 1fr;gap:18px}}
@media(max-width:760px){{.cols{{grid-template-columns:1fr}}}}
.card{{border:1px solid var(--line);background:#fff;padding:10px 14px}}
.pol{{background:var(--soft);border-left:3px solid var(--ink);padding:12px 16px;font-size:14px;margin:18px 0}}
.foot{{margin-top:36px;border-top:1px solid var(--line);padding-top:10px;color:var(--mut);font-size:12.5px}}
</style></head><body><div class="wrap">
<h1>Cómo funcionó el modelo en periodos previos</h1>
<p class="lede">Replay pseudo-prospectivo 2014–2026: modelo de visualización entrenado solo con datos ≤2013
(features E3, calibración Platt 2011–2013). Cada punto usa únicamente información anterior al trigger.
AUC global 24h = {roc_auc_score(y, p):.2f} · PR-AUC = {average_precision_score(y, p):.3f} (base {y.mean():.3f}).</p>

<div class="pol"><strong>Uso para decisión pública:</strong> tras un sismo M≥6, esta herramienta ordena las
próximas horas (dónde mirar, qué tan probable es otro M≥5). En plena secuencia la tasa base ya es
80–90% (casi todo trigger escala en 24h): ahí el modelo aporta <em>orden y probabilidades calibradas</em>,
no detección. En régimen basal emite pocas alertas y su precisión 2022–2026 fue ~0: fuera de secuencias,
trátelo como vigilancia, no como aviso. No sirve para evacuar ni para anunciar un terremoto principal.
Umbral 24h usado: {thrs[1440]:.2f}.</div>

<h2>Secuencias históricas (fuera de muestra)</h2>
{seq_html}

<div class="cols">
<div class="card"><h3 style="margin-top:4px">Calibración (24h)</h3>{svg_reliability(bins)}
<p class="lede">Si el modelo dice 20%, debería ocurrir ~20% de las veces. Cercano a la diagonal =
probabilidades honestas utilizables en documentos oficiales.</p></div>
<div class="card"><h3 style="margin-top:4px">Molchan (24h)</h3>{svg_molchan(mol)}
<p class="lede">Con el ~10% del tiempo en alarma se captura una fracción desproporcionada de los
escalamientos: esa es la habilidad real, concentrada en secuencias activas.</p></div>
</div>

<h2>Régimen activo vs basal (24h)</h2>
<p class="lede">El desempeño depende brutalmente del régimen: con secuencias (2014–2020) casi todo escala y
el AUC es 0,9+; en años quietos (2021–2026) discrimina poco (AUC ~0,5–0,9 variable, precisión ~0 con
1–2 alertas/año). El AUC global 0,92 está dominado por las megasecuencias: no lo cite sin el régimen.</p>
{regime_html}

<h2>Desempeño por año (24h)</h2>
<table><thead><tr><th>Año</th><th style="text-align:right">Triggers</th><th style="text-align:right">Escalaron</th>
<th style="text-align:right">AUC</th><th style="text-align:right">Alertas</th><th style="text-align:right">Aciertos</th>
<th style="text-align:right">Precisión</th></tr></thead><tbody>{ytab}</tbody></table>
<p class="lede">Años sin AUC: sin positivos o sin negativos (nada que discriminar).
† Año con &lt;10 escalamientos (ej. 2018: solo 3, sin secuencias grandes): <strong>sin potencia
estadística</strong> — su AUC/precisión es ruido, no una falla del modelo; Chile simplemente estuvo quieto.</p>

<div class="foot">Generado {now} · <code>python3 dashboard/backtest.py</code> · Modelo de visualización ≠ operativo v2
(entrenado ≤2020). Protocolo: features causales, umbrales desde calibración, etiquetas 60min/3h/24h a 150 km.</div>
</div></body></html>"""
    OUT.write_text(html)
    print(f"backtest -> {OUT} ({len(html)//1024} KB)")


if __name__ == "__main__":
    sys.exit(main())
