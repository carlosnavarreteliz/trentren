"""Plataforma unica (dashboard/index.html): prediccion en vivo + explorador
historico de probabilidades (replay 2014+) + desempeno + ciclo. Un solo archivo."""
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
import forecast_short as F

OUT = Path("dashboard/index.html")
CAMPAIGN = [
    ("Rondas 1+2 (20 familias/exp.)", "panel semanal M6", "todas nulas", "null"),
    ("E3 kernels ETAS+", "panel diario M5", "IG +0,0117 · p=0,017", "signal"),
    ("v3 homogeneización Mw", "trigger 60 min", "AUC 0,68→0,87", "signal"),
    ("E1/E2/E4/E5/C1 (tipos, nowcast, repeaters, exógenas, clima)", "varios", "nulos", "null"),
    ("M2 GRU · P2 Transformer · M3 motivos · M4 GBDT", "secuencias/tabular", "≤ logit simple", "null"),
    ("M5 36 años · P1b energía · M6 espacial", "largo plazo/lugar", "IC∋0 / LOZO ✗", "null"),
    ("R12 espacial por trigger (lat/lon, distancia, Benioff)", "T1 test 2024+", "IG +0,45 IC∋0 · no adoptado", "null"),
    ("R13 torneo: ETAS espacial / GBDT-full vs v4", "T1 test 2024+", "ETAS +0,000 · GBDT descartado en calib-CV", "null"),
    ("Clima ONI/PDO + precip. ERA5 (hidrocarga)", "panel diario M5", "IG +0,001 · p=0,27", "null"),
]


def live_rows(n=10):
    mp = F.MODEL_JSON_V4 if F.MODEL_JSON_V4.exists() else (
        F.MODEL_JSON_V3 if F.MODEL_JSON_V3.exists() else F.MODEL_JSON_V2)
    model = json.loads(mp.read_text())
    cols = model.get("cols", F.COLS_V2)
    mean, scale = np.array(model["scaler_mean"]), np.array(model["scaler_scale"])
    d = F.load_local()
    try:
        live = F.fetch_live_usgs(days=90, min_mag=4.0)
        if len(live):
            d = pd.concat([d, live[live.time > d.time.max()]], ignore_index=True)
            d = d.sort_values("time").reset_index(drop=True)
            d["tmin"] = (d.time - pd.Timestamp("1970-01-01")).dt.total_seconds() / 60.0
            src = f"USGS en vivo hasta {d.time.max()} UTC"
        else:
            src = f"catálogo local hasta {d.time.max()} UTC"
    except Exception as e:
        src = f"catálogo local hasta {d.time.max()} UTC (offline)"
    tm = d.tmin.to_numpy()
    la, lo, mg, dp = (d[c].to_numpy() for c in ["latitude", "longitude", "mag", "depth"])
    trig = d[d.mag >= 4.7].tail(60).reset_index()
    rows = []
    for _, r in trig.iterrows():
        f = F.trigger_features(r, int(r["index"]), tm, la, lo, mg, dp)
        xs = (np.array([[f[c] for c in cols]]) - mean) / scale
        probs, flags = {}, []
        for h in F.HORIZONS_MIN:
            m = model["horizons"][str(h)]
            s = float(xs[0] @ np.array(m["coef"]) + m["intercept"])
            p = float(1 / (1 + math.exp(-(m["platt_coef"] * s + m["platt_intercept"]))))
            probs[h] = p
            if p >= m["threshold"]:
                flags.append(F.H_LABEL[h])
        rows.append({"t": str(r.time)[:16], "m": r.mag, "z": F.zone_of(r.latitude),
                     "p60": probs[60], "p3": probs[180], "p24": probs[1440],
                     "al": ",".join(flags)})
    return rows[-n:], src, model, d


def replay_data(d):
    """Modelo viz <=2013 + Platt (slice final train) -> replay 2014+."""
    df = F.build_dataset(d)
    cols = F.COLS_V2
    tr = df[df.time < "2014-01-01"].reset_index(drop=True)
    nva = int(0.15 * len(tr))
    trf, trv = tr.iloc[:-nva], tr.iloc[-nva:]
    te = df[df.time >= "2014-01-01"].reset_index(drop=True)
    sc = StandardScaler().fit(trf[cols].to_numpy())
    P, THR = {}, {}
    for h in F.HORIZONS_MIN:
        clf = LogisticRegression(C=1.0).fit(sc.transform(trf[cols].to_numpy()), trf[f"y_{h}"].to_numpy())
        sv = clf.decision_function(sc.transform(trv[cols].to_numpy())).reshape(-1, 1)
        pl0 = LogisticRegression(C=1e6).fit(sv, trv[f"y_{h}"].to_numpy())
        pv = pl0.predict_proba(sv)[:, 1]
        THR[h] = float(np.quantile(pv, 0.95))
        ste = clf.decision_function(sc.transform(te[cols].to_numpy()))
        # Recalibración walk-forward: la Platt de cada año Y se ajusta solo con
        # triggers de años < Y (etiquetas ya maduras). Antes era una Platt fija
        # <=2013 que en 2014+ daba 65-74% falsos: eso era descalibración, no señal.
        Pall = np.zeros(len(te))
        yrs_te = te.time.dt.year.to_numpy()
        s_hist = list(sv.ravel()); y_hist = list(trv[f"y_{h}"].to_numpy())
        for Y in sorted(set(yrs_te)):
            mk = yrs_te == Y
            plY = LogisticRegression(C=1e6).fit(np.array(s_hist).reshape(-1, 1), np.array(y_hist))
            Pall[mk] = plY.predict_proba(ste[mk].reshape(-1, 1))[:, 1]
            s_hist += list(ste[mk]); y_hist += list(te.loc[mk, f"y_{h}"].to_numpy())
        P[h] = Pall
    t0 = pd.Timestamp("2000-01-01")
    znames = [z[0] for z in F.ZONES]
    RP = []
    for i in range(len(te)):
        r = te.iloc[i]
        RP.append([int((r.time - t0).days), round(float(r.lat), 2), round(float(r.lon), 2),
                   round(float(r.x2), 1), 0, round(float(P[60][i]), 3), round(float(P[180][i]), 3),
                   round(float(P[1440][i]), 3), int(r.y_60), int(r.y_180), int(r.y_1440),
                   znames.index(F.zone_of(r.lat))])
    # desempeno por ano (24h) para tabla compacta
    te = te.assign(pp=P[1440])
    yrs = []
    for yr, g in te.groupby(te.time.dt.year):
        yy = g.y_1440.to_numpy()
        al = g.pp.to_numpy() >= THR[1440]
        yrs.append([int(yr), len(g), int(yy.sum()), int(al.sum()),
                    int((al & (yy == 1)).sum())])
    return RP, THR, yrs, te


def cycle_rows(d):
    now = pd.Timestamp.now(tz="UTC").tz_localize(None)
    out = []
    for name, a, b in F.ZONES:
        z = d[d.latitude.between(a, b)]
        big = z[z.mag >= 7.5].sort_values("time")
        last = big.iloc[-1] if len(big) else None
        t_last = last.time if last is not None else pd.Timestamp("1904-01-01")
        yrs = (now - t_last).days / 365.25
        acc = z[(z.time >= t_last) & (z.mag >= 5.0)]
        ben = float(np.log10((10 ** (0.75 * acc.mag)).sum() + 1))
        out.append((name, f"{last.time.year}·M{last.mag:.1f}" if last is not None else "—",
                    round(yrs, 1), round(ben, 1), len(acc)))
    return out


def main():
    rows, src, model, d = live_rows()
    RP, THR, yrs, te = replay_data(d)
    bg = d[(d.time >= "2000-01-01") & (d.mag >= 4.5)]
    t0 = pd.Timestamp("2000-01-01")
    BG = [[int((t - t0).days), round(float(a), 2), round(float(b), 2), round(float(m), 1), int(z)]
          for t, a, b, m, z in zip(bg.time, bg.latitude, bg.longitude, bg.mag, bg.depth)]
    cyc = cycle_rows(d[d.time >= "1920-01-01"])
    n_alert = sum(1 for r in rows if r["al"])
    status = "SECUENCIA ACTIVA" if n_alert else "RÉGIMEN BASAL"
    scls = "bad" if n_alert else "ok"
    trs = ""
    for r in reversed(rows):
        trs += (f'<tr class="{"alert" if r["al"] else ""}"><td>{r["t"]}</td>'
                f'<td class="num big">{r["m"]:.1f}</td><td>{r["z"]}</td>'
                f'<td class="num">{r["p60"]*100:.1f}%</td><td class="num">{r["p3"]*100:.1f}%</td>'
                f'<td class="num">{r["p24"]*100:.1f}%</td>'
                f'<td>{"ALERTA " + r["al"] if r["al"] else "—"}</td></tr>')
    ytrs = "".join(
        f"<tr><td>{y[0]}{' †' if y[2] < 10 else ''}</td><td class='num'>{y[1]}</td><td class='num'>{y[2]}</td>"
        f"<td class='num'>{y[3]}</td><td class='num'>{y[4]}</td>"
        f"<td class='num'>{y[4] / max(y[3], 1):.2f}</td><td class='num'>{y[4] / max(y[2], 1):.2f}</td></tr>" for y in yrs)
    # Listado ANCHO de alertas del test congelado 2024+: 1 fila por trigger
    # (benchmark T1, probs SOTA v3 congeladas; umbrales v3 del benchmark).
    atrs, asum = "", ""
    try:
        t1 = pd.read_csv("benchmark/t1_test.csv", parse_dates=["time"])
        mv3 = json.loads(Path("results/short_term_model_v3.json").read_text())
        hlab = {60: "60 min", 180: "3 h", 1440: "24 h"}
        TH = {h: float(mv3["horizons"][str(h)]["threshold"]) for h in (60, 180, 1440)}
        for h in (60, 180, 1440):
            sub = t1[t1[f"p_sota_{h}"] >= TH[h]]
            pos = int(t1[f"y_{h}"].sum())
            prec = sub[f"y_{h}"].mean() if len(sub) else 0.0
            rec = sub[f"y_{h}"].sum() / max(pos, 1)
            f1 = 2 * prec * rec / max(prec + rec, 1e-9)
            asum += (f"<tr><td>{hlab[h]}</td><td class='num'>{TH[h]:.3f}</td><td class='num'>{len(sub)}</td>"
                     f"<td class='num'>{int(sub[f'y_{h}'].sum())}/{pos}</td>"
                     f"<td class='num'>{prec:.2f}</td><td class='num'>{rec:.2f}</td><td class='num'>{f1:.2f}</td>"
                     f"<td class='num'>{pos / len(t1):.3f}</td><td class='num'>{len(sub) / len(t1):.3f}</td>"
                     f"<td class='num'>×{prec / max(pos / len(t1), 1e-9):.1f}</td></tr>")
        alert_mask = np.zeros(len(t1), bool)
        for h in (60, 180, 1440):
            alert_mask |= (t1[f"p_sota_{h}"] >= TH[h]).to_numpy()
        wide = t1[alert_mask].copy()
        # CSV ancho canónico (reemplaza al tidy anterior)
        wcsv = pd.DataFrame({
            "trigger_time": wide.time, "lat": wide.lat.round(3), "lon": wide.lon.round(3),
            "mag_trigger": wide.mag, "zona": [F.zone_of(x) for x in wide.lat],
            "p_60min": wide.p_sota_60.round(4), "p_3h": wide.p_sota_180.round(4), "p_24h": wide.p_sota_1440.round(4),
            "alerta_60min": (wide.p_sota_60 >= TH[60]).astype(int), "alerta_3h": (wide.p_sota_180 >= TH[180]).astype(int),
            "alerta_24h": (wide.p_sota_1440 >= TH[1440]).astype(int),
            "escalo_60min": wide.y_60.astype(int), "escalo_3h": wide.y_180.astype(int), "escalo_24h": wide.y_1440.astype(int)})
        wcsv.to_csv("results/alert_ledger_2024.csv", index=False)
        for _, r in wide.iloc[::-1].iterrows():
            cells = ""
            for h, pc in ((60, "p_sota_60"), (180, "p_sota_180"), (1440, "p_sota_1440")):
                p = float(r[pc])
                if p >= TH[h]:
                    cells += f"<td class='num big'>{(p * 100):.1f}% {'✓' if int(r[f'y_{h}']) else '✗'}</td>"
                else:
                    cells += f"<td class='num' style='color:var(--mut)'>{(p * 100):.1f}%</td>"
            atrs += (f"<tr><td>{str(r.time)[:16]}</td><td class='num'>{r.mag:.1f}</td>"
                     f"<td class='num'>{r.lat:.2f}, {r.lon:.2f}</td><td>{F.zone_of(r.lat)}</td>{cells}</tr>")
    except Exception as e:
        atrs = f"<tr><td colspan='7'>ledger no disponible ({e})</td></tr>"
    crs = "".join(
        f"<tr><td>{c[0]}</td><td class='num'>{c[1]}</td><td class='num big'>{c[2]}</td>"
        f"<td class='num'>{c[3]}</td><td class='num'>{c[4]}</td></tr>" for c in cyc)
    ctrs = "".join(
        f"<tr><td>{a}</td><td>{b}</td><td class='num'>{c}</td>"
        f"<td><span class='pill {k}'>{'SEÑAL' if k == 'signal' else 'NULO'}</span></td></tr>"
        for a, b, c, k in CAMPAIGN)
    zones_js = json.dumps([z[0] for z in F.ZONES])
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    html = f"""<!DOCTYPE html>
<html lang="es"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Plataforma sísmica de Chile — probabilidades</title>
<style>
:root{{--ink:#16211f;--mut:#5d6b68;--line:#d8dee0;--ok:#0b7a6e;--bad:#c22e24;--amber:#a66a00;--bg:#e9edee;--card:#ffffff;--dark:#14201d}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 -apple-system,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif}}
.num{{font-variant-numeric:tabular-nums}}.big{{font-weight:700}}
.wrap{{max-width:1180px;margin:0 auto;padding:0 20px 56px}}
header{{background:var(--dark);color:#eef3f1;margin:0 -20px;padding:26px 28px 20px}}
header h1{{color:#fff}}
header .sub{{color:#b9c8c3}}
h1{{font-family:Georgia,"Times New Roman",serif;font-size:30px;margin:0 0 4px;letter-spacing:-.01em}}
h2{{font-family:Georgia,"Times New Roman",serif;font-size:21px;margin:30px 0 6px;padding:10px 14px;background:var(--card);border:1px solid var(--line);border-left:6px solid var(--ok)}}
.sub,.lede{{color:var(--mut);font-size:13.5px}}
header .lede,header .sub{{color:#b9c8c3}}
.kpis{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:10px;margin-top:16px}}
.kpi{{background:rgba(255,255,255,.07);border:1px solid rgba(255,255,255,.22);padding:10px 14px}}
section .kpi,.wrap>.kpis .kpi{{background:var(--card);border-color:var(--line)}}
.kpi .k{{font-size:10.5px;letter-spacing:.08em;color:var(--mut);text-transform:uppercase;font-weight:700}}
header .kpi .k{{color:#9fb3ac}}
.kpi .v{{font-size:22px;font-weight:700}}
table{{width:100%;border-collapse:collapse;font-size:13.5px;background:var(--card);margin-top:8px;border:1px solid var(--line)}}
th{{text-align:left;font-size:10.5px;text-transform:uppercase;letter-spacing:.07em;color:#e8efec;background:var(--dark);padding:8px}}
td{{border-bottom:1px solid var(--line);padding:7px 8px}}td.num{{text-align:right}}
tbody tr:nth-child(even) td{{background:#f4f7f6}}
tbody tr:hover td{{background:#e7f0ed}}
tr.alert td{{background:#fdecea}}tr.alert td:last-child{{color:var(--bad);font-weight:700}}
.pill{{display:inline-block;padding:1px 10px;font-size:12px;font-weight:700;border:1.5px solid}}
.pill.null{{color:var(--mut);border-color:var(--mut)}}.pill.signal{{color:#fff;background:var(--ok);border-color:var(--ok)}}
.warn{{background:#fff7e6;border:1px solid #e3c98a;border-left:6px solid var(--amber);padding:12px 16px;font-size:13.5px;margin:16px 0}}
.ctl{{display:flex;gap:10px;flex-wrap:wrap;align-items:end;background:var(--card);padding:12px 14px;margin-top:10px}}
.ctl label{{font-size:10.5px;text-transform:uppercase;letter-spacing:.07em;color:var(--mut);font-weight:700;display:block}}
.ctl input,.ctl select{{font-size:14px;padding:6px 8px;border:1px solid var(--line);background:#fff}}
button{{font-size:14px;padding:8px 16px;border:1.5px solid var(--dark);background:var(--dark);color:#fff;cursor:pointer;font-weight:600}}
button:hover{{background:var(--ok);border-color:var(--ok)}}
button.alt{{background:#fff;color:var(--ink)}}
canvas{{width:100%;border:1px solid var(--line);background:#fff;display:block}}
.explorer{{border:1px solid var(--line);background:var(--card);margin-top:10px}}
.explorer .ctl{{border:0;border-bottom:1px solid var(--line);margin:0}}
.mapgrid{{display:grid;grid-template-columns:470px 1fr;gap:0}}
@media(max-width:820px){{.mapgrid{{grid-template-columns:1fr}}}}
.side{{border-left:1px solid var(--line);padding:14px 16px;font-size:13px;background:#f7f9f8}}
@media(max-width:820px){{.side{{border-left:0;border-top:1px solid var(--line)}}}}
.side h3{{font-family:Georgia,serif;font-size:16px;margin:2px 0 6px}}
.legend{{display:flex;gap:12px;flex-wrap:wrap;font-size:12px;color:var(--mut);margin:8px 0 2px}}
.dot{{display:inline-block;width:10px;height:10px;border-radius:50%;vertical-align:-1px;margin-right:4px}}
#alertlist{{list-style:none;margin:6px 0 0;padding:0}}
#alertlist li{{border-bottom:1px solid var(--line);padding:6px 0}}
.zheat{{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-top:8px}}
@media(max-width:820px){{.zheat{{grid-template-columns:repeat(2,1fr)}}}}
.zcell{{border:1px solid var(--line);padding:8px;background:var(--card);font-size:11.5px;display:flex;justify-content:space-between;gap:8px;align-items:baseline}}
.zcell .zv{{font-size:16px;font-weight:700}}
.scroll{{max-height:360px;overflow:auto;border:1px solid var(--line);background:var(--card)}}
.scroll table{{margin:0;border:0}}
.scroll thead th{{position:sticky;top:0;z-index:1}}
.foot{{margin-top:38px;background:var(--dark);color:#9fb3ac;padding:14px 18px;font-size:12.5px}}
.foot code{{color:#d7e4df}}
</style></head><body><div class="wrap">
<header>
<h1>Plataforma de seguimiento de probabilidad sísmica — Chile</h1>
<div class="sub">Réplicas/escalamiento a 60 min · 3 h · 24 h (USGS homogeneizado a Mw, modelo v3-E3) ·
explorador histórico de probabilidades 2014–2026 · {src}</div>
<div class="kpis">
<div class="kpi"><div class="k">Estado (últ. triggers)</div><div class="v" style="color:var(--{'bad' if scls=='bad' else 'ok'})">{status}</div></div>
<div class="kpi"><div class="k">Alertas activas</div><div class="v">{n_alert}</div></div>
<div class="kpi"><div class="k">Modelo</div><div class="v">v4 recalibrado · AUC 0,87 (60m)</div></div>
<div class="kpi"><div class="k">Actualizado</div><div class="v" style="font-size:15px">{now}</div></div>
</div></header>

<div class="warn"><strong>Alcance honesto:</strong> esta plataforma pronostica <strong>réplicas</strong>
(clustering medido, en el frente publicado). El <strong>sismo principal</strong> no ha mostrado
precursores explotables en 25+ experimentos preregistrados sobre catálogos, waveforms, GNSS,
exógenas, redes neuronales, Transformer ni energía por segmento.</div>

<h2>1 · Predicción actual</h2>
<table><thead><tr><th>Hora UTC</th><th style="text-align:right">M</th><th>Zona</th>
<th style="text-align:right">P 60m</th><th style="text-align:right">P 3h</th><th style="text-align:right">P 24h</th><th>Alerta</th></tr></thead>
<tbody>{trs}</tbody></table>

<h2>2 · Explorador histórico de probabilidades</h2>
<p class="lede">Replay pseudo-prospectivo 2014–2026 (modelo entrenado solo con datos ≤2013).
Cada punto es un sismo M≥4,7 que gatilla pronóstico; el color es la probabilidad asignada
<em>antes</em> de saber el resultado y el anillo negro marca los que efectivamente escalaron a M≥5,0.
Mueve el cursor para auditar qué se dijo y qué pasó.</p>
<div class="explorer">
<div class="ctl">
<div><label>Inicio</label><input type="date" id="d0" value="2014-01-01"></div>
<div><label>Término</label><input type="date" id="d1" value="2026-10-26"></div>
<div><label>Horizonte</label><select id="hor"><option value="5">60 min</option>
<option value="6">3 h</option><option value="7" selected>24 h</option></select></div>
<div><label>Velocidad</label><select id="spd"><option value="3">lenta</option>
<option value="14" selected>media</option><option value="45">rápida</option></select></div>
<button id="play">▶ Reproducir</button><button id="reset" class="alt">Reiniciar</button>
<div style="flex:1;min-width:180px"><label>Cursor temporal</label><input type="range" id="slider" min="0" max="1000" value="0" style="width:100%"></div>
</div>
<div class="mapgrid">
<div>
<canvas id="map" width="640" height="620" style="max-width:400px;margin:0 auto"></canvas>
<p class="lede" style="padding:0 12px">Mapa en proporción real: 1° de latitud y 1° de longitud ocupan su distancia verdadera en km (cos 35°). Chile es así de angosto; la versión anterior estiraba la longitud ~5×.</p>
<div class="legend" style="padding:0 12px 10px">
<span><span class="dot" style="background:#b4231a"></span>alerta (p ≥ umbral)</span>
<span><span class="dot" style="background:#9a6a12"></span>vigilancia (p ≥ ½ umbral)</span>
<span><span class="dot" style="background:#0e7569"></span>elevada (p ≥ 3%)</span>
<span><span class="dot" style="background:#c9c2b4"></span>basal</span>
<span>○ anillo = escaló a M≥5,0</span><span>tamaño = magnitud del trigger</span>
</div>
</div>
<aside class="side">
<h3 id="kdate">—</h3>
<p style="margin:0;color:var(--mut)"><span id="kn">0</span> triggers · <span id="kal">0</span> alertas · <span id="khit">0</span> aciertos verificables</p>
<h3 style="margin-top:14px">Últimas alertas al cursor</h3>
<ul id="alertlist"><li style="color:var(--mut)">Sin alertas todavía en la ventana.</li></ul>
<p style="color:var(--mut);font-size:12px">Replay con recalibración walk-forward anual (cada año se calibra solo con el pasado). Aun así sobrepredice en años quietos (2025: pred 4,9% vs obs 1,6%): dentro de una secuencia la probabilidad por trigger es alta y la mayoría de los triggers individuales no escala. Las probabilidades auditables del test congelado están en la sección 3 y en <code>results/alert_ledger_2024.csv</code>.</p>
</aside>
</div>
</div>
<p class="lede" style="margin-top:10px">Riesgo máximo por zona, ventana móvil de 7 días al cursor:</p>
<div class="zheat" id="zheat"></div>
<canvas id="serie" width="1020" height="100" style="margin-top:10px"></canvas>

<h2>3 · Alertas del test congelado 2024+ (formato ancho: 1 fila por trigger)</h2>
<p class="lede">Probabilidades SOTA v3 congeladas del benchmark sobre los 945 triggers 2024–2026.
Definición: P(otro M≥5,0 en 150 km dentro del horizonte) tras un M≥4,7. Umbral = percentil 95 de calibración.
<strong>Azar</strong> = alertar al azar con el mismo presupuesto: su precisión esperada es la tasa base y su recall la fracción alertada.
La precisión absoluta es baja porque la tasa base es 1,6–3,7%; el lift mide cuánto se supera ese azar.</p>
<table><thead><tr><th>Horizonte</th><th style="text-align:right">Umbral</th><th style="text-align:right">Alertas</th>
<th style="text-align:right">Aciertos/total</th>
<th style="text-align:right">Precisión</th><th style="text-align:right">Recall</th><th style="text-align:right">F1</th>
<th style="text-align:right">Prec. azar</th><th style="text-align:right">Recall azar</th><th style="text-align:right">Lift</th></tr></thead>
<tbody>{asum}</tbody></table>
<p class="lede">Tabla ancha: una fila por trigger que alertó en algún horizonte. Negrita = superó el umbral de ese horizonte; ✓ escaló, ✗ no escaló. CSV: <code>results/alert_ledger_2024.csv</code>.</p>
<div class="scroll" style="margin-top:10px"><table><thead><tr><th>Trigger UTC</th><th style="text-align:right">M</th>
<th style="text-align:right">Lat, lon</th><th>Zona</th><th style="text-align:right">P 60 min</th><th style="text-align:right">P 3 h</th><th style="text-align:right">P 24 h</th></tr></thead>
<tbody>{atrs}</tbody></table></div>

<h2>4 · Desempeño por año (replay, horizonte 24h)</h2>
<table><thead><tr><th>Año</th><th style="text-align:right">Triggers</th><th style="text-align:right">Escalaron</th>
<th style="text-align:right">Alertas</th><th style="text-align:right">Aciertos</th><th style="text-align:right">Precisión</th><th style="text-align:right">Recall</th></tr></thead>
<tbody>{ytrs}</tbody></table>
<p class="lede">† &lt;10 escalamientos: sin potencia estadística (ej. 2018 quieto) — no interpretar.
En megasecuencias la tasa base llega a 80–90%: ahí el modelo ordena y calibra, no "detecta".</p>

<h2>5 · Ciclo por zona (descriptivo, sin poder predictivo demostrado)</h2>
<table><thead><tr><th>Zona</th><th style="text-align:right">Último M≥7,5</th><th style="text-align:right">Años</th>
<th style="text-align:right">log Benioff acum.</th><th style="text-align:right">N M≥5 desde entonces</th></tr></thead>
<tbody>{crs}</tbody></table>

<h2>6 · Historial de experimentos</h2>
<table><thead><tr><th>Línea</th><th>Diseño</th><th style="text-align:right">Resultado</th><th></th></tr></thead>
<tbody>{ctrs}</tbody></table>

<div class="foot">Generado {now} · <code>python3 dashboard/generate.py</code> · Replay pseudo-prospectivo
(features causales verificadas por truncación, diff=0,00) · Benchmark Chile-ST v1.1 ·
Log prospectivo: <code>forecast_short.py --prospective</code>.</div>
</div>
<script>
const RP = __RP__, BG = __BG__, THR = __THR__, ZN = __ZN__;
const T0 = Date.UTC(2000,0,1);
const cv = document.getElementById('map'), cx = cv.getContext('2d');
const cs = document.getElementById('serie'), cg = cs.getContext('2d');
let playing = false, cur = 1000, last = 0;
const dn = id => Math.floor((Date.parse(document.getElementById(id).value) - T0) / 86400000);
const ds = n => new Date(T0 + n * 86400000).toISOString().slice(0, 10);
const hidx = () => parseInt(document.getElementById('hor').value);
const hdays = i => [0.042, 0.125, 1][i - 5];
function pcol(p, thr) {{
  if (p >= thr) return '#b4231a';
  if (p >= thr * 0.5) return '#9a6a12';
  if (p >= 0.03) return '#0e7569';
  return '#c9c2b4';
}}
function draw() {{
  const a = dn('d0'), b = dn('d1'), hi = hidx(), thr = THR[hi];
  const tcur = a + Math.floor((b - a) * cur / 1000);
  document.getElementById('slider').value = cur;
  document.getElementById('kdate').textContent = ds(tcur);
  // Escala verdadera en km: px/°lon = px/°lat x cos(35°) -> sin estirar longitud
  const YS = 560 / 39, X = lo => 78 + (lo + 78) * YS * 0.819, Y = la => 26 + (-17 - la) * YS;
  cx.clearRect(0, 0, 640, 620);
  cx.fillStyle = '#fff'; cx.fillRect(0, 0, 640, 620);
  cx.strokeStyle = '#e8e2d8'; cx.fillStyle = '#6f675d'; cx.font = '11px sans-serif'; cx.lineWidth = 1;
  for (let la = -55; la <= -20; la += 5) {{ cx.beginPath(); cx.moveTo(78, Y(la)); cx.lineTo(X(-64), Y(la)); cx.stroke(); cx.fillText(la + '°', 42, Y(la) + 4); }}
  for (let lo = -76; lo <= -66; lo += 2) {{ cx.beginPath(); cx.moveTo(X(lo), 26); cx.lineTo(X(lo), 586); cx.stroke(); cx.fillText(lo + '°', X(lo) - 10, 604); }}
  cx.fillText('lat', 42, 18); cx.fillText('lon', X(-66) - 8, 604);
  const coast = [[-70.6,-18.3],[-70.3,-19.0],[-70.2,-20.2],[-70.2,-21.5],[-70.4,-22.5],[-70.3,-23.6],[-70.5,-24.8],[-70.7,-26.3],[-70.7,-27.5],[-70.9,-29.0],[-71.5,-30.6],[-71.7,-32.0],[-71.6,-33.0],[-71.7,-34.5],[-72.0,-36.0],[-72.4,-37.5],[-73.0,-39.0],[-73.7,-40.5],[-73.8,-42.5],[-73.9,-44.5],[-74.0,-47.0],[-74.3,-50.0],[-73.8,-53.0],[-71.5,-54.8]];
  const drawCoast = () => {{ cx.strokeStyle = '#fff'; cx.lineWidth = 4; cx.beginPath(); coast.forEach((c, i) => i ? cx.lineTo(X(c[0]), Y(c[1])) : cx.moveTo(X(c[0]), Y(c[1]))); cx.stroke(); cx.strokeStyle = '#191612'; cx.lineWidth = 1.8; cx.beginPath(); coast.forEach((c, i) => i ? cx.lineTo(X(c[0]), Y(c[1])) : cx.moveTo(X(c[0]), Y(c[1]))); cx.stroke(); cx.lineWidth = 1; }};
  const zb = [[-19.5,-17],[-21.5,-19.5],[-25,-21.5],[-29,-25],[-32.2,-29],[-34.5,-32.2],[-38.5,-34.5],[-44,-38.5],[-56,-44]];
  zb.forEach((z, i) => {{
    cx.fillStyle = '#6f675d'; cx.font = '11px sans-serif';
    cx.fillText(ZN[i], X(-64) + 14, (Y(z[0]) + Y(z[1])) / 2 + 4);
    cx.strokeStyle = '#e8e2d8'; cx.beginPath(); cx.moveTo(X(-64), Y(z[0])); cx.lineTo(X(-64) + 8, Y(z[0])); cx.stroke();
  }});
  for (const e of BG) {{
    if (e[0] < a || e[0] > tcur) continue;
    const age = (tcur - e[0]) / 240; if (age > 1) continue;
    cx.globalAlpha = 0.10 + 0.25 * (1 - age);
    cx.fillStyle = '#6f675d';
    cx.beginPath(); cx.arc(X(e[2]), Y(e[1]), 1.6, 0, 7); cx.fill();
  }}
  cx.globalAlpha = 1;
  let n = 0, al = 0, hit = 0;
  const zmax = new Array(9).fill(0);
  for (const e of RP) {{
    if (e[0] < a || e[0] > b) continue;
    const p = e[hi];
    if (e[0] >= tcur - 7 && e[0] <= tcur) zmax[e[11]] = Math.max(zmax[e[11]], p);
    if (e[0] > tcur) continue;
    n++;
    const isAl = p >= thr; if (isAl) al++;
    const mature = e[0] + hdays(hi) <= tcur;
    const yv = e[hi + 3];
    if (isAl && mature && yv) hit++;
    cx.globalAlpha = isAl ? 1 : 0.55;
    cx.fillStyle = pcol(p, thr);
    const r = (isAl ? 3.2 : 2.2) + (e[3] - 4.7) * 3;
    cx.beginPath(); cx.arc(X(e[2]), Y(e[1]), r, 0, 7); cx.fill();
    if (mature && yv) {{
      cx.strokeStyle = '#191612'; cx.lineWidth = 1.6;
      cx.beginPath(); cx.arc(X(e[2]), Y(e[1]), r + 2.5, 0, 7); cx.stroke(); cx.lineWidth = 1;
    }}
    if (e[3] >= 6.8) {{
      cx.fillStyle = '#191612'; cx.font = 'bold 12px sans-serif';
      cx.fillText('M' + e[3].toFixed(1), X(e[2]) + r + 3, Y(e[1]) + 4);
    }}
  }}
  cx.globalAlpha = 1; drawCoast();
  document.getElementById('kn').textContent = n;
  document.getElementById('kal').textContent = al;
  document.getElementById('khit').textContent = hit;
  const alerts = RP.filter(e => e[0] >= a && e[0] <= tcur && e[hi] >= thr).slice(-6).reverse();
  document.getElementById('alertlist').innerHTML = alerts.length ? alerts.map(e =>
    '<li><strong>' + ds(e[0]) + '</strong> · M' + e[3].toFixed(1) + ' · ' + ZN[e[11]] +
    '<br>probabilidad dada: <strong>' + (e[hi] * 100).toFixed(1) + '%</strong>' +
    (e[0] + hdays(hi) <= tcur ? (e[hi + 3] ? ' · <span style="color:#b4231a;font-weight:700">escaló</span>' : ' · no escaló') : ' · aún no verificable') + '</li>').join('')
    : '<li style="color:#6f675d">Sin alertas todavía en la ventana.</li>';
  document.getElementById('zheat').innerHTML = ZN.map((z, i) =>
    '<div class="zcell"><span>' + z + '</span><span><span style="color:' + pcol(zmax[i], thr) + '">●</span> <span class="zv">' +
    (zmax[i] * 100).toFixed(0) + '%</span></span></div>').join('');
  cg.clearRect(0, 0, 1020, 100);
  cg.strokeStyle = '#0e7569'; cg.lineWidth = 1.6; cg.beginPath();
  let started = false;
  const stepd = Math.max(Math.floor((b - a) / 300), 1);
  for (let dd = a; dd <= tcur; dd += stepd) {{
    let mx = 0;
    for (const e of RP) if (e[0] >= dd && e[0] < dd + stepd) mx = Math.max(mx, e[7]);
    const x = 40 + (dd - a) / Math.max(b - a, 1) * 940, y = 88 - Math.min(mx, 0.8) / 0.8 * 70;
    started ? cg.lineTo(x, y) : cg.moveTo(x, y); started = true;
  }}
  cg.stroke();
  cg.fillStyle = '#6f675d'; cg.font = '11px sans-serif';
  cg.fillText('P máxima 24h por ventana (replay)', 44, 14);
}}
function step(ts) {{
  if (!playing) return;
  if (ts - last > 50) {{
    cur += parseInt(document.getElementById('spd').value);
    if (cur >= 1000) {{ cur = 1000; playing = false; document.getElementById('play').textContent = '▶ Reproducir'; }}
    draw(); last = ts;
  }}
  if (playing) requestAnimationFrame(step);
}}
document.getElementById('play').onclick = e => {{
  if (cur >= 1000) cur = 0;
  playing = !playing;
  e.target.textContent = playing ? '⏸ Pausar' : '▶ Reproducir';
  if (playing) {{ last = 0; requestAnimationFrame(step); }}
}};
document.getElementById('reset').onclick = () => {{
  playing = false; cur = 0;
  document.getElementById('play').textContent = '▶ Reproducir'; draw();
}};
document.getElementById('slider').oninput = e => {{ cur = parseInt(e.target.value); draw(); }};
['d0', 'd1', 'hor'].forEach(id => document.getElementById(id).onchange = () => {{ cur = 0; draw(); }});
draw();
</script></body></html>"""
    html = (html.replace("__RP__", json.dumps(RP)).replace("__BG__", json.dumps(BG))
            .replace("__THR__", json.dumps({5: THR[60], 6: THR[180], 7: THR[1440]}))
            .replace("__ZN__", zones_js))
    OUT.write_text(html)
    print(f"plataforma unica -> {OUT} ({len(html)//1024} KB, replay {len(RP)} triggers)")


if __name__ == "__main__":
    sys.exit(main())
