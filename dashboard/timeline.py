"""Genera dashboard/timeline.html: visor interactivo de evolucion sismica.
Datos USGS homogeneizados 2000+, M>=4.5. Canvas + controles de fecha/play."""
import sys
sys.path.insert(0, ".")
import json
from datetime import datetime, timezone
from pathlib import Path
import forecast_short as F

OUT = Path("dashboard/timeline.html")


def main():
    d = F.load_local()
    s = d[(d.time >= "2000-01-01") & (d.mag >= 4.5)].copy()
    t0 = s.time.min().normalize()
    days = ((s.time - t0).dt.total_seconds() / 86400).round().astype(int)
    ev = [[int(a), round(float(b), 2), round(float(c), 2), round(float(e), 1), int(f)]
          for a, b, c, e, f in zip(days, s.latitude, s.longitude, s.mag, s.depth)]
    zones = [[z, a, b] for z, a, b in F.ZONES]
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    html = """<!DOCTYPE html>
<html lang="es"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Evolución sísmica de Chile — visor</title>
<style>
:root{--ink:#191612;--mut:#6f675d;--line:#e8e2d8;--ok:#0e7569;--bad:#b4231a;--bg:#fdfcf9;--soft:#f4f0e8;--amber:#9a6a12}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 "Avenir Next","Segoe UI",system-ui,sans-serif}
.wrap{max-width:1060px;margin:0 auto;padding:26px 22px 56px}
h1{font-family:Georgia,serif;font-size:27px;margin:0}
.sub{color:var(--mut);font-size:13.5px}
.ctl{display:flex;gap:10px;flex-wrap:wrap;align-items:end;margin:18px 0 10px;
border:1px solid var(--line);background:#fff;padding:12px 14px}
.ctl label{font-size:11px;text-transform:uppercase;letter-spacing:.06em;color:var(--mut);font-weight:600;display:block}
.ctl input,.ctl select{font-size:14px;padding:5px 7px;border:1px solid var(--line);background:#fff;color:var(--ink)}
button{font-size:14px;padding:7px 18px;border:1.5px solid var(--ink);background:var(--ink);color:#fff;cursor:pointer;font-weight:600}
button.alt{background:#fff;color:var(--ink)}
#slider{flex:1;min-width:220px}
.kpis{display:flex;gap:12px;flex-wrap:wrap;margin:10px 0}
.kpi{border:1px solid var(--line);background:#fff;padding:8px 14px;min-width:140px}
.kpi .k{font-size:11px;letter-spacing:.07em;color:var(--mut);text-transform:uppercase;font-weight:600}
.kpi .v{font-size:20px;font-weight:700}
canvas{width:100%;border:1px solid var(--line);background:#fff}
.foot{margin-top:26px;border-top:1px solid var(--line);padding-top:10px;color:var(--mut);font-size:12.5px}
a{color:var(--ok)}
</style></head><body><div class="wrap">
<h1>Evolución sísmica de Chile</h1>
<div class="sub">Elige fecha de inicio y término, y reproduce cómo avanzó la sismicidad (M≥4,5 Mw-equiv, 2000–hoy).
El color indica profundidad: <b style="color:#b4231a">rojo ≤60 km</b> · <b style="color:#9a6a12">ámbar 60–150</b> ·
<b style="color:#0e7569">verde &gt;150</b>. Tamaño = magnitud.</div>

<div class="ctl">
<div><label>Inicio</label><input type="date" id="d0" value="2010-01-01"></div>
<div><label>Término</label><input type="date" id="d1" value="2026-10-26"></div>
<div><label>Magnitud mín.</label><select id="mmin">
<option value="4.5">M ≥ 4,5</option><option value="5">M ≥ 5</option>
<option value="5.5">M ≥ 5,5</option><option value="6">M ≥ 6</option></select></div>
<div><label>Velocidad</label><select id="spd">
<option value="3">lenta</option><option value="12" selected>media</option><option value="40">rápida</option></select></div>
<button id="play">▶ Reproducir</button>
<button id="reset" class="alt">Reiniciar</button>
<div style="flex:1"><label>Cursor temporal</label><input type="range" id="slider" min="0" max="1000" value="0" style="width:100%"></div>
</div>

<div class="kpis">
<div class="kpi"><div class="k">Fecha cursor</div><div class="v" id="kdate">—</div></div>
<div class="kpi"><div class="k">Eventos en ventana</div><div class="v" id="kn">0</div></div>
<div class="kpi"><div class="k">Máx. en ventana</div><div class="v" id="kmax">—</div></div>
<div class="kpi"><div class="k">M≥6 acumulados</div><div class="v" id="k6">0</div></div>
</div>

<canvas id="map" width="1000" height="560"></canvas>
<canvas id="cum" width="1000" height="110" style="margin-top:10px"></canvas>
<p class="sub">Arriba: mapa (fondo por zona sísmica). Abajo: acumulado de eventos en la ventana y tasa por año.
Uso de gestión: recorrer 2010 (Maule), 2014 (Iquique), 2015 (Illapel) y comparar con periodos quietos
como 2018 — la diferencia visual entre "secuencia" y "régimen basal" es lo que el modelo cuantifica.</p>

<div class="foot">Generado __NOW__ · <code>python3 dashboard/timeline.py</code> · Datos: USGS homogeneizados a Mw
(Scordilis) vía <code>forecast_short.load_local</code> · Volver al <a href="index.html">panel principal</a> ·
<a href="backtest.html">backtest histórico</a></div>
</div>
<script>
const EV = __EV__;
const T0 = Date.UTC(2000,0,1);
const cv = document.getElementById('map'), cx = cv.getContext('2d');
const cc = document.getElementById('cum'), cg = cc.getContext('2d');
const LAT0=-47, LAT1=-16, LON0=-79, LON1=-65;
const ZCOL=['#f7f3ea','#fdfcf9'];
let playing=false, cur=0, last=0;
function dnum(id){ const v=document.getElementById(id).value; return Math.floor((Date.parse(v)-T0)/86400000); }
function dstr(n){ const d=new Date(T0+n*86400000); return d.toISOString().slice(0,10); }
function filt(){
  const mmin=parseFloat(document.getElementById('mmin').value);
  return EV.filter(e=>e[0]>=dnum('d0')&&e[0]<=dnum('d1')&&e[3]>=mmin);
}
function draw(){
  const evs=filt(), a=dnum('d0'), b=dnum('d1');
  const tcur=a+Math.floor((b-a)*cur/1000);
  document.getElementById('slider').value=cur;
  document.getElementById('kdate').textContent=dstr(tcur);
  const X=lo=>40+(lo-LON0)/(LON1-LON0)*920, Y=la=>20+(LAT1-la)/(LAT1-LAT0)*520;
  cx.clearRect(0,0,1000,560);
  const zones=[['Arica',-19.5,-17],['Iquique',-21.5,-19.5],['Antofagasta',-25,-21.5],['Atacama',-29,-25],
               ['Coquimbo',-32.2,-29],['Valparaíso',-34.5,-32.2],['Maule',-38.5,-34.5],['Valdivia',-44,-38.5],['Austral',-56,-44]];
  zones.forEach((z,i)=>{ cx.fillStyle=ZCOL[i%2]; cx.fillRect(40,Y(z[1]),920,Y(z[2])-Y(z[1]));
    cx.fillStyle='#6f675d'; cx.font='11px sans-serif'; cx.fillText(z[0],44,Y(z[2])+13); });
  cx.strokeStyle='#e8e2d8'; zones.forEach(z=>{ cx.beginPath(); cx.moveTo(40,Y(z[2])); cx.lineTo(960,Y(z[2])); cx.stroke(); });
  let n=0, mx=0, n6=0;
  for(const e of evs){
    if(e[0]>tcur) break;
    n++; if(e[3]>mx) mx=e[3]; if(e[3]>=6) n6++;
    const age=(tcur-e[0])/120; if(age>1) continue;
    cx.globalAlpha=Math.max(0.15,1-age*0.85);
    cx.fillStyle=e[4]<=60?'#b4231a':(e[4]<=150?'#9a6a12':'#0e7569');
    const r=2+(e[3]-4.5)*3.2;
    cx.beginPath(); cx.arc(X(e[2]),Y(e[1]),r,0,7); cx.fill();
    if(e[3]>=7){ cx.globalAlpha=1; cx.fillStyle='#191612'; cx.font='bold 12px sans-serif';
      cx.fillText('M'+e[3].toFixed(1),X(e[2])+r+2,Y(e[1])+4); }
  }
  cx.globalAlpha=1;
  document.getElementById('kn').textContent=n;
  document.getElementById('kmax').textContent=mx?('M'+mx.toFixed(1)):'—';
  document.getElementById('k6').textContent=n6;
  // acumulado + tasa anual
  cg.clearRect(0,0,1000,110);
  const per={}; let tot=0; const cum=[];
  for(const e of evs){ tot++; cum.push([e[0],tot]); const y=new Date(T0+e[0]*86400000).getUTCFullYear(); per[y]=(per[y]||0)+1; }
  const mxc=tot||1;
  cg.strokeStyle='#0e7569'; cg.lineWidth=2; cg.beginPath();
  cum.forEach((p,i)=>{ const x=40+(p[0]-a)/Math.max(b-a,1)*920, y=95-p[1]/mxc*75;
    i?cg.lineTo(x,y):cg.moveTo(x,y); });
  cg.stroke();
  const yrs=Object.keys(per).map(Number), ymx=Math.max(...Object.values(per),1);
  cg.fillStyle='#b4231a';
  yrs.forEach(y=>{ const x=40+(Date.UTC(y,0,1)-T0)/86400000; const xx=40+((x-a)/Math.max(b-a,1))*920;
    cg.fillRect(xx,95,Math.max(920/Math.max(b-a,1)*300,2),-per[y]/ymx*40); });
  cg.fillStyle='#191612'; cg.font='11px sans-serif';
  cg.fillText('acumulado: '+tot+' eventos · barras rojas: eventos/año',44,14);
}
function step(ts){
  if(!playing) return;
  if(ts-last>50){ cur+=parseInt(document.getElementById('spd').value);
    if(cur>=1000){ cur=1000; playing=false; document.getElementById('play').textContent='▶ Reproducir'; }
    draw(); last=ts; }
  if(playing) requestAnimationFrame(step);
}
document.getElementById('play').onclick=e=>{
  if(cur>=1000) cur=0;
  playing=!playing; e.target.textContent=playing?'⏸ Pausar':'▶ Reproducir';
  if(playing){ last=0; requestAnimationFrame(step); } };
document.getElementById('reset').onclick=()=>{ playing=false; cur=0;
  document.getElementById('play').textContent='▶ Reproducir'; draw(); };
document.getElementById('slider').oninput=e=>{ cur=parseInt(e.target.value); draw(); };
['d0','d1','mmin'].forEach(id=>document.getElementById(id).onchange=()=>{ cur=0; draw(); });
draw();
</script></body></html>"""
    html = html.replace("__EV__", json.dumps(ev)).replace("__NOW__", now)
    OUT.write_text(html)
    print(f"visor -> {OUT} ({len(html)//1024} KB, {len(ev)} eventos)")


if __name__ == "__main__":
    sys.exit(main())
