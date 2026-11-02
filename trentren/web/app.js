"use strict";
const $ = (s) => document.querySelector(s);
const NS = "http://www.w3.org/2000/svg";
const PAISES = {
  chile: { es: "Chile", en: "Chile", ja: "チリ" },
  japon: { es: "Japón", en: "Japan", ja: "日本" },
  eeuu: { es: "EE.UU.", en: "USA", ja: "米国" },
};
const LANGS = { es: "ES", en: "EN", ja: "日本語" };
const LOCALE = { es: "es-CL", en: "en-US", ja: "ja-JP" };
const PANELS = {
  chile: [{ box: [-80, -60.5, -56.4, -16.6], r: [0, 0, 1, 1] }],
  japon: [{ box: [123, 149.5, 24, 46], r: [0, 0, 1, 1] }],
  eeuu: [{ box: [-128.5, -111.5, 31, 50], r: [0, 0, 1, .7] }, { box: [-180, -129, 50, 72], r: [0, .72, .62, .28], lab: "Alaska" },
         { box: [-161, -154, 18.5, 22.5], r: [.64, .72, .36, .28], lab: "Hawaii" }],
};
const PLATE_LAB = { chile: [[-75.4, -27, "FOSA DE ATACAMA", -90]], japon: [[144.6, 38.5, "JAPAN TRENCH", -78], [134.5, 32.2, "NANKAI", -18]],
  eeuu: [[-126.6, 45, "CASCADIA", -88], [-121.3, 35.8, "SAN ANDREAS", -48]] };
const TOOL = {
  pronostico: { g: "∿", es: "Corre el modelo", en: "Runs the model", ja: "モデルを実行" },
  sismos_previos: { g: "◎", es: "Revisa sismos previos", en: "Checks prior quakes", ja: "直前の地震を確認" },
  analogos_historicos: { g: "⟲", es: "Busca análogos históricos", en: "Finds historical analogues", ja: "過去の類似事例" },
  contexto_zona: { g: "▦", es: "Memoria de la zona", en: "Zone memory", ja: "地域の記憶" },
  buscar_noticias: { g: "❝", es: "Busca noticias", en: "Searches the news", ja: "ニュースを検索" },
  exposicion: { g: "⌂", es: "Calcula exposición", en: "Computes exposure", ja: "影響範囲を算出" },
  web: { g: "⌕", es: "Busca en la web", en: "Searches the web", ja: "ウェブ検索" },
};
const AG_ORDER = ["sismologa", "medios", "proteccion"];
const ENUM = { bajo: { es: "bajo", en: "low", ja: "低" }, medio: { es: "medio", en: "medium", ja: "中" }, alto: { es: "alto", en: "high", ja: "高" },
  nulo: { es: "nulo", en: "none", ja: "なし" }, sin_cobertura: { es: "sin cobertura", en: "no coverage", ja: "報道なし" },
  calma: { es: "calma", en: "calm", ja: "平静" }, preocupacion: { es: "preocupación", en: "concern", ja: "懸念" }, alarma: { es: "alarma", en: "alarm", ja: "警戒" } };
const AG_COLOR = { sismologa: "var(--ember)", medios: "var(--cyan)", proteccion: "var(--gold)", coordinador: "var(--ink)" };
const AG_TOOLS = { sismologa: ["pronostico", "sismos_previos", "analogos_historicos", "contexto_zona"], medios: ["buscar_noticias", "web"],
  proteccion: ["exposicion", "contexto_zona"], coordinador: ["pronostico"] };
const AG_NAME = { sismologa: { es: "Sismóloga", en: "Seismologist", ja: "地震学者" }, medios: { es: "Vigía de medios", en: "Media watch", ja: "報道ウォッチ" },
  proteccion: { es: "Protección civil", en: "Civil protection", ja: "防災担当" }, coordinador: { es: "Trentren", en: "Trentren", ja: "トレントレン" } };

const qs = new URLSearchParams(location.search);
let PAIS = qs.get("pais") || localStorage.getItem("tt_pais") || "chile";
let LANG = qs.get("lang") || localStorage.getItem("tt_lang") || (navigator.language || "es").slice(0, 2);
if (!T[LANG]) LANG = "es";
if (!PAISES[PAIS]) PAIS = "chile";
let S = null, GEO = {}, HIST = null, SEL = null, HZ = 0, META = null;
const L = () => T[LANG];

// ---------- util ----------
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const tz = () => S?.info?.tz || "UTC";
const fmt = (iso, o) => new Intl.DateTimeFormat(LOCALE[LANG], { timeZone: tz(), ...o }).format(new Date(iso));
const fDH = (iso) => fmt(iso, { day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" });
const fD = (iso) => fmt(iso, { day: "numeric", month: "long", year: "numeric" });
const num = (x, d = 1) => Number(x).toLocaleString(LOCALE[LANG], { minimumFractionDigits: d, maximumFractionDigits: d });
const pct = (p) => (p == null ? "—" : p >= 0.995 ? ">99%" : p < 0.001 ? "<0.1%" : (p * 100).toLocaleString(LOCALE[LANG], { maximumFractionDigits: p < 0.1 ? 1 : 0 }) + "%");
function ago(iso) {
  if (!iso) return "—";
  const s = (Date.now() - new Date(iso)) / 1000, r = new Intl.RelativeTimeFormat(LOCALE[LANG], { numeric: "auto" });
  return s < 3600 ? r.format(-Math.max(1, Math.round(s / 60)), "minute") : s < 86400 ? r.format(-Math.round(s / 3600), "hour") : r.format(-Math.round(s / 86400), "day");
}
const el = (tag, a = {}, p) => { const n = document.createElementNS(NS, tag); for (const k in a) n.setAttribute(k, a[k]); if (p) p.appendChild(n); return n; };
const depthColor = (z) => (z < 70 ? "var(--ember)" : z < 300 ? "var(--gold)" : "var(--violet)");
const nivelName = (n) => L().nivel[n] || n;
function md(t) {
  return esc(t).replace(/\*\*(.+?)\*\*/g, "<b>$1</b>").split(/\n+/).map((l) => /^\s*[-•]\s+/.test(l) ? "<li>" + l.replace(/^\s*[-•]\s+/, "") + "</li>" : l ? `<p>${l}</p>` : "").join("");
}
function tip(m, html) {
  const t = $("#tip");
  if (!m) { t.style.opacity = 0; return; }
  t.innerHTML = html; t.style.opacity = 1;
  t.style.left = Math.min(m.clientX + 14, innerWidth - 280) + "px"; t.style.top = m.clientY + 14 + "px";
}

// ---------- estático / navegación ----------
function applyStatic() {
  document.documentElement.lang = LANG;
  document.body.classList.toggle("ja", LANG === "ja");
  document.querySelectorAll("[data-i]").forEach((n) => { const v = L()[n.dataset.i]; if (typeof v === "string") n.innerHTML = v; });
  $("#paises").innerHTML = Object.keys(PAISES).map((k) => `<button data-p="${k}" class="${k === PAIS ? "on" : ""}">${PAISES[k][LANG]}</button>`).join("");
  $("#idiomas").innerHTML = Object.keys(LANGS).map((k) => `<button data-l="${k}" class="${k === LANG ? "on" : ""}">${LANGS[k]}</button>`).join("");
  $("#subnav").innerHTML = Object.entries(L().nav).map(([k, v]) => `<a href="#${k}">${v}</a>`).join("");
  $("#hzTabs").innerHTML = L().hz.map((h, i) => `<button class="${i === HZ ? "on" : ""}" data-h="${i}">${h}</button>`).join("");
  $("#legend").innerHTML = L().leg.map((x, i) => `<span><i style="${i < 3 ? `background:${["var(--ember)", "var(--gold)", "var(--violet)"][i]}` : "border:1px solid var(--cyan)"}"></i>${esc(x)}</span>`).join("");
  $("#mapTitle").textContent = L().map_t;
  $("#clockLbl").textContent = `${L().hora} · ${PAISES[PAIS][LANG]}`;
}
$("#paises").addEventListener("click", (e) => { const p = e.target.dataset.p; if (p && p !== PAIS) { PAIS = p; localStorage.setItem("tt_pais", p); HIST = null; SEL = null; S = null; applyStatic(); refresh(); } });
$("#idiomas").addEventListener("click", (e) => { const l = e.target.dataset.l; if (l) { LANG = l; localStorage.setItem("tt_lang", l); applyStatic(); renderAll(); } });
$("#hzTabs").addEventListener("click", (e) => { if (e.target.dataset.h) { HZ = +e.target.dataset.h; applyStatic(); renderHistoria(); } });
setInterval(() => { if (S) $("#clock").textContent = fmt(new Date().toISOString(), { hour: "2-digit", minute: "2-digit", second: "2-digit" }); }, 1000);

// ---------- mapa ----------
async function geo(p) {
  if (!GEO[p]) GEO[p] = await (await fetch(`geo_${p}.json`)).json();
  return GEO[p];
}
async function drawMap() {
  if (!S) return;
  const g = await geo(PAIS), svg = $("#map");
  const W = svg.clientWidth || 380, H = svg.clientHeight || 800;
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  svg.innerHTML = `<defs><filter id="glow" x="-50%" y="-50%" width="200%" height="200%"><feGaussianBlur stdDeviation="2.2" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter></defs>`;
  const evs = HIST ? HIST.sismos : (S.sismos || []);
  const after = HIST ? HIST.despues_24h : [];
  const vivo = S.vivo && S.vivo.pais === PAIS ? S.vivo.evento : null;
  const panels = PANELS[PAIS];
  let n24 = 0;
  panels.forEach((pn, pi) => {
    const [lon0, lon1, lat0, lat1] = pn.box, [rx, ry, rw, rh] = [pn.r[0] * W, pn.r[1] * H, pn.r[2] * W, pn.r[3] * H];
    const cos = Math.cos(((lat0 + lat1) / 2) * Math.PI / 180);
    const k = Math.min(rw / ((lon1 - lon0) * cos), rh / (lat1 - lat0));
    const ox = rx + (rw - (lon1 - lon0) * cos * k) / 2, oy = panels.length > 1 ? ry + (rh - (lat1 - lat0) * k) / 2 : ry + 8;
    const P = (lon, lat) => [ox + (lon - lon0) * cos * k, oy + (lat1 - lat) * k];
    const cid = "clip" + pi;
    el("clipPath", { id: cid }, svg.querySelector("defs")).appendChild(el("rect", { x: ox, y: oy, width: (lon1 - lon0) * cos * k, height: (lat1 - lat0) * k }));
    const G = el("g", { "clip-path": `url(#${cid})` }, svg);
    if (panels.length > 1) el("rect", { x: ox, y: oy, width: (lon1 - lon0) * cos * k, height: (lat1 - lat0) * k, fill: "none", stroke: "var(--line)" }, svg);
    if (pn.lab) { const t = el("text", { x: ox + 8, y: oy + 16, fill: "var(--mut)", "font-size": 10, "letter-spacing": ".16em" }, svg); t.textContent = pn.lab.toUpperCase(); }
    const path = (r) => "M" + r.map(([x, y]) => P(x, y).map((v) => v.toFixed(1)).join(",")).join("L") + "Z";
    for (const r of g.others) el("path", { d: path(r), fill: "#0b0d12", stroke: "rgba(236,231,223,.08)", "stroke-width": .6 }, G);
    for (const r of g.main) el("path", { d: path(r), fill: "rgba(236,231,223,.065)", stroke: "rgba(236,231,223,.5)", "stroke-width": .7, "stroke-linejoin": "round" }, G);
    for (const pl of g.plates) el("polyline", { points: pl.c.map(([x, y]) => P(x, y).join(",")).join(" "), fill: "none", stroke: "var(--cyan)", "stroke-width": 1.2, opacity: .5, filter: "url(#glow)", "stroke-dasharray": "6 3" }, G);
    if (pi === 0) for (const [x, y, txt, rot] of PLATE_LAB[PAIS]) {
      const [px, py] = P(x, y); const t = el("text", { x: px, y: py, fill: "var(--cyan)", opacity: .55, "font-size": 9, "letter-spacing": ".28em", transform: `rotate(${rot} ${px} ${py})` }, G); t.textContent = txt;
    }
    if (PAIS === "chile" && !HIST) zoneBands(G, P);
    const inP = (e) => e.lon >= lon0 && e.lon <= lon1 && e.lat >= lat0 && e.lat <= lat1;
    const now = HIST ? new Date(HIST.t + "Z").getTime() : Date.now();
    for (const e of [...evs].filter(inP).sort((a, b) => a.mag - b.mag)) {
      const [x, y] = P(e.lon, e.lat), age = (now - new Date(e.t)) / 3.6e6;
      if (age < 24 && pi >= 0) n24++;
      const r = Math.max(1.6, Math.pow(Math.max(e.mag - 1.8, .4), 1.55) * (panels.length > 1 && pi ? .7 : 1.05));
      const c = el("circle", { cx: x, cy: y, r, fill: depthColor(e.prof), "fill-opacity": age < 24 ? .85 : .3, stroke: age < 24 ? "rgba(255,255,255,.5)" : "none", "stroke-width": .5 }, G);
      if (age < 6 && e.mag >= 3.5) ripple(G, x, y, r, depthColor(e.prof));
      c.addEventListener("mousemove", (m) => tip(m, `<b>M${num(e.mag)}</b> ${esc(e.tipo_mag || "")}<br>${esc(e.lugar || "")}<br>${fDH(e.t)} · ${Math.round(e.prof)} km · ${e.fuente}`));
      c.addEventListener("mouseleave", () => tip());
    }
    for (const e of after.filter(inP)) {
      const [x, y] = P(e.lon, e.lat);
      el("circle", { cx: x, cy: y, r: Math.max(2, Math.pow(e.mag - 1.8, 1.55)), fill: "none", stroke: "var(--cyan)", "stroke-width": .9, opacity: .8 }, G);
    }
    if (vivo && inP(vivo)) {
      const [x, y] = P(vivo.lon, vivo.lat), r = Math.pow(vivo.mag - 1.8, 1.55) * 1.05;
      el("circle", { cx: x, cy: y, r, fill: "none", stroke: "var(--gold)", "stroke-width": 1.4, "stroke-dasharray": "3 2" }, G);
      ripple(G, x, y, r, "var(--gold)");
      const t = el("text", { x: x + r + 6, y: y + 4, fill: "var(--gold)", "font-size": 10, "letter-spacing": ".14em" }, svg);
      t.textContent = `${L().modo[S.vivo.modo].toUpperCase()} M${num(vivo.mag)}`;
    }
  });
  $("#mapCount").textContent = HIST ? "" : L().map_c(evs.length, n24);
  const mh = $("#mapHist");
  mh.hidden = !HIST;
  if (HIST) mh.innerHTML = `${esc(L().hist_banner(fD(HIST.t + "Z")))} · <i style="color:var(--cyan)">○ ${esc(L().after24)}</i> <button id="backNow">${esc(L().back)}</button>`;
}
function zoneBands(G, P) {
  for (const z of S.zonas || []) {
    const [, y0] = P(0, z.lat_max), [, y1] = P(0, z.lat_min), [xa] = P(-76.5, 0), [xb] = P(-66.2, 0), [xl] = P(-65.6, 0);
    el("line", { x1: xa, x2: xb, y1: y0, y2: y0, stroke: "rgba(236,231,223,.06)", "stroke-dasharray": "2 4" }, G);
    const col = z.alerta_modelo ? "var(--accent)" : z.sismos_24h ? "var(--ink2)" : "var(--dim)", ym = (y0 + y1) / 2;
    el("text", { x: xl, y: ym - 2, fill: col, "font-size": 10 }, G).textContent = z.nombre.split(/ · | y /)[0];
    el("text", { x: xl, y: ym + 10, fill: "var(--dim)", "font-size": 9, "font-family": "JetBrains Mono" }, G).textContent = `${z.sismos_7d} · 7d`;
  }
}
function ripple(g, x, y, r, color) {
  for (let i = 0; i < 2; i++) {
    const c = el("circle", { cx: x, cy: y, r, fill: "none", stroke: color, "stroke-width": 1 }, g);
    c.innerHTML = `<animate attributeName="r" from="${r}" to="${r * 4 + 14}" dur="2.8s" begin="${i * 1.4}s" repeatCount="indefinite"/><animate attributeName="opacity" from=".9" to="0" dur="2.8s" begin="${i * 1.4}s" repeatCount="indefinite"/>`;
  }
}
document.addEventListener("click", (e) => { if (e.target.id === "backNow") { HIST = null; drawMap(); } });

// ---------- sismograma ----------
function drawSeismo() {
  const svg = $("#seismo"), W = svg.clientWidth || 1200, H = 56, mid = 32, now = Date.now(), span = 864e5;
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  const evs = (S?.sismos || []).filter((e) => now - new Date(e.t) < span);
  const amp = new Float32Array(Math.ceil(W) + 2);
  for (const e of evs) {
    const x0 = Math.round(W - ((now - new Date(e.t)) / span) * W), A = Math.min(22, 1.2 * Math.pow(10, .42 * (e.mag - 2)));
    for (let i = 0; i < 70 && x0 + i < W; i++) if (x0 + i >= 0) amp[x0 + i] = Math.max(amp[x0 + i], A * Math.exp(-i / (8 + e.mag * 3)));
  }
  let d = "", seed = 7; const rnd = () => ((seed = (seed * 16807) % 2147483647) / 2147483647 - .5);
  for (let x = 0; x < W; x += 1.5) d += (x ? "L" : "M") + x.toFixed(1) + "," + (mid + rnd() * 2 * (amp[Math.floor(x)] + .9) * (x % 3 < 1.5 ? 1 : -1)).toFixed(1);
  let marks = "";
  for (let h = 6; h < 24; h += 6) { const x = W - (h / 24) * W; marks += `<line x1="${x}" x2="${x}" y1="${H - 9}" y2="${H}" stroke="rgba(236,231,223,.15)"/><text x="${x + 4}" y="${H - 2}" fill="#5b5e66" font-size="9" font-family="JetBrains Mono">-${h}h</text>`; }
  const big = evs.filter((e) => e.mag >= 4.5).map((e) => `<text x="${W - ((now - new Date(e.t)) / span) * W + 3}" y="11" fill="var(--ember)" font-size="9.5" font-family="JetBrains Mono">M${num(e.mag)}</text>`).join("");
  svg.innerHTML = `<defs><linearGradient id="sg" x1="0" x2="1"><stop offset="0" stop-color="#ece7df" stop-opacity=".12"/><stop offset=".85" stop-color="#ece7df" stop-opacity=".55"/><stop offset="1" stop-color="var(--accent)"/></linearGradient></defs>${marks}<path d="${d}" fill="none" stroke="url(#sg)" stroke-width="1"/>${big}<circle cx="${W - 3}" cy="${mid}" r="3" fill="var(--accent)"><animate attributeName="opacity" values="1;.2;1" dur="1.6s" repeatCount="indefinite"/></circle>`;
}

// ---------- ahora ----------
function i18nOf(a) { return (a?.i18n && a.i18n[LANG]) || { titular: a?.titular, resumen: a?.resumen, acciones: a?.acciones || [] }; }
function renderAhora() {
  const v = S.vigente, nv = S.nivel || "vigilancia";
  document.body.dataset.nivel = nv;
  $("#pillNivel").textContent = nivelName(nv);
  $("#lvlWord").innerHTML = `<small>${esc(L().estado)} · ${esc(PAISES[PAIS][LANG])}</small>${esc(nivelName(nv))}`;
  if (v) {
    const x = i18nOf(v);
    $("#vigente").innerHTML = `<article class="card parte"><h3>${esc(x.titular)}</h3><p class="res">${esc(x.resumen)}</p>
      ${x.acciones?.length ? `<ol class="qh">${x.acciones.map((q) => `<li><span>${esc(q)}</span></li>`).join("")}</ol>` : ""}
      <div class="firma"><span>${esc(L().vig_by)}</span><span>${fDH(v.t)} · ${ago(v.t)}</span><span>${esc(L().grav)} <b>${v.gravedad}</b>/5</span><span>${esc(L().conf)} <b>${esc(v.confianza || "")}</b></span>
      <a href="#consejo" class="chip" data-ev="${esc(v.evento_id)}">${esc(L().reg(v.id))}</a></div></article>`;
  } else {
    $("#vigente").innerHTML = `<div class="card parte quiet"><h3>${esc(L().vig_none)}</h3><p class="res">${esc(L().nivel_txt[nv])}</p></div>`;
  }
  const ev = S.sismos || [], d7 = ev.reduce((m, e) => (e.mag > (m?.mag || 0) ? e : m), null);
  const n24 = ev.filter((e) => Date.now() - new Date(e.t) < 864e5).length;
  const k = [[n24, ""], [d7 ? "M" + num(d7.mag) : "—", d7 ? d7.lugar : ""], [(S.triggers || []).length, ""], [(S.consejos || []).length, ""]];
  $("#kpis").innerHTML = k.map(([v2, s], i) => `<div class="kpi"><div class="v">${esc(v2)}</div><div class="l">${esc(L().k[i])}</div>${s ? `<div class="s">${esc(s)}</div>` : ""}</div>`).join("");
  renderPulses();
}
function ring(h, lab) {
  const R = 28, C = 2 * Math.PI * R, col = h.alerta ? "var(--accent)" : "var(--ink2)";
  return `<div class="ring ${h.alerta && h.abierta ? "hot" : ""} ${h.abierta ? "" : "closed"}"><svg viewBox="0 0 72 72"><circle cx="36" cy="36" r="${R}" fill="none" stroke="rgba(255,255,255,.07)" stroke-width="4"/>
    <circle cx="36" cy="36" r="${R}" fill="none" stroke="${col}" stroke-width="4" stroke-linecap="round" stroke-dasharray="${C * Math.max(h.p, .012)} ${C}" transform="rotate(-90 36 36)"/>
    <text x="36" y="41" text-anchor="middle" fill="var(--ink)" font-family="JetBrains Mono" font-size="12.5">${pct(h.p)}</text></svg>
    <div class="lab">${lab}</div><div class="x">${h.abierta ? esc(L().normal(num(h.veces_base))) : esc(L().closed)}</div></div>`;
}
function renderPulses() {
  const tr = (S.triggers || []).slice(0, 3);
  if (!tr.length) { $("#pulses").innerHTML = `<div class="card pulse muted">${esc(L().sin_trig)}</div>`; return; }
  $("#pulses").innerHTML = tr.map((t) => `<div class="card pulse"><div class="ttl"><span class="m">M${num(t.mag)}</span>
    <span class="r"><div class="where">${esc(t.zona)}</div><div class="when">${fDH(t.t)} · ${Math.round(t.prof)} km</div></span></div>
    <div class="rings">${ring(t.horizontes["60min"], L().rings[0])}${ring(t.horizontes["3h"], L().rings[1])}${ring(t.horizontes["24h"], L().rings[2])}</div>
    <div class="foot">${esc(L().mayor7)}: <span class="mono">${pct(t.rj.p_mayor_7d)}</span></div></div>`).join("");
}

// ---------- cómo deciden ----------
function renderPipe() {
  const f = S.vivo && S.vivo.pais === PAIS ? S.vivo.fase : null;
  const act = { analisis: [0, 1, 2], debate: [3], decision: [4], fin: [5] }[f] || [];
  const done = { analisis: [0, 1], debate: [0, 1, 2], decision: [0, 1, 2, 3], fin: [0, 1, 2, 3, 4] }[f] || [];
  $("#pipe").innerHTML = `<div class="flow">${L().pipe.map(([h, d], i) => `<div class="node ${act.includes(i) ? "act" : ""} ${done.includes(i) ? "done" : ""}"><span class="n">${i + 1}</span><b>${esc(h)}</b><small>${esc(d)}</small></div>`).join('<span class="arr">→</span>')}</div>
    <div class="agents">${["sismologa", "medios", "proteccion", "coordinador"].map((k) => `<div class="agent" style="--c:${AG_COLOR[k]}">
      <div class="an">${esc(AG_NAME[k][LANG])}</div><div class="ar">${esc(L().roles[k])}</div>
      <div class="at">${AG_TOOLS[k].map((t) => `<span>${TOOL[t].g} ${esc(TOOL[t][LANG])}</span>`).join("")}</div></div>`).join("")}</div>`;
  const m = META;
  $("#metaBadge").innerHTML = `<span class="mdot ${m?.ok ? "ok" : "bad"}"></span>${L().meta(esc(m?.modelo || S.modelo_llm), S.meta?.llamadas ?? 0)} · <b>${m ? (m.ok ? L().meta_ok : L().meta_bad) : "…"}</b>`;
}

// ---------- consejo ----------
function fromDB(c) {
  const ag = {}, ev = c.evento;
  for (const k of [...AG_ORDER, "coordinador"]) ag[k] = { pasos: [], estado: "listo" };
  for (const d of c.deliberaciones) {
    const a = ag[d.agente] || (ag[d.agente] = { pasos: [] });
    if (d.ronda === 1) { a.informe = d.datos.informe; a.pasos = d.datos.pasos || []; a.modelo = d.datos.modelo; }
    if (d.ronda === 2) a.postura = d.datos.postura;
    if (d.ronda === 3) a.decision = d.datos.decision;
  }
  const ins = c.alertas.find((x) => x.etapa === "instantanea");
  return { evento: ev, modo: ev.modo, fase: "fin", agentes: ag, prensa: c.prensa, instantanea: ins && { nivel: ins.nivel, gravedad: ins.gravedad, latencia_s: ins.latencia_s } };
}
function currentCouncil() {
  if (S.vivo && S.vivo.pais === PAIS && (!SEL || SEL === "vivo")) return S.vivo;
  const c = (S.consejos || []).find((x) => x.evento.id === SEL) || (S.consejos || [])[0];
  return c ? fromDB(c) : null;
}
function gravDots(g) { return `<span class="gdots">${[1, 2, 3, 4, 5].map((i) => `<i class="${i <= (g || 0) ? "on" : ""}"></i>`).join("")}</span>`; }
function stepsHTML(pasos) {
  return pasos.map((s) => {
    if (s.tipo === "dice") return `<li class="say">${md(s.texto)}</li>`;
    const t = TOOL[s.tipo === "web" ? "web" : s.nombre] || { g: "◆", [LANG]: s.nombre };
    return `<li><span class="g">${t.g}</span><span class="tn">${esc(t[LANG])}</span><span class="tr">${esc(s.tipo === "web" ? "«" + s.texto + "»" : s.resumen)}</span></li>`;
  }).join("");
}
function renderCouncil() {
  const chips = [];
  if (S.vivo && S.vivo.pais === PAIS) chips.push(`<button class="${!SEL || SEL === "vivo" ? "on" : ""} live" data-c="vivo">● ${esc(L().live)}</button>`);
  for (const c of S.consejos || []) chips.push(`<button class="${SEL === c.evento.id || (!SEL && !S.vivo && c === S.consejos[0]) ? "on" : ""}" data-c="${esc(c.evento.id)}"><span class="mchip ${c.evento.modo}">${esc(L().modo[c.evento.modo])}</span> M${num(c.evento.mag)} ${esc((c.evento.lugar || "").slice(0, 34))}</button>`);
  $("#cpick").innerHTML = chips.join("");
  const v = currentCouncil();
  if (!v) { $("#council").innerHTML = `<div class="card empty">${esc(L().co_empty)}</div>`; return; }
  const ev = v.evento, live = v.fase !== "fin", coord = v.agentes.coordinador || {}, dec = coord.decision;
  const st = (a) => ({ pensando: L().thinking, debatiendo: L().debating, decidiendo: L().deciding, esperando: L().waiting, listo: L().done }[a.estado] || "");
  let h = `<div class="evhead"><span class="mchip ${v.modo}">${esc(L().modo[v.modo])}</span><span class="em">M${num(ev.mag)}</span>
    <span class="el">${esc(ev.lugar || "")}<small>${fDH(ev.t)} · ${Math.round(ev.prof)} km${live ? ` · <b class="livetxt">${esc(L().live)}<span class="thinking"><i></i><i></i><i></i></span></b>` : ""}</small></span>
    ${v.instantanea ? `<span class="inst">${L().inst(esc(nivelName(v.instantanea.nivel)), v.instantanea.gravedad, num(v.instantanea.latencia_s, 2))}</span>` : ""}</div>`;
  h += `<h4 class="rh">${esc(L().r1)}</h4><div class="r1">${AG_ORDER.map((k) => {
    const a = v.agentes[k] || { pasos: [] }, inf = a.informe || {};
    const tag = inf.escalamiento || inf.tono || inf.riesgo_tsunami;
    const list = inf.cifras || inf.hechos || inf.acciones || [];
    return `<div class="card ag" style="--c:${AG_COLOR[k]}"><div class="agh"><b>${esc(AG_NAME[k][LANG])}</b><span>${a.informe ? gravDots(inf.gravedad) : `<i class="st">${esc(st(a))}</i>`}</span></div>
      ${tag ? `<span class="tag">${esc(ENUM[tag]?.[LANG] || tag)}</span>` : ""}
      ${inf.argumento ? `<p class="arg">${esc(inf.argumento)}</p>` : `<p class="arg muted">${esc(st(a))}<span class="thinking"><i></i><i></i><i></i></span></p>`}
      ${list.length ? `<ul class="bul">${list.slice(0, 4).map((x) => `<li>${esc(x)}</li>`).join("")}</ul>` : ""}
      <details ${live ? "open" : ""}><summary>${esc(L().tools)} · ${a.pasos.filter((s) => s.tipo !== "dice").length}</summary><ul class="steps">${stepsHTML(a.pasos)}</ul></details></div>`;
  }).join("")}</div>`;
  const pos = AG_ORDER.filter((k) => v.agentes[k]?.postura);
  if (pos.length || v.fase === "debate") {
    h += `<h4 class="rh">${esc(L().r2)}</h4><div class="debate">${pos.map((k) => {
      const a = v.agentes[k], p = a.postura, g0 = a.informe?.gravedad;
      return `<div class="bub" style="--c:${AG_COLOR[k]}"><div class="bh"><b>${esc(AG_NAME[k][LANG])}</b>
        <span class="mono">${g0 ?? "?"} → ${p.gravedad ?? "?"}</span><span class="ag2 ${p.de_acuerdo ? "y" : "n"}">${esc(p.de_acuerdo ? L().agree : L().disagree)}</span></div><p>${esc(p.replica)}</p></div>`;
    }).join("")}${v.fase === "debate" ? `<div class="bub muted">${esc(L().debating)}<span class="thinking"><i></i><i></i><i></i></span></div>` : ""}</div>`;
  }
  if (dec || v.fase === "decision") {
    h += `<h4 class="rh">${esc(L().r3)}</h4>`;
    if (!dec) h += `<div class="card verdict muted">${esc(L().deciding)}<span class="thinking"><i></i><i></i><i></i></span></div>`;
    else {
      const x = dec.i18n?.[LANG] || {};
      h += `<div class="card verdict lv-${dec.nivel}"><div class="vh"><span class="vl">${esc(nivelName(dec.nivel))}</span>${gravDots(dec.gravedad)}<span class="vm">${esc(L().reg(dec.registro_id))} · ${esc(L().lat(num(dec.latencia_s, 0)))} · <span class="mono">${esc(dec.modelo || "")}</span></span></div>
        <h3>${esc(x.titular)}</h3><p class="res">${esc(x.resumen)}</p>${x.acciones?.length ? `<ol class="qh">${x.acciones.map((q) => `<li><span>${esc(q)}</span></li>`).join("")}</ol>` : ""}
        <p class="why">${esc(dec.razon_decision || "")}</p></div>`;
      if (dec.realidad) {
        const R = dec.realidad, lab = { "60min": L().rings[0], "3h": L().rings[1], "24h": L().rings[2], "7d": "7d" };
        h += `<div class="card reality"><h4>${esc(L().real)}</h4><div class="rgrid">${Object.entries(R).map(([k, r]) => `<div><b>${r.n_m5}</b><span>${esc(L().r_m5(lab[k]))}${r.mayor ? ` · M${num(r.mayor)}` : ""}</span></div>`).join("")}</div></div>`;
      }
    }
  }
  if (v.prensa?.length) h += `<h4 class="rh">${esc(L().prensa)} · ${v.prensa.length}</h4><div class="press">${v.prensa.slice(0, 12).map((p) => `<a href="${esc(p.url)}" target="_blank" rel="noopener"><span>${esc(p.medio || "")}</span>${esc(p.titulo)}</a>`).join("")}</div>`;
  $("#council").innerHTML = h;
}
$("#cpick").addEventListener("click", (e) => { const b = e.target.closest("button"); if (b) { SEL = b.dataset.c; renderCouncil(); } });
document.addEventListener("click", (e) => { const a = e.target.closest("[data-ev]"); if (a) { SEL = a.dataset.ev; renderCouncil(); } });

// ---------- registro ----------
function renderRegistro() {
  const r = S.registro || [];
  $("#regTable").innerHTML = `<thead><tr>${L().cols.map((c) => `<th>${esc(c)}</th>`).join("")}</tr></thead><tbody>${r.length ? r.map((a) => `<tr class="lv-${a.nivel}">
    <td class="mono">${fDH(a.t)}</td><td>M${a.mag != null ? num(a.mag) : "—"} <small>${esc((a.lugar || "").slice(0, 28))}</small></td>
    <td><span class="mchip ${a.modo}">${esc(L().modo[a.modo] || a.modo)}</span></td><td>${esc(L().etapa[a.etapa] || a.etapa)}</td>
    <td><b class="nv">${esc(nivelName(a.nivel))}</b></td><td class="mono">${a.gravedad ?? "—"}</td><td class="mono">${pct(a.p60)}</td><td class="mono">${pct(a.p24h)}</td>
    <td>${esc(a.autor || "")}</td><td class="mono">${a.latencia_s != null ? num(a.latencia_s, a.latencia_s < 10 ? 2 : 0) + " s" : "—"}</td></tr>`).join("") : `<tr><td colspan="10" class="muted">${esc(L().re_empty)}</td></tr>`}</tbody>`;
}

// ---------- historia ----------
function renderHistoria() {
  const H = S.historia;
  if (!H) return;
  const hk = ["60min", "3h", "24h"][HZ], A = H.anios;
  const W = 760, Hh = 230, pad = 28, bw = (W - pad * 2) / A.length;
  const mx = Math.max(...A.map((a) => Math.max(a[hk].escaladas, a[hk].alertas)), 1);
  const y = (v) => Hh - 34 - (Math.sqrt(v) / Math.sqrt(mx)) * (Hh - 64);
  let s = "";
  A.forEach((a, i) => {
    const m = a[hk], x = pad + i * bw, w = bw / 4;
    s += `<rect x="${x + bw * .12}" y="${y(m.escaladas)}" width="${w}" height="${Hh - 34 - y(m.escaladas)}" fill="rgba(236,231,223,.25)"><title>${m.escaladas}</title></rect>`;
    s += `<rect x="${x + bw * .12 + w + 2}" y="${y(m.alertas)}" width="${w}" height="${Hh - 34 - y(m.alertas)}" fill="none" stroke="var(--accent)"><title>${m.alertas}</title></rect>`;
    s += `<rect x="${x + bw * .12 + 2 * w + 4}" y="${y(m.aciertos)}" width="${w}" height="${Hh - 34 - y(m.aciertos)}" fill="var(--accent)"><title>${m.aciertos}</title></rect>`;
    s += `<text x="${x + bw / 2}" y="${Hh - 16}" text-anchor="middle" fill="var(--mut)" font-size="10.5" font-family="JetBrains Mono">${a.anio}</text>`;
    s += `<text x="${x + bw / 2}" y="${Math.min(y(m.escaladas), y(m.alertas)) - 6}" text-anchor="middle" fill="var(--ink2)" font-size="9.5" font-family="JetBrains Mono">${m.veces_azar ? "×" + num(m.veces_azar, m.veces_azar < 10 ? 1 : 0) : ""}</text>`;
  });
  const lg = L().ch.map((c, i) => `<span><i style="${["background:rgba(236,231,223,.25)", "border:1px solid var(--accent)", "background:var(--accent)"][i]}"></i>${esc(c)}</span>`).join("") + `<span class="mono">×n = ${esc(L().azar)}</span>`;
  $("#yearChart").innerHTML = `<svg viewBox="0 0 ${W} ${Hh}" width="100%">${s}</svg><div class="legend">${lg}</div>`;
  const t = H.total[hk];
  $("#histStats").innerHTML = [[t.n.toLocaleString(LOCALE[LANG]), ""], [t.alertas, `${t.aciertos} ✓`], [pct(t.precision), `${esc(L().azar)}: ×${num(t.veces_azar)}`], [pct(t.recall), `${t.aciertos}/${t.escaladas}`]]
    .map(([v, sub], i) => `<div class="stat"><div class="v">${v}</div><div class="l">${esc(L().hs[i])}</div><div class="s">${sub}</div></div>`).join("");
  $("#bigs").innerHTML = H.grandes.slice().reverse().map((g) => `<div class="card big"><div class="bt"><span class="m">M${num(g.mag)}</span><span><b>${esc(g.zona)}</b><small>${fD(g.t + "Z")} · ${g.prof} km</small></span></div>
    <div class="bl">${esc(L().big_before(g.previos_14d, g.previos_alerta))}</div>
    ${g.modelo ? `<div class="bl">${esc(L().big_at)}: <span class="mono">${pct(g.modelo["60min"])} · ${pct(g.modelo["3h"])} · ${pct(g.modelo["24h"])}</span></div>` : ""}
    <div class="bl strong">${esc(L().big_after(g.replicas_m5_24h, g.mayor_replica_24h ? num(g.mayor_replica_24h) : null))}</div>
    <div class="bb"><button data-map="${g.t}">${esc(L().btn_map)}</button><button class="go" data-rev="${g.t}">${esc(L().btn_rev)}</button></div></div>`).join("");
}
$("#bigs").addEventListener("click", async (e) => {
  const m = e.target.dataset.map, r = e.target.dataset.rev;
  if (m) { HIST = await (await fetch(`/api/momento?pais=${PAIS}&t=${encodeURIComponent(m)}`)).json(); drawMap(); if (innerWidth < 1000) $(".mapcol").scrollIntoView(); }
  if (r) launch("/api/revivir", { pais: PAIS, t: r }, e.target);
});

// ---------- simular ----------
function renderSims() {
  const E = S.info?.escenarios || {};
  $("#sims").innerHTML = Object.entries(E).map(([k, s]) => `<button class="simc" data-k="${k}"><div class="m">M${num(s.mag)}</div><div class="p">${esc(s.lugar)}</div><div class="d">${s.prof} km</div></button>`).join("");
}
$("#sims").addEventListener("click", (e) => { const b = e.target.closest(".simc"); if (b) launch("/api/simulacro", { pais: PAIS, escenario: b.dataset.k }, b); });
async function launch(url, body, btn) {
  btn.disabled = true;
  try {
    const j = await (await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) })).json();
    if (j.error) { alert(j.error); return; }
    SEL = "vivo"; HIST = null;
    $("#consejo").scrollIntoView();
  } finally { setTimeout(() => (btn.disabled = false), 4000); }
}

// ---------- método ----------
function renderMetodo() {
  const m = S.metricas;
  if (!m) return;
  const segs = [[L().train[0], m.periodo_train || "1990–2020", m.n_train, 31], [L().train[1], "2021–2023", m.n_calib, 3], [L().train[2], "2024 →", m.n_holdout, 3]];
  $("#trainbar").innerHTML = `<div class="tb">${segs.map(([n, p, c, w], i) => `<div class="seg s${i}" style="flex:${w}"><b>${esc(n)}</b><span>${esc(p)}</span><small>${(c || 0).toLocaleString(LOCALE[LANG])} ${esc(L().trig)}</small></div>`).join("")}</div>`;
  const a = m.holdout["60min"];
  $("#stats").innerHTML = [[a.auc != null ? num(a.auc, 2) : "—"], ["×" + num(a.precision / Math.max(a.base_rate, 1e-6), 0)], [Math.round(a.recall * 100) + "%"], ["20+"]]
    .map(([v], i) => `<div class="stat"><div class="v">${v}</div><div class="l">${esc(L().st[i])}</div><div class="s">${esc(L().st_s[i])}</div></div>`).join("");
  $("#zones").innerHTML = `<div class="zrow zhead"><span>${esc(L().zc[0])}</span><span>${esc(L().zc[1])}</span><span class="y">${esc(L().zc[2])}</span><span class="k">${esc(L().zc[3])}</span></div>` +
    (S.zonas || []).map((z) => `<div class="zrow"><span class="n">${esc(z.nombre)}<small>${esc(z.ultimo_gran || L().none)}</small></span>
    <span class="bar"><i style="width:${Math.min(100, z.anios_desde_gran || 0)}%"></i></span><span class="y">${z.anios_desde_gran != null ? num(z.anios_desde_gran) : "—"}</span><span class="k">${z.grandes_desde_1900}</span></div>`).join("");
}

// ---------- pie ----------
function renderFoot() {
  $("#links").innerHTML = (S.info?.links || []).map(([n, u]) => `<a href="${esc(u)}" target="_blank" rel="noopener">${esc(n)}</a>`).join("");
  $("#ftTsunami").textContent = S.info?.tsunami || "";
  $("#ftModel").textContent = L().model(S.modelo_llm);
  const u = S.uso || { in: 0, out: 0 };
  $("#uso").textContent = L().uso((u.in / 1e3).toFixed(0), (u.out / 1e3).toFixed(0), (u.in * 4e-6 + u.out * 2e-5).toFixed(2));
}

function renderAll() {
  if (!S) return;
  renderAhora(); renderPipe(); renderCouncil(); renderRegistro(); renderHistoria(); renderSims(); renderMetodo(); renderFoot(); drawSeismo(); drawMap();
}
async function refresh() {
  try { S = await (await fetch(`/api/estado?pais=${PAIS}`, { cache: "no-store" })).json(); renderAll(); }
  catch (e) { console.error(e); }
}
let rt = null, lt = 0;
const soon = () => { clearTimeout(rt); rt = setTimeout(refresh, 500); };
function stream() {
  const es = new EventSource("/api/stream");
  es.onmessage = (m) => {
    const msg = JSON.parse(m.data);
    if (!S) return;
    if (msg.tipo === "consejo") {
      S.vivo = msg.vivo;
      if (Date.now() - lt > 250) { lt = Date.now(); renderPipe(); renderCouncil(); }
    } else if (msg.tipo === "fin" || (msg.tipo === "estado" && msg.pais === PAIS)) soon();
  };
  es.onerror = () => { es.close(); setTimeout(stream, 5000); };
}
fetch("/api/meta").then((r) => r.json()).then((m) => { META = m; if (S) renderPipe(); }).catch(() => {});
applyStatic(); refresh(); if (!qs.has("nostream")) stream();
setInterval(refresh, 60000);
addEventListener("resize", () => { drawMap(); drawSeismo(); });
