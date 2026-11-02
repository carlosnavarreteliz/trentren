"use strict";
const $ = (s) => document.querySelector(s);
const NS = "http://www.w3.org/2000/svg";
const TZ = "America/Santiago";
const NIVEL = { vigilancia: "Vigilancia", atencion: "Atención", alerta: "Alerta" };
const NIVEL_TXT = {
  vigilancia: "Actividad dentro de lo normal para Chile.",
  atencion: "Hay una secuencia activa: el vigía la sigue de cerca.",
  alerta: "Sismo fuerte o escalamiento probable. Sigue a las autoridades.",
};
const TOOL = {
  consultar_sismos: ["◎", "Lee los sismos"], pronostico: ["∿", "Corre el modelo"], estado_zonas: ["▦", "Revisa las zonas"],
  buscar_noticias: ["❝", "Busca noticias"], fijar_nivel: ["▲", "Fija el nivel"], publicar_parte: ["✎", "Publica un parte"],
  programar_revision: ["◷", "Se agenda una revisión"], anotar_memoria: ["✦", "Anota en su memoria"],
};
const TRIG = { sismo: "Sismo detectado", ronda: "Ronda de vigilancia", revision: "Revisión autoagendada",
  arranque: "Despertar", simulacro: "Simulacro" };
const SIMS = [
  { k: "iquique", m: "7,6", p: "Frente a Iquique", d: "Interplaca, 28 km · costa de Tarapacá" },
  { k: "valparaiso", m: "6,4", p: "Costa de Valparaíso", d: "Interplaca, 32 km · zona central" },
  { k: "calama", m: "5,9", p: "Norte de Calama", d: "Intraplaca, 110 km · Antofagasta" },
  { k: "maule", m: "5,6", p: "Costa de Constitución", d: "Interplaca, 30 km · Maule" },
];
const SUGS = ["¿Hubo algún sismo sentido hoy?", "¿Qué tan probable es una réplica ahora?",
  "¿Se puede predecir un terremoto?", "¿Qué hago si tiembla fuerte en la costa?"];

let S = null, GEO = null, canal = "real", openEp = new Set();

// ---------- util ----------
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const fmtH = (iso) => new Intl.DateTimeFormat("es-CL", { timeZone: TZ, hour: "2-digit", minute: "2-digit" }).format(new Date(iso));
const fmtDH = (iso) => new Intl.DateTimeFormat("es-CL", { timeZone: TZ, day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" }).format(new Date(iso));
function ago(iso) {
  if (!iso) return "—";
  const s = (Date.now() - new Date(iso)) / 1000;
  if (s < 60) return "hace instantes";
  if (s < 3600) return `hace ${Math.round(s / 60)} min`;
  if (s < 86400) return `hace ${Math.round(s / 3600)} h`;
  return `hace ${Math.round(s / 86400)} d`;
}
const pct = (p) => (p == null ? "—" : p >= 0.995 ? ">99%" : p < 0.001 ? "<0,1%" : (p * 100).toLocaleString("es-CL", { maximumFractionDigits: p < 0.1 ? 1 : 0 }) + "%");
const dec = (x, d = 1) => Number(x).toLocaleString("es-CL", { minimumFractionDigits: d, maximumFractionDigits: d });
function md(t) {
  const lines = esc(t).split(/\n+/);
  let out = "", inList = false;
  for (let l of lines) {
    l = l.replace(/\*\*(.+?)\*\*/g, "<b>$1</b>").replace(/(^|\s)\*(\S.+?)\*(?=\s|$)/g, "$1<i>$2</i>");
    if (/^\s*[-•]\s+/.test(l)) { if (!inList) { out += "<ul>"; inList = true; } out += "<li>" + l.replace(/^\s*[-•]\s+/, "") + "</li>"; }
    else { if (inList) { out += "</ul>"; inList = false; } if (l.trim()) out += "<p style='margin:4px 0'>" + l + "</p>"; }
  }
  return out + (inList ? "</ul>" : "");
}
const el = (tag, attrs = {}, parent) => {
  const n = document.createElementNS(NS, tag);
  for (const k in attrs) n.setAttribute(k, attrs[k]);
  if (parent) parent.appendChild(n);
  return n;
};
const depthColor = (z) => (z < 70 ? "var(--ember)" : z < 300 ? "var(--gold)" : "var(--violet)");

// ---------- reloj ----------
setInterval(() => {
  $("#clock").textContent = new Intl.DateTimeFormat("es-CL", { timeZone: TZ, hour: "2-digit", minute: "2-digit", second: "2-digit" }).format(new Date());
}, 1000);

// ---------- mapa ----------
const BOX = { lon0: -80, lon1: -60.5, lat0: -56.4, lat1: -16.6 };
function drawMap() {
  const svg = $("#map");
  if (!GEO || !S) return;
  const W = svg.clientWidth || 380, Hh = svg.clientHeight || 800;
  const cos = Math.cos((36 * Math.PI) / 180);
  const k = Math.min(W / ((BOX.lon1 - BOX.lon0) * cos), Hh / (BOX.lat1 - BOX.lat0));
  const ox = (W - (BOX.lon1 - BOX.lon0) * cos * k) / 2;
  const P = (lon, lat) => [ox + (lon - BOX.lon0) * cos * k, (BOX.lat1 - lat) * k];
  svg.setAttribute("viewBox", `0 0 ${W} ${Hh}`);
  svg.innerHTML = "";
  const defs = el("defs", {}, svg);
  defs.innerHTML = `<filter id="glow" x="-50%" y="-50%" width="200%" height="200%"><feGaussianBlur stdDeviation="2.4" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
    <linearGradient id="land" x1="0" y1="0" x2="1" y2="0"><stop offset="0" stop-color="#ece7df" stop-opacity=".09"/><stop offset="1" stop-color="#ece7df" stop-opacity=".03"/></linearGradient>`;
  const path = (rings) => rings.map((r) => "M" + r.map(([x, y]) => P(x, y).map((v) => v.toFixed(1)).join(",")).join("L") + "Z").join("");
  // vecinos
  for (const n of ["Argentina", "Peru", "Bolivia"])
    for (const poly of GEO[n] || []) el("path", { d: path(poly), fill: "#0b0d12", stroke: "rgba(236,231,223,.07)", "stroke-width": ".6" }, svg);
  // zonas (bandas)
  const zg = el("g", {}, svg);
  const zonas = S.zonas || [];
  zonas.forEach((z) => {
    const [, y0] = P(0, z.lat_max), [, y1] = P(0, z.lat_min);
    const [xa] = P(-76.5, 0), [xb] = P(-66.2, 0), [xl] = P(-65.6, 0);
    el("line", { x1: xa, x2: xb, y1: y0, y2: y0, stroke: "rgba(236,231,223,.06)", "stroke-dasharray": "2 4" }, zg);
    const heat = z.alerta_modelo ? "var(--accent)" : z.sismos_24h ? "var(--ink2)" : "var(--dim)";
    const ym = (y0 + y1) / 2;
    const t = el("text", { x: xl, y: ym - 3, fill: heat, "font-size": 10, "font-family": "Inter", "letter-spacing": ".06em" }, zg);
    t.textContent = z.nombre.split(" · ")[0];
    const t2 = el("text", { x: xl, y: ym + 10, fill: "var(--dim)", "font-size": 9.5, "font-family": "JetBrains Mono" }, zg);
    t2.textContent = `${z.sismos_7d} sismos 7d` + (z.mayor_7d ? ` · máx ${dec(z.mayor_7d)}` : "");
    if (z.sismos_24h) el("rect", { x: xl - 7, y: ym - 10, width: 2, height: 22, fill: heat, rx: 1 }, zg);
  });
  // Chile
  for (const poly of GEO.Chile) el("path", { d: path(poly), fill: "url(#land)", stroke: "rgba(236,231,223,.5)", "stroke-width": ".7", "stroke-linejoin": "round" }, svg);
  // fosa
  for (const line of GEO.trench) {
    const pts = line.filter(([x, y]) => y > BOX.lat0 && y < BOX.lat1 && x < -66).map(([x, y]) => P(x, y).join(","));
    if (pts.length < 2) continue;
    el("polyline", { points: pts.join(" "), fill: "none", stroke: "var(--cyan)", "stroke-width": 1.3, opacity: .55, filter: "url(#glow)", "stroke-dasharray": "6 3" }, svg);
  }
  const [tx, ty] = P(-75.6, -27);
  const ft = el("text", { x: tx, y: ty, fill: "var(--cyan)", opacity: .55, "font-size": 9.5, "letter-spacing": ".3em", transform: `rotate(-90 ${tx} ${ty})`, "font-family": "Inter" }, svg);
  ft.textContent = "FOSA DE ATACAMA";
  const [px, py] = P(-78.6, -40);
  const oc = el("text", { x: px, y: py, fill: "rgba(98,208,227,.28)", "font-size": 12, "font-style": "italic", "font-family": "Instrument Serif", transform: `rotate(-90 ${px} ${py})` }, svg);
  oc.textContent = "Océano Pacífico";
  // sismos
  const now = Date.now();
  const ev = [...(S.sismos || [])].filter((e) => e.lat > BOX.lat0 && e.lat < BOX.lat1 && e.lon > BOX.lon0 && e.lon < BOX.lon1).sort((a, b) => a.mag - b.mag);
  const g = el("g", {}, svg);
  let n24 = 0;
  for (const e of ev) {
    const [x, y] = P(e.lon, e.lat);
    const age = (now - new Date(e.t)) / 3.6e6;
    if (age < 24) n24++;
    const r = Math.max(1.6, Math.pow(Math.max(e.mag - 1.8, 0.4), 1.55) * 1.05);
    const c = el("circle", { cx: x, cy: y, r, fill: depthColor(e.prof), "fill-opacity": age < 24 ? 0.85 : 0.32, stroke: age < 24 ? "rgba(255,255,255,.5)" : "none", "stroke-width": .5 }, g);
    if (age < 6 && e.mag >= 3.5) ripple(g, x, y, r, depthColor(e.prof));
    c.addEventListener("mousemove", (m) => tip(m, `<b>M${dec(e.mag)}</b> <span class="mono">${esc(e.tipo_mag || "")}</span><br>${esc(e.lugar || "")}<br>${fmtDH(e.t)} · ${Math.round(e.prof)} km · ${e.fuente}`));
    c.addEventListener("mouseleave", () => tip());
  }
  const sim = simActivo();
  if (sim) {
    const [x, y] = P(sim.lon, sim.lat);
    const r = Math.pow(sim.mag - 1.8, 1.55) * 1.05;
    el("circle", { cx: x, cy: y, r, fill: "none", stroke: "var(--gold)", "stroke-width": 1.2, "stroke-dasharray": "3 2" }, g);
    ripple(g, x, y, r, "var(--gold)");
    const t = el("text", { x: x + r + 6, y: y + 4, fill: "var(--gold)", "font-size": 10, "letter-spacing": ".18em", "font-family": "Inter" }, g);
    t.textContent = "SIMULACRO M" + dec(sim.mag);
  }
  $("#mapCount").textContent = `${ev.length} sismos en 7 días · ${n24} en 24 h`;
}
function ripple(g, x, y, r, color) {
  for (let i = 0; i < 2; i++) {
    const c = el("circle", { cx: x, cy: y, r, fill: "none", stroke: color, "stroke-width": 1 }, g);
    c.innerHTML = `<animate attributeName="r" from="${r}" to="${r * 4 + 14}" dur="2.8s" begin="${i * 1.4}s" repeatCount="indefinite"/>
      <animate attributeName="opacity" from=".9" to="0" dur="2.8s" begin="${i * 1.4}s" repeatCount="indefinite"/>`;
  }
}
function tip(m, html) {
  const t = $("#tip");
  if (!m) { t.style.opacity = 0; return; }
  t.innerHTML = html; t.style.opacity = 1;
  t.style.left = Math.min(m.clientX + 14, innerWidth - 270) + "px"; t.style.top = m.clientY + 14 + "px";
}

// ---------- sismograma ----------
function drawSeismo() {
  const svg = $("#seismo");
  const W = svg.clientWidth || 1200, H = 64, mid = 38;
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  const now = Date.now(), span = 864e5;
  const evs = (S?.sismos || []).filter((e) => now - new Date(e.t) < span);
  const amp = new Float32Array(Math.ceil(W));
  for (const e of evs) {
    const x0 = Math.round(W - ((now - new Date(e.t)) / span) * W);
    const A = Math.min(26, 1.2 * Math.pow(10, 0.42 * (e.mag - 2)));
    for (let i = 0; i < 70 && x0 + i < W; i++) if (x0 + i >= 0) amp[x0 + i] = Math.max(amp[x0 + i], A * Math.exp(-i / (8 + e.mag * 3)));
  }
  let d = "", seed = 7;
  const rnd = () => ((seed = (seed * 16807) % 2147483647) / 2147483647 - 0.5);
  for (let x = 0; x < W; x += 1.5) {
    const a = amp[Math.floor(x)] + 0.9;
    d += (x ? "L" : "M") + x.toFixed(1) + "," + (mid + rnd() * 2 * a * (x % 3 < 1.5 ? 1 : -1)).toFixed(1);
  }
  let marks = "";
  for (let h = 0; h <= 24; h += 6) {
    const x = W - (h / 24) * W;
    marks += `<line x1="${x}" x2="${x}" y1="${H - 10}" y2="${H}" stroke="rgba(236,231,223,.15)"/>`;
    if (h && h < 24) marks += `<text x="${x + 4}" y="${H - 3}" fill="#5b5e66" font-size="9" font-family="JetBrains Mono">-${h}h</text>`;
  }
  const big = evs.filter((e) => e.mag >= 4).map((e) => {
    const x = W - ((now - new Date(e.t)) / span) * W;
    return `<text x="${x + 3}" y="12" fill="var(--ember)" font-size="9.5" font-family="JetBrains Mono">M${dec(e.mag)}</text>`;
  }).join("");
  svg.innerHTML = `<defs><linearGradient id="sg" x1="0" x2="1"><stop offset="0" stop-color="#ece7df" stop-opacity=".12"/><stop offset=".85" stop-color="#ece7df" stop-opacity=".55"/><stop offset="1" stop-color="var(--accent)"/></linearGradient></defs>
    ${marks}<path d="${d}" fill="none" stroke="url(#sg)" stroke-width="1"/>${big}
    <circle cx="${W - 3}" cy="${mid}" r="3" fill="var(--accent)"><animate attributeName="opacity" values="1;.2;1" dur="1.6s" repeatCount="indefinite"/></circle>`;
}

// ---------- situación ----------
function simActivo() {
  const s = S?.simulacro;
  if (!s || !s.inicio || Date.now() - new Date(s.inicio) > 3 * 3.6e6) return null;
  return s.evento;
}
function renderSituacion() {
  const sim = simActivo();
  $("#tabs").style.display = sim ? "flex" : "none";
  if (!sim) canal = "real";
  document.querySelectorAll("#tabs button").forEach((b) => b.classList.toggle("on", b.dataset.c === canal));
  const src = canal === "sim" ? S.simulacro : S;
  const nv = (src.nivel && src.nivel.nivel) || "vigilancia";
  document.body.dataset.nivel = canal === "sim" ? (S.nivel?.nivel || "vigilancia") : nv;
  $("#pillNivel").textContent = NIVEL[S.nivel?.nivel || "vigilancia"];
  $("#pillZona").textContent = S.nivel?.zona ? "· " + S.nivel.zona : "";
  $("#lvlWord").innerHTML = `<small>${canal === "sim" ? "Simulacro · nivel decidido por el agente" : "Estado de situación · Chile"}</small>
    <span style="color:${canal === "sim" ? "var(--gold)" : "var(--accent)"}">${NIVEL[nv]}</span>`;
  const p = src.parte;
  const band = canal === "sim" ? `<div class="sim-band">Simulacro · M${dec(sim.mag)} ${esc(sim.lugar)} · evento ficticio</div>` : "";
  if (!p) {
    $("#parte").innerHTML = band + `<div class="card parte"><h3>${NIVEL_TXT[nv]}</h3><p class="res">${esc(src.nivel?.razon || "Trentren aún no publica un parte de situación.")}</p>
      <div class="firma"><span>Trentren · agente autónomo</span></div></div>`;
    return;
  }
  const f = (p.fuentes || []).filter((x) => x && x.url);
  $("#parte").innerHTML = band + `<article class="card parte">
    <h3>${esc(p.titular)}</h3><p class="res">${esc(p.resumen)}</p>
    ${p.que_hacer?.length ? `<ol class="qh">${p.que_hacer.map((q) => `<li><span>${esc(q)}</span></li>`).join("")}</ol>` : ""}
    <div class="firma"><span>Redactado por <b>Trentren</b>, agente autónomo</span><span>${fmtDH(p.t)} · ${ago(p.t)}</span><span>Confianza <b>${esc(p.confianza)}</b></span>
      <span style="flex-basis:100%">Razón del nivel: ${esc(src.nivel?.razon || "")}</span></div>
    ${f.length ? `<div class="chips">${f.map((x) => `<a class="chip" href="${esc(x.url)}" target="_blank" rel="noopener" title="${esc(x.titulo)}">${esc(x.titulo || x.url)}</a>`).join("")}</div>` : ""}
  </article>`;
}
document.querySelectorAll("#tabs button").forEach((b) => b.addEventListener("click", () => { canal = b.dataset.c; renderSituacion(); }));

// ---------- mente ----------
function stepHTML(s, isNew) {
  if (s.tipo === "pensamiento") return `<li data-g="·" class="${isNew ? "new" : ""}"><span class="md">${md(s.texto)}</span></li>`;
  if (s.tipo === "web") return `<li data-g="⌕" class="web ${isNew ? "new" : ""}"><span class="tn">Busca en la web</span><span class="tr">«${esc(s.texto)}»</span></li>`;
  if (s.tipo === "error") return `<li data-g="!" class="err">${esc(s.texto)}</li>`;
  const [g, lab] = TOOL[s.nombre] || ["◆", s.nombre];
  let det = s.resumen || "";
  if (s.nombre === "buscar_noticias") det = `«${s.args?.consulta || ""}» → ${det}`;
  if (s.nombre === "fijar_nivel") det = `${(NIVEL[s.args?.nivel] || s.args?.nivel || "").toUpperCase()} · ${s.args?.razon || ""}`;
  if (s.nombre === "publicar_parte") det = s.args?.titular || "";
  if (s.nombre === "programar_revision") det = `en ${s.args?.minutos} min · ${s.args?.motivo || ""}`;
  if (s.nombre === "anotar_memoria") det = s.args?.nota || "";
  return `<li data-g="${g}" class="${isNew ? "new" : ""}"><span class="tn">${lab}</span><span class="tr">${esc(det)}</span></li>`;
}
let liveCount = 0;
function renderMind() {
  const a = S.actual;
  let html = "";
  if (a) {
    const isNew = a.pasos.length > liveCount;
    html += `<div class="card live"><div class="hd" style="display:flex;gap:12px;align-items:center;font-size:12px;color:var(--mut)">
      <span class="trig ${a.disparador.tipo}">${TRIG[a.disparador.tipo] || a.disparador.tipo}</span><span>deliberando ahora<span class="thinking"><i></i><i></i><i></i></span></span></div>
      <p class="dec" style="margin:10px 0 0;color:var(--ink2);font-size:14px">${esc(a.disparador.texto)}</p>
      <ul class="steps">${a.pasos.map((s, i) => stepHTML(s, isNew && i >= liveCount)).join("")}</ul><div class="wave"></div></div>`;
    liveCount = a.pasos.length;
  } else {
    liveCount = 0;
    html += `<div class="card idle"><svg class="eye" viewBox="0 0 40 40" style="color:var(--ink)"><use href="#sigil"/></svg>
      <div><div style="color:var(--ink)">Trentren está vigilando.</div><div style="font-size:12.5px;color:var(--mut)">Lee CSN y USGS cada 2 minutos.
      ${S.proxima_ronda ? `Próxima revisión autónoma a las <span class="mono">${fmtH(S.proxima_ronda)}</span>.` : ""}
      ${S.notas?.length ? `<br>Última nota en su memoria: «${esc(S.notas[S.notas.length - 1].nota.slice(0, 160))}»` : ""}</div></div></div>`;
  }
  for (const e of (S.episodios || []).slice(0, 14)) {
    const dur = e.fin ? Math.round((new Date(e.fin) - new Date(e.inicio)) / 1000) : 0;
    html += `<div class="card ep ${openEp.has(e.id) ? "open" : ""}" data-id="${e.id}">
      <div class="hd"><span class="trig ${e.disparador.tipo}">${TRIG[e.disparador.tipo] || e.disparador.tipo}</span>
      <span>${fmtDH(e.inicio)}</span><span>${e.pasos.length} pasos · ${dur} s</span><span style="margin-left:auto">${openEp.has(e.id) ? "ocultar razonamiento ▴" : "ver razonamiento ▾"}</span></div>
      <div class="dec md">${md(e.decision || "")}</div>
      ${e.acciones?.length ? `<div class="acts">${e.acciones.map((x) => `<span class="act">${esc(x)}</span>`).join("")}</div>` : ""}
      <ul class="steps"><li data-g="◉"><span class="tn">Motivo</span><span class="tr">${esc(e.disparador.texto)}</span></li>${e.pasos.map((s) => stepHTML(s)).join("")}</ul></div>`;
  }
  $("#mind").innerHTML = html;
  $("#agentState").innerHTML = a ? `<span style="color:var(--accent)">deliberando…</span>` : `vigilando${S.proxima_ronda ? " · " + fmtH(S.proxima_ronda) : ""}`;
}
$("#mind").addEventListener("click", (ev) => {
  const c = ev.target.closest(".ep");
  if (!c || ev.target.closest("a")) return;
  const id = c.dataset.id;
  openEp.has(id) ? openEp.delete(id) : openEp.add(id);
  renderMind();
});

// ---------- pulso ----------
function ring(h, lab) {
  const p = h.p, R = 30, C = 2 * Math.PI * R;
  const frac = Math.max(p, 0.012);
  const col = h.alerta ? "var(--accent)" : "var(--ink2)";
  return `<div class="ring ${h.alerta && h.abierta ? "hot" : ""} ${h.abierta ? "" : "closed"}">
    <svg viewBox="0 0 76 76"><circle cx="38" cy="38" r="${R}" fill="none" stroke="rgba(255,255,255,.07)" stroke-width="4"/>
    <circle cx="38" cy="38" r="${R}" fill="none" stroke="${col}" stroke-width="4" stroke-linecap="round" stroke-dasharray="${C * frac} ${C}" transform="rotate(-90 38 38)"/>
    <text x="38" y="43" text-anchor="middle" fill="var(--ink)" font-family="JetBrains Mono" font-size="13">${pct(p)}</text></svg>
    <div class="lab">${lab}</div><div class="x">${h.abierta ? `×${dec(h.veces_base)} lo normal` : "ventana cerrada"}</div></div>`;
}
function renderPulses() {
  const cards = [];
  const sf = S.simulacro_pronostico;
  if (sf && simActivo()) cards.push({ ...sf, sim: true });
  for (const t of (S.triggers || []).slice(0, 5)) cards.push(t);
  if (!cards.length) { $("#pulses").innerHTML = `<div class="card pulse" style="color:var(--mut)">Sin sismos M≥4,7 en los últimos 7 días.</div>`; return; }
  $("#pulses").innerHTML = cards.map((t) => {
    const h = t.horizontes;
    const zona = t.sim ? t.lugar : (S.zonas.find((z) => z.zona === t.zona)?.nombre || t.zona);
    return `<div class="card pulse ${t.sim ? "sim" : ""}"><div class="ttl"><span class="m" style="${t.sim ? "color:var(--gold)" : ""}">M${dec(t.mag)}</span>
      <span style="text-align:right"><div class="where">${esc(zona)}</div><div class="when">${t.sim ? "SIMULACRO · " : ""}${fmtDH(t.t)} · ${Math.round(t.prof)} km</div></span></div>
      <div class="rings">${ring(h["60min"], "1 hora")}${ring(h["3h"], "3 horas")}${ring(h["24h"], "24 horas")}</div>
      <div style="font-size:11.5px;color:var(--mut);margin-top:10px;border-top:1px solid var(--line);padding-top:8px">
      Probabilidad de que sea el anuncio de uno mayor (7 días): <span class="mono" style="color:var(--ink2)">${pct(t.rj.p_mayor_7d)}</span> · ${h["24h"].fuente}</div></div>`;
  }).join("");
}

// ---------- memoria ----------
function renderZones() {
  const z = S.zonas || [];
  $("#zones").innerHTML = `<div class="zrow zhead"><span>Zona</span><span>Tiempo desde el último gran sismo</span><span style="text-align:right">años</span><span style="text-align:right">acoplam.</span></div>` +
    z.map((r) => `<div class="zrow"><span class="n">${esc(r.nombre)}<small>${esc(r.ultimo_gran)} · ruptura gigante ${r.ultima_ruptura_gigante} (M${r.mag_ruptura_gigante})</small></span>
      <span class="bar"><i style="width:${Math.min(100, (r.anios_desde_gran || 0) / 1)}%"></i></span>
      <span class="y">${r.anios_desde_gran != null ? dec(r.anios_desde_gran) : "—"}</span><span class="k">${r.acoplamiento != null ? Math.round(r.acoplamiento * 100) + "%" : "—"}</span></div>`).join("");
}

// ---------- ciencia ----------
function renderStats() {
  const m = S.metricas?.holdout;
  if (!m) return;
  const a = m["60min"], b = m["24h"];
  $("#nHold").textContent = a.n;
  $("#stats").innerHTML = [
    [dec(a.auc, 2), "", "Capacidad de distinguir", "AUC a 1 hora en sismos nunca vistos (0,5 = azar)"],
    ["×" + dec(a.precision / a.base_rate, 0), "", "Mejor que el azar", `cuando alerta acierta ${pct(a.precision)} vs ${pct(a.base_rate)} al azar`],
    [Math.round(a.recall * 100), "<small>%</small>", "Escaladas anticipadas", `a 1 hora (${a.hits} de ${a.pos}); a 24 h baja a ${pct(b.recall)}`],
    ["20", "<small>+</small>", "Hipótesis descartadas", "para predecir sismos principales: todas nulas"],
  ].map(([v, s, l, sub]) => `<div class="stat"><div class="v">${v}${s}</div><div class="l">${l}</div><div class="s">${sub}</div></div>`).join("");
}

// ---------- cabecera ----------
function renderMeta() {
  const f = S.fuentes || {};
  $("#fresh").innerHTML = `CSN ${ago(f.csn)} · USGS ${ago(f.usgs)}`;
  $("#llm").textContent = S.modelo_llm === "muse-wonder-1.0" ? "Muse Wonder (Meta)" : S.modelo_llm;
  const u = S.uso || { in: 0, out: 0 };
  $("#uso").textContent = `Consumo del agente: ${(u.in / 1e3).toFixed(0)}k tokens de entrada · ${(u.out / 1e3).toFixed(0)}k de salida (≈ US$${(u.in * 4e-6 + u.out * 2e-5).toFixed(2)})`;
}

function renderAll() {
  if (!S) return;
  renderSituacion(); renderMind(); renderPulses(); renderZones(); renderStats(); renderMeta(); drawSeismo(); drawMap();
}

// ---------- datos ----------
async function refresh() {
  try {
    const r = await fetch("/api/estado", { cache: "no-store" });
    S = await r.json();
    renderAll();
  } catch (e) { $("#agentState").textContent = "sin conexión"; }
}
let rt = null;
function soon() { clearTimeout(rt); rt = setTimeout(refresh, 400); }
function stream() {
  const es = new EventSource("/api/stream");
  es.onmessage = (m) => {
    const msg = JSON.parse(m.data);
    if (!S) return;
    if (msg.tipo === "episodio" || msg.tipo === "paso") { S.actual = msg.ep; renderMind(); }
    else soon();
  };
  es.onerror = () => { es.close(); setTimeout(stream, 5000); };
}

// ---------- preguntar ----------
$("#sugs").innerHTML = SUGS.map((s) => `<button type="button">${esc(s)}</button>`).join("");
$("#sugs").addEventListener("click", (e) => { if (e.target.tagName === "BUTTON") { $("#askQ").value = e.target.textContent; ask(); } });
$("#askForm").addEventListener("submit", (e) => { e.preventDefault(); ask(); });
async function ask() {
  const q = $("#askQ").value.trim();
  if (!q) return;
  const box = $("#answer");
  $("#askBtn").disabled = true;
  box.className = "answer md show";
  box.innerHTML = `<span style="color:var(--mut)">Trentren está revisando sus datos<span class="thinking"><i></i><i></i><i></i></span></span>`;
  try {
    const r = await fetch("/api/preguntar", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ pregunta: q }) });
    const j = await r.json();
    if (j.error) throw new Error(j.error);
    const used = (j.pasos || []).filter((p) => p.tipo !== "pensamiento").map((p) => p.tipo === "web" ? "web: " + p.texto : (TOOL[p.nombre]?.[1] || p.nombre));
    box.innerHTML = md(j.respuesta) +
      (used.length ? `<div class="acts" style="margin-top:12px">${used.map((u) => `<span class="act">${esc(u)}</span>`).join("")}</div>` : "") +
      ((j.citas || []).length ? `<div class="chips">${j.citas.map((c) => `<a class="chip" target="_blank" rel="noopener" href="${esc(c.url)}">${esc(c.titulo)}</a>`).join("")}</div>` : "");
  } catch (e) {
    box.innerHTML = `<span style="color:var(--red)">${esc(e.message)}</span>`;
  } finally { $("#askBtn").disabled = false; }
}

// ---------- simulacros ----------
$("#sims").innerHTML = SIMS.map((s) => `<button class="simc" data-k="${s.k}"><div class="m">M${s.m}</div><div class="p">${s.p}</div><div class="d">${s.d}</div></button>`).join("");
$("#sims").addEventListener("click", async (e) => {
  const b = e.target.closest(".simc");
  if (!b) return;
  document.querySelectorAll(".simc").forEach((x) => (x.disabled = true));
  try {
    const r = await fetch("/api/simulacro", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ escenario: b.dataset.k }) });
    const j = await r.json();
    if (j.error) alert(j.error);
    else { canal = "sim"; $("#mente").scrollIntoView(); }
  } finally { setTimeout(() => document.querySelectorAll(".simc").forEach((x) => (x.disabled = false)), 90000); }
});

// ---------- arranque ----------
fetch("geo.json").then((r) => r.json()).then((g) => { GEO = g; drawMap(); });
refresh(); stream();
setInterval(refresh, 60000);
setInterval(() => { if (S) { renderMeta(); drawSeismo(); } }, 30000);
addEventListener("resize", () => { drawMap(); drawSeismo(); });
