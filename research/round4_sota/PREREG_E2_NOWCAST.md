# Preregistro E2: nowcasting de Rundle (round4_sota)

**Idea no testeada en el repo:** Earthquake Nowcasting (Rundle et al.): ciclos de
eventos pequeños en tiempo natural → Earthquake Potential Score (EPS) = CDF del
conteo actual. Método publicado, jamás probado en Chile en este repo.

**Diseño:**
- Región harness, USGS. Pequeños M≥4.7 desde 1990 (Mc≤4.5 en 1990-2026 salvo 80s).
- Grandes M≥6.0. Ciclos = nº pequeños entre grandes consecutivos.
- CDF empírica con ciclos completados ≤2012 (train). EPS(t) = P(N ≤ n_actual).
- Test 2013+: en cada evento pequeño, label = 1 si hay M≥6.0 en los próximos 180d
  (caja completa, sin bins: nowcasting es temporal puro).
- Métrica primaria: AUC de EPS vs label. H0: AUC=0,5 (p permutación 5000).
- Comparación secundaria honesta: EPS vs baseline "días desde el último M≥6"
  (ΔAUC). Nowcasting suele recapitular tiempo-desde-último; si no lo supera,
  no hay novedad.
- Umbral p<0,05, 1 test primario.

**Veredicto:** nulo si p≥0,05 o si EPS no supera al baseline de tiempo.
