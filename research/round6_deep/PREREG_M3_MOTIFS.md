# Preregistro M3: motivos secuenciales pre-mainshock (round6)

**Pregunta:** ¿hay secuencias simbólicas de eventos (patrones) enriquecidas
antes de sismos grandes que los conteos no capturan?

**Diseño:**
- Símbolos (18): clase mag (A: ML2.5-3.5, B: 3.5-4.5, C: ≥4.5) × profundidad
  (S ≤60 km, D >60) × tipo (I Interplate, N Intraslab, X otro) — ej. "ASI".
- Casos: eventos ML≥5.5 "mainshock-proxy" (sin evento mayor en ±30d/150 km),
  Riedel 2017-2021. Secuencia = últimos 20 eventos ML≥2.5 en 150 km previos.
- Controles: mismo epicentro, fecha aleatoria (mismo año ±1), sin ML≥5.5 en
  ±60d; 5 por caso. Últimos 20 eventos previos a esa fecha.
- Minería: n-gramas contiguos n=2,3,4; soporte mínimo 3 casos.
  Enriquecimiento = tasa en casos vs controles.
- Descubrimiento: casos 2017-2019. Significancia por permutación de etiquetas
  (1000) + Bonferroni sobre motivos con soporte.
- **Validación temporal:** motivos significativos se testean SIN cambios en
  casos 2020-2021 (p<0,05, un test por motivo validado).

**Veredicto:** sin motivo que valide → nulo. Con validación → señal candidata.
