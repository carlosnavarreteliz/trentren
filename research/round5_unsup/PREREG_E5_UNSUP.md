# Preregistro E5: minería no-supervisada de precursores (round5)

**Pregunta:** sin hipótesis física previa, ¿los días anómalos en el espacio
multivariado exógeno preceden a sismos principales M≥6 más que al azar?

**Matriz diaria X (1990-2026, ~22 cols, solo datos ≤ D−1 para el día D):**
- Clima espacial (6): kp_max, dst_min, f107_omni, np, v, bz
- Rotación terrestre (4): x, y, UT1-UTC, LOD derivado (−diff UT1, lageado)
- Carga oceánica (2): media y media|·| de eta_anom_fd diaria
- Hidroclima (6): medias nacionales P_land, P_andes, SWE_andes + anomalías
  (valor − media móvil causal 30d)/sd causal
- Ruido estaciones (2): mediana db diaria + nº targets (familia control)
- GNSS (2): media|z30| diaria + nº estaciones con dato
- Mareas: se excluyen (deterministas, nulas univariadas; declarado)
- NaNs → mediana de train por columna (train 1990-2012). Estandarización en train.

**Detectores (ajuste solo 1990-2012):**
- Primario: IsolationForest (200 árboles, contamination=0,05) → score anomalía.
- Secundario: error de reconstrucción PCA (95% varianza).

**Test (casos-controles apareados):**
- Casos: mainshocks M≥6 ZBZ (harness) en 2013+, score = media scores días D−30..D−1.
- Controles: 3 por caso, mismo bin lat 1°, mismo mes calendario ±15d, año≠caso,
  sin M≥6 en ±90d. Misma ventana de 30d.
- Estadístico primario: media(casos) − media(controles); p por permutación de
  etiquetas intra-sets apareados (10.000). Umbral p<0,05, 1 test primario.
- Secundarios (descriptivos + p nominal): PCA-error, horizontes 7d/90d,
  ablación por familia (qué aporta a los scores, no al test).

**Prior declarado:** las 6 familias dieron nulo univariado semanal; interacciones
multivariadas son posibles pero el prior es débil. Un nulo aquí cierra la puerta
exógena con estos datos.

**Veredicto:** nulo si p≥0,05.
