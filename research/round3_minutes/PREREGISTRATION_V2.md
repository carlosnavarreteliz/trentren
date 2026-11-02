# Preregistro: piloto v2 minutos con microsismicidad (round3_minutes)

**Pregunta:** con catálogo denso homogéneo (Riedel ML≥2.5, estaciones fijas
2017-2021), ¿la microsismicidad de los últimos 10-60 minutos anticipa si un
sismo ML≥4 será seguido por otro ML≥5 dentro de 60 min / 150 km?

**Diseño (fijado antes de ver métricas; aprende del v1):**
- Catálogo: Riedel Primary_Catalogue.csv completo (footprint 18-42°S).
- Triggers: ML≥4.0. Train: 2017-01-01 → 2018-12-31. Test: 2019-01-01 → 2021-11-15
  (corte para etiquetas completas). Test esperado: ~3100 triggers, ~84 positivos.
- Target `y`: 1 si existe ≥1 evento ML≥5.0 con `t_trig < t ≤ t_trig+60min`
  y distancia ≤150 km.
- Features (causales, solo `t < t_trig`, radio 150 km salvo x2):
  - `x1`: log1p(nº ML≥2.5 en últimos 60 min)
  - `x2`: magnitud del trigger
  - `x3`: log10(1 + minutos desde el ML≥4 previo)
  - `x4`: magnitud máxima en últimas 24 h (piso 4.0)
  - `x5`: log1p(nº ML≥2.5 en últimos 10 min) ← la novedad: minuto a minuto
- Modelo: logística sklearn (C=1, estandarizado), ajuste solo en train.
- Baselines: (a) logit solo-x2 (mag trigger); (b) climatología train-rate (referencia,
  sabidamente vulnerable a cambio de régimen); (c) climatología test-rate (oráculo
  de referencia para aislar efecto régimen, no es competidor).
- **Métrica primaria: IG full vs mag-only** (evita la trampa del v1, donde casi todo
  el IG era ajuste de régimen vs climatología descalibrada).
- Secundarias: IG vs ambos climas, AUC, PR-AUC.
- Significancia: IC95% bootstrap bloques anuales + p permutación (5000).
  Umbral p<0,05 nominal, piloto exploratorio.
- Diagnóstico preregistrado: fracción de positivos con evento mayor en 30 días
  previos / 150 km (réplicas) — si domina, el veredicto lo dirá.

**Veredicto `nulo`** si IC95% de IG full-vs-mag-only incluye 0 o p≥0,05.
