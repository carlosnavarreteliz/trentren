# Preregistro: piloto minutos (round3_minutes)

**Pregunta:** ¿Se puede anticipar, al momento de un sismo M≥4.7, si otro sismo M≥5.0
ocurrirá dentro de 60 minutos y 150 km? (escalamiento a minutos, catálogo USGS solamente)

**Diseño (fijado antes de ver métricas):**
- Triggers: eventos M≥4.7, región harness (lat −46…−17, lon −78…−66).
- Train: triggers 1990-01-01 → 2017-12-31. Test: 2018-01-01 → 2026-10-01
  (corte 1-oct para que todas las etiquetas de 60 min estén completas).
- Target primario `y`: 1 si existe ≥1 evento M≥5.0 con `t_trig < t ≤ t_trig+60min`
  y distancia ≤150 km del trigger. Test esperado: ~1055 triggers, ~31 positivos.
- Features (causales, solo `t < t_trig`, radio 150 km salvo x2):
  - `x1`: log1p(nº M≥4.7 en últimos 60 min antes del trigger)
  - `x2`: magnitud del trigger
  - `x3`: log10(1 + minutos desde el M≥4.7 previo en 150 km)
  - `x4`: magnitud máxima en 150 km en últimas 24 h antes del trigger (piso 4.2)
- Modelo: regresión logística sklearn (C=1, inputs estandarizados), ajuste solo en train.
- Baselines: (a) climatología (tasa de train); (b) logit solo con x2 (mag del trigger).
- Métrica primaria: IG (nats/evento positivo) del modelo full vs climatología en test.
- Secundarias: IG full vs mag-only, ROC-AUC, PR-AUC, tasa base.
- Significancia: IC 95% por bootstrap de bloques anuales (2000 remuestreos) +
  p por permutación de etiquetas en test (5000, aproximado; se reporta como tal).
- Umbral: p < 0,05 nominal (piloto exploratorio, sin corrección; se declara).
- Veredicto `nulo` si el IC 95% de IG incluye 0 o p ≥ 0,05.

**Limitaciones declaradas:** ~31 positivos → solo efectos grandes son detectables;
ventanas de 60 min de triggers cercanos se solapan (dependencia parcial);
catálogo USGS revisado (pseudo-prospectivo, con latencia real de minutos-horas).
