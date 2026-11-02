# Preregistro M5: ¿más historia rescata features ricas? (round6)

**Pregunta (pedida explícita del usuario):** con 36 años USGS (1990-2026) en vez
de 5, ¿features ricas + modelo flexible superan al logit simple de 3 features
en escalamiento 60 min M≥4.7→M≥5.0/150km?

**Diseño:** triggers USGS M≥4.7 (como v1). Train 1990-2017, test 2018-2026-10.
Features (~35, causales): conteos {60m,3h,24h,7d,30d,365d} × {total, ≤60km,
>60km}, maxmag {24h,7d,30d}, ETAS-lite Omori (2 p), días-since M≥5/M≥6,
b-positive k20/50, trigger mag/depth, tasa bin, tendencia profundidad.
Modelos: HGBT + logit-rico. Baseline: logit 3 feats (mag, tasa 60m, tasa 24h).
Primario: IG vs baseline, permutación 5000, IC95% anual. p<0,05.

**Veredicto:** nulo si p≥0,05 o IC incluye 0.
