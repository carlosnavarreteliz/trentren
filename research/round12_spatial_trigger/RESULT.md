# R12: rasgos espaciales por trigger (lat/lon, distancia, acumulación radial) sobre v3 — NO ADOPTADO

Pedido del usuario: lat/lon, radio/onda expansiva, acumulación y distancia entre sismos.
Test T1 congelado (train ≤2020, calib 2021-23, test 2024+, n=945). Rasgos causales nuevos:
lat, lon, profundidad del trigger, distancia al M≥4.7 previo más cercano (24h-7d),
kernels con decaimiento exp(-d/50) y exp(-d/150) ponderados por 10^(M-4.7)/(1+dt),
y energía Benioff acumulada 7d en 300 km. Logit+Platt idéntico a v3, solo cambian las columnas.

| H | IG vs v3 (nats/pos) | IC95% bootstrap-año | AUC base→esp | PR base→esp | positivos |
|---|---|---|---|---|---|
| 60 min | +0.448 | [-0.113, +0.675] | 0.869→0.904 | 0.200→0.200 | 15 |
| 3 h | +0.526 | [-0.291, +0.754] | 0.836→0.900 | 0.134→0.176 | 16 |
| 24 h | +0.127 | [-0.304, +0.263] | 0.666→0.653 | 0.078→0.109 | 35 |

Ablación IG: solo lat/lon/prof {+0.28, +0.30, -0.00}; solo distancia+acumulación {+0.17, +0.16, +0.11}.

Veredicto: estimador puntual positivo pero IC95% cruza 0 en los 3 horizontes
(test con 15-35 positivos no resuelve efectos así). Consistente con round11
(espacial suave en panel: gana in-distribución, colapsa en leave-one-zone-out
Antofagasta IG -0.91). No se productiviza. Reevaluar solo con test prospectivo
acumulado o más años; la lat/lon probablemente memoriza tasas zonales.
Script de referencia: sesión 2026-10-27 (/tmp/spatial_test2.py), datos benchmark/t1_test.csv + forecast_short.build_dataset.
