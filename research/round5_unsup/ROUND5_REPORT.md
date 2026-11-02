# Campaña round5: minería no-supervisada + empaquetado (2026-10-27)

## E5: anomalías exógenas multivariadas — NULO

Matriz diaria 1990-2026 (22 features, 6 familias), IsolationForest + PCA
ajustados 1990-2012, test apareado pre-M≥6 (58 casos): IF diff +0,0004 p=0,39;
PCA diff −0,0024 p~0,48. Cierra la puerta exógena con estos datos.
Detalle: `PREREG_E5_UNSUP.md`, `E5_RESULT.md`, `e5_unsup.py`.

## Benchmark propio: Chile-ST-Benchmark v1.0

`benchmark/BENCHMARK.md`. T1: escalamiento por trigger (test 2024+ congelado,
342 triggers, SOTA = forecast_short v2); T2: panel diario (SOTA = E3-kernels).
Scorer verificado (`score.py`: auto-IG=0, dummy −2,58).

## Dashboard

`dashboard/index.html` (8 KB, estático, sin JS, offline) generado por
`dashboard/generate.py`: pronóstico en vivo + backtest + historial campañas.
Regenerar tras cada reentrenamiento.

## Conclusión minería mainshock (todas las campañas)

Catálogo (micro)sísmico, waveforms, tipos, repeaters, nowcasting, exógenas
uni- y multivariadas, redes globales: **ninguna anticipa el sismo principal**.
Lo único real y productivizado: clustering a minutos-horas-días (E3 + v2).
