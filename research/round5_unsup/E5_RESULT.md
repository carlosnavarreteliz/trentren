# E5: minería no-supervisada exógena — NULO

Matriz diaria 1990-2026, 22 features de 6 familias (clima espacial, rotación
terrestre, carga oceánica, hidroclima, ruido estaciones, GNSS), todo lageado
causal, ajuste 1990-2012. Test: 58 mainshocks M≥6 con 3 controles apareados.

- IsolationForest (primario): media casos 0,4562 vs controles 0,4558,
  diff +0,0004, **p=0,39 → NULO**.
- PCA-error (secundario nominal): diff −0,0024, p~0,48 → nulo.
- Top días anómalos (2017-06, 2025-06, 2026-07...) no coinciden con mainshocks.

Conclusión: las anomalías multivariadas exógenas no preceden a M≥6. Combinado
con ronda 1 (14 familias nulas univariadas), se cierra la puerta exógena con
estos datos. Script: `e5_unsup.py`.
