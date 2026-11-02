# Preregistro E1: excitación cruzada por tipo (round4_sota)

**Idea no testeada en el repo:** el catálogo Riedel clasifica cada evento
(Interplate/Intraslab/Outerrise/Crustal/NC). La hipótesis Bouchon (losa profunda
carga la interfaz) se testeó con un corte crudo de profundidad (ronda 1: nulo,
p=0,044 signo opuesto). Nunca con tipos físicos ni a escala de días.

**Pregunta:** dado un sismo ML≥4.0 "quiet" (sin evento mayor en 30d/150km previos),
¿la composición por TIPO de la microsismicidad previa anticipa escalamiento a
ML≥5.0 en 7 días / 150 km, más allá de la tasa total?

**Diseño (fijado tras conteo de viabilidad, antes de ver efectos):**
- Catálogo Riedel primario limpio: sin NaN-Ml, sin 401 duplicados exactos, ordenado.
- Triggers quiet ML≥4.0. Train: 2017-03 → 2019-12. Test: 2020-01 → 2021-11-15.
  Esperado test: ~291 triggers, ~59 positivos.
- Target: ≥1 ML≥5.0 en 7d/150km post-trigger.
- Baseline (3 feat): mag trigger, log1p(total ML≥2.5 en 24h), log1p(total en 7d).
- Full (+4 feat): log1p(Interplate 7d), log1p(Intraslab 7d), log1p(Outerrise 7d),
  log1p(Crustal 7d). NC excluido.
- Modelo: logística C=1 estandarizada. Métrica primaria: IG full-vs-baseline por
  positivo en test. IC95% bootstrap anual (pocos años: se reporta ancho) +
  p permutación 5000. Umbral p<0,05 (1 test primario).
- Predicción direccional (Bouchon): coef(Intraslab) > 0. Se reporta signo.

**Veredicto:** nulo si IC95% incluye 0 o p≥0,05. Si positivo: replicar en ventana
alternativa + chequeo de secuencia dominante (leave-one-month-out).
