# Recalibración total + auditoría de datos (2026-10-27, pedido del usuario)

## Formato
- Ledger de alertas ANCHO: 1 fila por trigger (`results/alert_ledger_2024.csv`),
  columnas p_60min/p_3h/p_24h + alerta_* + escalo_* por horizonte. Reemplaza al tidy.

## Baseline aleatorio (test congelado 2024+, mismo presupuesto de alertas)
Alertar k triggers al azar: P_esperada = tasa base, R_esperada = k/n (sim 5000x).
| H | base | Modelo P / R | Azar P / R | Lift | P(azar >= aciertos) |
|---|---|---|---|---|---|
| 60 min | 1,59% | 0,18 / 0,40 | 0,016 / 0,035 | x11,5 | <0,0002 |
| 3 h | 1,69% | 0,16 / 0,31 | 0,017 / 0,033 | x9,5 | <0,0002 |
| 24 h | 3,70% | 0,16 / 0,11 | 0,037 / 0,026 | x4,3 | 0,013 |
Verdad completa: la precisión absoluta ES pésima (0,16-0,18) porque el evento
es raro (1,6-3,7%); aun así es 4-12x el azar. A 24h el recall 0,11 es el punto débil real.

## Auditoría de datos (script /tmp/data_audit.py)
- Crudo limpio: 54.855 ev, 0 ids duplicados, 0 nulos en mag/depth/lat-lon.
- Mezcla de magnitudes por era: 1970-89 domina mb; 1990-2008 md/ml; 2009-26 mb+mwr.
  La homogeneización Scordilis (v3+) es obligatoria; sin ella el AUC cae a 0,68.
- Colapso de reporte: M2,5-4,0 = 20.512 ev (1990-2008) vs 108 (2022-26). USGS ciego a microsismos hoy.
- Tasas de escalado por era: triggers 1990+ escalan 10,7%/14,8%/22,4% (60m/3h/24h)
  vs 2024+ solo 1,6%/1,7%/3,7%: el test es un régimen quieto sin megasecuencia.
  Toda métrica de 2024+ (15-35 positivos) es frágil por régimen, no solo por modelo.
- No independencia: 2.265/11.961 triggers a <60 min del anterior (ventanas solapadas).
- Caja incluye 12% de eventos al este de -68 (trasarco argentino) con otra física.

## Recalibración
- v4 (`forecast_short.py --train-v4` -> `results/short_term_model_v4.json`): mismo score,
  Platt CROSS-FITTED en 5 bloques temporales de calib 2021-23 (antes: Platt y umbral
  sobre los mismos datos in-sample). Umbral desde probs cross-fitted.
  Resultado holdout: AUC idéntico (ranking no cambia), P 0,16/0,16/0,18, R 0,40/0,38/0,14.
  Diagnóstico guardado en el JSON (slope/ECE/pred-vs-obs + azar por horizonte).
- Deriva que NO se corrige con el test: en 2024+ predice 0,012/0,013/0,031 vs observado
  0,016/0,017/0,037 (subpredice ~25%). Corregirla exige recalibrar con datos ya
  maduros en operación real (walk-forward anual), nunca ajustando sobre el test congelado.
- Replay del dashboard: recalibración walk-forward anual (Platt de cada año Y ajustada
  solo con triggers < Y). Antes: Platt fija <=2013 que daba 65-74% falsos en 2014+.

## ¿De dónde sacar datos densos?
- Formas de onda C1 (red CSN): FDSN EarthScope vivo (verificado: 107 estaciones).
  Vía para reconstruir catálogo denso 2022-26 con picking EQTransformer en GPU (Colab):
  es un proyecto de reproceso (TB de descarga, meses), no un download.
- Catálogos densos ya hechos terminan 2021: Riedel (Zenodo 15284376) e IPOC (GFZ
  doi:10.5880/GFZ.4.1.2023.004), ya usados en ronda 2 (nulos para mainshock).
- Sin canal nuevo, más modelo no cambia nada: el límite es Mc y régimen.
