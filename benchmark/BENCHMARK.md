# Chile-ST-Benchmark v1.0 (benchmark propio)

Benchmark versionado para pronóstico sísmico de corto plazo en Chile.
Congelado 2026-10-27. Regla de oro: el test (2024+) no se mira para diseñar;
solo se puntúa.

## Tareas

**T1 — Escalamiento por trigger (minutos-horas).** Triggers = sismos USGS Mw-equiv ≥4.7
en caja (-56..-17, -78..-66). Target: ≥1 Mw-equiv ≥5.0 en 150 km dentro de H ∈ {60 min,
3 h, 24 h}. Splits: train ≤2020, calib 2021-2023, **test 2024-01→2026-10**
(congelado en `benchmark/t1_test.csv`, 945 triggers). Métricas por horizonte:
IG vs SOTA (nats/pos), ROC-AUC, PR-AUC, Brier, precisión/recall @umbral
(percentil 95 calib). Referencia SOTA: `p_sota_*` = forecast_short v3 (frozen).

> **v1.1 (2026-10-28):** magnitudes homogeneizadas a Mw (Scordilis mb/ms).
> La v1.0 (etiquetas mixtas USGS, 342 triggers) quedó deprecada: 78% de los
> eventos 2024+ son `mb` (mb 4,7 ≈ Mw 5,0), el ruido de etiquetas deprimía
> el AUC de 0,87 a 0,68 con el mismo modelo (fair test en OPERATIONAL.md).

**T2 — Panel diario espacial.** Panel harness 1d M≥5.0, `y_all` (primario) e
`y_main` (secundario), test 2013+. Métrica: IG vs baseline ETAS-logit +
surrogate phase_randomize. Referencia SOTA: E3-kernels (+0,0117, p=0,017).
Se corre con `research/harness.py` + features del candidato (ver E3 como plantilla).

## Reglas

1. Features estrictamente causales (t < t_start / t_trigger). 2. Nada ajustado
   en test (umbrales, pesos, selección). 3. Reportar las 3 métricas + IC95%
   bootstrap por año. 4. T1: mejora = IG vs `p_sota` con IC95% > 0.
   5. T2: mejora = IG vs ETAS-logit con p surrogate < 0,05 (1 test).

## Uso T1

```bash
python3 benchmark/freeze_t1.py   # regenera t1_test.csv (solo mantenedor)
python3 benchmark/score.py mi_modelo.csv --h 60   # puntúa envío
```

Formato envío: CSV con columnas `trigger_idx,p_60,p_180,p_1440`
(`trigger_idx` = fila en t1_test.csv).
