# Pronóstico operativo de corto plazo (minutos-horas)

**Herramienta:** `forecast_short.py` (raíz del repo). Modelo vigente: `results/short_term_model_v3.json`.

## v3 (2026-10-28): homogeneización de magnitudes — mejora real de datos

El catálogo USGS reporta 78% `mb` en 2024+ (mb 4,7 ≈ Mw 5,0, Scordilis 2006):
las etiquetas mixtas eran ruido que deprimía todo. Fair test (misma tarea
Mw-equiv, mismos triggers/targets): features mixtas AUC 0,874 ≈ homogeneizadas
0,866 — el modelo siempre rankeó bien; las **etiquetas** estaban mal.
v3 convierte mb/ms→Mw al cargar (y en vivo). Holdout 2024+ (945 triggers):
60 min AUC **0,87** (v2: 0,68), PR-AUC 0,20 (base 0,016), Brier 0,0146.
Umbrales: percentil 95 de calibración (presupuesto alerta 5%; F0.5 nunca
alertaba en 3h/24h). Benchmark actualizado a v1.1 (test recongelado, 945).

**Qué hace:** ante cada sismo M≥4.7, estima P(otro M≥5.0 en 150 km) a 60 min, 3 h y
24 h. Tres logísticas calibradas (Platt) sobre 4 features causales de clustering.
**Pronostica réplicas/escalamiento, no sismos principales.**

**Entrenamiento:** train ≤2020, calibración 2021-2023, holdout 2024+. Umbral por
horizonte = max F0.5 en calibración (regla pro-precisión, fijada pre-holdout).

## Backtest honesto v2 (E3 portado; holdout 2024+, n=342 triggers)

Modelo `short_term_model_v2.json`: features con split interfaz/intraslab +
memoria 3h (lección de E3). v1 entre paréntesis.

| Horizonte | Pos | AUC v2 (v1) | PR-AUC v2 (v1) [base] | Alertas | Aciertos | Prec/Rec |
|---|---|---|---|---|---|---|
| 60 min | 8 | 0,68 (0,65) | 0,099 (0,067) [0,023] | 14 | 1 | 0,07/0,12 |
| 3 h | 9 | 0,65 (0,61) | 0,069 (0,049) [0,026] | 11 | 1 | 0,09/0,11 |
| 24 h | 14 | 0,55 (0,55) | 0,057 (0,054) [0,041] | 18 | 2 | 0,11/0,14 |

Lectura: v2 mejora a 60 min y 3 h, plano a 24 h — consistente con E3 (804
positivos en panel diario). Con 8-14 positivos el holdout NO confirma nada por
sí solo; es chequeo de consistencia, no prueba independiente. Holdout quieto →
precisión baja por diseño en régimen basal; el valor aparece en secuencias
activas. No usar los IG-vs-mag (+2 a +3,8): inflados por cambio de régimen.

## Uso

```bash
python3 forecast_short.py --train              # reentrena + recalibrar + backtest
python3 forecast_short.py --fetch-live         # pronóstico actual con USGS en vivo
python3 forecast_short.py --fetch-live --n 5   # últimos 5 triggers
python3 forecast_short.py --prospective        # congela predicciones en results/prospective_log.csv
python3 forecast_short.py --score-prospective  # evalúa entradas ya maduradas del log
```

**Protocolo prospectivo:** correr `--prospective` tras cada trigger nuevo (o a
diario). El log guarda (issued_at, trigger, P por horizonte, modelo). Solo las
entradas emitidas antes del horizonte son evidencia prospectiva real; la primera
carga (2026-10-28, 60 triggers históricos) es retrospectiva y sirve de formato.
La pantalla incluye además ESTADO POR ZONA (7d): max P24h por segmento.

**Deriva de calibración (observada 2026-10-28):** en la primera carga
retrospectiva del log, las entradas maduras muestran sub-pronóstico ~×2
(60 min: pred media 0,032 vs observado 0,068). La calibración Platt 2021-2023
envejece con el régimen; vigilar con `--score-prospective` y reentrenar si
el sesgo persiste >1 trimestre. El AUC (ranking) se mantiene; es el nivel
absoluto el que deriva.

## Limitaciones

- Latencia USGS real (minutos-horas); pseudo-prospectivo sobre catálogo revisado.
- Mc≈4,7: ciego a microsismicidad; Riedel (ML≥2,5) termina en 2021, no sirve en vivo.
- Umbrales backtesteados en periodo quieto: en una secuencia grande sonarían muchas
  alertas (correcto: P sube de verdad) — reentrenar tras el próximo M≥7.
- Reemplaza operativamente a `predict_chile.py` (PI-ST-Fusion), desacreditado en
  `research/AUDIT.md`. No se borró ese archivo por provenance.
