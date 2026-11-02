# Trentren: an AI council that watches Chile's earthquakes

Trentren is a real-time earthquake watch system for Chile. Every two minutes it reads the
CSN (Centro Sismológico Nacional) and USGS feeds. When a significant earthquake occurs, it
writes a rule-based pre-alert in under a second. Then a council of AI agents takes over,
each with its own tools:

- **Seismologist**: runs the escalation model, recent nearby seismicity and historical analogues.
- **Media watcher**: reads Google News Chile headlines about the event.
- **Civil protection**: exposed cities, estimated intensity and tsunami condition.

The agents read each other's reports and debate. A coordinator (Trentren) then issues the
alert level (`vigilancia`, `atencion`, `alerta`), which is stored in a SQLite database and
streamed to a live web map. Historical earthquakes can be replayed "as of" their date to see
how the council reasons with only the information available at the time.

Trentren does not claim to predict earthquakes. Its forecasts cover escalation only: the
probability that an event is followed by a larger one. That model is benchmarked against
standard seismology baselines (see `benchmark/` and `research/AUDIT.md`).

## Meta models used

Every AI model Trentren calls at runtime is a Meta model:

| Model | Where it runs | What it does |
|---|---|---|
| `muse-wonder-1.0` | Meta Model API (`https://api.meta.ai/v1/responses`), called from `trentren/council.py` | Powers all four agents: tool calling, the debate round, and the final alert decision |

The model is set with `META_MODEL` and the endpoint with `META_BASE_URL` (see `.env.example`).
No other LLM or third-party AI service is called. The escalation forecast
(`forecast_short.py`, v4) is an in-house statistical model: Platt-calibrated logistic
regressions on causal clustering features, fitted to the USGS catalog. It runs locally and the agents call it as a tool.

## Run Trentren

```bash
cp .env.example .env          # set META_API_KEY
python3 -m trentren           # watcher + web at http://127.0.0.1:8787
```

Requires Python 3.9+, `numpy`, `pandas` and `scikit-learn`.

---

*The rest of this README (in Spanish) documents the underlying research.*

# PI-ST-Fusion: Modelo de Pronóstico Sísmico Informado por la Física para la Subducción de Chile

Este repositorio contiene el estudio completo, catálogo histórico e instrumental (1904–2026, $N=54,855$ sismos), ingeniería de características geofísicas 3D (*Slab2* + acoplamiento geodésico GPS + $b$-positive de van der Elst + *slab pull* de Bouchon), benchmark **CSEP Zero-Leakage** frente a 5 modelos de la literatura y la herramienta operacional en tiempo real (`predict_chile.py`) para predecir probabilidades de terremotos en Chile.

## Estructura del Repositorio

- [`ESTUDIO_PREDICCION_SISMICA_CHILE.md`](file:///Users/cnavarreteliz/Documents/Repos/earthquakes/ESTUDIO_PREDICCION_SISMICA_CHILE.md): Monografía científica completa del estudio, revisión de literatura (incluyendo estado del arte hasta 2026), formulación matemática, demostración de las dos paradojas descubiertas, tablas CSEP y pronóstico operacional Octubre–Noviembre 2026.
- [`predict_chile.py`](file:///Users/cnavarreteliz/Documents/Repos/earthquakes/predict_chile.py): Herramienta CLI ejecutable en tiempo real para emitir pronósticos sísmicos por zona sismotectónica o coordenada específica en Chile.
- [`src/download_data.py`](file:///Users/cnavarreteliz/Documents/Repos/earthquakes/src/download_data.py): Descarga y consolidación del catálogo sísmico de Chile ($56^\circ\text{S}\text{–}17^\circ\text{S}$) desde USGS ComCat FDSN API y parametrización de segmentos tectónicos y geometría Slab2.
- [`src/physics_features.py`](file:///Users/cnavarreteliz/Documents/Repos/earthquakes/src/physics_features.py): Cálculo estrictamente causal de 46 variables físicas en 6 bloques (ETAS anisotrópico de subducción, termodinámica $b^+$, acoplamiento profundo-a-somero, bloqueo geodésico Slab2, energía de Benioff y fase mareal).
- [`src/models_and_evaluation.py`](file:///Users/cnavarreteliz/Documents/Repos/earthquakes/src/models_and_evaluation.py): Entrenamiento y evaluación CSEP (1980–2013 Train, 2014–2017 Val, 2018–2026 Test Ciego Prospectivo) de los 6 modelos (`Poisson-GR`, `Classical-ETAS`, `Reyes-Panakkat-ML`, `Neural-Point-Process`, `Anisotropic-Slab-ETAS` y `PI-ST-Fusion`).
- [`src/generate_figures.py`](file:///Users/cnavarreteliz/Documents/Repos/earthquakes/src/generate_figures.py): Generación de las 4 figuras científicas de alta resolución en `results/figures/`.
- `data/raw/chile_usgs_1900_2026.csv`: Catálogo crudo con 54,855 sismos en Chile.
- `data/processed/chile_features_m40.csv`: Catálogo homogeneizado ($M_c = 4.0$, 31,048 eventos) con las 108 columnas de estado físico y etiquetas multi-horizonte.
- `results/`: Métricas CSEP (`benchmark_metrics.json`), predicciones de test (`catalog_with_predictions.csv`), atribución de variables (`feature_importance_pre_mainshock.csv`), pronóstico vivo (`live_chile_forecast_oct2026.csv`) y figuras (`results/figures/`).

## Herramienta operativa vigente: corto plazo (minutos-horas)

> `predict_chile.py` (PI-ST-Fusion) está desacreditado por `research/AUDIT.md`.
> La herramienta honesta es `forecast_short.py` (réplicas/escalamiento, v2 con E3).

```bash
python3 forecast_short.py --fetch-live   # pronóstico actual con USGS en vivo
python3 forecast_short.py --train        # reentrena + recalibrar + backtest
python3 dashboard/generate.py            # regenera dashboard/index.html
python3 benchmark/score.py envio.csv --h 60  # puntúa contra Chile-ST-Benchmark v1.0
```

## Uso Rápido (legacy, desacreditado)

```bash
# Emitir pronóstico operacional a 30 días actualizando en vivo con la API de USGS:
python3 predict_chile.py --fetch-live

# Pronóstico a 14 días para una coordenada específica (ej. Valparaíso / Santiago):
python3 predict_chile.py --lat -33.0 --lon -71.6 --days 14
```

