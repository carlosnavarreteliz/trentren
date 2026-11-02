# Informe ronda 1: variables no consideradas para pronosticar terremotos en Chile

_Generado el 2026-10-17 por el workflow `chile-quake-hidden-variables` (18 agentes). Detalle por hipótesis en `hypotheses/*/RESULTS.md`; auditoría en `AUDIT.md`; literatura en `LITERATURE.md`; calibración en `calibration/CALIBRATION_V2.md`._

## 1. Resumen ejecutivo

**No, hoy no se pueden predecir terremotos en Chile en el sentido de "dónde, cuándo y de qué tamaño". Tampoco encontramos una variable nueva que mejore de forma demostrable el pronóstico probabilístico estándar (ETAS).**

**Qué se probó**
- 14 familias de variables poco usadas, cada una con su preregistro: mareas, rotación terrestre, clima espacial, carga hidroclimática, carga oceánica, GNSS, acoplamiento intraslab–interfase, gatillamiento telesísmico, rotación de esfuerzos, búsqueda masiva en metadatos del catálogo, quiescencia y migración, volcanismo, ruido de estaciones y estructura estática.
- Todas se evaluaron fuera de muestra: ajuste en 1976–2012 y prueba en 2013-01 → 2026-10.
- Debían sumar información sobre un baseline de agrupamiento: ETAS más conteos recientes.

**Resultado**
- **Las 14 son nulas** en su test primario.
- Ningún p-valor llega al umbral de Bonferroni (0,0036).
- Los tres p < 0,05 nominales son estos:
  - acoplamiento vertical: p = 0,044;
  - gatillamiento telesísmico: p = 0,020 en sismos principales (`y_main`) y 0,026 en todos los eventos (`y_all`).
- Los tres tienen el signo opuesto a la hipótesis o se explican porque la tasa de sismos principales del periodo de prueba es un 32% más alta. Ninguno se replica. Con unos 20 tests primarios, es lo que se espera por azar.
- La calibración final con 60 variables de ruido puro dio estos falsos positivos con phase_randomize:
  - 8,3% en `y_all` y 0% en `y_main` a p < 0,05;
  - 0 a p < 0,0036.

**Lo único con habilidad real es el agrupamiento (réplicas)**
- En el panel diario M≥6, el baseline ETAS da un factor de ganancia de 1,72 por evento frente a una tasa de fondo constante (+0,55 nats).
- Con alarma en el 1% de las celdas-día con mayor probabilidad, captura el 20% de las celdas con M≥6. Una tasa de fondo constante (Poisson) captura el 3%.
- Esa habilidad viene casi por completo de las secuencias de Iquique 2014, Illapel 2015 y Chiloé 2016.
- Para **sismos principales**, nada supera claramente a la tasa histórica:
  - semanal: +0,006 nats;
  - diario: +0,045, con IC del 95% [+0,002, +0,094].

**Modelo previo (PI-ST-Fusion)**
- En el test primario 2013+ queda por debajo de un Reasenberg–Jones genérico: −0,085 nats/evento, p = 0,96.
- Sus variables "físicas" aportan +0,001 nats (p = 0,24).
- Su pronóstico en vivo no está calibrado: todas las zonas quedan bajo la tasa histórica (Antofagasta 7,2% frente a 22–25%).
- Los umbrales de alerta se editaron sin un backtest.

**Qué significa un nulo aquí**
- Con 72–85 celdas positivas en el periodo de prueba, no se pueden detectar ni descartar efectos menores a unos 0,02–0,03 nats/evento, o sea un 2–3% de ganancia de probabilidad.
- Sí podemos afirmar que **ninguna de estas variables es un predictor útil** con los datos públicos actuales.
- El mejor resultado nominal, +0,021 nats (factor 1,02), cambiaría las probabilidades semanales en un 1–2% relativo.

**Qué cambiaría el cuadro**
- Datos más finos, no más variables exóticas: el catálogo CSN/IPOC o catálogos ML con Mc ≈ 1–2, GNSS denso en tiempo real y sismómetros de fondo oceánico (OBS) en la fosa.
- Un test **prospectivo** preregistrado contra ETAS 2D. El diseño está en la sección 8 del informe completo.
- La carpeta `research/round2/` (catálogo mejorado, sorpresa ETAS, transferencia global) aún no tiene un `RESULTS.md` y no está incluida.

## 5. Tabla de las 14 hipótesis

Valores de IG (ganancia de información) en nats por evento frente al baseline ETAS + agrupamiento. Panel semanal M≥6, regresión logística, prueba en 2013–2026, con 72 celdas `y_main` y 85 `y_all`. Los p-valores salen de surrogates, en su mayoría phase_randomize con n = 1000.

Ningún resultado se mandó a verificación adversaria, porque ninguno llegó al umbral de positivo. Los autores revisaron fuga de información y robustez.

| # | Hipótesis | IG primario (`y_main`) | p | IG `y_all` (p) | Veredicto |
|---|---|---|---|---|---|
| 1 | Mareas (sensibilidad tidal) | −0,026 | 0,98 | −0,023 (0,86) | Nulo |
| 2 | Rotación terrestre: LOD con rezago de 5 años / marea polar | +0,003 / −0,051 | 0,24 / 0,91 | +0,002 (0,14) / −0,031 (0,87) | Nulo |
| 3 | Clima espacial (densidad de protones, panel diario) | +0,002 | 0,19 | +0,000 (0,33) | Nulo; no replica ni globalmente (ρ = +0,004) |
| 4 | Carga hidroclimática | +0,005 | 0,23 | +0,012 (0,055) | Nulo |
| 5 | Carga oceánica (presión en el fondo oceánico) | −0,002 | 0,41 | +0,012 (0,035, n = 200) | Nulo |
| 6 | Transitorios GNSS (2007+) | +0,003 | 0,15 | −0,001 (0,34) | Nulo |
| 7 | Intraslab / outer-rise (Bouchon) | +0,021 | **0,044** | +0,005 (0,23) | Nulo: signo opuesto y cambio de nivel |
| 8 | Gatillamiento telesísmico | +0,018 | **0,020** | +0,016 (**0,026**) | Nulo: signo opuesto y no replica |
| 9 | Rotación de esfuerzos (GCMT) | −0,051 | 0,93 | −0,017 (0,72) | Nulo |
| 10 | Búsqueda en catálogo y metadatos (164 variables, FDR) | 0 (ninguna seleccionada) | — | 0 | Nulo |
| 11 | Quiescencia y migración | −0,051 | 0,71 | −0,013 (0,47) | Nulo |
| 12 | Acoplamiento volcánico | −0,001 | 0,32 | +0,002 (0,25) | Nulo |
| 13 | Ruido de estaciones (MUSTANG) | −0,013 | 0,42 | −0,020 (0,51) | Nulo |
| 14 | Estructura estática | −0,019 | 0,82 (surrogate espacial) | −0,008 (0,81) | Nulo |

**Referencia práctica del baseline**

Tasas recalculadas en esta sesión con el harness. La columna "Captura" es la fracción de celdas con M≥6 que caen dentro de la alarma.

| Panel y objetivo | Factor de ganancia frente a Poisson | Captura con alarma en 1% / 10% de las celdas | Captura de Poisson con 1% / 10% |
|---|---|---|---|
| Diario M6, `y_all` | 1,72 | 20% / 37% | 3% / 26% |
| Semanal M6, `y_all` | 1,25 | 13% / 28% | 5% / 25% |
| Semanal M6, `y_main` | 1,01 | 1% / 25% | 6% / 29% |

La fuente de cada número está en los `RESULTS.md` de `/Users/cnavarreteliz/Documents/Repos/earthquakes/research/hypotheses/`, en `AUDIT.md` y `LITERATURE.md` de `/Users/cnavarreteliz/Documents/Repos/earthquakes/research/`, y en `/Users/cnavarreteliz/Documents/Repos/earthquakes/research/calibration/CALIBRATION_V2.md` (estado: final). El script de las tasas de captura está en `/private/tmp/claude-501/-Users-cnavarreteliz-Documents-Repos-earthquakes/a9d8f28c-dfc3-4e03-bbcb-ed0af0e86ec5/scratchpad/rep/hit.py`.
