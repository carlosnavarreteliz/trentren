# Preregistro P1b: renovación por segmento + energía acumulada (round7)

**Hipótesis (usuario):** los grandes se repiten con energía acumulada por zona;
variables de acumulación deberían predecir el próximo grande.

**Contexto medido (P1a):** M≥8 nacional: 12 eventos, gaps 0-25 años, CV=0,75
(Poisson=1, incertidumbre ±0,21 con N=11 → sin reloj de 25 años; la serie
1939/1960/1985/2010 mezcla segmentos e ignora 1995/2014/2015).

**Diseño P1b (versión física, por segmento):**
- 8 zonas (las de forecast_short). Panel anual 1960-2021 (issue 1-ene),
  target: ≥1 evento M≥7,5 en la zona dentro de 5 años.
- Features (causales): log1p(años desde último M≥7,5 zonal),
  log momento acumulado desde entonces (M≥5, log10 M0=1.5M+9.1),
  Benioff acumulado (Σ10^(0.75M)), déficit = años×66 mm/año,
  tasa zonal 20 años (M≥5/año).
- Baseline: Poisson (tasa zonal de train). Modelo: logit C=1.
- Split: train ≤1990, test 1991-2021. Métrica: IG vs Poisson, AUC,
  IC95% bootstrap por décadas-zona (2000). Umbral p<0,05 nominal.
- Potencia declarada: ~10-12 positivos en test → solo efectos grandes.

**Veredicto:** nulo si IC incluye 0. Es el test justo de la idea de energía.
