# P1b: renovación por segmento + energía — NULO (artefacto corregido)

Primera corrida: IG +10,95 "SEÑAL" — **artefacto**: el Poisson por zona tenía
tasa 0 en 3 zonas (sin positivos en train) → p≈0 castiga ~20 nats/acierto.
Contra Poisson suavizado: IG +0,29 pero **AUC 0,495** (azar): gana solo por
nivel de probabilidad (hedging), cero discriminación de qué zona-año es riesgoso.
Coefs grandes con AUC 0,5 = sobreajuste sin ranking.

Veredicto: las variables de energía acumulada / años-desde-el-último (la
hipótesis 1939-1960-1985-2010) **no ordenan el riesgo** fuera de muestra.
Scripts: `p1b_cycle.py` (+ diagnóstico en el historial de la sesión).
