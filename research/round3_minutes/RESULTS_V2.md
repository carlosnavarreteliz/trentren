# Resultados: piloto v2 minutos con microsismicidad (round3_minutes)

**Veredicto: señal nominal vs mag-only, pero es clustering — y lo de minutos
aporta cero. Sin evidencia de predicción del mainshock a minutos.**

## Números (Riedel; train 2017-2018: 2318 trig/95 pos; test 2019-2021: 3078/86)

| Métrica | Valor |
|---|---|
| IG full vs mag-only (primaria) | +0,47 nats/ev, IC95% [+0,07, +0,79], p≈0,0002 |
| IG full vs sin-x5 (ablación 10 min) | **+0,003 nats/ev ≈ 0** |
| AUC full / mag-only | 0,87 / 0,82 |
| PR-AUC full vs base | 0,22 vs 0,03 |
| Coef x5 (microsismos 10 min) | 0,02 (vs 0,74 x1, 0,80 x2, 0,57 x4) |
| Positivos que son réplicas (mayor previo 30d/150km) | 57/86 = **66%** |

## Lectura honesta

1. El modelo supera al baseline de magnitud, pero toda la ganancia viene de
   features de clustering horario (tasa 60 min, mag del trigger, mag máxima 24h).
2. La novedad de este piloto —conteo de microsismos ML≥2.5 en los últimos
   **10 minutos**— tiene coeficiente ~0 y ablación +0,003 nats: **no aporta nada**.
3. 66% de los aciertos son réplicas dentro de secuencias activas. Para los ~29
   restantes no hay potencia ni diseño para distinguir mainshocks.
4. IPOC (norte 2007-2021) no sirvió para este piloto: 0 positivos fuera de la
   ventana de Iquique 2014 (12, todos de una sola secuencia). Los 5,1 GB de
   formas de onda cubren 3 estaciones × 3 ventanas: caso de estudio, no entrenamiento.

## Conclusión

Tercera confirmación independiente (ronda 1, ronda 2, v1, v2): a 60 minutos el
catálogo denso **no agrega nada sobre el clustering que ETAS ya captura**.
Predecir el sismo principal minutos antes sigue sin evidencia con estos datos.
Script: `pilot_minutes_v2.py`. Preregistro: `PREREGISTRATION_V2.md` (sin cambios;
se excluyeron 9253 eventos Riedel sin Ml, documentado en el script).
