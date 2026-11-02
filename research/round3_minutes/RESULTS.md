# Resultados: piloto minutos (round3_minutes)

**Veredicto: señal nominal según preregistro, pero explicada por réplicas y cambio
de régimen. Sin evidencia de predicción del sismo principal a minutos.**

## Números (test 2018-01 → 2026-10, 1055 triggers M≥4.7, 31 positivos)

| Métrica | Valor |
|---|---|
| Tasa train / test | 12,0% / 2,9% (cambio de régimen fuerte) |
| IG full vs climatología (primaria) | +2,51 nats/evento, IC95% [+1,75, +4,39], p≈0,0002 |
| IG full vs mag-only | +2,24 nats/evento |
| AUC full / mag-only | 0,81 / 0,65 |
| PR-AUC full vs base | 0,21 vs 0,03 |

## Diagnóstico (por qué no es predicción de mainshocks)

1. **+1,81 de los +2,51 nats vienen solo de acertar la tasa base del periodo test.**
   La climatología usa la tasa de train (12%, inflada por Maule 2010 / Iquique 2014 /
   Illapel 2015) y sobre-predice todo el test (2,9%). Cualquier modelo que prediga
   "menos" gana por ajuste de régimen, no por discriminar qué trigger escala.
2. **65% de los 31 positivos (20/31) tenían un evento mayor en los 30 días previos
   dentro de 150 km: son réplicas.** El modelo aprende clustering (x1 alto, x3 bajo,
   x4 alto = secuencia activa), lo mismo que ETAS ya captura.
3. Quedan ~11 positivos sin mayor previo: sin potencia para afirmar nada sobre
   sismos principales, consistente con los nulos de ronda 1 y ronda 2.

## Conclusión

A 60 minutos el piloto replica el único resultado robusto de todo el repo:
**las réplicas sí se pronostican (clustering); los sismos principales, no.**
Con el catálogo USGS no hay señal de predicción del mainshock a minutos.
Script: `pilot_minutes.py`. Preregistro: `PREREGISTRATION.md` (sin cambios).
