# Campaña round7: recurrencia/energía + Transformer (2026-10-28)

## P1: la hipótesis de los ~25 años y energía acumulada

- Nacional M≥8: 12 eventos (1906-2015), gaps 0-25 años, **CV=0,75** (±0,21):
  sin reloj. La serie 1939/1960/1985/2010 mezcla 4 segmentos e ignora
  1995/2001/2014/2015. Físicamente la recurrencia es por segmento.
- P1b (por segmento, energía Benioff/momento acumulados + años-desde):
  primera métrica engañosa (IG +10,95 por Poisson degenerado); corregida:
  **AUC 0,495 = azar**. La energía acumulada no ordena el riesgo zonal
  fuera de muestra. Detalle: `P1B_RESULT.md`.

## P2: Transformer con atención (protocolo idéntico a M2)

IG −0,86, AUC 0,623 ≈ GRU 0,618 < logit 0,663. La atención tampoco.
Tercera arquitectura secuencial que confirma el mismo techo.

## Dashboard v2

Rediseño: KPIs de estado, barras de probabilidad por trigger, panel de
ciclo por zona (descriptivo, etiquetado sin poder predictivo), backtest,
historial de 24 líneas. `dashboard/index.html` (13 KB, sin JS).

## Conclusión

Recurrencia/energía: descriptivamente real (el margen acumula y libera),
predictivamente nula con estos datos. Transformer: nulo como GRU y GBDT.
El techo sigue siendo clustering (E3, +4%).
