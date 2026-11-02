# Campaña round6: arquitecturas nuevas, máximo esfuerzo (2026-10-27/28)

Pedido: el modelo es "muy malo"; buscar con esfuerzo algo que los humanos
pasen por alto; más historia; modelos nuevos.

## Diagnóstico del caso criticado (Coquimbo 2026-10-14)

En los 30 días/150 km previos al M5.1 había **0 eventos M≥4,7** (max M4,4).
En el catálogo USGS ese sismo nació del silencio total: ninguna feature
podía marcarlo. El M5,0 7 min después, igual. Límite del dato, no del modelo.

## Experimentos

| # | Enfoque | Resultado |
|---|---|---|
| M2 | SeqGRU historia cruda (nuevo) | **peor**: IG −0,64, AUC 0,62 vs 0,66 |
| M3 | Motivos simbólicos pre-mainshock (nuevo) | nulo: 0/29 candidatos validan |
| M4 | GBDT 43 features densas (nuevo) | **peor**: IG −0,18 / −0,03 |
| M5 | 36 años + 32 features + GBDT | nulo: IC95% cruza 0, AUC idéntico |

## Lectura

Cuatro arquitecturas distintas (recurrente, simbólica, boosting, lineal
rica) convergen al mismo techo: la señal de clustering que ya capturan
3 resúmenes. Más capacidad empeora (sobreajuste, n≈2.800); más historia
no mueve el AUC (0,87 en todos). El cuello de botella es información,
no modelamiento: con catálogos (incluso densos, ML≥2,5), el sismo
principal no anuncia su llegada en forma explotable.

## Qué sí mejoró y sigue vigente

E3 (+4% sobre ETAS, p=0,017) productivizado en forecast_short v2.
Benchmark Chile-ST v1.0 y dashboard actualizados con esta campaña.
