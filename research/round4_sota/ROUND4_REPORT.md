# Campaña round4_sota: informe final (2026-10-26/27)

Objetivo: superar el baseline SOTA (ETAS) verificando data, consiguiendo más
datos y entrenando con protocolo prospectivo honesto.

## Data: verificación + conseguida

- Verificación (`verify_catalogs.py`): USGS limpio (0 duplicados, Mc 3,7→4,9 por
  década, mezcla mb/md/ml/Mw sin homogeneizar — limitación declarada); Riedel
  primario con 401 duplicados exactos + 9.253 NaN-Ml + desordenado → limpio en
  `primary_clean.parquet` (612.267 ev); IPOC limpio.
- Descargado de Zenodo 15284376: catálogo secundario (1,7 MB, 77% sin Ml —
  marginal), picks secundarios (8,4 MB) y picks primarios (506 MB).
- Borrado: formas de onda 5,1 GB (`research/data/` 6,7→1,6 GB, luego +0,5 GB picks).
- Extensión post-2021: densa no existe públicamente (Riedel/IPOC terminan 2021);
  USGS cubre M≥4,7 hasta hoy (en vivo verificado 2026-10-26). Reprocesar CSN
  2022-2026 ≈ TBs + GPU: fuera de alcance, declarado.

## Experimentos (todos preregistrados)

| # | Idea (no testeada antes) | Resultado |
|---|---|---|
| E1 | Excitación cruzada por tipo Riedel | NULO (+0,008, p=0,10; intraslab signo opuesto) |
| E2 | Nowcasting Rundle | NULO (AUC 0,48, p=0,95; label original vacuo, enmienda registrada) |
| E3 | Kernels ETAS por profundidad + memoria 1-3d | **SEÑAL: +0,0117 nats (+4%), p=0,017 (n=1000)** |
| E4 | Repeaters vía S-P | NULO (+0,011, p=0,069; chaining single-linkage) |

## E3: la mejora (clustering, no mainshocks)

Panel diario M≥5, test 2013+, 804 positivos. Réplica M5.5 (+0,015, p=0,015).
LOOYO 14/14 años estable (+0,010/+0,014). Ablación aditiva (profundidad +0,0086,
conteos +0,0035). Pasa p<0,05 preregistrado; en el borde de Bonferroni de
campaña (0,0170 vs 0,0167 E1-E3). Rango 2-10% creíble según literatura.
En `y_main`: nulo (p=0,40) — mainshocks siguen impredecibles.

## Productivización

`forecast_short.py` v2 con features E3 (split interfaz/intraslab + memoria 3h):
mejora 60 min y 3 h en holdout (AUC 0,65→0,68 y 0,61→0,65), plano a 24 h.
Chequeo de consistencia (8-14 pos), no prueba. En vivo verificado.

## Conclusión

Mejora significativa pequeña y robusta sobre ETAS en pronóstico de clustering
a días (E3), productivizada. Predicción del sismo principal (60 min, días,
nowcasting, tipos, repeaters): nula en todos los datos disponibles.
