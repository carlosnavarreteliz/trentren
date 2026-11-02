# Auditoría de datos faltantes (2026-10-28)

Pregunta: ¿el modelo no predice nada o faltan datos? Respuesta en dos partes:
el modelo SÍ predice réplicas (AUC 0,68 a 60 min; E3 +4% sobre ETAS en panel
diario, p=0,017). Para el sismo principal, además de 24 nulos, **hay datos
clave que nunca estuvieron disponibles**. Inventario con números:

## Lo que el catálogo USGS no ve (medido)

| Brecha | Número | Consecuencia |
|---|---|---|
| Microsismicidad colapsada | M2,5-3,9: 1.474/año (2000s) → **35/año (2020s)**; M≥4,7 ~115-271/año | Ciego a foreshocks pequeños desde ~2010 |
| Catálogos densos terminan 2021 | Riedel/IPOC: 26 mil ev/año ML≥2,5 vs **~900/año** USGS 2022+ | **2022-2026 sin microsismicidad utilizable (×30 menos)** |
| Offshore (fosa) | Riedel solo 4,4% ev con lon<−72°; la nucleación megathrust es offshore | Redes terrestres ven mal donde nacen los M8-9; sin OBS |
| Magnitudes mezcladas | mb 23k, md 20k, ml 5,5k, mw ~5k (sin Mw homogeneizada) | b-value/energía comparan peras con manzanas entre eras |
| Mc por década | 4,9 (80s) → 3,7 (2000s) → 4,5 (2010s+) | Todo conteo <M4,5 es artefacto de red, no física |

## Canales físicos nunca obtenidos (no testeables aquí)

| Canal | Estado en el repo | Costo estimado para hacerlo bien |
|---|---|---|
| Formas de onda continuas 2007+ (toda la red) | Solo piloto: 3 estaciones × 3 ventanas (5,1 GB, **borrado por pedido del usuario**) | ~2 TB + ~500 CPU-h (medido en piloto Iquique) |
| GNSS alta tasa (1-5 Hz, precursors 2h tipo Bletery) | Solo soluciones diarias (245 MB derivados) | ~1-6 GB/día × 130 estaciones; acceso NGL/CSN |
| Catálogo CSN/IPOC post-2021 en tiempo real | No existe públicamente procesado; USGS es el sustituto ciego | Reproceso tipo Riedel: GPU + meses de cómputo |
| OBS fondo oceánico en la fosa | Inexistente integrado | Campañas navales; no descargable |
| InSAR sistemático (Sentinel-1) | No descargado | TB en bruto; procesamiento pesado |
| Mecanismos focales densos | GCMT esparcido (testeado semanal: nulo) | Requiere catálogo de tensores propio |

## Lo que SÍ se testeó y dio nulo (con estos datos)

24 líneas: 14 familias exógenas semanales, 6 ronda 2 (sorpresa ETAS/neuronal,
transferencia global, microsismos, waveforms piloto), v1/v2 minutos,
E1 tipos, E2 nowcasting, E3 (**única señal: clustering**), E4 repeaters S-P,
E5 anomalías multivariadas IF/PCA, M2 SeqGRU, M3 motivos, M4 GBDT, M5 36 años,
P1b energía por segmento, P2 Transformer.

## Conclusión

"Algo debe haber" es posible pero no demostrable: los canales con mejor
fundamento físico para un precursor (waveforms continuas a escala, GNSS alta
tasa, microsismicidad post-2021, OBS) **nunca estuvieron en esta carpeta**.
Todo lo que estuvo, se testeó. La siguiente inversión real no es otro modelo:
es conseguir 1 de esos 4 canales y repetir el protocolo preregistrado.
