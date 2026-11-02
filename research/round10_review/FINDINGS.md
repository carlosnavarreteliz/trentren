# Round10: revisión profunda (memorización, anomalías, patrones)

## A) Memorización/fuga: DESCARTADA empíricamente
300 triggers 2024+: features recomputadas con catálogo TRUNCADO en el trigger
vs completo → **max diff = 0,00 exacto**. Features sin lat/lon (conteos, mag,
profundidad, tiempos): no puede memorizar lugares. Splits cronológicos, test
congelado, log prospectivo con issued_at. Diferencia con PI-ST auditado: aquél
usaba coordenadas de entrenamiento como feature (memorización espacial real).

## B) Años anómalos: la anomalía real es la PROFUNDIDAD reportada
- 2018/2022/2025: n(M≥4,7) normal (~300/año), n(M≥6) bajo (3/6/3): quietud
  tectónica real en grandes, no caída de datos.
- **Artefacto descubierto:** mediana de profundidad del catálogo: 35 km (2010)
  → 60 (2012) → **103-114 km (2018-2026)**. Cambio de práctica de reporte USGS
  (profundidades fijas por defecto ~33 km pre-2016 vs hipocentros profundos
  reportados después). Post-2011 (expansión CSN) la composición cambia era a era.
  Implicación: features con corte 60 km mezclan dos regímenes de medición;
  E3 sobrevivió porque LOOYO fue estable en ambas eras, pero queda declarado.

## C) Patrones directos (revisión propia, sin modelo)
- Techo empírico de escalamiento 24h por magnitud del trigger:
  M4,7-5: 17% · M5-5,5: 27% · M5,5-6: 38% · M≥6: 50%. Gradiente físico limpio:
  ahí vive toda la predictibilidad de corto plazo.
- Foreshocks Riedel: 44% de M≥5,5 tuvo ML≥4 en 7d/50km previos vs 35% en
  controles desplazados (+9 pp, marginal): los foreshocks EXISTEN pero apenas
  discriminan — por eso todos los tests predictivos dan nulo más allá de la tasa.

## Caminos no explorados (sugerencias, sin canales físicos nuevos)
1. **E3 con etiquetas Mw homogeneizadas en el PANEL** (E3 original usó mags
   mixtas para y): el fair test de triggers sugiere que el panel puede mejorar.
2. **Coulomb con fallas finitas SRCMOD** (rank #1 literatura, nunca hecho):
   descargas KB-MB por evento; kernels orientados por geometría de falla.
3. **Traffic-light Gulia-Wiemer bien hecho** en Riedel (b-positivo tras
   M≥5,5 con targets declusterizados y test temporal estricto).
4. **Z-value/RTL diario en Riedel homogéneo** (quiescencia testeada solo
   semanal en USGS ruidoso).
5. Features robustas al artefacto de profundidad (rangos relativos por era,
   no cortes absolutos de 60 km).
