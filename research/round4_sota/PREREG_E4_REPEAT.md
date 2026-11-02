# Preregistro E4: repeaters vía S-P (round4_sota)

**Idea no testeada en el repo:** terremotos repetidos/co-localizados como
creepmeters (Uchida & Bürgmann 2019; LITERATURE rank 10). El repo la descartó
porque exige correlación de waveforms — pero los picks Riedel (P+S por estación)
permiten una versión barata: eventos co-localizados comparten tiempos S-P
casi idénticos en estaciones comunes. Sin formas de onda.

**Diseño:**
- Pares (i,j) de eventos Riedel primario limpio (ML≥2.5) con ≥6 estaciones comunes
  con picks P y S en ambos (enmienda: la mediana es 8 estaciones/evento, ≥8 dejaría
  muy pocos pares; decidido por cobertura, antes de ver efectos). Similares si
  RMS(diff de S-P) < 0,5 s (umbral fijo).
- Familias = componentes conexas con ≥3 miembros (single-linkage).
- Alcance piloto (fijado por conteo, antes de ver efectos): solo tipo Interplate
  (creep de la placa es la hipótesis física), ML≥2.5, bins 1° lat −33/−32/−31
  (890+676+974 eventos 2019-2021; pares intra-bin ≈1,1M, cómputo viable).
  Periodo 2017-2021.
- Test (mismo protocolo que E1): triggers quiet ML≥4.0, target ML≥5.0 en 7d/150km,
  train 2017-03→2019-12, test 2020-01→2021-11-15.
- Feature causal: log1p(nº eventos pasados en 150km/30d que pertenecen a una
  familia con ≥2 miembros previos al trigger). Solo pares con ambos eventos
  < t_trigger (estrictamente causal).
- Baseline: mag trigger + log1p(total 24h) + log1p(total 7d) (igual que E1).
- Métrica primaria: IG full-vs-baseline por positivo. p permutación 5000 +
  IC95% mensual. Umbral p<0,05, 1 test primario.

**Veredicto:** nulo si p≥0,05 o IC incluye 0. Descriptivos (nº familias, tamaños,
distribución espacial) se reportan sin test.
