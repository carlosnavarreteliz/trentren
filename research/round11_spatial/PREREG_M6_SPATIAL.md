# Preregistro M6: modelo espacial + profundidad (round11)

**Pregunta:** ¿agregar lugar (base espacial suave) e interacciones con
profundidad supera a v3 en la tarea congelada T1 v1.1?

**Features nuevas (sobre COLS_V2):** lat, lon, sin/cos(lat), base RBF suave
(7×4 centros, σ=3° — suave a propósito: no puede memorizar epicentros),
interacciones mag×(depth≤60), lat×(depth≤60).
**Modelos:** logit L2 + HGBT. Train ≤2020, Platt 2021-2023, test 2024+ congelado.
**Primario:** IG vs p_sota v3 (bootstrap anual IC95%). 
**Anti-memorización:** leave-one-zone-out en las 3 zonas con más test triggers
(entrenar sin la zona, evaluar en ella): si IG colapsa a ≤0 fuera de zona,
es memorización espacial y se descarta.
**Veredicto:** mejora solo si IG>0 con IC>0 Y sobrevive leave-one-zone-out.
