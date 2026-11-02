# Preregistro C1: clima (ENSO/PDO) como modulador sobre ETAS (round9)

**Pregunta:** ¿ONI/PDO agregan información al pronóstico diario M≥5 (panel
harness, y_all) más allá de ETAS? Ronda 1 testeó precipitación ERA5 binned
(nulo); nunca los índices ONI/PDO como covariables diarias sobre ETAS.

**Diseño:** panel harness diario M≥5.0 (split y baseline estándar).
Features globales (broadcast, lageadas 1d, mensual→diario fwd-fill):
x_oni, x_pdo, x_oni_sin (ONI × sin(2π·día-del-año)).
Modelo logit. Primario: IG vs baseline `y_all`. phase_randomize n=200
(protocolo ronda 1) + IC95% anual. Umbral p<0,05 nominal.

**Veredicto:** nulo si p≥0,05 o IC incluye 0.
