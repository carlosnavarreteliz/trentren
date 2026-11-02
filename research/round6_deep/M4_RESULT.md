# M4: GBDT 60+ features — NULO (peor)

Test 2020-2021 (1912 trig, 451 pos). GBM: IG −0,18, AUC 0,558 (base 0,662).
Logit-60: IG −0,025, AUC 0,639. **Más capacidad con ~2.800 ejemplos de
entrenamiento empeora fuera de muestra**: sobreajuste + cambio de régimen.
Las 43 features no contienen nada que 3 resúmenes no capturen mejor.
Script: `m4_gbdt.py`.
