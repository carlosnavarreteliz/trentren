# M5: historia larga (36 años) con features ricas — NULO

Test 2018-2026 (878 triggers, 25 positivos): GBM IG +0,15 (IC95% [−0,23,
+0,46]), logit-rico +0,07 (IC95% [−0,24, +0,28]). **IC incluye 0 → nulo**
por criterio preregistrado. AUC idéntico entre modelos (0,867-0,875):
todos convergen a la misma señal de clustering. El p de permutación (0,0002)
es anti-conservador para IG calibrado; manda el IC bootstrap (declarado en
preregistro como criterio junto a p — con IC cruzando 0, nulo).

Con 4× más historia, las features ricas tampoco despegan. Script: `m5_history.py`.
