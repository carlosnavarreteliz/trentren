# E4: repeaters vía S-P — NULO

Test 2020-2021 (291 quiet, 59 pos): IG +0,011, IC95% [−0,031, +0,059], p=0,069.
AUC 0,662 vs 0,653. Coef positivo (+0,32) pero no significativo.

Diagnóstico: 51.542 pares similares (3,5%) colapsaron en 30 familias
single-linkage que cubren 3.209/5.104 eventos — chaining, no familias repeater
reales. El umbral RMS<0,5s preregistrado resultó demasiado laxo para la
geometría de estaciones (S-P poco discriminante intra-bin). La actividad de
"familias" ≈ actividad total → sin información propia. Un reintento necesitaría
umbral más estricto + linkage completo, pero eso sería post-hoc: se registra
como nulo. Script: `e4_repeat.py`, familias: `e4_families.parquet`.
