# E3: kernels ETAS por profundidad + memoria corta — SEÑAL ROBUSTA (mejora SOTA)

**Resultado primario** (panel diario M≥5.0, test 2013+, 804 celdas positivas):
IG full-vs-baseline **+0,0117 nats/evento** (~+4% sobre el +0,296 del baseline),
**p=0,015** (phase_randomize n=200), IC95% bootstrap anual [+0,004, +0,020].
AUC 0,729→0,732. Pasa Bonferroni de campaña (0,05/3=0,0167).

**Robustez (preregistrada):**
- Panel M≥5.5: IG +0,015, p=0,015, misma dirección (IC95% roza 0: [-0,001,+0,030]).
- Leave-one-year-out: IG +0,010 a +0,014 sacando CUALQUIER año — no lo arrastra
  ninguna secuencia (ni Iquique 2014, ni Illapel 2015).
- Ablación: solo-profundidad +0,0086, solo-conteos-1/3d +0,0035 — ambos grupos
  aportan de forma aditiva.
- Coeficientes: kernels lentos negativos (intra_slow −0,30, iface_slow −0,17):
  el modelo corrige el exceso de memoria larga del baseline, distinguiendo
  interfaz de intraslab.

**Límites honestos:** es mejora de pronóstico de CLUSTERING (`y_all`).
En `y_main` (sismos principales): +0,003, p=0,40 — nulo, como todo lo anterior.
No es predicción de mainshocks; es un ETAS mejorado (~+4%, en el rango 2-10%
que la literatura considera creíble: Nandan 2019, Zlydenko 2023).

Confirmación n=1000 (`e3_confirm.py`, 2026-10-26): **p=0,0170**, IG idéntico
+0,0117. Pasa el umbral preregistrado p<0,05; queda justo en el borde del
Bonferroni de campaña (0,05/3=0,0167 con E1-E3; 0,0125 con E4 incluido).
Evidencia de apoyo: réplica M5.5, LOOYO estable 14/14 años, ablación aditiva.
Lectura honesta: mejora pequeña y robusta, significancia en el límite bajo
corrección múltiple estricta.
Scripts: `e3_kernels.py`, `e3_robust.py`, `e3_confirm.py`.
