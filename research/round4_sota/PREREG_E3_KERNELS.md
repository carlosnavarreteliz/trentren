# Preregistro E3: kernels ETAS mejorados (round4_sota)

**Pregunta:** ¿un ETAS con productividad dependiente de profundidad + memoria
ultracorta supera al baseline ETAS-logit del harness en pronóstico diario?

**Motivación:** el baseline usa 12 kernels fijos sin distinción interfaz/intraslab
y conteos mínimos de 7 días. Literatura (Nandan 2019: ETAS espacialmente variable;
Zlydenko 2023: +4% sobre ETAS) dice que hay 2-10% de mejora disponible dentro de
la familia ETAS. Esto ataca al SOTA en su propio juego (clustering, `y_all`).

**Diseño:**
- Panel harness diario M≥5.0 (`step=1, horizon=1, target=5.0`), split estándar
  (train ≤2012, test 2013+). Target primario `y_all`; secundario `y_main`.
- 8 features causales nuevas (eventos M≥4.7, t<t_start):
  - Omori interfaz (depth≤60): (p=1.3,c=0.02,α=1.1,s0=0.3) y (p=0.9,c=0.5,α=1.1,s0=1.0)
  - Omori intraslab (depth>60): los mismos dos
  - log1p(conteos M≥4.7 en bins ±1, últimos 1 y 3 días)
  (memoria de 1-3 días no existe en el baseline: mínimo 7d)
- Modelo: logit (primario) + GBM-limpio? No: solo logit, igual que baseline.
- Métrica primaria: IG full-vs-base por evento en test (`y_all`).
- Significancia: protocolo ronda 1 — `phase_randomize` n=200 (primario) + IC95%
  bootstrap anual. Umbral p<0,05 (1 test primario; se declara exploratorio).
- Robustez si positivo: panel M≥5.5, `y_main`, leave-one-year-out.

**Veredicto:** nulo si p≥0,05 o IC95% incluye 0.
