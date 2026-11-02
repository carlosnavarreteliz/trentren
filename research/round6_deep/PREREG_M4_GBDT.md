# Preregistro M4: GBDT con 60+ features (round6)

**Pregunta:** ¿un modelo tabular fuerte con features multi-escala ricas supera
al logit simple (E1) en escalamiento ML≥4→ML≥5/7d con catálogo denso?

**Features (causales, 150 km salvo estáticas):** conteos ML≥2.5 en
{10min,1h,6h,24h,7d,30d} × splits {total, Interplate, Intraslab, profundidad≤60};
max mag {24h,7d,30d}; b-positive k=20 y k=50 (diffs ≥0,2 ML, 150 km);
suma ETAS-lite (Omori p=1,08 y 1,3); días desde último ML≥5 y ML≥6 en 150 km;
tendencia de profundidad (media 7d − 30d); mag/profundidad/distancia del trigger;
tasa histórica del bin lat 1° (train); hora del día (sen/cos, solo referencia).
≈60 columnas.

**Modelo:** HistGradientBoosting (300 iters, lr 0,05, hojas 31, min_leaf 50,
L2) + comparador logit mismo set. Baseline: E1-logit (3 feats).
**Protocolo:** train ≤2019 (val = últ. 15% para early-stop HGBT), test 2020-01
→ 2021-11-15. Primario: IG vs E1-logit, permutación 5000, IC95% mensual.
Umbral p<0,05. Si señal: top-importancias + drop-secuencia dominante.

**Veredicto:** nulo si p≥0,05 o IC incluye 0.
