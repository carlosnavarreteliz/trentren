# Preregistro M2: SeqGRU sobre secuencias densas (round6)

**Pregunta:** ¿una GRU sobre la secuencia cruda de los últimos 32 eventos
(ML≥2.5, 150 km) supera al logit de resúmenes (E1) en escalamiento ML≥4→ML≥5/7d?

**Motivación:** ningún experimento previo alimentó historia cruda a una red;
sorpresa_neural (ronda 2) fue TPP sobre panel semanal USGS, otro diseño.
Si hay estructura temporal que los resúmenes pierden, aquí puede aparecer.

**Diseño:**
- Catálogo Riedel limpio. Triggers ML≥4.0, 2017-06→2021-11-15.
  Train ≤2019-12 (últ. 15% temporal = val), test 2020-01+.
- Secuencia: últimos ≤32 eventos ML≥2.5 en 150 km/90d previos al trigger
  (orden cronológico). Features/evento: log10(Δt_min), log10(dist_km), mag,
  depth/150, one-hot tipo (5). Padding a la izquierda con ceros.
- Cabeza: features del trigger (mag, depth) concatenadas al estado final.
- GRU (2 capas, hidden 64, dropout 0,2) + MLP. BCE con peso de clase.
  Adam lr 1e-3, ≤15 épocas, early-stop en val (paciencia 3). Semilla 5.
- Primario: IG vs logit-resumen (b_mag, b_tot24, b_tot7), por positivo, test.
  p permutación 5000 + IC95% mensual. Umbral p<0,05.
- Secundarios: AUC/PR-AUC, subset quiet de E1, descriptivo mainshock-like
  de los positivos capturados.

**Veredicto:** nulo si p≥0,05 o IC incluye 0. Si señal: adversarial antes
de creer (drop-secuencia dominante, permutar orden temporal como control).
