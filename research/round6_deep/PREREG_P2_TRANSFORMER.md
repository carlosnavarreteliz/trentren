# Preregistro P2: Transformer con atención (round7)

**Pregunta (pedido explícito):** ¿un Transformer encoder sobre la secuencia de
32 eventos supera al logit-resumen donde la GRU (M2) fracasó?

**Protocolo idéntico a M2** (mismo dataset de secuencias Riedel, splits,
target ML≥4→ML≥5/7d, baseline logit [mag, tot24, tot7], IG primario,
permutación 5000, IC95% mensual) para comparabilidad directa.
Arquitectura: embedding lineal 9→64 + pos-encoding sinusoidal, encoder
2 capas / 4 cabezas / d_ff 128, dropout 0,2, mean-pool con máscara,
cabeza MLP (+features trigger). Adam 1e-3, ≤15 épocas, early-stop val.

**Veredicto:** nulo si p≥0,05 o IC incluye 0. Se reporta AUC vs GRU (0,618)
y vs logit (0,663) como referencia.
