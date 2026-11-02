# M2: SeqGRU sobre secuencias densas — NULO (peor que logit)

Test 2020-2021 (1917 triggers, 451 pos): **IG vs logit-resumen = −0,64**,
IC95% [−0,98, −0,41], p=0,9998. AUC 0,618 (GRU) vs 0,663 (logit 3 features).
PR-AUC 0,321 vs base 0,235 pero con IG muy negativo: ranking peor e impreciso.

Segunda confirmación de que lo neuronal rinde peor que resúmenes simples con
~2.000 secuencias de entrenamiento (cf. ronda 2 surprise_neural). La historia
cruda no esconde señal que el resumen no capture; el cuellos es tamaño de datos,
no arquitectura. Script: `m2_seqgru.py`, preds en `m2_preds.npz`.
