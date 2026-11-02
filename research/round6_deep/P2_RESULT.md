# P2: Transformer con atención — NULO (peor)

Test 2020-2021 (1917 trig, 451 pos): IG −0,86, AUC 0,623 ≈ GRU (0,618),
bajo logit (0,663). La atención no rescata lo que la recurrencia no pudo:
con ~2.800 secuencias, capacidad secuencial profunda pierde contra
3 resúmenes. Tercera arquitectura secuencial/tabular que confirma el techo.
Script: `p2_transformer.py` (protocolo idéntico a M2).
