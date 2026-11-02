# R13: torneo "usar lo mejor" — ETAS espacial vs GBDT-full vs v4-logit (2026-10-27)

Mismo protocolo T1 (train ≤2020, calib 2021-23, test congelado 2024+, n=945).
Contendores: v4-logit (vigente), ETAS espacial (intensidad Omori+decaimiento
exp(-d/150) por trigger, Platt en calib), ETAS+logit (stacking), GBDT-full
(HistGB 300 árboles, features v4 + espaciales R12 + tasa ETAS).

Test 2024+ (IG vs v4, bootstrap por año):
- ETAS-esp solo: AUC 0,50-0,54, IG -4,7/-4,1/-1,3 (malo: la logit ya contiene
  su información vía conteos/recencia/máx-mag; el kernel crudo no aporta).
- ETAS+logit: IG +0,001 en todo (el stacking le da peso ~0 al ETAS).
- GBDT-full: AUC mayor (0,89/0,88/0,66) y a 24h IG +0,186 IC95 [+0,05,+0,30],
  LOYO-test positivo los 3 años. PERO la selección honesta sin mirar test lo
  descarta: en calib-CV (5 bloques 2021-23) GBDT tiene AUC 0,41 a 24h e
  IG -4,1 vs logit; a 60 min IG -5,0. Adoptarlo sería elegir con el test.

Veredicto: NO se adopta GBDT. v4-logit sigue siendo lo mejor seleccionable
ex-ante. La ganancia GBDT-24h queda como hipótesis a confirmar en prospectivo,
no como modelo. Lo "mejor" que falta no es arquitectura: es el catálogo denso
post-2021 (picking GPU sobre formas de onda C1/CSN vía FDSN-EarthScope).
Scripts: /tmp/best_contest.py, /tmp/tournament.py, /tmp/gbdt_check.py (sesión 2026-10-27).
