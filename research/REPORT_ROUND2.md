# Informe ronda 2: aprender de lo improbable

_Generado el 2026-10-18 por el workflow `chile-quake-round2`. Detalle por experimento en `round2/*/RESULTS.md` y `round2/*/summary.json`._

## Ronda 2: resumen ejecutivo

**Ninguna de las seis apuestas supera al baseline ETAS-logit.** El umbral de Bonferroni es α = 0,0083 y ningún test primario se le acerca. El blanco principal, y_main, son los sismos principales M≥6 de Chile declusterizados: 72 en el test 2013–2026. En y_main la ganancia va de −0,016 a +0,002 nats/sismo, con p ≥ 0,42.

| Experimento | Pregunta | Ganancia en y_main (nats/sismo) | p | Veredicto |
|---|---|---|---|---|
| surprise_etas | ¿La sismicidad que el ETAS no esperaba anticipa M≥6? | −0,0018 | 0,49 | nulo |
| surprise_neural | ¿La sorpresa de una red neuronal o un autoencoder anticipa M≥6? | −0,0039 | 0,45 | nulo |
| global_transfer_gbm | ¿Un modelo GBM entrenado con 16 subducciones del mundo pronostica mejor Chile? | −0,009 | 0,63 | nulo |
| global_transfer_neural | ¿Una red preentrenada en el mundo y ajustada a Chile mejora el pronóstico? | +0,0021 | 0,42 | nulo |
| enhanced_catalog | ¿Los sismos pequeños (IPOC ML≥3, objetivo M≥5) ayudan? | −0,016 | 0,62 | nulo |
| waveform_pilot_iquique | ¿La señal sísmica continua distinguía los meses previos a Iquique 2014? | sin cálculo | ~1/3 como mínimo | datos insuficientes |

Ningún experimento pasó por una verificación adversarial independiente. Cotejé todas las cifras contra cada `summary.json` y `RESULTS.md`. Como todos los tests principales son nulos, ningún resultado positivo sobrevive.

**Resultados laterales**
- **Aprender de otras zonas evita el sobreajuste, pero no le gana al baseline.** Un modelo GBM entrenado solo con Chile se sobreajusta y pierde entre 0,20 y 0,30 nats/sismo. Sumar las otras subducciones recupera esa pérdida: entre +0,23 y +0,29 frente al GBM chileno (p ≤ 0,002). Un modelo que nunca vio Chile lo pronostica igual de bien que el modelo local, porque el agrupamiento de sismos funciona igual en todas partes. Aun así, solo llega al nivel del ETAS-logit, sin superarlo.
- **La red global deja una pista que no es evidencia.** Usada sola, sin el ETAS al lado, supera al ETAS-logit en unos +0,04 a +0,05 nats/sismo en y_main. Pero ese análisis se hizo después de ver los datos, y su p (0,002 a 0,008) viene de un remuestreo por año que aquí exagera la significancia. Cuando la misma red se suma al ETAS en el test oficial, la ganancia desaparece (p = 0,42).
- **La señal Anderson-Darling (surprise_etas) es frágil.** Mide qué tan raros son los tiempos entre sismos según el ETAS. Da +0,011 con p = 0,006, pero fue la mejor de 11 variables probadas. Corrigiendo por esa selección, p ≈ 0,07. Casi toda la señal viene de Iquique 2014 y casi desaparece al cambiar la magnitud mínima de 4,7 a 5,0.
- **Los sismos pequeños no anticipan nada.** Antes de 227 sismos M≥5 no hay exceso de foreshocks frente a los controles: 4,8% de los casos supera el umbral, cuando el azar daría 5%. Un catálogo dio p = 0,01, pero era un falso positivo del test por defecto cuando los datos cubren solo unos pocos años.
- **La señal continua antes de Iquique es el enjambre que ya se conocía.** La tasa, la migración y el b bajo previos a Iquique corresponden al enjambre que el catálogo rutinario ya registraba. El catálogo fino no da ventaja de tiempo, y la anomalía desaparece al quitar las réplicas.

**Qué significa para la pregunta de fondo**
- **Aprender de lo improbable:** con los datos actuales, no. Hay solo 5 semanas con M≥7 en el test chileno. Juntar los M≥7 del mundo hizo que la red pronosticara de más: unas 13 semanas esperadas frente a 5 observadas. Perdió frente a la ley de Gutenberg-Richter.
- **Variables que nadie consideró:** se dejó buscar a redes neuronales, autoencoders, modelos globales, millones de microsismos y señal sísmica continua. Todos terminan reaprendiendo el agrupamiento que el ETAS ya captura, y no apareció ningún precursor.
- **Límite de los nulos:** con 72 eventos, los nulos solo descartan efectos mayores a unos 0,01–0,05 nats/sismo, según el experimento. Efectos de ese tamaño servirían poco para dar alertas.

**Lecciones de método**
- Con datos que cubren pocos años, el test por defecto del harness exagera la significancia. Hay que usar uno restringido a esa ventana.
- El remuestreo por año no sirve como evidencia para comparar modelos.
- El test oficial castiga a los modelos entrenados afuera. Para evaluarlos de forma justa hace falta un test preregistrado del modelo usado solo.
- Con magnitud mínima 4,7 el ETAS queda casi crítico (factor de ramificación 0,98). Hay que desconfiar de las señales que desaparecen con 5,0.

**Qué haría falta para avanzar de verdad**
- **Datos:** un catálogo de Chile homogéneo y con versiones fechadas, relocalizado con deep learning (completitud ML ~2 desde 2007). Señal sísmica continua de toda la red, con muchos controles: unas 330 mil estaciones-día, ~2 TB y ~500 horas de CPU. GNSS de alta tasa con detección de sismos lentos, e InSAR. Catálogos globales homogéneos.
- **Cómputo:** GPU para un ETAS espacio-temporal 3D y para redes espaciales, y simulaciones ETAS masivas para calibrar los tests.
- **Test prospectivo:** congelar hoy, con fecha y huella digital (hash), tres candidatos: la red global usada sola frente al ETAS-logit, la señal Anderson-Darling con magnitud mínima 5,0 y el GBM global. Después, evaluarlos solo con datos futuros, al estilo CSEP. Para detectar 0,02 nats/sismo con α = 0,0083 se necesitan varios cientos de M≥6. Chile da unos 5–6 al año, así que eso tomaría décadas, o algunos años con un protocolo global que junte todas las subducciones. Los M≥7 deben tratarse como un problema de calibrar la cola de la distribución, no de clasificación.
