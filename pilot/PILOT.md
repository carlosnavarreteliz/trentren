# Piloto Mc-bajo — Coquimbo, 7 días (alcance cerrado para hackatón)

## Pregunta (una sola)
¿Un catálogo denso (Mc ~2,5) construido por picking con IA sobre formas de onda
reales de la red CSN ve los microsismos que USGS no ve, en la ventana donde
después hubo un M6,5? Si no los ve, se cierra la línea. Si los ve, se mide si
cambian las features de escalamiento.

## Alcance (no negociable, esto es TODO el piloto)
- 1 segmento: Coquimbo-Illapel (lat -32,2..-29,0, lon -73..-69,5).
- 1 ventana: 2019-11-01 a 2019-11-08 (7 días; M6,5 el 2019-11-04 en la caja).
- 4 estaciones CSN (red C1 vía FDSN-EarthScope): CO02, CO03, CO05, CO06.
- 1 modelo: EQTransformer pretrained (SeisBench, pesos "original"), sin entrenar nada.
- Asociación simple por agrupar picks P entre estaciones (ventana 10 s). Sin magnitudes
  nuevas: la verdad de comparación es Riedel (ML) y USGS.
- Runtime objetivo: < 45 min en un T4 de Colab. Descarga estimada ~260 MB.

## Verdad ya medida localmente (sin GPU, hoy)
En esa ventana/caja: Riedel ML>=2,5 = 178 eventos; USGS Mw>=2,5 = 6; USGS >=4,7 = 2.
=> Hay ~30x más sismicidad pequeña que la que usa el modelo vigente. El piloto
de picking debe recuperar una fracción sustancial de esos 178 para ser creíble.

## Métrica de éxito (definida antes de correr)
1. Picks->eventos asociados >= 60 (de 178 Riedel ML>=2,5) con >=70% de match
   temporal-espacial contra Riedel (|dt|<=10 s tras asociar, o match a evento USGS).
2. Reportar falsos: eventos asociados sin match Riedel/USGS.
3. Entregable demo: mini-catálogo CSV + figura antes/después (USGS 6 vs denso ~178)
   y las features de escalamiento del trigger M6,5 calculadas con ambos catálogos.
Si (1) no se cumple, el piloto se reporta como nulo y NO se escala.

## Qué NO es este piloto
No predice sismos, no entrena modelos, no usa el test 2024+, no toca el dashboard
salvo la figura final. Escalarlo (2022-2026, todo Chile) es otro proyecto.

## Demo hackatón (3 min)
1. Problema: el modelo ve 6 eventos donde hay 178 (figura antes/después).
2. Live: notebook Colab GPU baja waveforms CSN y detecta en minutos.
3. Cierre honesto: detectar más != predecir; el piloto mide si la señal existe
   donde el catálogo oficial es ciego (rondas previas: 26 experimentos nulos
   con catálogos pobres; esta es la primera prueba con el canal correcto).

## Cómo correr
Abrir `pilot/colab_pilot.ipynb` en Colab (Runtime > GPU T4) y ejecutar todo.
Yo (el asistente) no puedo entrar a tu cuenta de Google: el notebook queda
listo para un clic; la primera celda verifica GPU y falla rápido si no hay.
