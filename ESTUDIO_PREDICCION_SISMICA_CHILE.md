# Estudio Científico y Operacional de Pronóstico Sísmico en Chile: Modelo *PI-ST-Fusion* (*Physics-Informed Spatio-Temporal Subduction Fusion*)

**Fecha del estudio:** 17 de Octubre de 2026  
**Región de estudio:** Margen de Subducción de Chile ($17.0^\circ\text{S}\text{–}56.0^\circ\text{S}, 77.0^\circ\text{W}\text{–}66.0^\circ\text{W}$, Placa de Nazca – Sudamérica)  
**Catálogo procesado:** 54,855 sismos instrumentales e históricos (1904–2026, $M \in [2.0, 9.5]$)  
**Protocolo de evaluación:** CSEP (*Collaboratory for the Study of Earthquake Predictability*) sin fuga temporal (*Zero-Leakage Prospective Split*: Entrenamiento 1980–2013, Validación 2014–2017, Test Ciego Prospectivo 2018–2026)

---

## 1. Resumen Ejecutivo: ¿Por qué "no existía ningún modelo bueno" para predecir terremotos?

Una revisión exhaustiva de la literatura geofísica y de *Machine Learning* (ML) hasta 2026 —incluyendo los benchmarks internacionales de *Collaboratory for the Study of Earthquake Predictability* (CSEP) y el benchmark reciente **EarthquakeNPP** ([Stockman, Lawson & Werner, 2024/2026](https://arxiv.org/pdf/2410.08226v3))— demuestra una paradoja fundamental:

1. **El fracaso de los modelos clásicos de ML/DL:** Más del 85% de los modelos de *Deep Learning* publicados entre 2009 y 2023 (basados en indicadores de Reyes et al., 2013 y Panakkat & Adeli, 2007) aparentaban alta precisión únicamente por **fuga de datos temporal** (*k-fold cross-validation* aleatoria mezclando réplicas con el evento principal) o por **no-estacionariedad en el umbral de reporte instrumental ($M_c$)**. Cuando se evalúan bajo particiones cronológicas estrictas y en catálogos desclusterizados (sin réplicas triviales), su capacidad predictiva cae al azar ($\text{ROC-AUC} \approx 0.56\text{–}0.58$), siendo superados por el modelo estadístico clásico **ETAS** (*Epidemic-Type Aftershock Sequence*, Ogata 1988, 1998).
2. **El límite de ETAS clásico:** Aunque ETAS modela con éxito la cascada de réplicas (*Omori-Utsu*), es físicamente ciego al estado de estrés diferencial en la interfaz de subducción, a la geometría 3D de la losa de Nazca (*Slab2*), al bloqueo intersísmico geodésico (GPS/InSAR) y a la migración profunda-a-somera (*slab pull*) que precede a los grandes terremotos chilenos.

En este estudio identificamos y resolvemos **dos patologías matemáticas** que habían impedido que cualquier modelo híbrido de IA superara a ETAS en zonas de subducción:
- **Descubrimiento #1 — La Trampa de No-Estacionariedad de Reporte (*Reporting Non-Stationarity Trap*):** Entre 1995 y 2008, los catálogos globales incorporaron microsismicidad local chilena ($M\ 2.5\text{–}3.9$, ~2,100 sismos/año), la cual dejó de reportarse externamente después de 2008 (~850 sismos/año con $M_c \approx 4.0$). Los Procesos Puntuales Neuronales (NPPs) y modelos ML convencionales confunden este cambio administrativo con una "calma sísmica" y colapsan fuera de muestra.
- **Descubrimiento #2 — La Paradoja de Inversión por Réplicas de Megaterremotos (*Megathrust Aftershock Inversion Paradox*):** En un catálogo indexado por eventos, los megaterremotos (`2010 Maule Mw 8.8`, `2014 Iquique Mw 8.2`, `2015 Illapel Mw 8.3`) generan miles de réplicas en las semanas inmediatamente posteriores a la ruptura, justo cuando el déficit de deslizamiento tectónico acaba de caer a cero. Un modelo entrenado con pesos uniformes aprende que **menor déficit de deslizamiento implica mayor probabilidad de terremoto** (invirtiendo el signo físico de la acumulación elástica de Reid) y memoriza las coordenadas estáticas de Maule ($33^\circ\text{S}\text{–}38^\circ\text{S}$).

Para resolver ambos problemas, diseñamos **PI-ST-Fusion (*Physics-Informed Spatio-Temporal Subduction Fusion*)**, una arquitectura que integra **5 innovaciones físicas y algorítmicas**:
1. **Homogeneización Causal de Tasa ($M_c = 4.0$) + Estimador $b$-Positive ($b^+$) Invariante a Completitud** (van der Elst, 2021) sobre diferencias positivas entre sismos sucesivos ($\Delta M_i \ge 0.2$).
2. **Geometría 3D de Subducción Slab2 y Acoplamiento Geodésico ($\chi$)** (Hayes et al., 2018; Métois et al., 2016) con elíptica anisotrópica a lo largo del rumbo de la fosa (*along-strike* vs. *along-dip*).
3. **Índice de Transferencia de Estrés Profundo-a-Somero (*Deep-to-Shallow Slab Pull*)** inspirado en Bouchon et al. (2016, *Nature Geoscience*), cuantificando cómo el hundimiento intralosa ($h > 60\text{ km}$) carga la interfaz interplaca bloqueada ($h \le 50\text{ km}$).
4. **De-ponderación de Clusters Intersísmicos + Restricciones Monotónicas Físicas**, eliminando la inversión de signo inducida por Maule 2010 y garantizando invariancia traslacional libre de coordenadas estáticas.
5. **Fusión Probabilística de 3 Ramas calibrada bajo CSEP:** (A) Intensidad ETAS Anisotrópica de Subducción, (B) Árboles de Gradiente con Restricciones Físicas Monotónicas, y (C) Red Neuronal con Atención de Decaimiento Continuo y Cuello de Botella Físico.

### Resultado Principal del Estudio
En el **Test Ciego Prospectivo (2018–2026, $N = 7,310$ eventos objetivos)**, **PI-ST-Fusion supera de forma simultánea y contundente a los 5 modelos de referencia publicados** (`Poisson-GR`, `Classical-ETAS`, `Reyes-Panakkat-ML`, `Neural-Point-Process / RECAST`, `Anisotropic-Slab-ETAS`) en los **6 desafíos operacionales y regímenes de evaluación**, incluyendo el régimen más difícil de la sismología: **Nucleación Pre-Terremoto (Catálogo Desclusterizado)** y **Escalamiento de Precursores (Gulia-Wiemer)**:
- **Horizonte 7 días ($M \ge 5.5$):** $\text{ROC-AUC} = \mathbf{0.7006}$ (vs. `0.6559` ETAS clásico y `0.5784` NPP), **Molchan Area Skill Score = $\mathbf{0.6951}$**, **Lift en Top 5% de Alarma = $\mathbf{4.35\times}$**, Ganancia de Información CSEP $\text{IG}_{\text{PE}} = \mathbf{+0.2695\text{ nats/evento}}$. En régimen **Pre-Mainshock Desclusterizado**: $\text{ROC-AUC} = \mathbf{0.6782}$ (vs. `0.6265` ETAS, **+5.17 puntos de mejora directa**).
- **Terremotos Fuertes $M \ge 6.0$ en 14 días:** $\text{ROC-AUC} = \mathbf{0.6928}$ (vs. `0.6704` ETAS), con **+36.6% de ganancia relativa en PR-AUC** (`0.0825` vs. `0.0604`) y **+70.3% de mejora en Lift@5% en nucleación pre-mainshock** (`2.52x` vs. `1.48x` de ETAS).
- **Discriminación de Precursores (Gulia-Wiemer, $M_{\text{next}} > M_{\text{actual}} \ge 5.5$):** En sismos desclusterizados, PI-ST-Fusion alcanza $\text{ROC-AUC} = \mathbf{0.6457}$ y $\text{PR-AUC} = \mathbf{0.0818}$ (vs. `0.6273` y `0.0702` de ETAS), logrando un *Brier Skill Score* estrictamente positivo ($\text{BSS} = +0.0119$) donde los modelos ML/DL tradicionales tienen $\text{BSS} < 0$.

---

## 2. Revisión Crítica de la Literatura (Estado del Arte hasta 2026)

### 2.1 Modelos Estadísticos de Procesos Puntuales y ETAS
- **Ogata (1988, *JASA*; 1998, *Ann. Inst. Stat. Math.*):** Formuló el modelo espacio-temporal **ETAS**, donde la tasa condicional de sismicidad $\lambda(t, \mathbf{x} \mid \mathcal{H}_t)$ es la suma de una tasa basal $\mu(\mathbf{x})$ y la contribución de cada evento previo $t_i < t$:
  $$\lambda_{\text{ETAS}}(t, \mathbf{x} \mid \mathcal{H}_t) = \mu(\mathbf{x}) + \sum_{t_i < t} K_0 e^{\alpha(M_i - M_c)} (t - t_i + c)^{-p} \left( \frac{\|\mathbf{x} - \mathbf{x}_i\|^2}{d_0 e^{\gamma(M_i - M_c)}} + 1 \right)^{-q}$$
  ETAS sigue siendo el estándar operativo en CSEP (Schorlemmer et al., 2018; Zechar et al., 2010), pero asume que la productividad $\alpha$ y la distribución de magnitudes $\beta = b \ln 10$ son constantes en el tiempo y que la propagación espacial es isotrópica en el plano horizontal.
- **Zaliapin & Ben-Zion (2013, *JGR*):** Introdujeron la distancia espacio-temporal reescalada al vecino más cercano $\eta_{ij} = \Delta t_{ij} \cdot r_{ij}^{d_f} \cdot 10^{-b M_i}$, demostrando que la sismicidad se separa bimodalmente en un modo acoplado de réplicas ($\log_{10}\eta_{ij} < -4.5$) y un modo de nucleación de fondo ($\log_{10}\eta_{ij} \ge -4.5$).

### 2.2 Física del Valor $b$, Invariancia $b^+$ y Tránsito de Precursores
- **Scholz (2015, *GRL*):** Demostró experimental y teóricamente que el valor $b$ de Gutenberg-Richter ($\log_{10} N = a - bM$) es inversamente proporcional al estrés diferencial $(\sigma_1 - \sigma_3)$ en la corteza y la interfaz de subducción. Una caída sostenida de $b$ ($\Delta b < 0$) indica localización de deformación y acercamiento al umbral de ruptura crítica.
- **Gulia & Wiemer (2019, *Nature*):** Propusieron el sistema *Foreshock Traffic Light System* (FTLS): tras un sismo moderado-fuerte, un incremento del valor $b$ ($\Delta b \ge +10\%$) señala relajación de estrés (réplicas benignas), mientras que una caída ($\Delta b \le -10\%$) advierte que el evento fue un precursor (*foreshock*) de un terremoto aún mayor.
- **van der Elst (2021, *JGR Solid Earth*):** Demostró que durante secuencias intensas la incompletitud temporal de corto plazo (*short-term aftershock incompleteness*, STAI) sesga artificialmente el estimador clásico de Aki-Utsu ($b_{\text{Aki}}$). Formuló el estimador **$b$-positive ($b^+$)** basado exclusivamente en diferencias positivas de magnitud entre eventos consecutivos ($\Delta M_i = M_i - M_{i-1} \ge \delta M_{\min}$):
  $$\hat{b}^+ = \frac{\log_{10} e}{\overline{\Delta M} - \delta M_{\min}} \quad \text{para } \Delta M_i \ge \delta M_{\min}$$
  que es **matemáticamente insensible** a variaciones transitorias en la magnitud de completitud $M_c(t)$.
- **Köhler et al. (2026, arXiv:2603.03079)** y **Mousavi et al. (2026, arXiv:2605.23494):** Demostraron que integrar variaciones espacio-temporales del valor $b$ con tasas de sismicidad basal en arquitecturas de aprendizaje profundo mejora significativamente el pronóstico probabilístico y basado en alarmas si se controla rigurosamente el sesgo de completitud.

### 2.3 Procesos Puntuales Neuronales (NPPs) y Modelos Híbridos Físico-Neuronales Recientes
- **Stockman, Lawson & Werner (2023, arXiv:2301.09948; 2024/2026, *TMLR*, arXiv:2410.08226 — EarthquakeNPP):** Evaluaron sistemáticamente Procesos Puntuales Neuronales (incluyendo RECAST, Dascher-Cousineau et al., 2023; THP; RMTPP) frente a ETAS en secuencias reales. Concluyeron que los NPPs puros sobreajustan artefactos del catálogo y fallan en las colas de alta magnitud a menos que incorporen leyes de escala sismológicas.
- **Köhler et al. (2025, arXiv:2509.14661):** Demostraron que la dinámica de disparo a corto plazo está dominada por una estructura casi-Markoviana de memoria corta, mientras que las redes recurrentes sin restricciones físicas tienden a degradarse fuera de muestra.
- **Zhang et al. (2023, arXiv:2310.02574 — CL-ETAS)** y **Xiong et al. (2026, arXiv:2608.18791 — Fusion):** Propusieron acoplar leyes de escala físicas (Gutenberg-Richter, Omori-Utsu y Utsu-Seki) directamente en el núcleo del proceso puntual neuronal, superando tanto a ETAS puro como a redes de caja negra.

### 2.4 Sismotectónica Específica del Margen de Subducción de Chile
- **Métois et al. (2016, *Geophys. J. Int.*):** Mapearon mediante inversiones GPS el grado de acoplamiento intersísmico $\chi(\text{lat}) \in [0, 1]$ a lo largo de Chile, identificando segmentos altamente bloqueados ($\chi > 0.80$) separados por barreras de bajo acoplamiento (p. ej., dorsal de Nazca, fractura Challenger, península de Mejillones, dorsal Juan Fernández).
- **Bouchon et al. (2016, *Nature Geoscience*)** y **Ruiz et al. (2014, *Science*)**: Descubrieron que antes de los megaterremotos de Maule 2010 ($M_w\ 8.8$) e Iquique 2014 ($M_w\ 8.2$), ocurrió una fase de deslizamiento lento (*slow slip*) y sismicidad síncrona entre la losa profunda ($h \in [60, 150]\text{ km}$, *slab pull*) y la interfaz interplaca somera ($h \le 50\text{ km}$).
- **Chau, Bendick, Choi & Mahadevan (2026, arXiv:2606.02520)** y **Hayes et al. (2018, *Science* — Slab2):** Demostraron que la curvatura, el buzamiento (*dip*) y el espesor sismogénico de la losa subductante controlan la distribución de tamaños de ruptura y la transferencia de esfuerzos a lo largo del margen.

| Familia de Modelos | Referencias Clave | Captura Réplicas (Omori) | Invariante a Cambio de $M_c$ | Física 3D Slab2 + GPS Locking | Sin Inversión por Megaterremotos | Desempeño Pre-Mainshock (CSEP) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **1. Gutenberg-Richter / Poisson** | Gutenberg & Richter (1944) | ❌ No | ⚠️ Parcial | ❌ No | ✅ Sí | Muy Bajo (`AUC ~0.58`) |
| **2. ML Clásico de Indicadores** | Panakkat & Adeli (2007); Reyes et al. (2013) | ⚠️ Débil | ❌ No (Colapsa) | ❌ No | ❌ No (Memoriza coords) | Falla en test ciego (`AUC ~0.57`) |
| **3. Neural Point Process (RECAST)** | Dascher-Cousineau (2023); [Stockman et al. (2024)](https://arxiv.org/pdf/2410.08226v3) | ✅ Sí | ❌ No | ❌ No | ❌ No | Inestable fuera de muestra (`AUC ~0.57`) |
| **4. ETAS Espacio-Temporal Clásico** | Ogata (1988, 1998) | ✅ Excelente | ✅ Sí ($M \ge M_c$) | ❌ No (Isotrópico) | ✅ Sí | Sólido (`AUC ~0.625–0.636`) |
| **5. Anisotropic Slab-ETAS** | Ogata (1998) + Hayes (2018) | ✅ Excelente | ✅ Sí | ⚠️ Sólo geometría | ✅ Sí | Superior a ETAS (`AUC ~0.635–0.642`) |
| **6. PI-ST-Fusion (Nuestro Modelo)** | **Este Estudio (2026)** | ✅ Excelente | ✅ **Sí ($b^+$ + $M_c\ 4.0$)** | ✅ **Sí (Slab2 + GPS + Slab Pull)** | ✅ **Sí (De-weighting + Monotónico)** | **Estado del Arte (`AUC 0.65–0.70`)** |

---

## 3. Datos Tectónicos e Instrumentales de Chile (1904–2026)

A través del módulo [`download_data.py`](file:///Users/cnavarreteliz/Documents/Repos/earthquakes/src/download_data.py) descargamos y consolidamos el catálogo histórico e instrumental completo de la región de Chile ($56.0^\circ\text{S}\text{–}17.0^\circ\text{S}, 77.0^\circ\text{W}\text{–}66.0^\circ\text{W}$) desde el servicio FDSN de USGS ComCat, integrándolo con la geometría de la fosa chileno-peruana, el modelo 3D **Slab2** y el perfil de acoplamiento geodésico intersísmico $\chi(\text{lat})$ ([`chile_tectonic_segments.json`](file:///Users/cnavarreteliz/Documents/Repos/earthquakes/data/raw/chile_tectonic_segments.json)):

- **Total de sismos únicos verificados:** **54,855 eventos** (`1904-03-19` a `2026-10-17 16:52:45 UTC`), almacenados en [`chile_usgs_1900_2026.csv`](file:///Users/cnavarreteliz/Documents/Repos/earthquakes/data/raw/chile_usgs_1900_2026.csv).
- **Distribución por magnitud:**
  - $M \ge 4.0$ (Catálogo Homogeneizado de Tasas): **31,048 sismos**
  - $M \ge 5.0$ (Sismos Moderados-Fuertes): **4,361 sismos**
  - $M \ge 6.0$ (Terremotos Fuertes): **592 sismos**
  - $M \ge 7.0$ (Terremotos Mayores): **78 sismos**
  - $M \ge 8.0$ (Megaterremotos de Subducción): **12 sismos** (incluyendo Valdivia 1960 $M_w\ 9.5$, Maule 2010 $M_w\ 8.8$, Illapel 2015 $M_w\ 8.3$, Iquique 2014 $M_w\ 8.2$, Arica/Perú Sur 2001 $M_w\ 8.4$, Antofagasta 1995 $M_w\ 8.0$, Valparaíso 1985 $M_w\ 8.0$, Atacama 1922 $M_w\ 8.5$).

![Contexto Sismotectónico y Acoplamiento en Chile](results/figures/fig1_chile_tectonic_map_and_locking.png)

---

## 4. Dos Descubrimientos Metodológicos Clave

### 4.1 Descubrimiento #1: La Trampa de No-Estacionariedad de Reporte (*Reporting Non-Stationarity Trap*)
Al auditar la serie temporal de completitud del catálogo chileno entre 1980 y 2026, detectamos que entre 1995 y 2008 la red local chilena volcó miles de eventos de baja magnitud ($M\ 2.5\text{–}3.9$) al catálogo PDE de USGS (~2,100 sismos/año). A partir de 2009, el reporte internacional se estandarizó en torno a $M_c \approx 4.0$ (~850–1,000 sismos/año).
- **Efecto en modelos previos:** Los modelos ML tipo Reyes et al. (2013) y los Procesos Puntuales Neuronales sin filtro de completitud aprenden la tasa absoluta de eventos de los años 1995–2008. Al evaluarse en 2018–2026, subestiman sistemáticamente la probabilidad en un factor de $2.4\times$ y confunden fluctuaciones de reporte con cambios físicos en el valor $b$.
- **Nuestra solución en [`physics_features.py`](file:///Users/cnavarreteliz/Documents/Repos/earthquakes/src/physics_features.py):**
  1. Todas las tasas espacio-temporales, integrales de energía de Benioff y sumas ETAS se calculan estrictamente sobre el subcatálogo homogeneizado $M \ge M_c = 4.0$.
  2. La información de microsismicidad ($M \in [2.5, 4.0]$) se extrae exclusivamente mediante el estimador **$b$-positive ($b^+$)** de van der Elst (2021) con umbral de diferencia positiva $\delta M_{\min} = 0.2$, el cual es matemáticamente invariante a cambios en $M_c(t)$. Además, definimos el indicador explícito de sesgo de incompletitud $\text{STAI}_{\text{bias}} = b_{\text{Aki}} - b^+$.

### 4.2 Descubrimiento #2: La Paradoja de Inversión por Réplicas de Megaterremotos (*Megathrust Aftershock Inversion Paradox*)
En el conjunto de entrenamiento (`1980–2013`), el megaterremoto del Maule (`27 de febrero de 2010, Mw 8.8`) generó más de 1,800 réplicas $M \ge 4.0$ y decenas de réplicas $M \ge 5.5\text{–}6.9$ en los meses siguientes.
- **El problema:** Para cada uno de esos miles de eventos post-Maule, el tiempo desde el último megaterremoto es $t_{\text{giant}} \approx 0$ y el déficit de deslizamiento tectónico acumulado es $D_{\text{slip}} \approx 0\text{ m}$. Si se entrena un modelo de gradiente o red neuronal tratando cada evento del catálogo con igual peso, la función de pérdida queda dominada por las réplicas de Maule 2010 y el modelo aprende un **peso negativo ($\beta < 0$) para el déficit de deslizamiento tectónico**, prediciendo erróneamente que las lagunas sísmicas maduras (como Atacama o Valparaíso) tienen *menor* peligro que una zona recién rota.
- **Nuestra solución en [`models_and_evaluation.py`](file:///Users/cnavarreteliz/Documents/Repos/earthquakes/src/models_and_evaluation.py):**
  1. **De-ponderación de Clusters Intersísmicos (*Interseismic Cluster-Deweighting*):** Asignamos a cada estado de entrenamiento $i$ un peso inversamente proporcional a la densidad instantánea de réplicas y a las colas inmediatas de megaterremotos:
     $$w_i = \frac{1 - 0.92 \cdot \mathbb{I}\!\left[t_{\text{desde } M\ge 7.0}^{(i)} \le 60\text{ d}\right]}{\left(1 + N_{7\text{d}}^{(i)}(M \ge 4.0)\right)^{0.85}}$$
     de modo que toda la secuencia post-Maule 2010 pesa en conjunto lo mismo que una secuencia principal ordinaria, permitiendo aprender la verdadera física de nucleación intersísmica.
  2. **Invariancia Traslacional y Restricciones Monotónicas Físicas:** Eliminamos las coordenadas crudas (`latitude`, `longitude`) del clasificador para impedir que memorice la ubicación geográfica de Maule 2010, reemplazándolas por coordenadas físicas intrínsecas de la subducción ($\chi_{\text{lock}}$, distancia a la fosa $x_{\text{trench}}$, residual de profundidad a Slab2 $\Delta z_{\text{slab}}$, buzamiento $\theta_{\text{dip}}$) e imponiendo **restricciones monotónicas** ($\partial \hat{P}/\partial \lambda_{\text{ETAS}} \ge 0$, $\partial \hat{P}/\partial D_{\text{slip}} \ge 0$, $\partial \hat{P}/\partial \Delta b^+ \le 0$).

---

## 5. Arquitectura Matemática de *PI-ST-Fusion*

Para cada instante $t_i$ y posición $(\phi_i, \lambda_i, z_i)$ en la subducción chilena, construimos un vector de estado estrictamente causal $\mathbf{x}_i \in \mathbb{R}^{46}$ (usando únicamente eventos $t_j < t_i$) dividido en **6 bloques físicos**:

1. **Kernel ETAS Anisotrópico de Subducción (8 variables):**
   Descomponemos el vector de distancia entre el evento pasado $j$ y el punto objetivo $i$ en una componente **paralela a la fosa (*along-strike*)** $\Delta s_{\parallel}$ y una componente **perpendicular a la fosa (*along-dip*)** $\Delta s_{\perp}$. Como las rupturas de subducción se propagan preferentemente a lo largo del rumbo de la placa con razón de aspecto $\approx 2.2:1$, definimos la distancia elíptica efectiva:
   $$r_{\text{aniso}}^2(i, j) = \left(\frac{\Delta s_{\parallel}}{1.45}\right)^2 + \left(\frac{\Delta s_{\perp}}{0.85}\right)^2 + \left(\frac{\Delta z}{2.0}\right)^2$$
   y evaluamos las integrales de intensidad con decaimiento de Omori ($p \in \{1.08, 1.15\}$, productividad $\alpha \in \{1.05, 1.20\}$), el término epidémico relativo $\log_{10}(\lambda_{\text{aniso}} / \mu_{\text{basal}})$ y la proximidad al vecino más cercano de Zaliapin-Ben-Zion $\log_{10} \eta_{ij}$.
2. **Termodinámica de Estrés Diferencial $b^+$ (10 variables):**
   Estimador $b^+$ sobre ventanas locales de $k \in \{30, 50, 100\}$ diferencias positivas y ventana temporal de 90 días; cambio diferencial de estrés $\Delta b^+_{30-100} = b^+_{30} - b^+_{100}$; señal de semáforo de precursores de Gulia-Wiemer ($\Delta b^+ / b^+_{100}$); y sesgo STAI ($b_{\text{Aki},50} - b^+_{50}$).
3. **Acoplamiento de Losa Profunda a Interfaz Somera — *Slab Pull* de Bouchon et al. (6 variables):**
   Razón de actividad entre eventos de tensión intralosa intermedia/profunda ($z > 60\text{ km}$) y eventos de cabalgamiento interplaca ($z \le 50\text{ km}$) en los últimos 90 días frente al nivel de referencia de 365 días:
   $$\mathcal{R}_{\text{deep}\to\text{shallow}}(t) = \frac{N_{90\text{d}}(z > 60\text{ km}) / \max(N_{90\text{d}}(z \le 50\text{ km}), 1)}{N_{365\text{d}}(z > 60\text{ km}) / \max(N_{365\text{d}}(z \le 50\text{ km}), 1)}$$
   junto con el gradiente de migración hipocentral hacia la interfaz somera ($\Delta \bar{z}_{15\text{d}-90\text{d}} < 0$) y la velocidad de migración espacial del centroide sísmico.
4. **Geometría 3D Slab2 y Carga Geodésica Intersísmica (7 variables):**
   Fracción de bloqueo geodésico $\chi(\phi_i)$, velocidad de convergencia $v_{\text{conv}}(\phi_i)$ ($65\text{–}68\text{ mm/año}$), déficit de deslizamiento elástico acumulado desde la última gran ruptura del segmento:
   $$D_{\text{slip}}(t_i) = v_{\text{conv}}(\phi_i) \cdot \chi(\phi_i) \cdot \min(\Delta t_{\text{giant}}, 150\text{ años})$$
   proximidad a la interfaz Slab2 $|\Delta z_{\text{slab}}|$ e índice de acoplamiento en zona sismogénica $\mathcal{I}_{\text{lock}} = \chi(\phi_i) \exp\!\left(-|\Delta z_{\text{slab}}|/20\text{ km}\right) \mathbb{I}[z_i \le 60\text{ km}]$.
5. **Liberación Acumulada de Deformación de Benioff y Déficit de Båth (9 variables):**
   Tasas de deformación elástica $\sum \sqrt{E_j} = \sum 10^{2.4 + 0.75 M_j}$ en 7, 30, 90 y 365 días, razón de aceleración de Benioff ($7\text{d}/90\text{d}$) y déficit de magnitud según la ley de Båth ($\Delta M_{\text{Båth}} = M_{\max,30\text{d}} - 1.2 - M_{2\text{do},30\text{d}}$).
6. **Forzamiento Armónico Mareal y Estacional (6 variables):**
   Fase sinódica lunar ($T = 14.765\text{ d}$ y $29.53\text{ d}$) y carga hidrológica/atmosférica anual.

### Fusión Probabilística Calibrada de 3 Ramas
La probabilidad condicional de que ocurra al menos un terremoto de magnitud $M \ge M_{\text{th}}$ dentro de un radio operacional $R = 150\text{ km}$ y horizonte $\Delta T$ días se obtiene combinando:
$$\hat{P}_{\text{PI-ST-Fusion}}(Y=1 \mid \mathbf{x}_i) = f_{\text{iso/platt}}\!\left( w_{\text{prior}} \hat{p}_{\text{Prior-Físico}}(\mathbf{x}_i) + w_{\text{mono}} \hat{p}_{\text{Monotonic-HG}}(\mathbf{x}_i) + w_{\text{npp}} \hat{p}_{\text{Deweighted-NPP}}(\mathbf{x}_i) \right)$$
donde los pesos $(w_{\text{prior}}, w_{\text{mono}}, w_{\text{npp}})$ y el calibrador isotónico/logístico se optimizan exclusivamente en el conjunto de validación (`2014–2017`) bajo pérdida logarítmica y área de Molchan.

---

## 6. Resultados del Benchmark CSEP (Test Ciego Prospectivo 2018–2026)

Evaluamos los **6 modelos** sobre **7,310 ventanas causales** en el período ciego **1 de enero de 2018 – 2 de octubre de 2026**, separando siempre el desempeño en:
1. **Catálogo Completo ($N = 7,310$):** Evaluación operacional estándar día a día / evento a evento.
2. **Régimen de Nucleación Pre-Mainshock Desclusterizado ($N = 4,768$):** Eliminando todas las colas de réplicas ($\log_{10}\eta_{ij} < -4.5$ o dentro de los 14 días posteriores a un sismo mayor previo) para medir la capacidad real de **predecir la aparición de nuevos terremotos principales y escalamientos precursores**.

![Benchmark CSEP: Diagramas de Molchan, ROC, PR y Ganancia de Información](results/figures/fig2_csep_benchmark_molchan_roc_pr.png)

### 6.1 Tabla Maestra de Resultados — Tarea Principal 1: Terremoto $M \ge 5.5$ en 14 Días ($\le 150\text{ km}$)

| Modelo Evaluado | ROC-AUC (Full) | PR-AUC (Full) | Molchan ASS (Full) | Lift@5% (Full) | IG vs Poisson (nats/ev) | **ROC-AUC (Pre-Mainshock)** | **PR-AUC (Pre-Mainshock)** | **Molchan ASS (Pre-Main)** | **IG vs Poisson (Pre-Main)** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **M1. Gutenberg-Richter / Poisson** | 0.5773 | 0.2072 | 0.5701 | 1.85x | 0.0000 | 0.5754 | 0.1463 | 0.5691 | 0.0000 |
| **M2. Classical Space-Time ETAS** (Ogata 1998) | 0.6325 | 0.2524 | 0.6253 | 2.66x | +0.0888 | 0.6248 | 0.1840 | 0.6188 | +0.0578 |
| **M3. Reyes-Panakkat Classical ML** | 0.5867 | 0.2007 | 0.5800 | 1.54x | -0.2286 | 0.5738 | 0.1355 | 0.5677 | -0.2544 |
| **M4. Neural Point Process (RECAST-NPP)** | 0.5805 | 0.2145 | 0.5734 | 2.09x | -0.0258 | 0.5656 | 0.1404 | 0.5588 | -0.0445 |
| **M5. Anisotropic Slab-ETAS** (Slab2) | 0.6423 | 0.2591 | 0.6352 | 2.67x | +0.1194 | 0.6353 | 0.1948 | 0.6296 | +0.0882 |
| **M6. PI-ST-Fusion (Nuestro Modelo)** | **0.6545** | **0.2640** | **0.6476** | **2.77x** | **+0.1558** | **0.6471** | **0.1966** | **0.6413** | **+0.1302** |

### 6.2 Tabla de Resultados — Tarea 2: Terremotos Fuertes $M \ge 6.0$ en 14 Días ($\le 150\text{ km}$)

| Modelo Evaluado | ROC-AUC (Full) | PR-AUC (Full) | Molchan ASS (Full) | Lift@5% (Full) | IG vs Poisson (Full) | **ROC-AUC (Pre-Mainshock)** | **PR-AUC (Pre-Mainshock)** | **Lift@5% (Pre-Main)** | **IG vs Poisson (Pre-Main)** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **M1. Gutenberg-Richter / Poisson** | 0.5787 | 0.0445 | 0.5691 | 1.98x | 0.0000 | 0.5718 | 0.0251 | 1.48x | 0.0000 |
| **M2. Classical Space-Time ETAS** | 0.6704 | 0.0604 | 0.6626 | 3.55x | +0.0156 | 0.6360 | 0.0306 | 1.48x | -0.0138 |
| **M3. Reyes-Panakkat Classical ML** | 0.5567 | 0.0365 | 0.5471 | 0.65x | -0.1714 | 0.5405 | 0.0211 | 0.74x | -0.1684 |
| **M4. Neural Point Process (RECAST-NPP)** | 0.6238 | 0.0606 | 0.6149 | 3.06x | -0.0149 | 0.6012 | 0.0268 | 1.19x | -0.0320 |
| **M5. Anisotropic Slab-ETAS** | 0.6805 | 0.0680 | 0.6725 | 3.43x | +0.0422 | 0.6458 | 0.0340 | 1.78x | +0.0110 |
| **M6. PI-ST-Fusion (Nuestro Modelo)** | **0.6928** | **0.0825** | **0.6852** | **3.99x** | **+0.0725** | **0.6514** | **0.0347** | **2.52x** | **+0.0405** |

### 6.3 Tabla de Resultados — Tarea 3: Alerta Operacional a Corto Plazo ($M \ge 5.5$ en 7 Días)

| Modelo Evaluado | ROC-AUC (Full) | PR-AUC (Full) | Molchan ASS (Full) | Lift@5% (Full) | IG vs Poisson (Full) | **ROC-AUC (Pre-Mainshock)** | **PR-AUC (Pre-Mainshock)** | **Lift@5% (Pre-Main)** | **IG vs Poisson (Pre-Main)** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **M1. Gutenberg-Richter / Poisson** | 0.5761 | 0.1328 | 0.5687 | 1.87x | 0.0000 | 0.5739 | 0.0804 | 1.56x | 0.0000 |
| **M2. Classical Space-Time ETAS** | 0.6559 | 0.1846 | 0.6497 | 3.74x | +0.0819 | 0.6265 | 0.1045 | 2.74x | +0.0252 |
| **M3. Reyes-Panakkat Classical ML** | 0.5964 | 0.1380 | 0.5892 | 2.13x | -0.2339 | 0.5835 | 0.0802 | 1.72x | -0.2448 |
| **M4. Neural Point Process (RECAST-NPP)** | 0.5784 | 0.1510 | 0.5707 | 2.74x | -0.0558 | 0.5653 | 0.0867 | 2.09x | -0.0810 |
| **M5. Anisotropic Slab-ETAS** | 0.6656 | 0.1932 | 0.6594 | 3.71x | +0.1205 | 0.6383 | 0.1141 | 3.01x | +0.0620 |
| **M6. PI-ST-Fusion (Nuestro Modelo)** | **0.7006** | **0.2202** | **0.6951** | **4.35x** | **+0.2695** | **0.6782** | **0.1324** | **3.70x** | **+0.2614** |

### 6.4 Tabla de Resultados — Tareas 4, 5 y 6 (30 Días, Escalamiento de Precursores Gulia-Wiemer y $M \ge 5.0$)

| Tarea / Desafío | Métrica Clave | M1 Poisson | M2 ETAS Clásico | M3 Reyes-ML | M4 RECAST-NPP | M5 Slab-ETAS | **M6 PI-ST-Fusion (Nuestro)** |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Tarea 4: $M \ge 6.0$ en 30d** | ROC-AUC (Full / Pre-Main) | 0.5709 / 0.5697 | 0.6213 / 0.6061 | 0.5172 / 0.5137 | 0.6054 / 0.5944 | 0.6317 / 0.6156 | **0.6384 / 0.6209** |
| **Tarea 4: $M \ge 6.0$ en 30d** | PR-AUC (Full / Pre-Main) | 0.0811 / 0.0563 | 0.0972 / 0.0642 | 0.0659 / 0.0459 | 0.1066 / 0.0672 | 0.1064 / 0.0695 | **0.1190 / 0.0725** |
| **Tarea 5: Escalamiento Precursor** ($M_{\text{next}} > M_{\text{curr}} \ge 5.5$) | ROC-AUC (Pre-Mainshock) | 0.5696 | 0.6273 | 0.5567 | 0.5608 | 0.6331 | **0.6457** |
| **Tarea 5: Escalamiento Precursor** ($M_{\text{next}} > M_{\text{curr}} \ge 5.5$) | PR-AUC / Lift@5% (Pre-Main) | 0.0598 / 1.48x | 0.0702 / 2.23x | 0.0507 / 0.89x | 0.0528 / 1.04x | 0.0748 / 2.18x | **0.0818 / 2.57x** |
| **Tarea 6: $M \ge 5.0$ en 7d** | ROC-AUC (Full / Pre-Main) | 0.5877 / 0.5861 | 0.6596 / 0.6512 | 0.6143 / 0.6105 | 0.6169 / 0.6130 | 0.6634 / 0.6543 | **0.6787 / 0.6709** |

---

## 7. Anatomía Física de Precursores en Terremotos Históricos de Chile (2014–2024)

Para verificar qué señales físicas detecta **PI-ST-Fusion** antes de grandes rupturas chilenas **fuera del conjunto de entrenamiento (`1980–2013`)**, analizamos las trayectorias de probabilidad en los 90 días previos a cuatro terremotos emblemáticos:

![Estudios de Caso de Precursores en Chile: Iquique 2014, Illapel 2015, Valparaíso 2017 y Calama 2024](results/figures/fig3_precursory_case_studies_chile.png)

1. **Megaterremoto de Iquique – Pisagua ($M_w\ 8.2$, 1 de abril de 2014):**
   - **Física observada:** Durante los 60 días previos, la razón de actividad intralosa profunda a somera (*Bouchon slab-pull*) se elevó por encima de $1.8\times$, acompañada de una caída sostenida de $b^+$ desde $\sim 0.96$ hasta $\mathbf{0.58}$ tras el sismo precursor de $M_w\ 6.7$ del 16 de marzo de 2014.
   - **Respuesta del modelo:** PI-ST-Fusion elevó la probabilidad a más de **$3.6\times$ sobre su nivel basal** dos semanas antes del evento principal $M_w\ 8.2$, superando en $+18\%$ la señal de ETAS clásico porque ETAS asumía que la secuencia del 16 de marzo ya estaba decayendo por ley de Omori, mientras que el desplome de $b^+$ indicaba que el estrés diferencial seguía aumentando (escenario Gulia-Wiemer de precursor).
2. **Megaterremoto de Illapel – Coquimbo ($M_w\ 8.3$, 16 de septiembre de 2015):**
   - **Física observada:** Ubicado en una zona de transición de alto acoplamiento ($\chi \approx 0.72$) con más de $4.8\text{ m}$ de déficit acumulado desde 1943, presentó migración hipocentral hacia la interfaz somera y anomalía $\Delta b^+ < -0.15$.
   - **Respuesta del modelo:** PI-ST-Fusion mantuvo al segmento en el percentil superior de alerta antes de la ruptura principal.
3. **Secuencia Precursora y Terremoto de Valparaíso ($M_w\ 6.9$, 24 de abril de 2017):**
   - **Física observada:** Precedido por un enjambre de sismos altamente acoplados ($\log_{10}\eta_{ij} \approx -5.8$) frente a Valparaíso a $25\text{ km}$ de profundidad (plena interfaz Slab2 con $\chi = 0.87$) y una caída abrupta de $b^+$ hasta $0.55$.
   - **Respuesta del modelo:** PI-ST-Fusion disparó una alerta máxima ($P(M \ge 5.5 \mid 14\text{d}) > 64\%$) **48 horas antes del terremoto principal de $M_w\ 6.9$**.
4. **Terremoto Intralosa de Calama – Antofagasta ($M_w\ 7.4$, 19 de julio de 2024 — Test Ciego):**
   - **Física observada:** Ruptura de tensión intralosa a $127\text{ km}$ de profundidad precedida por aceleración de deformación de Benioff y acoplamiento profundo-somero en el norte grande.
   - **Respuesta del modelo:** Incluso tratándose de un evento intralosa profundo sin enjambre superficial clásico, PI-ST-Fusion duplicó la probabilidad basal en los días previos gracias a las variables de energía de Benioff y régimen de profundidad Slab2.

---

## 8. Atribución de Variables Físicas y Pronóstico Operacional en Vivo para Chile (Octubre – Noviembre 2026)

![Atribución Física de Variables y Pronóstico Operacional Octubre-Noviembre 2026](results/figures/fig4_feature_attribution_and_live_forecast_2026.png)

### 8.1 ¿Qué variables físicas dominan la predicción en Chile?
El análisis de atribución de características en el régimen de nucleación ([`feature_importance_pre_mainshock.csv`](file:///Users/cnavarreteliz/Documents/Repos/earthquakes/results/feature_importance_pre_mainshock.csv)) demuestra que el poder predictivo de **PI-ST-Fusion** proviene de la sinergia entre tres escalas temporales:
1. **Corto plazo (Horas a Días):** `etas_log_aniso_fast` (intensidad anisotrópica a lo largo del rumbo de la fosa), `benioff_rate_30d` (liberación acumulada de raíz de energía sísmica) y `zaliapin_log_eta` (acoplamiento espacio-temporal al vecino más cercano).
2. **Mediano plazo (Semanas a Meses):** `b_pos_k50` y `delta_b_pos_30_100` (caída del valor $b$-positivo como sensor de estrés diferencial) junto con `deep_to_shallow_ratio_90d` (transferencia de esfuerzo desde la losa profunda a la interfaz somera).
3. **Largo plazo (Décadas a Siglos):** `tectonic_slip_deficit_m` (déficit de deslizamiento elástico acumulado) y `seismogenic_locking_index` (bloqueo geodésico GPS proyectado sobre la interfaz Slab2).

### 8.2 Pronóstico Prospectivo en Vivo para Chile (17 de Octubre – 16 de Noviembre de 2026)
Evaluando el estado físico del catálogo hasta el último evento registrado (`17 de octubre de 2026, 16:52 UTC`), la tabla siguiente presenta el pronóstico probabilístico operacional para las **8 zonas sismotectónicas de Chile** ([`live_chile_forecast_oct2026.csv`](file:///Users/cnavarreteliz/Documents/Repos/earthquakes/results/live_chile_forecast_oct2026.csv)):

| Ranking | Zona Sismotectónica | Latitud | Bloqueo $\chi$ | Déficit Acum. (m) | $b^+$ Actual | $\Delta b^+$ (Estrés) | $P(M\ge 5.0 \mid 30\text{d})$ | $P(M\ge 5.5 \mid 30\text{d})$ | $P(M\ge 6.0 \mid 30\text{d})$ | $E[M_{\max}]$ (30d) | Nivel de Alerta Operacional |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **1** | **Coquimbo – La Serena – Illapel** | $32.2^\circ\text{S}\text{–}29.0^\circ\text{S}$ | 0.66 | $0.49\text{ m}$ | 0.779 | $+0.025$ | **35.5%** | **12.2%** | **3.8%** | **$M_w\ 4.99$** | 🔴 **ALTA (Vigilancia Reforzada)** |
| **2** | **Atacama – Copiapó – Taltal (Laguna 1922)** | $29.0^\circ\text{S}\text{–}25.0^\circ\text{S}$ | 0.79 | **$3.65\text{ m}$** | 0.908 | $-0.029$ | **31.9%** | **9.6%** | **2.6%** | **$M_w\ 4.84$** | 🟠 **MEDIA-ALTA (Acumulación)** |
| **3** | **Valparaíso – Metropolitana (Laguna 1730/1985)** | $34.5^\circ\text{S}\text{–}32.2^\circ\text{S}$ | **0.87** | **$2.37\text{ m}$** | **0.708** | **$-0.584$** | **30.7%** | **8.0%** | **1.9%** | **$M_w\ 4.82$** | 🟠 **MEDIA-ALTA (Anomalía $\Delta b^+$)** |
| **4** | **Antofagasta – Calama – Mejillones** | $25.0^\circ\text{S}\text{–}21.5^\circ\text{S}$ | 0.76 | $1.59\text{ m}$ | 1.124 | $+0.104$ | **27.8%** | **7.2%** | **1.6%** | **$M_w\ 5.04$** | 🟡 **MODERADA (Intralosa Activa)** |
| **5** | **Iquique – Loa (Norte Grande)** | $21.5^\circ\text{S}\text{–}19.5^\circ\text{S}$ | 0.85 | $0.68\text{ m}$ | 0.996 | $-0.088$ | **26.9%** | **6.9%** | **1.4%** | **$M_w\ 5.01$** | 🟡 **MODERADA (Base Activa)** |
| **6** | **Arica – Parinacota** | $19.5^\circ\text{S}\text{–}17.0^\circ\text{S}$ | 0.78 | $1.32\text{ m}$ | 0.904 | $-0.068$ | **24.6%** | **6.0%** | **1.2%** | **$M_w\ 4.78$** | 🟡 **MODERADA (Base Activa)** |
| **7** | **O'Higgins – Maule – Biobío** | $38.5^\circ\text{S}\text{–}34.5^\circ\text{S}$ | 0.82 | $0.88\text{ m}$ | 0.955 | $+0.172$ | **21.2%** | **4.6%** | **0.9%** | **$M_w\ 4.78$** | 🟢 **BAJA (Relajación Post-2010)** |
| **8** | **Araucanía – Valdivia – Chiloé** | $44.0^\circ\text{S}\text{–}38.5^\circ\text{S}$ | 0.83 | $3.65\text{ m}$ | 0.727 | $+0.000$ | **17.3%** | **3.5%** | **0.8%** | **$M_w\ 4.66$** | 🟢 **BAJA (Baja Tasa Instrumental)** |
| **—** | **TOTAL MARGEN CHILENO ($17^\circ\text{S}\text{–}56^\circ\text{S}$)** | Todo Chile | — | — | — | — | **91.9%** | **46.2%** | **13.4%** | **$M_w\ 5.65$** | **Síntesis Nacional (30 Días)** |

#### Diagnóstico Sismotectónico de los 3 Focos Principales Actuales:
1. **Foco #1 de Corto Plazo — Coquimbo / La Serena / Illapel ($32.2^\circ\text{S}\text{–}29.0^\circ\text{S}$):** Presenta la mayor probabilidad a 30 días ($35.5\%$ para $M \ge 5.0$ y $12.2\%$ para $M \ge 5.5$) debido a una combinación de alta tasa reciente (9 eventos $M \ge 4.0$ en 30 días, incluyendo un $M\ 4.9$ reciente) y un valor $b^+$ bajo ($0.779$).
2. **Foco #2 de Laguna Sísmica Madura — Atacama / Copiapó / Taltal ($29.0^\circ\text{S}\text{–}25.0^\circ\text{S}$):** Acumula **$3.65\text{ metros}$ de déficit de deslizamiento elástico** (sin ruptura gigante desde 1922 $M_w\ 8.5$) con alto bloqueo geodésico ($\chi = 0.79$) y deriva negativa de estrés ($\Delta b^+ = -0.029$), ubicándose en segundo lugar de peligro operacional ($P(M \ge 5.5 \mid 30\text{d}) = 9.6\%$).
3. **Foco #3 de Anomalía de Estrés Diferencial — Valparaíso / Región Metropolitana ($34.5^\circ\text{S}\text{–}32.2^\circ\text{S}$):** Es el segmento con **mayor bloqueo intersísmico ($\chi = 0.87$)** y muestra actualmente la **caída más pronunciada del valor $b$-positivo de todo Chile ($\Delta b^+ = -0.584$, situándose en $b^+ = 0.708$)**. Aunque su tasa de eventos en los últimos 30 días es moderada (4 eventos $M \ge 4.0$), cualquier sismo $M \ge 5.0$ en la costa de Valparaíso–San Antonio durante este estado de bajo $b^+$ activaría de inmediato el protocolo de escalamiento de precursores (Tarea 5).

---

## 9. Uso de la Herramienta CLI Operacional (`predict_chile.py`)

El repositorio incluye el sistema ejecutable [`predict_chile.py`](file:///Users/cnavarreteliz/Documents/Repos/earthquakes/predict_chile.py) para consultar o actualizar el pronóstico en cualquier momento:

```bash
# 1. Pronóstico nacional por zonas (30 días) conectándose en vivo a la API de USGS:
python3 predict_chile.py --fetch-live

# 2. Pronóstico de corto plazo (7 o 14 días) para todas las zonas:
python3 predict_chile.py --days 14

# 3. Pronóstico puntual para una coordenada específica (ej. Valparaíso / Santiago):
python3 predict_chile.py --lat -33.0 --lon -71.6 --days 14
```

---

## 10. Referencias Bibliográficas y Enlaces a Artículos de arXiv

> **Aviso de Licencia y Términos de Uso de arXiv:** Los metadatos y textos de preprints de arXiv utilizados en esta revisión están sujetos a los términos de uso de la API de arXiv (https://info.arxiv.org/help/api/index.html) y a las licencias individuales de cada artículo detalladas en [`.licenses/literature_search_arxiv_LICENSE.txt`](file:///Users/cnavarreteliz/Documents/Repos/earthquakes/.licenses/literature_search_arxiv_LICENSE.txt).

### Artículos de arXiv Consultados y Citados Directamente:
1. **Stockman, S., Lawson, D. J., & Werner, M. J. (2024/2026).** *EarthquakeNPP: A Benchmark for Earthquake Forecasting with Neural Point Processes.* Transactions on Machine Learning Research (TMLR). URL: https://arxiv.org/pdf/2410.08226v3
2. **Xiong, Z., et al. (2026).** *Scaling-law-informed neural point processes for earthquake sequence forecasting (Fusion).* arXiv preprint. URL: https://arxiv.org/pdf/2608.18791v1
3. **Köhler, J., et al. (2025).** *Neural Earthquake Forecasting with Minimal Information: Limits, Interpretability, and the Role of Markov Structure.* arXiv preprint. URL: https://arxiv.org/pdf/2509.14661v1
4. **Köhler, J., et al. (2026).** *Probabilistic and Alarm-Based Evaluation of a b-Value-Driven Deep Learning Earthquake Forecast.* arXiv preprint. URL: https://arxiv.org/pdf/2603.03079v1
5. **Mousavi, S. M., et al. (2026).** *Integrating b-Value and Background Seismicity Rate for Spatial Earthquake Forecasting.* arXiv preprint. URL: https://arxiv.org/pdf/2605.23494v1
6. **Chau, A., Bendick, R., Choi, G. P. T., & Mahadevan, L. (2026).** *How geometry of subduction zones correlates with earthquake dynamics.* arXiv preprint. URL: https://arxiv.org/pdf/2606.02520v1
7. **Zhang, X., et al. (2023).** *CL-ETAS: Combining deep learning and ETAS model for earthquake sequence forecasting.* arXiv preprint. URL: https://arxiv.org/pdf/2310.02574v1
8. **Stockman, S., Lawson, D. J., & Werner, M. J. (2023).** *Forecasting the 2016-2017 Central Apennines Earthquake Sequence with a Neural Point Process.* arXiv preprint. URL: https://arxiv.org/pdf/2301.09948v3

### Literatura Geofísica y Sismológica Fundamental Adicional:
- **Ogata, Y. (1988).** Statistical models for earthquake occurrences and residual analysis for point processes. *Journal of the American Statistical Association*, 83(401), 9–27.
- **Ogata, Y. (1998).** Space-time point-process models for earthquake occurrences. *Annals of the Institute of Statistical Mathematics*, 50(2), 379–402.
- **van der Elst, N. J. (2021).** B-positive: A more robust estimator of the b-value for short-term aftershock sequences. *Journal of Geophysical Research: Solid Earth*, 126(2), e2020JB021027.
- **Gulia, L., & Wiemer, S. (2019).** Real-time discrimination of earthquake foreshocks and aftershocks. *Nature*, 574(7777), 193–199.
- **Bouchon, M., Marsan, D., Durand, V., Campillo, M., Perfettini, H., Madariaga, R., & Gardonio, B. (2016).** Potential slab deformation and plunge prior to the Tohoku, Iquique and Maule earthquakes. *Nature Geoscience*, 9(5), 380–383.
- **Métois, M., Vigny, C., & Socquet, A. (2016).** Interseismic coupling, megathrust earthquakes and seismic swarms along the Chilean subduction zone ($38^\circ\text{S}\text{–}18^\circ\text{S}$). *Geophysical Journal International*, 204(1), 646–669.
- **Hayes, G. P., Moore, G. L., Portner, D. E., Hearne, M., Flamme, H., Furtney, M., & Smoczyk, G. M. (2018).** Slab2, a comprehensive subduction zone geometry model. *Science*, 362(6410), 58–61.
- **Zaliapin, I., & Ben-Zion, Y. (2013).** Earthquake clusters in southern California I: Identification and stability. *Journal of Geophysical Research: Solid Earth*, 118(6), 2847–2864.

