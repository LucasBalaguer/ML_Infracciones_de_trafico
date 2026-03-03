# 🚦 Predicción de Gravedad en Infracciones de Tráfico

## 📌 Descripción del problema

El objetivo de este proyecto es desarrollar un modelo de Machine Learning capaz de predecir, cuando un asegurado recibe una multa y la notifica a la aseguradora,  si esa infracción es grave o no, y subir la prima en consecuencia

Desde el punto de vista de negocio, este modelo permitiría:

- Detectar perfiles de mayor riesgo.
- Priorizar acciones preventivas.
- Optimizar campañas de concienciación.
- Apoyar la toma de decisiones en organismos de tráfico.

La variable objetivo utilizada es **GRAVEDAD**, construida a partir de la variable `PUNTOS`:

- `0` → No grave (≤ 3 puntos)
- `1` → Grave (≥ 4 puntos)

---

## 📊 Dataset utilizado

- **Nombre del archivo:** `dataset_definitivo.csv`
- **Formato:** CSV
- **Tipo:** Dataset estructurado
- **Origen:** Público. Descargado desde data.gob.es
- **[Acceso](https://datos.gob.es/es/catalogo/e00130502-fichero-de-microdatos-de-sanciones-con-detraccion-de-puntos-2023)**

### Preprocesamiento realizado

- Eliminación de variables con alta correlación o fuga de información (`PUNTOS`, `CUANTIA`, etc.).
- Conversión de variables categóricas a formato numérico:
  - `SEXO` → Variable binaria
  - `NOVEL` → Variable binaria
  - `EDAD` → Codificación ordinal por rangos
- División del dataset:
  - 80% entrenamiento
  - 20% test
  - División estratificada según la variable objetivo

---

## 🤖 Solución adoptada

Se plantea un problema de **clasificación binaria supervisada**.

### Modelos evaluados

Se compararon distintos modelos priorizando la métrica **recall**, ya que el objetivo principal es minimizar los falsos negativos (casos graves no detectados):

- K-Nearest Neighbors (KNN)
- Regresión Logística (con `class_weight="balanced"`)
- Random Forest
- LightGBM

### Optimización

- Validación cruzada con `StratifiedKFold`
- Búsqueda de hiperparámetros mediante:
  - `GridSearchCV`
  - `RandomizedSearchCV`
- Ajuste manual del umbral de decisión (threshold tuning)

### Modelo final

El modelo seleccionado fue **LightGBM optimizado**, con ajuste del umbral de decisión para maximizar el recall en la clase grave.

## 🛠 Tecnologías utilizadas

### Lenguaje
- Python 3.x

### Librerías principales
- pandas
- numpy
- scikit-learn
- lightgbm
- xgboost
- seaborn
- matplotlib
- scipy

### Técnicas aplicadas
- Preprocesamiento de datos
- Codificación de variables
- Escalado (`StandardScaler`)
- Train/Test split estratificado
- Validación cruzada
- Optimización de hiperparámetros
- Ajuste de umbral
- Matriz de confusión
- Classification report


## 📈 Principales resultados

Con el modelo final seleccionado (**LightGBM optimizado + ajuste de umbral**), se obtuvieron los siguientes resultados:

- **Recall (clase grave):** ≈ 0.93  
- **Accuracy:** (HAY QUE ACTUALIZAR ESTO)  
- **F1-score:** (HAY QUE ACTUALIZAR ESTO)  

### Interpretación de resultados

- El modelo identifica aproximadamente el **93% de los casos graves reales**.
- Se prioriza la minimización de **falsos negativos**, ya que no detectar una infracción grave tiene mayor impacto que generar un falso positivo.
- El modelo es adecuado para contextos donde la detección temprana del riesgo es crítica.

En conclusión, la solución desarrollada cumple el objetivo principal de maximizar la capacidad de detección de casos graves, manteniendo un equilibrio razonable con el resto de métricas.

---

## 👩‍💻 Autores

**Alba Rodríguez**  
- [GitHub](https://github.com/albarodriguez7) 

**Carlos D'Olhaberriague**  
- [GitHub](https://github.com/Carlos72293)  
  
**Lucas Cavalcante**  
- [GitHub](https://github.com/LucasBalaguer)