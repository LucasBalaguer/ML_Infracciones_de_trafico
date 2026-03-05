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

- **Recall (clase grave):** ≈ 0.64  
- **Accuracy:** ≈ 0.61 
- **F1-score:** ≈ 0.69  

### Interpretación de resultados

- El modelo identifica aproximadamente el 64% de los casos graves reales.
- Se ha priorizado la detección de casos graves mediante el ajuste del umbral de decisión.
- El modelo constituye una base sólida y funcional sobre la que seguir optimizando el rendimiento.

En conclusión, la solución desarrollada cumple el objetivo de construir un modelo operativo y alineado con negocio, aunque todavía presenta oportunidades claras de mejora en la detección de casos graves.

---

## 👩‍💻 Autores

**Alba Rodríguez**  
- [GitHub](https://github.com/albarodriguez7) 

**Carlos D'Olhaberriague**  
- [GitHub](https://github.com/Carlos72293)  
  
**Lucas Cavalcante**  
- [GitHub](https://github.com/LucasBalaguer)


----


# 🚦 Traffic Violation Severity Prediction

## 📌 Problem Description

The objective of this project is to develop a Machine Learning model capable of predicting whether a traffic violation is severe or not when an insured driver reports a fine to the insurance company, and adjust the premium accordingly.

From a business perspective, this model would enable:

- Identification of higher-risk profiles.  
- Prioritization of preventive actions.  
- Optimization of awareness campaigns.  
- Support for decision-making in traffic authorities.  

The target variable used is **GRAVEDAD** (SEVERITY), built from the variable `PUNTOS` (points deducted from the driver’s license):

- `0` → Non-severe (≤ 3 points)  
- `1` → Severe (≥ 4 points)  

---

## 📊 Dataset Used

- **File name:** `dataset_definitivo.csv`  
- **Format:** CSV  
- **Type:** Structured dataset  
- **Source:** Public dataset downloaded from data.gob.es  
- **Access:** https://datos.gob.es/es/catalogo/e00130502-fichero-de-microdatos-de-sanciones-con-detraccion-de-puntos-2023  

### Preprocessing Performed

- Removal of highly correlated variables or those causing data leakage (`PUNTOS`, `CUANTIA`, etc.).  
- Conversion of categorical variables into numerical format:
  - `SEXO` → Binary variable  
  - `NOVEL` → Binary variable  
  - `EDAD` → Ordinal encoding by age ranges  
- Dataset split:
  - 80% training  
  - 20% test  
  - Stratified split based on the target variable  

---

## 🤖 Proposed Solution

The problem is framed as a **supervised binary classification task**.

### Evaluated Models

Several models were compared, prioritizing the **recall** metric, since the main objective is to minimize false negatives (severe cases not detected):

- K-Nearest Neighbors (KNN)  
- Logistic Regression (with `class_weight="balanced"`)  
- Random Forest  
- LightGBM  

### Optimization

- Cross-validation using `StratifiedKFold`  
- Hyperparameter tuning through:
  - `GridSearchCV`  
  - `RandomizedSearchCV`  
- Manual adjustment of the decision threshold (threshold tuning)  

### Final Model

The selected model was **Optimized LightGBM**, with decision threshold adjustment to maximize recall for the severe class.

---

## 🛠 Technologies Used

### Language
- Python 3.x  

### Main Libraries
- pandas  
- numpy  
- scikit-learn  
- lightgbm  
- xgboost  
- seaborn  
- matplotlib  
- scipy  

### Applied Techniques
- Data preprocessing  
- Variable encoding  
- Feature scaling (`StandardScaler`)  
- Stratified train/test split  
- Cross-validation  
- Hyperparameter optimization  
- Threshold adjustment  
- Confusion matrix analysis  
- Classification report  

---

## 📈 Key Results

With the final selected model (**Optimized LightGBM + threshold adjustment**), the following results were obtained:

- **Recall (severe class):** ≈ 0.64  
- **Accuracy:** ≈ 0.61  
- **F1-score:** ≈ 0.69  

### Results Interpretation

- The model correctly identifies approximately **64% of real severe cases**.  
- Detection of severe cases was prioritized through decision threshold adjustment.  
- The model provides a solid and functional baseline upon which performance can continue to be optimized.  

In conclusion, the developed solution achieves the objective of building an operational and business-aligned model, although there are still clear opportunities for improvement in detecting severe cases.

---

## 👩‍💻 Authors

**Alba Rodríguez**  
- GitHub: https://github.com/albarodriguez7  

**Carlos D'Olhaberriague**  
- GitHub: https://github.com/Carlos72293  

**Lucas Cavalcante**  
- GitHub: https://github.com/LucasBalaguer  