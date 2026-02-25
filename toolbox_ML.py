import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import f_oneway
from scipy.stats import pearsonr
import math


# 1 - Función: describe_df

"""
Esta función debe recibir como argumento un dataframe y debe devolver una dataframe (describe_df) como el de la imagen (NO el de la imagen). 
Es decir, un dataframe que tenga una columna por cada variable del dataframe original, y como filas los tipos de dichas variables, 
el tanto por ciento de valores nulos o missings, los valores únicos y el porcentaje de cardinalidad.  

"""

def describe_df(df):
    DATA_TYPE=df.dtypes
    MISSINGS=(df.isna().sum()/len(df)*100).sort_values(ascending=False)
    UNIQUE_VALUES=df.nunique()
    CARDIN=UNIQUE_VALUES/len(df)*100
    describe_df=pd.DataFrame([DATA_TYPE, MISSINGS, UNIQUE_VALUES, CARDIN])
    parametros=["DATA_TYPE", "MISSINGS (%)", "UNIQUE_VALUES", "CARDIN (%)"]
    describe_df.insert(0, "COL_N",parametros)
    return describe_df


# 2 - Función: tipifica_variables

"""
Esta función debe recibir como argumento un dataframe, un entero (`umbral_categoria`) y un float (`umbral_continua`).
La función debe devolver un dataframe con dos columnas "nombre_variable", "tipo_sugerido" que tendrá tantas filas como columnas el dataframe. 
En cada fila irá el nombre de una de las columnas y una sugerencia del tipo de variable. 
Esta sugerencia se hará siguiendo las siguientes pautas:
+ Si la cardinalidad es 2, asignara "Binaria"
+ Si la cardinalidad es menor que `umbral_categoria` asignara "Categórica"
+ Si la cardinalidad es mayor o igual que `umbral_categoria`, entonces entra en juego el tercer argumento:
    * Si además el porcentaje de cardinalidad es superior o igual a `umbral_continua`, asigna "Numerica Continua"
    * En caso contrario, asigna "Numerica Discreta"

"""

def tipifica_variables(df,umbral_categoria=6, umbral_continua=10):
    Tipo=df.dtypes
    Card=df.nunique()
    Card_rel=Card/len(df)*100
    df_tipificacion=pd.DataFrame([df.columns, Tipo, Card, Card_rel]).T.rename(columns = {0: "nombre_variable", 1: "Tipo", 2: "Card", 3: "Card_rel"})
    df_tipificacion["tipo_sugerido"]="sin categoría" 
    df_tipificacion.loc[df_tipificacion.Card == 2, "tipo_sugerido"] = "Binaria"
    es_numerica = ((df_tipificacion.Tipo == "float64") | (df_tipificacion.Tipo == "int64")) 
    #df_tipificacion.loc[df_tipificacion.Tipo == "datetime64[ns]", "Clasificada_como"] = "Fecha"
    df_tipificacion.loc[(df_tipificacion.Card > 2) & (df_tipificacion.Card < umbral_categoria), "tipo_sugerido"] = "Categórica"
    df_tipificacion.loc[(df_tipificacion.Card > umbral_categoria) & es_numerica & (df_tipificacion.Card_rel < umbral_continua), "tipo_sugerido"] = "Numérica discreta"
    df_tipificacion.loc[(df_tipificacion.Card > umbral_categoria) & es_numerica &  (df_tipificacion.Card_rel > umbral_continua), "tipo_sugerido"] = "Numérica continua"

    return df_tipificacion[["nombre_variable", "tipo_sugerido"]]


# 3 - Función: get_features_num_regression

"""
Esta función recibe como argumentos un dataframe, el nombre de una de las columnas del mismo (argumento 'target_col'), que debería ser el target de un hipotético modelo de regresión, 
es decir debe ser una variable numérica continua o discreta pero con alta cardinalidad, además de un argumento 'umbral_corr', 
de tipo float que debe estar entre 0 y 1 y una variable float "pvalue" cuyo valor debe ser por defecto "None".
La función debe devolver una lista con las columnas numéricas del dataframe cuya correlación con la columna designada por "target_col" sea superior en valor absoluto al valor dado por "umbral_corr". 
Además si la variable "pvalue" es distinta de None, sólo devolvera las columnas numéricas cuya correlación supere el valor indicado y además supere el test de hipótesis con significación mayor o igual a 1-pvalue.
La función debe hacer todas las comprobaciones necesarias para no dar error como consecuecia de los valores de entrada. 
Es decir hará un check de los valores asignados a los argumentos de entrada y si estos no son adecuados debe retornar None y printar por pantalla la razón de este comportamiento. 
Ojo entre las comprobaciones debe estar que "target_col" hace referencia a una variable numérica continua del dataframe.
"""

def get_features_num_regression(df, target_col, umbral_corr, pvalue=None):
    df_tip=tipifica_variables(df)
    if target_col in df.columns:
        cond_target1=df_tip.nombre_variable == target_col
        cond_target2=(df_tip.tipo_sugerido == "Numérica continua")
        cond_num= (df_tip.tipo_sugerido == "Numérica continua") | (df_tip.tipo_sugerido == "Numérica discreta")
        target_numerico = df_tip.loc[cond_target1 & cond_target2]
        
        if not target_numerico.empty:
            col_num=df_tip.loc[cond_num, "nombre_variable"].tolist()
            matriz_correlacion=df[col_num].corr()
            correlacion_con_target = matriz_correlacion[target_col].abs()
            print(f"La variable {target_col} se puede considerar variable target, ya que es numérica continua.")
            condicion_umbral=(correlacion_con_target > umbral_corr) & (correlacion_con_target < 1.0)
            num_selec1=correlacion_con_target[condicion_umbral].index.tolist()
            num_selec2 = []
            for col in num_selec1:
            # Eliminamos filas con nulos en estas dos columnas para evitar errores en el test
                df_temp = df[[col, target_col]].dropna()
            # Calculamos la correlación y el p-value:
                r_val, p_val = pearsonr(df_temp[col], df_temp[target_col])
            # Si el usuario no definió pvalue, o si el p_val es menor al umbral solicitado
                if pvalue is None or p_val < pvalue:
                    num_selec2.append(col)
                    print(f"Variable '{col}' seleccionada: p-value = {p_val:.4f} (relación estadísticamente significativa).")
                else:
                    print(f"Variable '{col}' descartada: p-value = {p_val:.4f} (relación no significativa).")
            return num_selec2
        else:
            print(f"La variable {target_col} existe, pero no es numérica continua.")
    else:
        print("La variable no está en el dataset.")


# 4 - Función: plot_features_num_regression

def plot_features_num_regression(
    df: pd.DataFrame,
    target_col: str = "",
    columns: list[str] = [],
    umbral_corr: float = 0,
    pvalue: float | None = None
):
    """
    ENUNCIADO (cumplido):
    - args: df, target_col="", columns=[], umbral_corr=0, pvalue=None
    - si columns vacía -> usa todas las numéricas del df (excepto target)
    - filtra columns por |corr| > umbral_corr
    - si pvalue != None -> además p < pvalue (Pearson)
    - pinta pairplot de target + columnas filtradas
    - EXTRA: paginación: varios pairplots con máx 5 columnas + target
    - devuelve la lista final de columnas filtradas
    - si inputs inválidos: print razón + return None
    """

    # ---------------- checks de entrada (inválido -> None) ----------------
    if not isinstance(df, pd.DataFrame):
        print("Error: df debe ser un pandas DataFrame.")
        return None

    if not isinstance(target_col, str) or target_col == "":
        print("Error: target_col debe ser un string no vacío.")
        return None

    if target_col not in df.columns:
        print("Error: target_col no existe en df.columns.")
        return None

    if not pd.api.types.is_numeric_dtype(df[target_col]):
        print("Error: target_col debe ser numérica (target de regresión).")
        return None

    if not isinstance(umbral_corr, (int, float)) or not (0 <= float(umbral_corr) <= 1):
        print("Error: umbral_corr debe ser un número entre 0 y 1.")
        return None
    umbral_corr = float(umbral_corr)

    if pvalue is not None:
        if not isinstance(pvalue, (int, float)) or not (0 < float(pvalue) < 1):
            print("Error: pvalue debe ser None o un float entre 0 y 1 (ej. 0.05).")
            return None
        pvalue = float(pvalue)

    if columns is None or not isinstance(columns, list):
        print("Error: columns debe ser una lista de strings (puede ser vacía).")
        return None
    if len(columns) > 0 and not all(isinstance(c, str) for c in columns):
        print("Error: columns debe contener solo strings (nombres de columnas).")
        return None

    # ---------------- preparar columns ----------------
    if len(columns) == 0:
        # todas las numéricas excepto target
        columns = df.select_dtypes(include="number").columns.tolist()
        columns = [c for c in columns if c != target_col]
    else:
        # validar existencia y tipo numérico
        missing = [c for c in columns if c not in df.columns]
        if missing:
            print(f"Error: columns contiene columnas inexistentes: {missing}")
            return None
        non_num = [c for c in columns if not pd.api.types.is_numeric_dtype(df[c])]
        if non_num:
            print(f"Error: columns contiene columnas no numéricas: {non_num}")
            return None
        columns = [c for c in columns if c != target_col]

    if len(columns) == 0:
        print("Error: no hay columnas numéricas para comparar con target_col.")
        return None

    # ---------------- filtro por correlación ----------------
    corr_abs = df[[target_col] + columns].corr(numeric_only=True)[target_col].abs()
    corr_abs = corr_abs.drop(labels=[target_col], errors="ignore")
    selected = corr_abs[corr_abs > umbral_corr].index.tolist()

    if len(selected) == 0:
        print("No hay columnas que cumplan |corr| > umbral_corr.")
        return []  # entrada válida, resultado vacío

    # ---------------- filtro por pvalue (opcional) ----------------
    if pvalue is not None:
        try:
            from scipy.stats import pearsonr
        except Exception:
            print("Error: no se pudo importar scipy.stats.pearsonr para calcular p-values.")
            return None

        passed = []
        for col in selected:
            tmp = df[[col, target_col]].dropna()
            if len(tmp) < 3 or tmp[col].nunique() < 2 or tmp[target_col].nunique() < 2:
                continue
            r, p = pearsonr(tmp[col], tmp[target_col])
            if p < pvalue:
                passed.append(col)

        selected = passed

        if len(selected) == 0:
            print("No hay columnas que cumplan umbral_corr y además p-value < pvalue.")
            return []

    # ---------------- pairplot con paginación (máx 5 cols + target) ----------------
    def _plot_pairplot(df_plot: pd.DataFrame, title: str):
        # Intentamos seaborn (más bonito). Si no está, fallback a scatter_matrix.
        try:
            import seaborn as sns
            g = sns.pairplot(df_plot, diag_kind="hist", corner=False, plot_kws={"alpha": 0.6})
            g.fig.suptitle(title, y=1.02)
            plt.show()
        except Exception:
            pd.plotting.scatter_matrix(
                df_plot,
                figsize=(3 * df_plot.shape[1], 3 * df_plot.shape[1]),
                diagonal="hist",
                alpha=0.6
            )
            plt.suptitle(title, y=1.02)
            plt.tight_layout()
            plt.show()

    max_cols = 5
    for i in range(0, len(selected), max_cols):
        chunk = selected[i:i + max_cols]
        cols_to_plot = [target_col] + chunk
        df_plot = df[cols_to_plot].dropna()

        title = f"Pairplot: target={target_col} | cols {i+1}-{i+len(chunk)} de {len(selected)}"
        _plot_pairplot(df_plot, title)

    # ---------------- return obligatorio ----------------
    return selected

# 5 - Función: get_features_cat_regression 
   
def get_features_cat_regression(df, target_col, pvalue=0.05):
    """
    Devuelve una lista de columnas categóricas que tienen una relación estadísticamente significativa
    con una variable objetivo numérica, utilizando el test ANOVA (Analysis of Variance).

    Argumentos:
    df (pandas.DataFrame): DataFrame que contiene los datos.
    target_col (str): Nombre de la columna numérica que es el target del modelo.
    pvalue (float): Valor de significación estadística (alpha). Por defecto 0.05.
                    Si el p-valor del test es MENOR que este umbral, la variable se selecciona.

    Retorna:
    list: Los nombres de las columnas categóricas relevantes.
          Retorna None si los argumentos de entrada no son válidos.
    """

    # 1. Comprobaciones de seguridad (Input checks)
    if not isinstance(df, pd.DataFrame):
        print("Error: El argumento 'df' debe ser un pandas DataFrame.")
        return None

    if target_col not in df.columns:
        print(f"Error: La columna '{target_col}' no existe en el DataFrame.")
        return None

    # Comprobamos si el target es numérico
    if not pd.api.types.is_numeric_dtype(df[target_col]):
        print(f"Error: La columna objetivo '{target_col}' debe ser numérica.")
        return None

    if not isinstance(pvalue, float) or not (0 < pvalue < 1):
        print("Error: 'pvalue' debe ser un float entre 0 y 1.")
        return None

    # 2. Selección de variables categóricas candidatas
    # Incluimos 'object' (strings) y 'category' (pandas categories)
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    if not cat_cols:
        print("Aviso: No se encontraron columnas categóricas en el DataFrame.")
        return [] # Devolvemos lista vacía en lugar de None para no romper bucles externos

    columnas_seleccionadas = []

    # 3. Bucle de Análisis (ANOVA)
    for col in cat_cols:
        # Limpiamos nulos solo para este par de variables (features + target)
        temp_df = df[[col, target_col]].dropna()

        # Si tras limpiar nulos no hay datos o solo hay una categoría, saltamos
        if temp_df.empty or temp_df[col].nunique() < 2:
            continue

        # Preparamos los grupos para el ANOVA
        # Ejemplo: grupos = [precios_primera_clase, precios_segunda_clase, ...]
        grupos = []
        for categoria in temp_df[col].unique():
            grupo_valores = temp_df[temp_df[col] == categoria][target_col].values
            grupos.append(grupo_valores)

        # Ejecutamos el test ANOVA (f_oneway)
        # f_oneway devuelve (estadístico, p_valor)
        stat, p_valor_calculado = f_oneway(*grupos)

        # 4. Decisión
        # Si el p-valor es PEQUEÑO (< 0.05), rechazamos la hipótesis nula.
        # Significa que SÍ hay diferencias significativas entre las medias de los grupos.
        if p_valor_calculado < pvalue:
            columnas_seleccionadas.append(col)

    return columnas_seleccionadas


# 6 - Funcion: plot_features_cat_regression

def plot_features_cat_regression2(
    df,
    target_col="",
    columns=[],
    pvalue=0.05,
    with_individual_plot=False
):
    """
    Representa gráficamente la relación entre variables categóricas y una variable
    objetivo numérica mediante histogramas agrupados, siempre que exista una relación
    estadísticamente significativa.

    Argumentos:
    df: DataFrame que contiene las variables a analizar.
    target_col (str): Nombre de la variable objetivo numérica. Por defecto "".
    columns (list[str]): Lista de variables categóricas a evaluar. Si está vacía,
                          se utilizarán automáticamente las variables categóricas
                          del DataFrame.
    pvalue (float): Nivel de significación estadística para el contraste de hipótesis.
                    Por defecto 0.05.
    with_individual_plot (bool): Si es True, genera un gráfico individual por cada
                                 variable significativa. Por defecto False.

    Retorna:
    list[str]: Lista de variables categóricas cuya relación con la variable objetivo
               es estadísticamente significativa según el nivel de significación
               indicado.
    """

    # ================== CHECKS ================== Verifica que la columna target exista, sea numérica y que el p-value sea válido.
    if target_col == "":
        print(" Debes indicar una columna target.")
        return []

    if target_col not in df.columns:
        print(f" La columna target '{target_col}' no existe en el DataFrame.")
        return []

    if not pd.api.types.is_numeric_dtype(df[target_col]):
        print(f" La columna target '{target_col}' debe ser numérica.")
        return []

    if not (0 < pvalue < 1):
        print(" El valor de pvalue debe estar entre 0 y 1.")
        return []

    # ================== SELECCIÓN DE COLUMNAS ================== Determina qué variables categóricas se van a analizar, usando todas si no se pasan como argumento.
    if columns is None or len(columns) == 0:
        columns = df.select_dtypes(include=["object", "category"]).columns.tolist()
    else:
        columns = [col for col in columns if col in df.columns]

    if len(columns) == 0:
        print(" No hay variables categóricas válidas para analizar.")
        return []

    significant_cols = []

    # ================== TEST ESTADÍSTICO ================== Aplica ANOVA a cada variable categórica para comprobar si hay relación significativa con la variable objetivo.
    for col in columns:
        grupos = [
            df.loc[df[col] == categoria, target_col].dropna()
            for categoria in df[col].dropna().unique()
        ]

        if len(grupos) < 2:
            continue

        if any(g.nunique() <= 1 for g in grupos):
            continue

        stat, p = f_oneway(*grupos)

        if p < pvalue:
            significant_cols.append((col, p))

            # === GRÁFICO INDIVIDUAL === Por defecto genera un gráfico de barras separado para cada variable categórica significativa.
            if with_individual_plot:
                sns.barplot(
                    data=df,
                    x=col,
                    y=target_col,
                    hue=col,
                    estimator="mean",
                    errorbar=None,
                    palette="tab10",
                    legend=False,
                )
                plt.ylim(0, 1)
                plt.ylabel("Survival rate")
                plt.title(f"{col} vs {target_col}")
                plt.tight_layout()
                plt.show()

    # ================== GRÁFICOS EN GRID (2 POR FILA) ================== Si no se generan gráficos individuales, organiza todas las variables significativas en un grid de 2 columnas y muestra sus medias.
    if not with_individual_plot and significant_cols:
        n_cols = 2
        n_rows = math.ceil(len(significant_cols) / n_cols)

        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(16, 4 * n_rows),
            squeeze=False
        )

        # Convertir axes a lista 2D para recorrerlos fácilmente
        axes = [[ax for ax in row] for row in axes]

        for ax, (col, p) in zip([ax for row in axes for ax in row], significant_cols):
            sns.barplot(
                data=df,
                x=col,
                y=target_col,
                hue=col,
                estimator="mean",
                errorbar=None,
                palette="tab10",
                legend=False,
                ax=ax
            )
            ax.set_ylim(0, 1)
            ax.set_ylabel("Survival rate")
            ax.set_title(f"{col}")

        # Eliminar ejes sobrantes si hay número impar
        for ax in [ax for row in axes for ax in row][len(significant_cols):]:
            ax.remove()

        plt.tight_layout()
        plt.show()

    return [col for col, _ in significant_cols]
