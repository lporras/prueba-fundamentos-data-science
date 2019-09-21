#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
File: herramientas_prueba.py
Author(s):
    *   Jose Gonzalez
    *   Catalina Cerda
    *   Luis Alfredo Porras Páez
Email(s):
    *   cotegl@gmail.com
    *   catalina.cerda@usach.cl
    *   lporras16@gmail.com
Github[s]:
    * https://github.com/catalinacerda
    * https://github.com/cotegl
    * https://github.com/lporras
Description:
    Este archivo python contiene sub rutinas utilizadas para brindar soluciones
    a la prueba final del 2do Módulo: Fundamentos de Data Science Generación 9!
"""

# Se importan librerías python para
# procesar datos y generar gráficos
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score

def get_missing_percent_info(df, column):
    """
    Definición:
    Dado un Dataframe y una columna retorna
    el porcentaje de datos perdidos de dicha columna.

    Parámetros de ingreso:
    - df: Tipo de dato: DataFrame. Base de Datos.
    - column: Tipo de dato: String. Nombre de la columna del Dataframe (df)
    Retorno:
    - Porcentaje de datos perdidos: Tipo de dato: String
    """
    non_nan_rows = len(df[df[column].isna()])
    total_rows = len(df)
    msg = f"El porcentaje de datos perdidos de la variable '{column}' es: {round(non_nan_rows/total_rows * 100, 2)}%"
    return (msg)

def get_total_percent_of_nan_rows(df):
    """
    Definición:
    Dado un Dataframe se retorna un mensaje del porcentaje
    de datos perdidos.

    Parámetros de ingreso:
    - df: Tipo de dato: DataFrame. Base de Datos.

    Retorno:
    - Porcentaje total de datos perdidos del df: Tipo de dato: String
    """
    dropped_df = df.dropna()
    msg = f"El porcentaje de datos perdidos si removemos todos los valores N/A es de: {round(len(dropped_df)/len(df)*100, 2)}%"
    return (msg)

def inverse_logit(x):
    """
    Definición:
    Dada una seríe x, esta funcion retorna
    el mapeo de log-odds a probabilidad.

    Parámetros de Ingreso:
    - x: Tipo de dato: Pandas.Serie

    Retorno:
    - Nueva Serie
    """
    return 1 / (1 + np.exp(-x))

def report_scores (y_hat, y_test):
    """
    Definición:
    Dado un vector predicho y un vector de test.
    Imprime los Scores del Modelo.

    Parámetros de Ingreso:
    - y_hat: Vector de Predicho.
    - y_test: Vector de test.
    """
    mse = mean_squared_error(y_test, y_hat).round(2)
    r2 = r2_score(y_test, y_hat).round(2)
    print("Scores del Modelo")
    print("Mean Squared Error: ", mse)
    print("R-cuadrado: ", r2)

def print_logit_estimations(modelo_logit, predicate):
    """
    Definición:
    Dado un modelo de Regresión Logística Clásica, y un predicado
    Entonces se imprime la probabilidad de la predicción del caso 1.

    Parámetros de Ingreso:
    - modelo_logit: Modelo de Regresión Logística de Statsmodels
    - predicate: Tipo de dato: String. El Predicado caso 1.
    """
    estimate_odds = 0
    prob_1 = 0
    msg = ''

    for (i, param) in enumerate(modelo_logit.params.index):
        estimate_odds = estimate_odds + modelo_logit.params[param]
    print(f"Estimate_odds: {estimate_odds}")
    prob_1 = inverse_logit(estimate_odds) * 100
    msg = f"La probabilidad para un individuo con las variables suministradas en el modelo de tener un {predicate} es de: {round(prob_1, 3)}%"
    print(msg)
    return None

# Función para graficar variables en un Data Frame
def visualize_rows (df, width=12, height=12, title=None):
    """
    Definición:
    Dado un DataFrame se itera por filas
    y genera gráficos dependiendo al tipo de columna

    Parámetros de ingreso:
    - df: Tipo de dato: DataFrame.  Base de Datos a visualizar
    - width: Tipo de dato: Float. Usada para configurar el ancho de cada gráfico
    - height: Tipo de dato: Float. Usada para configurar el alto de cada gráfico
    - title: Tipo de dato: String. Usada para el título de cada gráfico.
        - Si no se envía ningun título, entonces tomará por defecto el nombre de cada
        columna del argumento df.
        De lo contrario asignará a cada gráfico el titulo recibido.
    Retorno:
    - None
    """
    for n, i in enumerate(df):
        plt.rcParams['figure.figsize'] = (width, height)
        plt.subplot((len(list(df.columns))/3)+1,3,n+1)
        if df[i].dtypes == float:
            sns.distplot(df[i].dropna())
            plt.xlabel("")
        elif df[i].dtypes == "object":
            sns.countplot(df[i])
            plt.xlabel("")
        else:
            sns.distplot(df[i].dropna(),kde=False)
            plt.xlabel("")

        if title is not None:
            plt.title(title)
        else:
          plt.title(i)

        plt.tight_layout()
        plt.show()
    return None

# Función Binarizadora de columnas de un Data Frame
def binarize_columns(dataframe, columns):
    """
    Definición:
    Dado un DataFrame y una lista de columnas
    Binariza cada columna en una nueva columna b_variable
    tomando como criterio:
    - Asignar 1 a las categorías minoritarias.
    - Asignar 0 al resto.
    Parámetros de ingreso:
    - dataframe: Tipo de dato: DataFrame.  Base de Datos a modificar (Pandas DataFrame)
    - columns: Tipo de dato: Array. Lista de nombres de las columnas a binarizar
    Retorno:
    - Nuevo DataFrame
    """
    tmp_dataframe = dataframe.copy()
    for variable in columns:
        binary_column = f"b_{variable}"
        print(f"Analizando Datos de la variable: {variable}")
        variable_value_counts = dataframe[variable].value_counts()
        print(variable_value_counts)
        max_variables_count = variable_value_counts.max()
        print(f"El valor más frecuente tiene: {max_variables_count} registros")
        mayor_variable = variable_value_counts[variable_value_counts == max_variables_count].index[0]
        print(f"El valor más frecuente es: {mayor_variable}")
        tmp_dataframe[binary_column] = np.where(tmp_dataframe[variable] == mayor_variable, 0, 1)
        print(f"Analizando Datos Binarizados de la variable: {variable}")
        print(tmp_dataframe[binary_column].value_counts())
    return tmp_dataframe
