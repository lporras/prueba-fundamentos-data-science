#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
File: utils.py
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
    Este archivo python contiene subrutinas utilizadas para brindar soluciones
    a la prueba final del 2do Módulo: Fundamentos de Data Science Generación 9!
"""

#Importar librerias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Función para graficar variables en un Data Frame
def visualize_rows (df):
    """
    Definición:
    Dado un DataFrame se itera por filas
    y genera gráficos dependiendo al tipo de columna

    Parámetros de ingreso:
    - df: Tipo de dato: DataFrame.  Base de Datos a visualizar
    - columns: Tipo de dato: Array. Lista de nombres de las columnas a binarizar

    Retorno:
    - None
    """
    for n, i in enumerate(df):
        plt.rcParams['figure.figsize'] = (10, 6)
        plt.subplot((len(list(df.columns))/3)+1,3,n+1)
        if df[i].dtypes == float:
            sns.distplot(df[i].dropna())
            plt.title(i)
            plt.xlabel("")
        elif df[i].dtypes == "object":
            sns.countplot(df[i])
            plt.title(i)
            plt.xlabel("")
        else:
            sns.distplot(df[i].dropna(),kde=False)
            plt.title(i)
            plt.xlabel("")
    plt.tight_layout()
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
