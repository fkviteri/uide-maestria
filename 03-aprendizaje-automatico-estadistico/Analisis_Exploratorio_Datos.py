# -*- coding: utf-8 -*-
"""
@author: IVAN
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar el conjunto de datos del Titanic
titanic_df = pd.read_csv('titanic.csv', encoding='utf-8', sep=';')
# titanic_df = sns.load_dataset("titanic")



# Ver las primeras filas del conjunto de datos
print(titanic_df.head(10))

# Estadísticas descriptivas
print(titanic_df.describe())

# Estadísticas de variables categóricas
print(titanic_df.describe(include=['object']))

# Histogramas de Edades (frecuencias):
plt.figure(figsize=(8, 6)) # Crear una nueva figura con tamaño 8x6 pulgadas
# Divide el rango de edades en 30 intervalos o barras. No considera valores nulos (nan)
# Añade una curva KDE (Kernel Density Estimate) sobre el histograma, que representa una estimación suave de la distribución de probabilidad de los datos
sns.histplot(titanic_df['Age'].dropna(), bins=30, kde=True)
plt.xlabel('Edad')
plt.ylabel('Frecuencia')
plt.title('Distribución de Edades en el Titanic')
plt.show()

# Crear el diagrama de barras de las clases del target (Survived)
# plt.figure(figsize=(8, 6))
sns.countplot(data=titanic_df, x='Survived')
plt.xlabel('Survived')
plt.ylabel('Count')
plt.title('Distribución de sobrevivientes')
plt.xticks([0, 1], ['No', 'Si'])
plt.show()
print('Conteo de valores del target:\n', titanic_df['Survived'].value_counts())


# Gráfico de Barras de Supervivencia por Clase:
# El parámetro hue se utiliza para agregar una dimensión adicional de categorización a 
# las barras del gráfico y se dividirán y colorearán según los valores de la variable 'Survived'.
plt.figure(figsize=(8, 6))
sns.countplot(data=titanic_df, x='Pclass', hue='Survived')
plt.xlabel('Clase')
plt.ylabel('Cantidad')
plt.title('Supervivencia por Clase en el Titanic')
plt.legend(title='Sobreviviente', labels=['No', 'Sí'])
plt.show()

# Gráfico de Torta de Género:
plt.figure(figsize=(8, 6))
titanic_df['Sex'].value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.title('Distribución de Género en el Titanic')
plt.ylabel('')
plt.show()

# Análisis de Correlación
# Se podría eliminar aquellas variables que tengan una correlación baja o nula
# con la variable objetivo, ya que podrían no aportar información relevante al modelo. 
# Además, se podría quitar ciertas variables que están altamente correlacionadas entre sí.

# Seleccionar las variables numéricas principales para el análisis de correlación y el target
variables_numericas = ['Age', 'SibSp', 'Parch', 'Fare', 'Survived']

# age_mean = titanic_df['Age'].mean()
# titanic_df['Age'] = titanic_df['Age'].fillna(age_mean)

# Crear una submatriz de correlación
correlation_matrix = titanic_df[variables_numericas].corr()
# crea una máscara para ocultar la parte superior de la matriz de correlación
# con k=0 no incluye la diagonal principal y con k=1 si
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool), k=1)

# Crear un mapa de calor de correlación
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', linewidths=.5)
plt.title('Matriz de Correlación entre Variables Numéricas del Titanic')
plt.show()

# Aplicar una máscara para mostrar solo correlaciones moderadas/altas mayores a 0.4
mask = np.abs(correlation_matrix) < 0.4
correlation_matrix[mask] = np.nan
# Crear un mapa de calor de correlación con valores significativos
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', linewidths=.5)
plt.title('Matriz de Correlación (moderadas / altas)')
plt.show()

# Crear el box-plot de la variable "edad"
plt.figure(figsize=(10, 6))
sns.boxplot(x=titanic_df['Age'])
plt.title('Box-Plot de la Edad')
plt.xlabel('Edad')
plt.show()

# Eliminar valores atípicos (filas)
Q1 = titanic_df['Age'].quantile(0.25)
Q3 = titanic_df['Age'].quantile(0.75)
IQR = Q3 - Q1 # rango intercuartil
limite_superior = Q3 + 1.5*IQR
limite_inferior = Q1 - 1.5*IQR
filtered_titanic_df = titanic_df[(titanic_df['Age'] >= limite_inferior) & (titanic_df['Age'] <= limite_superior)]

# Crear el box-plot de la variable "edad"
plt.figure(figsize=(10, 6))
sns.boxplot(x=filtered_titanic_df['Age'])
plt.title('Box-Plot de la Edad (sin atípicos)')
plt.xlabel('Edad')
plt.show()
