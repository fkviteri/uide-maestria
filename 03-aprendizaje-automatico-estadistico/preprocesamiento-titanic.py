# -*- coding: utf-8 -*-
"""
@author: IVAN
"""

"""
En un proyecto típico de análisis de datos, el preprocesamiento de datos generalmente 
precede al análisis exploratorio de datos (EDA). Sin embargo, en la práctica, 
estas fases pueden superponerse y retroalimentarse, ya que algunas etapas del EDA
 pueden revelar la necesidad de más preprocesamiento de datos. 
"""

########## librerías a utilizar ##########

#Se importan la librerías a utilizar
import numpy as np
import pandas as pd


########## Importando la data ##########

#Importar los datos de los archivos .csv almacenados
df_test = pd.read_csv('titanic_test.csv')
df_train = pd.read_csv('titanic_train.csv')

print(df_test.head())
print(df_train.head())

# la columna parch se refiere al número de padres/niños (Parents/Children) que viajan con el pasajero.
# la columna sibsp, que indica el número de hermanos/esposos a bordo con el pasajero.

########## Entendimiento de la data ##########

#Verifica la cantidad de datos que hay en los dataset
print('Cantidad de datos:')
print(df_train.shape)
print(df_test.shape)

#Verifica el tipo de datos contenida en ambos dataset
print('Tipos de datos:')
print(df_train.info())
print(df_test.info())

#Verifica los datos faltantes de los dataset
print('Datos faltantes:')
print(pd.isnull(df_train).sum())
print(pd.isnull(df_test).sum())

#Verifica las estadísticas básicas del dataset
print('Estadísticas del dataset:')
print(df_train.describe())
print(df_test.describe())

########## Preprocesamiento de la data ##########

# Transforma los datos de la variable sexo (categórico) en números
df_train['Sex'].replace(['female','male'],[0,1],inplace=True)
df_test['Sex'].replace(['female','male'],[0,1],inplace=True)

#Transforma los datos de embarque (categórico) en números
df_train['Embarked'].replace(['Q','S', 'C'],[0,1,2],inplace=True)
df_test['Embarked'].replace(['Q','S', 'C'],[0,1,2],inplace=True)

#Reemplazo los datos faltantes en la edad por la media de esta variable
print(df_train["Age"].mean())
print(df_test["Age"].mean())
promedio = 30
df_train['Age'] = df_train['Age'].replace(np.nan, promedio)
df_test['Age'] = df_test['Age'].replace(np.nan, promedio)

#Crea varios grupos/rangos de edades
#Rangos: 0-8, 9-15, 16-18, 19-25, 26-40, 41-60, 61-100
bins = [0, 8, 15, 18, 25, 40, 60, 100]
names = ['1', '2', '3', '4', '5', '6', '7']
df_train['Age'] = pd.cut(df_train['Age'], bins, labels = names)
df_test['Age'] = pd.cut(df_test['Age'], bins, labels = names)

#Se elimina la columna de "Cabin" ya que tiene muchos datos perdidos
# El parámetro axis=1 indica que se deben eliminar columnas en lugar de filas (axis=0).
# El parámetro inplace indica si la operación se realiza directamente en el 
# DataFrame original o devolvuelve una nueva copia con las filas o columnas eliminadas.
df_train.drop(['Cabin'], axis = 1, inplace=True)
df_test.drop(['Cabin'], axis = 1, inplace=True)

#Elimina las columnas que se considera que no son necesarias para el analisis
df_train = df_train.drop(['PassengerId','Name','Ticket'], axis=1)
df_test = df_test.drop(['Name','Ticket'], axis=1)

#Se elimina las filas con datos perdidos
df_train.dropna(axis=0, how='any', inplace=True)
df_test.dropna(axis=0, how='any', inplace=True)

#Verifica los datos
print(pd.isnull(df_train).sum())
print(pd.isnull(df_test).sum())

print(df_train.shape)
print(df_test.shape)

print(df_test.head())
print(df_train.head())

# Guardar el DataFrame en un archivo CSV
# El parámetro index=False evita que los índices del DataFrame
# se guarden como una columna en el archivo CSV
df_train.to_csv('train_procesado.csv', index=False, sep=',', encoding='utf-8')


# NORMALIZACIÓN
# Las variables categóricas NO se normalizan; se codifican (One-Hot, Label Encoding, etc.)
from sklearn.preprocessing import MinMaxScaler

# Normalizar solo la columna Fare (la única numérica continua)
# Se usa fit_transform para el train y transform para test (buena práctica en ML)
scaler = MinMaxScaler(feature_range=(0,1))
df_train['Fare_normalized'] = scaler.fit_transform(df_train[['Fare']])
df_test['Fare_normalized'] = scaler.transform(df_test[['Fare']])

# (Opcional) Si deseas eliminar la columna Fare original:
df_train.drop(columns=['Fare'], inplace=True)
# df_test.drop(columns=['Fare'], inplace=True)

df_train.to_csv('train_procesado_2.csv', index=False, sep=',', encoding='utf-8')