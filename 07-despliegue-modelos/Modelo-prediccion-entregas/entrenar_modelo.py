import pandas as pd
import numpy as np
import joblib
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix

# Cargar datos simulados
df = pd.read_csv("entregas.csv")
print("Datos cargados: filas=", df.shape[0], " columnas=", df.shape[1])
print("Primeras filas:\n", df.head().to_string(index=False))
print("Dtypes:\n", df.dtypes)

# Preprocesamiento
columnas_numericas = ['distancia_km', 'hora_dia']
columnas_categoricas = ['vehiculo_tipo']

preprocesador = ColumnTransformer([
    ('num', StandardScaler(), columnas_numericas),
    ('cat', OneHotEncoder(), columnas_categoricas)
])

# Generar variable objetivo
np.random.seed(42)
df['retraso'] = (df['distancia_km'] > 60).astype(int)  # Regla simplificada
print("Distribución de la etiqueta 'retraso':\n", df['retraso'].value_counts())

X = df.drop(columns=['retraso'])
y = df['retraso']
print("Características (X) primeras filas:\n", X.head().to_string(index=False))
print("Etiqueta (y) primeros valores:\n", y.head().to_string())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipeline = Pipeline([
    ('preprocesamiento', preprocesador),
    ('modelo', LogisticRegression())
])

pipeline.fit(X_train, y_train)

# Evaluación
accuracy = pipeline.score(X_test, y_test)
print(f"Precisión en test: {accuracy:.2f}")
from sklearn.metrics import accuracy_score
y_pred = pipeline.predict(X_test)
print("Classification report:\n", classification_report(y_test, y_pred))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

# Guardar el modelo en carpeta models
os.makedirs("models", exist_ok=True)
joblib.dump(pipeline, 'models/modelo_entregas.pkl')
print("Modelo guardado en models/modelo_entregas.pkl")
