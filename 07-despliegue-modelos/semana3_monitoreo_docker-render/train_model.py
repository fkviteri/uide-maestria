"""
train_model.py
---------------
     Genera datos simulados de sensores industriales (temperatura, vibración)
     y entrena un RandomForestClassifier que predice si habrá un "retraso"
     (falla) según esas lecturas.

     Este mismo modelo (mismo dataset, mismos hiperparámetros) es el
     que ya usamos en la Semana 2 para el monitoreo con Evidently. Aquí
     lo reutilizamos tal cual, solo que ahora en vez de monitorearlo,
     lo exponemos como servicio de predicción.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# -------------------- Rutas del proyecto --------------------
# QUÉ: Definimos carpetas relativas a la ubicación de este archivo.
# POR QUÉ: Así el script funciona igual sin importar desde dónde se ejecute
#          (tu PC, el contenedor Docker, o el servidor de Render).
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = DATA_DIR / "datos_sensor.csv"
MODEL_PATH = MODELS_DIR / "modelo_sensores.pkl"


def generar_datos():
    """
          Crea un dataset sintético con dos grupos: sensores en estado
         "normal" y sensores en estado de "falla".
          Este CSV sirve como dataset de entrenamiento y también
         como referencia (baseline) para la detección de drift en /monitor.
    """
    np.random.seed(42)

    normal = pd.DataFrame({
        "temperatura": np.random.normal(loc=70, scale=5, size=1000),
        "vibracion": np.random.normal(loc=0.5, scale=0.1, size=1000),
        "retraso": 0,
    })

    falla = pd.DataFrame({
        "temperatura": np.random.normal(loc=85, scale=5, size=200),
        "vibracion": np.random.normal(loc=0.9, scale=0.1, size=200),
        "retraso": 1,
    })

    df = pd.concat([normal, falla], ignore_index=True).sample(
        frac=1, random_state=42
    )
    df.to_csv(CSV_PATH, index=False)
    print(f"OK -> datos guardados en {CSV_PATH}")
    return df


def entrenar_modelo(df: pd.DataFrame):
    """
          Entrena un RandomForestClassifier para predecir la columna
         'retraso' a partir de 'temperatura' y 'vibracion'.
          El modelo entrenado es lo que la API va a "cargar" para
         responder cada petición de /predict.
    """
    X = df[["temperatura", "vibracion"]]
    y = df["retraso"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    modelo = RandomForestClassifier(random_state=42)
    modelo.fit(X_train, y_train)

    acc = accuracy_score(y_test, modelo.predict(X_test))
    print(f"Precisión holdout: {acc:.3f}")

    joblib.dump(modelo, MODEL_PATH)
    print(f"Modelo guardado -> {MODEL_PATH}")


if __name__ == "__main__":
    datos = generar_datos()
    entrenar_modelo(datos)
