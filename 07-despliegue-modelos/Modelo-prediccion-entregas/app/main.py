from fastapi import FastAPI
import joblib
import pandas as pd
import os
from pathlib import Path

# Ruta base: carpeta del proyecto
BASE_DIR = Path(__file__).resolve().parents[1]

# Ruta por defecto del modelo
DEFAULT_MODEL_PATH = BASE_DIR / "models" / "modelo_entregas.pkl"

# Permite sobreescribir con variable de entorno
MODEL_PATH = Path(os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH)))

# Carga del modelo
modelo = joblib.load(MODEL_PATH)
print(f"Modelo cargado desde: {MODEL_PATH}")

app = FastAPI(
    title="API de Predicción de Entregas",
    description="""
Predice la probabilidad de retraso en entregas logísticas.

**Grupo 7:**

- Azadobay Cuitopala Jose Daniel
- Maldonado de la Cruz Walter Javier 
- Viteri Ayala Flavia Kamila
""",
    version="1.0.0"
)

@app.get("/")
def root():
    return {"mensaje": "API de predicción de entregas activa"}

from typing import Literal

@app.post("/predecir")
def predecir(
    distancia_km: float,
    vehiculo_tipo: Literal["camion", "furgoneta", "auto", "bicicleta"],
    hora_dia: int
):
    df = pd.DataFrame([[distancia_km, vehiculo_tipo, hora_dia]],
                       columns=['distancia_km', 'vehiculo_tipo', 'hora_dia'])
    print("Entrada recibida para predecir:\n", df.to_string(index=False))
    resultado = modelo.predict_proba(df)[0][1]
    salida = {"probabilidad_retraso": round(float(resultado), 2)}
    print("Salida de la predicción:\n", salida)
    return salida
