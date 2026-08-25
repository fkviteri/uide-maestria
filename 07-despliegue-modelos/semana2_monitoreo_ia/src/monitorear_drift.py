from pathlib import Path
import numpy as np
import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

root = Path(__file__).resolve().parents[1]

# Cargar datos originales
df = pd.read_csv(root / "data" / "datos_sensor.csv")

# Conjunto de referencia: 70% de los datos originales (mismo criterio que el entrenamiento)
ref = df.sample(frac=0.7, random_state=42)[['temperatura', 'vibracion']]

# Nuevo batch simulado con una distribucion distinta (deriva deliberada)
np.random.seed(7)
curr = pd.DataFrame({
    'temperatura': np.random.normal(loc=90, scale=5, size=200),
    'vibracion': np.random.normal(loc=1.0, scale=0.1, size=200)
})

# Ejecutar el reporte de Data Drift comparando referencia vs batch actual
report = Report(metrics=[DataDriftPreset()])
resultado = report.run(reference_data=ref, current_data=curr)

# Guardar el reporte HTML
(root / "reports").mkdir(parents=True, exist_ok=True)
ruta_reporte = root / "reports" / "reporte_drift.html"
resultado.save_html(str(ruta_reporte))
print(f"OK -> {ruta_reporte}")

# Resumen rapido de drift (cuantas columnas mostraron deriva)
resumen = resultado.dict()
valor = resumen["metrics"][0]["value"]
print(f"Columnas con drift: {valor.get('count')} | Proporcion: {valor.get('share')}")

# --- Alertas de mantenimiento predictivo (limites k-sigma) ---
k_sigma = 2

media_referencia_temp = ref['temperatura'].mean()
desviacion_referencia_temp = ref['temperatura'].std()
limite_superior_temp = media_referencia_temp + k_sigma * desviacion_referencia_temp

media_referencia_vib = ref['vibracion'].mean()
desviacion_referencia_vib = ref['vibracion'].std()
limite_superior_vib = media_referencia_vib + k_sigma * desviacion_referencia_vib

media_temperatura_actual = curr['temperatura'].mean()
media_vibracion_actual = curr['vibracion'].mean()

print(f"Limite superior temperatura: {limite_superior_temp:.2f} | Media actual: {media_temperatura_actual:.2f}")
print(f"Limite superior vibracion: {limite_superior_vib:.2f} | Media actual: {media_vibracion_actual:.2f}")

if media_temperatura_actual > limite_superior_temp:
    print("ALERTA: temperatura fuera del rango esperado.")

if media_vibracion_actual > limite_superior_vib:
    print("ALERTA: vibración fuera del rango esperado.")