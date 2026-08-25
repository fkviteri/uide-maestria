import numpy as np
import pandas as pd
# Generar datos normales de funcionamiento
np.random.seed(42)
normal = pd.DataFrame({
'temperatura': np.random.normal(loc=70, scale=5, size=1000),
'vibracion': np.random.normal(loc=0.5, scale=0.1, size=1000),
'retraso': 0 # sin falla
})

# Generar datos con falla (deriva simulada)
falla = pd.DataFrame({
'temperatura': np.random.normal(loc=85, scale=5, size=200),
'vibracion': np.random.normal(loc=0.9, scale=0.1, size=200),
'retraso': 1 # falla detectada
})

# Dataset completo
df = pd.concat([normal, falla]).sample(frac=1).reset_index(drop=True)
df.to_csv("datos_sensor.csv", index=False)