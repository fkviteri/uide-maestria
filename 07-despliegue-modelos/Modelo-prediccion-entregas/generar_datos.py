import pandas as pd
import numpy as np

# Simular eventos en tiempo real
def generar_evento():
    return {
        'distancia_km': np.random.uniform(1, 100),
        'vehiculo_tipo': np.random.choice(['camion', 'furgoneta', 'auto', 'bicicleta']),
        'hora_dia': np.random.randint(0, 24)
    }

if __name__ == "__main__":
    # Fijar semilla para el generador de números aleatorios
    np.random.seed(42)
    datos = [generar_evento() for _ in range(500)]
    # Mostrar un ejemplo del formato generado (dict)
    print("Primer registro generado:", datos[0])

    # Convertir a DataFrame y mostrar primeras filas y tipos
    df = pd.DataFrame(datos)
    print("DataFrame (primeras filas):")
    print(df.head().to_string(index=False))
    print("Dtypes:")
    print(df.dtypes)

    # Guardar en CSV
    df.to_csv("entregas.csv", index=False)
    print(f"Generados {len(df)} eventos en entregas.csv")
