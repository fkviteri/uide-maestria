"""
test_api.py
-----------
    Prueba los endpoints principales de la API (/health y /predict)
     contra cualquier URL base -- local o desplegada en Render.

    Tener un script de verificación reutilizable evita probar
     "a mano" con curl cada vez, y sirve como evidencia de que el
     despliegue funciona correctamente.

    Uso:
     python test_api.py                          -> prueba localhost:8000
     python test_api.py http://localhost:8000     -> local explícito
     python test_api.py https://sensores-api-XXXX.onrender.com  -> Render
"""

import sys
import requests

# QUÉ: Toma la URL base desde el argumento de línea de comandos, o usa
#      localhost:8000 como valor por defecto.
BASE_URL = sys.argv[1].rstrip("/") if len(sys.argv) > 1 else "http://localhost:8000"


def probar_health():
    r = requests.get(f"{BASE_URL}/health", timeout=30)
    print(f"[/health] status={r.status_code} body={r.json()}")
    assert r.status_code == 200, "El endpoint /health no respondió OK"


def probar_predict(temperatura: float, vibracion: float, etiqueta: str):
    r = requests.post(
        f"{BASE_URL}/predict",
        params={"temperatura": temperatura, "vibracion": vibracion},
        timeout=30,
    )
    print(f"[/predict] {etiqueta}: status={r.status_code} body={r.json()}")
    assert r.status_code == 200, f"Falló /predict para caso '{etiqueta}'"


def probar_monitor():
    r = requests.post(
        f"{BASE_URL}/monitor",
        params={"size": 100, "t_mean": 90, "t_std": 5, "v_mean": 1.0, "v_std": 0.1},
        timeout=30,
    )
    print(f"[/monitor] status={r.status_code} body={r.json()}")
    assert r.status_code == 200, "El endpoint /monitor no respondió OK"


if __name__ == "__main__":
    print(f"Probando API en: {BASE_URL}\n")
    probar_health()
    probar_predict(70, 0.5, "condición normal")
    probar_predict(90, 1.0, "condición de falla")
    probar_monitor()
    print("\n✅ Todas las pruebas pasaron.")
