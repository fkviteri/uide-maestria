"""
main.py
-------
    API REST (FastAPI) que carga el modelo entrenado en train_model.py
     y expone endpoints de predicción, monitoreo de drift, y un dashboard
     visual con Gradio.

    Separamos esto del entrenamiento (train_model.py) para que la API
     arranque rápido -> solo CARGA el modelo ya entrenado, no lo entrena.

    Esta misma app corre igual en tres contextos distintos:
     1) Local con uvicorn directo (python -m uvicorn main:app ...)
     2) Local dentro de un contenedor Docker
     3) En producción en Render (con puerto asignado dinámicamente)
     El código de la API no necesita saber en cuál de los tres está --
     esa diferencia se resuelve fuera del código, a nivel de Dockerfile
     y de la variable de entorno PORT.
"""

import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# -------------------- Rutas base y archivos --------------------
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
REPORTS_DIR = BASE_DIR / "reports"

for d in (MODELS_DIR, DATA_DIR, REPORTS_DIR):
    try:
        d.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass  # en el contenedor ya se crean estos permisos en el Dockerfile

CSV_PATH = DATA_DIR / "datos_sensor.csv"
MODEL_PATH = MODELS_DIR / "modelo_sensores.pkl"

# -------------------- Red de seguridad: auto-generación --------------------
if not CSV_PATH.exists():
    rng = np.random.default_rng(42)
    normal = pd.DataFrame({
        "temperatura": rng.normal(70, 5, 300),
        "vibracion": rng.normal(0.5, 0.1, 300),
        "retraso": 0,
    })
    falla = pd.DataFrame({
        "temperatura": rng.normal(85, 5, 60),
        "vibracion": rng.normal(0.9, 0.1, 60),
        "retraso": 1,
    })
    df0 = pd.concat([normal, falla], ignore_index=True)
    df0.to_csv(CSV_PATH, index=False)

if not MODEL_PATH.exists():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(CSV_PATH)
    X = df[["temperatura", "vibracion"]]
    y = df["retraso"]
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    clf = RandomForestClassifier(random_state=42).fit(Xtr, ytr)
    joblib.dump(clf, MODEL_PATH)

modelo = joblib.load(MODEL_PATH)

# -------------------- FastAPI --------------------
app = FastAPI(
    title="API Sensores + Dashboard",
    description="API REST para predicción de fallas industriales",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    """Endpoint de latido: confirma que la API y sus archivos están OK."""
    return {
        "status": "ok",
        "model": MODEL_PATH.name,
        "csv": CSV_PATH.exists(),
    }


@app.post("/predict")
def predict(temperatura: float, vibracion: float):
    """Predicción individual: ¿esta lectura indica riesgo de falla?"""
    X = pd.DataFrame(
        [[temperatura, vibracion]], columns=["temperatura", "vibracion"]
    )
    proba = float(modelo.predict_proba(X)[0][1])
    pred = int(proba >= 0.5)
    return {"prediccion": pred, "prob_retraso": round(proba, 4)}


@app.post("/monitor")
def monitor(size: int, t_mean: float, t_std: float, v_mean: float, v_std: float):
    """
    Detección de drift simplificada: compara la media de un lote nuevo
    contra el límite de referencia (media + 2 desviaciones estándar).
    """
    rng = np.random.default_rng(7)
    curr = pd.DataFrame({
        "temperatura": rng.normal(t_mean, t_std, size),
        "vibracion": rng.normal(v_mean, v_std, size),
    })
    ref = pd.read_csv(CSV_PATH)[["temperatura", "vibracion"]]

    t_upper = ref["temperatura"].mean() + 2 * ref["temperatura"].std()
    v_upper = ref["vibracion"].mean() + 2 * ref["vibracion"].std()

    return {
        "limits": {
            "temp_upper": round(float(t_upper), 3),
            "vib_upper": round(float(v_upper), 3),
        },
        "current_means": {
            "temp": round(float(curr["temperatura"].mean()), 3),
            "vib": round(float(curr["vibracion"].mean()), 3),
        },
        "alerts": {
            "temp": bool(curr["temperatura"].mean() > t_upper),
            "vib": bool(curr["vibracion"].mean() > v_upper),
        },
    }


# -------------------- Dashboard (Gradio) --------------------
import gradio as gr
import plotly.express as px
from gradio.routes import mount_gradio_app


def make_dashboard():
    def _predecir_ui(t, v):
        r = predict(temperatura=t, vibracion=v)
        etiqueta = "Retraso" if r["prediccion"] == 1 else "Sin retraso"
        return f"{etiqueta} (p={r['prob_retraso']})"

    def _monitor_ui(sz, tm, ts, vm, vs):
        r = monitor(sz, tm, ts, vm, vs)
        a = r["alerts"]
        return (
            f"Temp>{a['temp']} | Vib>{a['vib']} | "
            f"limites={r['limits']} actuales={r['current_means']}"
        )

    def _graficos():
        df = pd.read_csv(CSV_PATH)
        f1 = px.histogram(df, x="temperatura", nbins=40, title="Hist. temperatura")
        f2 = px.histogram(df, x="vibracion", nbins=40, title="Hist. vibración")
        return f1, f2

    with gr.Blocks(title="Panel de Sensores", analytics_enabled=False) as demo:
        gr.Markdown("## Panel de Sensores")

        with gr.Row():
            with gr.Column():
                temperatura = gr.Slider(50, 110, value=75, step=1, label="Temperatura")
                vibracion = gr.Slider(0.2, 1.5, value=0.8, step=0.01, label="Vibración")
                salida_pred = gr.Label(label="Resultado")
                gr.Button("Predecir").click(
                    _predecir_ui, inputs=[temperatura, vibracion], outputs=salida_pred
                )

            with gr.Column():
                size = gr.Slider(50, 1000, value=200, step=10, label="Tamaño batch")
                t_mean = gr.Slider(60, 110, value=90, step=1, label="Temp. media batch")
                t_std = gr.Slider(1, 10, value=5, step=0.5, label="Temp. std batch")
                v_mean = gr.Slider(0.3, 1.5, value=1.0, step=0.05, label="Vib. media batch")
                v_std = gr.Slider(0.05, 0.5, value=0.1, step=0.01, label="Vib. std batch")
                salida_mon = gr.Label(label="Resumen drift")
                gr.Button("Calcular drift").click(
                    _monitor_ui,
                    inputs=[size, t_mean, t_std, v_mean, v_std],
                    outputs=salida_mon,
                )

        gr.Markdown("## Distribuciones históricas")
        fig_t = gr.Plot()
        fig_v = gr.Plot()
        demo.load(_graficos, None, [fig_t, fig_v])
        gr.Button("Actualizar gráficos").click(_graficos, None, [fig_t, fig_v])

    return demo

GRADIO_ROOT_PATH = os.environ.get("GRADIO_ROOT_PATH", "")

demo = make_dashboard()
app = mount_gradio_app(app, demo, path="/ui", root_path=GRADIO_ROOT_PATH)


# -------------------- Arranque directo --------------------
if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
