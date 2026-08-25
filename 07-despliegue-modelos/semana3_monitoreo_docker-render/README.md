# API Sensores — Despliegue en Docker y Render

API REST (FastAPI + Gradio) que predice fallas industriales a partir de
lecturas de temperatura y vibración de sensores. Reutiliza el modelo
entrenado en la Semana 2 (RandomForestClassifier).

## Estructura del proyecto

```
├── Dockerfile          <- En la raíz (obligatorio para Render)
├── main.py             <- API FastAPI + dashboard Gradio
├── train_model.py       <- Entrena y guarda el modelo (se corre en el build)
├── requirements.txt
├── test_api.py          <- Prueba /health, /predict, /monitor
├── .dockerignore
└── README.md
```

## Cómo funciona el puerto en cada entorno

El puerto **nunca está fijo en el código** — siempre se lee de la
variable de entorno `PORT`, con `8000` como valor por defecto:

| Entorno | Cómo se define `PORT` |
|---|---|
| `python main.py` (sin Docker) | Si no defines `PORT`, usa `8000` por defecto |
| Docker local | Tú lo defines con `-e PORT=8000` al correr el contenedor |
| Render | Render lo inyecta automáticamente (normalmente `10000`) |

---

## 1. Probar localmente sin Docker

```bash
pip install -r requirements.txt
python train_model.py          # genera data/ y models/
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Abrir:
- http://localhost:8000/docs (Swagger)
- http://localhost:8000/ui (Dashboard Gradio)
- http://localhost:8000/health

## 2. Probar con Docker localmente

```bash
# Construir la imagen (el build ya entrena el modelo internamente)
docker build -t sensores-api .

# Ejecutar, mapeando el puerto 8000 de tu PC al PORT=8000 del contenedor
docker run --rm -p 8000:8000 -e PORT=8000 sensores-api
```

Verificar:

```bash
curl http://localhost:8000/health
curl -X POST "http://localhost:8000/predict?temperatura=70&vibracion=0.5"
curl -X POST "http://localhost:8000/predict?temperatura=90&vibracion=1.0"

# o con el script de pruebas
python test_api.py http://localhost:8000
```

> Nota: si no pasas `-e PORT=8000`, el contenedor usará el valor por
> defecto definido en el Dockerfile (`ENV PORT=10000`), y en ese caso
> deberías mapear `-p 8000:10000` en vez de `-p 8000:8000`.

## 3. Desplegar en Render

### Paso 1 — Subir a GitHub

```bash
git init
git add .
git commit -m "API sensores con FastAPI, Gradio y Docker"
git remote add origin https://github.com/TU_USUARIO/sensores-api-render.git
git branch -M main
git push -u origin main
```

### Paso 2 — Crear el Web Service en Render

1. Crear cuenta en https://render.com (con GitHub, sin tarjeta de crédito).
2. "New +" → "Web Service" → conectar el repositorio.
3. Configurar:

   | Campo | Valor |
   |---|---|
   | Name | sensores-api |
   | Runtime | Docker (auto-detectado) |
   | Instance Type | Free |

4. "Deploy Web Service". Render **inyecta su propia variable `PORT`**
   automáticamente — no hace falta configurarla a mano en el dashboard.

### Paso 3 — Verificar

```bash
python test_api.py https://sensores-api-XXXX.onrender.com
```

- `/health` → JSON con `status: ok`
- `/docs` → Swagger interactivo
- `/ui` → Dashboard Gradio

### Cold start (free tier)

El servicio se duerme tras ~15 min de inactividad. La primera petición
tras dormir tarda 30-60 segundos. Antes de una demo, hacer una petición
a `/health` unos minutos antes para "despertar" el servicio.

### Si el dashboard Gradio no carga en /ui detrás de Render

Agregar en el dashboard de Render, en Environment Variables:

```
GRADIO_ROOT_PATH = https://sensores-api-XXXX.onrender.com/ui
```

(`main.py` ya está preparado para leer esta variable automáticamente.)
