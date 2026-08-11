# Modelo de Predicción de Retrasos en Entregas

Ejercicio Práctico Guiado Semana 1: Despliegue de Modelos de IA en Producción.

Simula el flujo de una empresa de logística que predice la probabilidad de
retraso en una entrega a partir de la distancia del envío, el tipo de
vehículo asignado y la hora del día en que inicia la entrega.

## Estructura del proyecto

```
.
├── app/
│   └── main.py            # API FastAPI que expone el modelo
├── models/
│   └── modelo_entregas.pkl  # Modelo entrenado (se genera con entrenar_modelo.py)
├── docker-compose.yml
├── Dockerfile
├── entregas.csv            # Datos sintéticos (se genera con generar_datos.py)
├── entrenar_modelo.py       # Preprocesamiento + entrenamiento del modelo
├── generar_datos.py         # Generación de datos sintéticos en tiempo real
├── requirements.txt
└── README.md
```

## Uso local (sin Docker)

```bash
pip install -r requirements.txt

# 1. Generar datos sintéticos
python generar_datos.py

# 2. Entrenar el modelo y guardarlo en models/
python entrenar_modelo.py

# 3. Levantar la API
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

La documentación interactiva queda disponible en http://localhost:8000/docs

### Ejemplo de consumo del endpoint

```bash
curl -X POST "http://localhost:8000/predecir?distancia_km=75&vehiculo_tipo=camion&hora_dia=14"
```

Respuesta:

```json
{"probabilidad_retraso": 0.87}
```

## Uso con Docker

```bash
# Generar datos y entrenar el modelo antes de construir la imagen
python generar_datos.py
python entrenar_modelo.py

# Construir y correr
docker build -t modelo-entregas:semana1 .
docker run -p 8000:8000 modelo-entregas:semana1
```

O con Docker Compose:

```bash
docker compose up --build
```

## Flujo del proyecto

1. **Simulación de datos y preprocesamiento**: `generar_datos.py` simula
   eventos de entrega en tiempo real y los guarda en `entregas.csv`.
2. **Entrenamiento del modelo**: `entrenar_modelo.py` preprocesa las
   variables (escalado numérico + one-hot encoding) y entrena una regresión
   logística para clasificar si una entrega llegará tarde, guardando el
   pipeline entrenado en `models/modelo_entregas.pkl`.
3. **Despliegue con FastAPI**: `app/main.py` carga el modelo y expone el
   endpoint `POST /predecir`.
4. **Dockerización**: `Dockerfile` empaqueta el servicio para que pueda
   ejecutarse de forma portable en cualquier entorno.
