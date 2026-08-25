# Semana 2 - Monitoreo del Modelo e IA aplicada a la Industria 4.0

## Descripción

Proyecto correspondiente a la Semana 2 de la materia **Despliegue de Modelos de IA**.

El proyecto simula datos de sensores industriales (temperatura y vibración),
entrena un modelo de clasificación para identificar posibles fallas y utiliza
**Evidently** para detectar cambios en la distribución de los datos (data
drift). Adicionalmente, incluye una lógica sencilla de alertas de
mantenimiento predictivo basada en límites de k-sigma, y expone todo el
sistema mediante una API con **FastAPI** y una interfaz visual en HTML.

## Estructura del proyecto

```
semana2_monitoreo_ia/
├── data/               # Datos simulados de sensores (datos_sensor.csv)
├── models/             # Modelo entrenado (modelo_sensores.pkl)
├── reports/            # Reporte de drift (Evidently) y app visual HTML
├── src/
│   ├── generar_datos.py     # Genera los datos sintéticos de sensores
│   ├── entrenar_modelo.py   # Entrena y guarda el modelo de clasificación
│   ├── monitorear_drift.py  # Genera el reporte de drift con Evidently
│   └── app.py                # API con FastAPI (predicción y monitoreo)
├── README.md
└── requirements.txt
```

## Instalación

```
pip install -r requirements.txt
```

## Orden de ejecución

```
python src/generar_datos.py
python src/entrenar_modelo.py
python src/monitorear_drift.py
python -m uvicorn src.app:app --host 127.0.0.1 --port 8000 --reload
```

## URLs

- Aplicación: http://127.0.0.1:8000/
- Swagger: http://127.0.0.1:8000/docs
- Health: http://127.0.0.1:8000/health
- Reporte Evidently: http://127.0.0.1:8000/report

## Interpretación del modelo

- `0` = funcionamiento normal
- `1` = posible falla

El modelo se entrena con dos sensores (temperatura y vibración) y predice la
probabilidad de que las lecturas actuales correspondan a una condición de
falla. El sistema de monitoreo compara un nuevo lote de datos contra los
datos de referencia para detectar si hubo un cambio relevante en su
distribución (data drift), y calcula límites simples de alerta usando la
media y desviación estándar del conjunto de referencia.
