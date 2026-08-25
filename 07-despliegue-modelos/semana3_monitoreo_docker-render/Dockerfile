# QUÉ: Imagen base ligera de Python 3.10.
# POR QUÉ: "slim" trae lo mínimo necesario -> imagen más chica, build
#      más rápido, menos superficie de ataque.
FROM python:3.10-slim

# QUÉ: Variables de entorno globales del contenedor.
# POR QUÉ:
#   DEBIAN_FRONTEND=noninteractive -> evita que apt-get pida confirmaciones
#      interactivas durante el build (que se colgaría sin nadie para
#      responder).
#   PYTHONUNBUFFERED=1 -> los print() de Python salen inmediatamente en
#      los logs (útil para depurar en Render, donde solo ves logs, no
#      la terminal).
#   PIP_NO_CACHE_DIR=1 -> pip no guarda caché de paquetes descargados,
#      lo que reduce el tamaño final de la imagen.
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# QUÉ: Dependencias a nivel de sistema operativo (no de Python).
# POR QUÉ: build-essential es necesario porque algunas librerías de
#      Python (como parte de scikit-learn/numpy) compilan código C
#      durante la instalación.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential git && \
    rm -rf /var/lib/apt/lists/*

# QUÉ: Crea un usuario sin privilegios de administrador.
# POR QUÉ: Por defecto Docker corre como root. Ejecutar la app con un
#      usuario limitado reduce el daño posible si alguien explota una
#      vulnerabilidad -- principio de "menor privilegio".
RUN useradd -m -d /home/appuser -s /bin/bash appuser

WORKDIR /app

# QUÉ: Copiamos SOLO requirements.txt primero, instalamos, y recién
#      después copiamos el resto del código.
# POR QUÉ: Docker cachea cada instrucción por capas. Si copiaras todo
#      el código de una vez, cualquier cambio en main.py invalidaría
#      la caché de "pip install" y tendrías que reinstalar todo de
#      nuevo en cada build. Separándolo así, mientras no cambies
#      requirements.txt, esta capa se reutiliza -> builds mucho más
#      rápidos.
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# QUÉ: Carpetas de runtime que la app necesita en tiempo de ejecución.
# POR QUÉ: Se crean con permisos del usuario appuser para que la app
#      pueda escribir CSVs/reportes sin errores de permisos.
RUN mkdir -p /app/data /app/reports /app/models && \
    chown -R appuser:appuser /app

# QUÉ: Copiamos el resto del código fuente, ya con el dueño correcto.
COPY --chown=appuser:appuser . /app

# QUÉ: A partir de aquí, todo corre como appuser, no como root.
USER appuser

# QUÉ: Entrena el modelo UNA VEZ, durante el build de la imagen.
# POR QUÉ: Así el contenedor arranca en producción con el modelo ya
#      listo -- ningún usuario espera a que se entrene en su primer
#      request. También hace que la imagen sea reproducible: siempre
#      contiene el mismo modelo, sin importar cuántas veces se reinicie
#      el contenedor.
RUN python train_model.py

# ============================================================
# CONFIGURACIÓN DE PUERTO -- el punto clave de esta semana
# ============================================================
# QUÉ: Declaramos un valor de PORT por defecto (10000, el que usa Render).
# POR QUÉ: Esta línea es solo un valor por defecto -- Render, al arrancar
#      el contenedor, INYECTA su propia variable de entorno PORT
#      (normalmente 10000, aunque puede variar), que SOBRESCRIBE este
#      valor automáticamente. Nosotros no elegimos ese número; Render sí.
ENV PORT=10000

# QUÉ: Documenta (a nivel informativo) qué puerto expone el contenedor.
# POR QUÉ: EXPOSE no abre puertos por sí solo -- es metadata para quien
#      lea el Dockerfile o use "docker run -P". El mapeo real de puertos
#      se hace con "-p" al correr el contenedor (ver README).
EXPOSE $PORT

# ============================================================
# CMD -- formato SHELL (no exec) para poder resolver $PORT
# ============================================================
# QUÉ: Comando que arranca la API cuando el contenedor inicia.
# POR QUÉ FORMATO SHELL Y NO EXEC:
#   CMD ["uvicorn", "main:app", "--port", "$PORT"]  <- NO funciona.
#      Docker pasa "$PORT" como texto literal (los 5 caracteres),
#      no lo interpreta como variable.
#   CMD uvicorn main:app --host 0.0.0.0 --port $PORT  <- SÍ funciona.
#      Este formato (sin corchetes) hace que Docker ejecute el comando
#      a través de "/bin/sh -c", que SÍ sabe expandir variables de
#      entorno antes de correr uvicorn.
# PARA QUÉ: Esto es lo que permite que el MISMO Dockerfile funcione:
#   - En tu PC: "docker run -p 8000:8000 -e PORT=8000 sensores-api"
#   - En Render: Render inyecta su propio PORT (ej. 10000) y todo
#     funciona sin que tengas que tocar el Dockerfile.
CMD uvicorn main:app --host 0.0.0.0 --port $PORT
