# Usamos una imagen oficial y ligera de Python
FROM python:3.10-slim

# Establecer el directorio de trabajo dentro del contenedor
WORKDIR /app

# Instalar dependencias del sistema necesarias
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copiamos TODO el proyecto primero (requirements.txt instala -e ., que
# necesita el codigo fuente presente en este mismo paso)
COPY . /app

# Instalamos las dependencias despues de tener el codigo en el contenedor
RUN pip install --no-cache-dir -r requirements.txt

# Exponer el puerto en el que corre FastAPI
EXPOSE 8000

# Indica a model_loader.py que cargue el modelo directo del .pkl,
# sin pasar por MLflow (que no existe dentro del contenedor)
ENV ENVIRONMENT=docker

# Comando para ejecutar la aplicacion con Uvicorn
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
