# Usar una imagen base de Python 3.12.6
FROM python:3.12.6-slim

# Establecer el directorio de trabajo dentro del contenedor
WORKDIR /app

# Instalar las dependencias del sistema necesarias para OpenCV y otras bibliotecas
RUN apt-get update && apt-get install -y \
    poppler-utils \
    libpoppler-cpp-dev \
    pkg-config \
    libgl1 \
    libglib2.0-0 \
    libgstreamer1.0-0 \
    libgstreamer-plugins-base1.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Copiar el archivo de requisitos al contenedor
COPY requirements.txt .

# Instalar las dependencias de Python (incluye Flask)
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el resto de la aplicación al contenedor
COPY . .

# Exponer el puerto donde el servicio Flask estará escuchando
EXPOSE 5000

# Establecer el comando predeterminado para iniciar Flask
CMD ["python3", "app.py"]
