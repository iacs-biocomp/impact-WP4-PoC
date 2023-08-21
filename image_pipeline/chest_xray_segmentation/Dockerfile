# Elegir la imagen base
FROM python:3.6

# Establecer el directorio de trabajo dentro del contenedor
WORKDIR /module

# Copiar los archivos de dependencias
COPY requirements.txt .

# Instalar las dependencias del proyecto
RUN pip install -r requirements.txt

# Copiar el contenido del directorio chest_xray_segmentation al contenedor
COPY . /scripts

# Ejecutar el script de segmentacion
ENTRYPOINT ["python", "/scripts/lung_segmentation.py"]