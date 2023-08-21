# Pipeline de imagen IMPaCT

Esta pipeline consiste en la ejecución de diversos contenedores docker que harán las funciones de segmentación de las imágenes, extracción de características radiómicas y mapeo a las tablas
con la estructura de la extensión de omop utilizada para IMPaCT.

## Estructura de carpetas

Para que esta pipeline funcione correctamente se requiere una estructuración de carpetas específica:

    Repositorio
        |_Pacientes*
                |_Procedimientos**
                        |_original***
                        |_segmentation****
                        |_radiomics*****
                        |_dicom_headers******
                        |_omop_tables*******

    *: carpetas que contendrán cada paciente. Deben estar llamadas con el patient_id.
    **: carpetas que contendrán cada procedimiento dentro del paciente. Deben estar llamadas con el procedure_ocurrence_id.
    ***: carpeta con el DICOM de la placa de tórax (DICOM de una placa de tórax en posición AP).
    ****: carpeta donde se guardará la segmentacion.
    *****: carpeta donde se guardarán los valores de radiómica.
    ******: carpeta donde se guardarán las tags dicom necesarias.
    *******: carpeta donde se guardarán las tablas generadas con el formato de la extensión de omop.

Por ejemplo un procedimiento con id 1234 y del paciente 9876 debería tener la siguiente estructura antes de empezar con el análisis:

    Repositorio
        |_9876
            |_1234
                |_original -> img.dcm
                |_segmentation
                |_radiomics
                |_dicom_headers
                |_omop_tables

## 1. Segmentación

Este paso de la pipeline, generará una máscara de segmentación de los pulmones a partir de una imagen de entrada (.dcm), que deberá ser una
RX de tórax en posición AnteroPosterior. Este código es una adaptación de uno desarrollado por Quibim ©.
Para empezar, para ejecutar este paso de la pipeline habrá que construir la imagen (este paso no es necesario si ya se ha construido). Para ello habrá que ir hasta el directorio "chest_xray_segmentation" y desde la terminal ejecutar el siguiente comando:

```no-highlight
$ docker build -t <nombre_de_la_imagen>:<tag> .
```

Posteriormente ejecutaremos nuestra imagen montando como volumen el directorio con nuestra estructura definida anteriormente de la siguiente forma:

```no-highlight
$ docker run --rm  -v <ruta/absoluta/del/directorio/estructurado>:/input <nombre_de_la_imagen>:<tag> 
```

Cuando ejecutemos esto ya se habrá creado en nuestra carpeta segmentation el archivo .nrrd con la máscara y podremos pasar al siguiente paso.

## 2. Radiómica

Esta segunda etapa de la pipeline extraerá tanto las características radiómicas de la imagen como las dicom tags necesarias para
las tablas definidas en nuestra extensión OMOP para el proyecto IMPaCT. Para poder ser ejecutada deberá existir tanto la imagen .dcm de la RX de tórax así como la máscara generada en el paso anterior
Para este paso también deberemos construir la imagen de extracción de radiómica si no lo hemos hecho. Para ello habrá que ir hasta el directorio "IMPaCT_radiomics" y desde la terminal ejecutar el siguiente comando:

```no-highlight
$ docker build -t <nombre_de_la_imagen>:<tag> .
```

Posteriormente ejecutaremos nuestra imagen montando como volumen el directorio con nuestra estructura definida anteriormente de la siguiente forma:

```no-highlight
$ docker run --rm  -v <ruta/absoluta/del/directorio/estructurado>:/input <nombre_de_la_imagen>:<tag> 
```

Cuando ejecutemos esto ya se habrá creado en nuestra carpeta radiomics el archivo .csv con los valores de radiómica y podremos pasar al siguiente paso.

## 3. Mapeo a extensión OMOP

En esta última etapa, a partir de todo lo generado anteriormente (info de dicom tags y radiómicas), se construirán unas tablas con el formato de nuestras tablas de la extensión de OMOP. 
Como en los dos anteriores, en este paso también hay que construir la imagen si no se ha hecho. Para ello habrá que ir hasta el directorio "IMPaCT_mapping" y desde la terminal ejecutar el siguiente comando:

```no-highlight
$ docker build -t <nombre_de_la_imagen>:<tag> .
```

Ahora debemos dirigirnos al directorio "config" y modificar la id inicial que se nos habrá asignado en los tres campos: "imaging_occurrence_id_start", "imaging_feature_id_start", "measurement_id_start"

Posteriormente ejecutaremos nuestra imagen montando como volumenes el directorio con nuestra estructura definida anteriormente y el directorio "config" de la siguiente forma:

```no-highlight
$ docker run --rm  -v <ruta/absoluta/del/directorio/estructurado>:/input  -v <ruta/absoluta/del/directorio/config>:/config <nombre_de_la_imagen>:<tag> 
```

Cuando ejecutemos esto ya se habrá creado en nuestra carpeta omop_tables los csvs con la estructura de las tablas a rellenar en la extensión de omop