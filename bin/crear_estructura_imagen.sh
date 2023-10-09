#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
echo "Script directory: $SCRIPT_DIR"

SOURCE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../repositorio/dicom_originales" &> /dev/null && pwd )"
TARGET_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../repositorio/imagen" &> /dev/null && pwd )"

# creamos la estructura de tablas
if [ ! -d "$TARGET_DIR" ]; then
    echo "Creando carpeta para imagenes"
    mkdir "$TARGET_DIR"
fi


# leemos el fichero "estructura_imagen.csv", y construimos la estructura de carpetas de acuerdo con él
input_file="${SOURCE_DIR}/estructura_imagen.csv"

tail -n +2 "$input_file" | while read -r line
do 
   arrIN=(${line//;/ })
   person=${arrIN[0]}
   image=${arrIN[1]}
   procedure=${arrIN[2]}

   if [ ! -d "$TARGET_DIR/$person" ]; then
      echo "Creando estructura de carpetas para el paciente $person"
      mkdir "$TARGET_DIR/$person"
   fi

   if [ ! -d "$TARGET_DIR/$person/$procedure" ]; then
      mkdir "$TARGET_DIR/$person/$procedure"
      mkdir "$TARGET_DIR/$person/$procedure/segmentation"
      mkdir "$TARGET_DIR/$person/$procedure/radiomics"
      mkdir "$TARGET_DIR/$person/$procedure/dicom_headers"
      mkdir "$TARGET_DIR/$person/$procedure/omop_tables"
      mkdir "$TARGET_DIR/$person/$procedure/original"
   fi
   if [ -f "$SOURCE_DIR/${image}.dcm" ]; then
#      mv -f "$SCRIPT_DIR/../repositorio/dicom_originales/$c.dcm" "$PWD/$c/Procedimientos/original"
      cp -f "$SOURCE_DIR/${image}.dcm" "$TARGET_DIR/$person/$procedure/original"
   fi

done
