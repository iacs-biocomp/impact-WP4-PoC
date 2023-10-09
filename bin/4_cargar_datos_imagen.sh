#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
echo "Script directory: $SCRIPT_DIR"

IMAGE_REPO="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../repositorio/imagen" &> /dev/null && pwd )"
echo "Image repository: $IMAGE_REPO"

DATA_REPO="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../repositorio/datos" &> /dev/null && pwd )"
echo "Data repository: $IMAGE_PIPELINE"

DOCKER_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../data_pipeline" &> /dev/null && pwd )"
echo "Docker dir: $DOCKER_DIR"

mkdir "${DATA_REPO}/imagen"

cd "${IMAGE_REPO}"

for f in * 
do 
    echo "Procesando paciente $f"
    cd "${IMAGE_REPO}/$f"
    for p in *
    do
        cp "$PWD/${p}/omop_tables"/*.csv "${DATA_REPO}/imagen/"
        docker compose -f "${DOCKER_DIR}/docker-compose.yml" exec -it omop psql -U omop -d omop -f /data/import_image_data.sql
    done
done
rm -rf "${DATA_REPO}/imagen"