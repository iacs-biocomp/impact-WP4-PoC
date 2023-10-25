#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
echo "Script directory: $SCRIPT_DIR"

DATA_REPO="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../repositorio/datos" &> /dev/null && pwd )"
echo "Data repository: $IMAGE_PIPELINE"

DOCKER_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../data_pipeline" &> /dev/null && pwd )"
echo "Docker dir: $DOCKER_DIR"

docker compose -f "${DOCKER_DIR}/docker-compose.yml" exec -it omop psql -U omop -d omop -f /data/import_data.sql