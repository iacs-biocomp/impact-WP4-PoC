#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
echo "Script directory: $SCRIPT_DIR"

IMAGE_REPO="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../repositorio/imagen" &> /dev/null && pwd )"
echo "Image repository: $IMAGE_REPO"

IMAGE_PIPELINE="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../image_pipeline" &> /dev/null && pwd )"
echo "Image pipeline sources: $IMAGE_PIPELINE"

cd "$IMAGE_PIPELINE/IMPaCT_radiomics"
echo "$PWD"

if [ $# -gt 0 ] && [ $1 = "-create" ] 
then
    echo "Construyendo imagen docker del extractor de radiómica"
    docker build -t impact-wp4-poc/radiomics:0.1 .
fi

docker run --rm  -v "${IMAGE_REPO}":/input impact-wp4-poc/radiomics:0.1
