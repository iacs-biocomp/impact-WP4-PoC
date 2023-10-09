#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
echo "Script directory: $SCRIPT_DIR"

IMAGE_REPO="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../repositorio/imagen" &> /dev/null && pwd )"
echo "Image repository: $IMAGE_REPO"

IMAGE_PIPELINE="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../image_pipeline" &> /dev/null && pwd )"
echo "Image pipeline sources: $IMAGE_PIPELINE"

cd "$IMAGE_PIPELINE/IMPaCT_mapping"
echo "$PWD"

echo "Construyendo imagen docker del metadatador"
docker build -t impact-wp4-poc/metadata:0.1 .


docker run --rm  -v "${IMAGE_REPO}":/input -v "${IMAGE_PIPELINE}/IMPaCT_mapping/config":/config impact-wp4-poc/metadata:0.1