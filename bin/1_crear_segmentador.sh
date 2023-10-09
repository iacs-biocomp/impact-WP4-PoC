#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
echo "Script directory: $SCRIPT_DIR"

IMAGE_REPO="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../repositorio/imagen" &> /dev/null && pwd )"
echo "Image repository: $IMAGE_REPO"

IMAGE_PIPELINE="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../image_pipeline" &> /dev/null && pwd )"
echo "Image pipeline sources: $IMAGE_PIPELINE"

cd "$IMAGE_PIPELINE/chest_xray_segmentation"
echo "$PWD"

echo "Construyendo imagen docker del segmentador"
docker build -t impact-wp4-poc/segementador:0.1 .


docker run --rm -v "${IMAGE_REPO}":/input impact-wp4-poc/segementador:0.1
