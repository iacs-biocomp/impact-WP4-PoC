#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
echo "Script directory: $SCRIPT_DIR"


# creamos la estructura de tablas
cd $SCRIPT_DIR/../repositorio/imagen
echo "Borrando carpetas bajo "$PWD

rm -rf $PWD/*


