$script_dir = $PSScriptRoot
echo "Script directory: $script_dir"
cd $script_dir\..\repositorio\imagen
$image_repo = $pwd
echo "Image directory: $image_repo"
cd $script_dir\..\image_pipeline
$image_pipeline = $pwd
echo "Image pipeline directory: $image_pipeline"

cd $image_pipeline\IMPaCT_mapping
echo $pwd

echo "Construyendo imagen docker del generador de tablas OMOP"
docker build -t impact-wp4-poc/metadata:0.1 .

docker run --rm  -v ${image_repo}:/input -v ${image_pipeline}\IMPaCT_mapping\config:/config impact-wp4-poc/metadata:0.1

cd $script_dir