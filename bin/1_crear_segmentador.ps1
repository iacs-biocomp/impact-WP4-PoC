$script_dir = $PSScriptRoot
echo "Script directory: $script_dir"
cd $script_dir\..\repositorio\imagen
$image_repo = $pwd
echo "Image directory: $image_repo"
cd $script_dir\..\image_pipeline
$image_pipeline = $pwd
echo "Image pipeline directory: $image_pipeline"
cd $image_pipeline\chest_xray_segmentation 
echo $pwd

echo "Construyendo imagen docker del segmentador"
docker build -t impact-wp4-poc/segmentador:0.1 .

docker run --rm -v ${image_repo}:/input impact-wp4-poc/segmentador:0.1
cd $script_dir