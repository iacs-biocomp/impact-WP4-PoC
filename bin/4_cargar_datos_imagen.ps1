$script_dir = $PSScriptRoot
echo "Script directory: $script_dir"

cd $script_dir\..\repositorio\imagen
$image_repo = $pwd
echo "Image directory: $image_repo"

cd $script_dir\..\repositorio\datos
$data_repo = $pwd
echo "Data directory: $data_repo"

cd $script_dir\..\data_pipeline
$docker_dir = $pwd
echo "Docker dir: $docker_dir"

mkdir "${data_repo}/imagen"

cd $image_repo
echo $pwd

Get-ChildItem -Directory | ForEach-Object {
	cd $($_.FullName)
	Echo "entrando en $($_.FullName)"
	Get-ChildItem -Directory | ForEach-Object {
           Copy-Item -Path "$($_.FullName)\omop_tables\*.csv" -Destination $data_repo\imagen\
           docker compose -f "${docker_dir}/docker-compose.yml" exec -it omop psql -U omop -d omop -f /data/import_image_data.sql
	}
}

cd $script_dir
