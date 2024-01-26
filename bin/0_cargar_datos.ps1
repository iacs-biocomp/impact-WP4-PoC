$script_dir = $PSScriptRoot
echo "Script directory: $script_dir"
cd $script_dir\..\repositorio\datos
$data_dir = $pwd
echo "Data directory: $data_dir"
cd $script_dir\..\data_pipeline
$docker_dir = $pwd
echo "Docker repository: $docker_dir"
cd $script_dir 

docker compose -f "$docker_dir/docker-compose.yml" exec -it omop psql -U omop -d omop -f /data/import_data.sql