$script_dir = $PSScriptRoot
echo "Script directory: $script_dir"

cd $script_dir\..\repositorio\dicom_originales
$source_dir = $pwd
echo "Source directory: $source_dir"


if (Test-Path -Path $script_dir\..\repositorio\imagen) {
    "Path exists!"
} else {
    mkdir $script_dir\..\repositorio\imagen
}
cd $script_dir\..\repositorio\imagen
$target_dir = $pwd
echo "Target directory: $target_dir"
cd $script_dir 

$input_file="$source_dir\estructura_imagen.csv"

$reader = [System.IO.File]::OpenText($input_file)
$reader.ReadLine()
while($null -ne ($line = $reader.ReadLine())) {
   $splitArray = $line.Split(";")
   $person = $splitArray[0]
   $image = $splitArray[1]
   $procedure = $splitArray[2]
   Echo "[ $person - $image - $procedure ]"

   if (Test-Path -Path $target_dir\$person) {
      Echo "directorio $target_dir\$person ya existe!"
   } else {
      echo "Creando estructura de carpetas para el paciente $person"
      mkdir "$target_dir\$person"
   }

   if (-Not(Test-Path -Path $target_dir\$person\$procedure) ) {
      mkdir "$target_dir\$person\$procedure"
      mkdir "$target_dir\$person\$procedure\segmentation"
      mkdir "$target_dir\$person\$procedure\radiomics"
      mkdir "$target_dir\$person\$procedure\dicom_headers"
      mkdir "$target_dir\$person\$procedure\omop_tables"
      mkdir "$target_dir\$person\$procedure\original"
   }   

   if (Test-Path -Path $source_dir\$image.dcm -PathType Leaf) {
      Copy-Item -Path $source_dir\$image.dcm -Destination $target_dir\$person\$procedure\original\$image.dcm
   }

}
