**Update 15-03**

- Se ha añadido el archivo cdm_source.sql, este realiza una inserción en la tabla cdm_source del esquema OMOP la cual indica la versión y datos generales del CDM. Modificar dicho archivo con los datos correspondientes
- Se ha modificado el 0_cargar_datos añadiendo la ejecución de este .sql
> [!NOTE]
> Si ya se tiene poblada la BBDD hay que ejecutar únicamente el comando sobre el contenedor de omop
>
> docker exec -it omop psql -U omop -d omop -f /data/cdm_source.sql

- En r-scripts se ha creado un archivo denominado execute.R, este se encarga de ejecutar tanto Achilles, como CDMInspection, como CatalogueExport
- El Dockerfile de R contiene ahora la instalación de nuevos paquetes y librerías. Asimismo, se ha actualizado el sources.list
- El docker-compose levanta sendos contenedores y lanza el execute.R
> [!TIP]
> Si se quiere visualizar la ejecución del archivo interior
> 
> docker logs r-base
- Por último, en la carpeta r-scripts/envio se generan los archivos de interés.



