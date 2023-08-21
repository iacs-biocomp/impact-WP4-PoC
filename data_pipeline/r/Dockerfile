# Utilizar la imagen base de R
FROM r-base:latest

# Instalar las dependencias necesarias para RPostgreSQL
RUN apt-get update && apt-get install -y libpq-dev

# Instalar las librerías DBI y RPostgreSQL en R
RUN Rscript -e "install.packages(c('DBI', 'RPostgreSQL'))"
ADD CatalogueExport.bigan_1.0.2_R_x86_64-pc-linux-gnu.tar.gz /tmp
if (!require("devtools")) install.packages("devtools")
RUN Rscript -e "install.packages('/tmp/CatalogueExport.bigan',repos=NULL, type='source')"

# Copiar tus scripts R al contenedor
#COPY scripts /scripts

# Establecer el directorio de trabajo
#WORKDIR /scripts

# Establecer el comando por defecto
#CMD ["Rscript", "your_script.R"]

#CMD ["Rscript", "/your_script.R"]   
