# Utilizar la imagen base de R
FROM r-base:4.3.0
#added
ADD sources.list /etc/apt/sources.list.d/
RUN echo "deb https://deb.debian.org/debian sid main" > /etc/apt/sources.list.d/debian-unstable.list
# Instalar las dependencias necesarias para RPostgreSQL
RUN apt-get update && grep -ri stable /etc
RUN apt-get install -y libfreetype6=2.12.1+dfsg-5 libfreetype-dev=2.12.1+dfsg-5 libfreetype6-dev=2.12.1+dfsg-5
RUN apt-get install -y libasound2-data=1.2.11-1 
RUN apt-get install -y libpq-dev libcurl4-openssl-dev libfontconfig1-dev libxml2-dev libharfbuzz-dev libfribidi-dev libpng-dev libtiff5-dev libjpeg-dev
# Dependencias de CDMInspection
RUN apt-get install -y libgirepository1.0-dev libcairo2-dev libbz2-dev liblzma-dev libssl-dev 

# Install OpenJDK-8
RUN apt-get update && \
    apt-get install -y openjdk-8-jdk ant && \
    ln -s /usr/lib/jvm/java-8-openjdk-amd64 /usr/lib/jvm/default-java && \
    apt-get clean;    
# Fix certificate issues
RUN apt-get update && \
    apt-get install ca-certificates-java && \
    apt-get clean && \
    update-ca-certificates -f;

# Setup JAVA_HOME -- useful for docker commandline
ENV JAVA_HOME /usr/lib/jvm/java-8-openjdk-amd64/
RUN export JAVA_HOME

# Dependencias de CDMInspection, hay que reinstalar porque el install de openjdk las borra
RUN apt-get install -y libcairo2-dev libpng16-16 libgirepository1.0-dev libbz2-dev liblzma-dev libssl-dev libharfbuzz-dev

# Instalar las librerías DBI y RPostgreSQL en R
RUN Rscript -e "install.packages(c('DBI', 'RPostgreSQL'))"
ADD CatalogueExport.bigan_1.0.2_R_x86_64-pc-linux-gnu.tar.gz /tmp
ADD postgresql-42.2.18.jar /usr/lib/jdbcDrivers/
RUN Rscript -e "install.packages(c('curl', 'systemfonts', 'xml2', 'textshaping', 'roxygen2', 'rversions', 'urlchecker', 'ragg', 'httr', 'pkgdown'))"
RUN Rscript -e "if (!require('devtools')) install.packages('devtools')"
RUN Rscript -e "install.packages(c('hms', 'progress', 'vroom', 'dplyr', 'tidyr', 'readr', 'dbplyr', 'backports', 'checkmate'))"
# CDMInspection
RUN Rscript -e "install.packages(c('officer', 'gdtools', 'flextable', 'gfonts'))"

#RUN Rscript -e "install.packages('remotes')"
#RUN Rscript -e "remotes::install_github('ohdsi/SqlRender')"

RUN R CMD javareconf
RUN Rscript -e "install.packages('rJava', repos='https://rforge.net')"
RUN Rscript -e "install.packages('SqlRender') "
RUN Rscript -e "install.packages('textshaping') "

ENV DATABASECONNECTOR_JAR_FOLDER /usr/lib/jdbcDrivers

RUN Rscript -e "install.packages(c('DatabaseConnector','ParallelLogger','rjson', 'remotes', 'R.utils'))"
RUN Rscript -e "remotes::install_github('OHDSI/Achilles')"
RUN Rscript -e "packageVersion('curl')"
RUN Rscript -e "remotes::install_github('OHDSI/DataQualityDashboard')"
RUN Rscript -e "remotes::install_github('OHDSI/ROhdsiWebApi')"
RUN Rscript -e "remotes::install_github('EHDEN/CdmInspection')"
RUN Rscript -e "install.packages('/tmp/CatalogueExport.bigan',repos=NULL, type='source')"

# Copiar tus scripts R al contenedor
# COPY scripts /scripts

# Establecer el directorio de trabajo
# WORKDIR /scripts

# Establecer el comando por defecto
# CMD ["Rscript", "conexion.R"]

#CMD ["Rscript", "/your_script.R"]



