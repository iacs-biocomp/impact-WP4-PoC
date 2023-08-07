# Utilizar la imagen base de R
FROM r-base:latest

# Instalar las dependencias necesarias para RPostgreSQL
RUN apt-get update && apt-get install -y libpq-dev

# Instalar las librerías DBI y RPostgreSQL en R
RUN Rscript -e "install.packages(c('DBI', 'RPostgreSQL'))"
ADD CatalogueExport.bigan_1.0.2_R_x86_64-pc-linux-gnu.tar.gz /tmp
ADD postgresql-42.2.18.jar /usr/lib/jdbcDrivers/


# Install OpenJDK-8
RUN apt-get update && \
    apt-get install -y openjdk-8-jdk && \
    apt-get install -y ant && \
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

RUN apt-get install -y libcurl4-openssl-dev libfontconfig1-dev libxml2-dev libharfbuzz-dev libfribidi-dev libfreetype6-dev libpng-dev libtiff5-dev libjpeg-dev

RUN Rscript -e "install.packages(c('curl', 'systemfonts', 'xml2', 'textshaping', 'roxygen2', 'rversions', 'urlchecker', 'ragg', 'pkgdown'))"


RUN Rscript -e "if (!require('devtools')) install.packages('devtools')"
RUN Rscript -e "install.packages(c('hms', 'progress', 'vroom', 'dplyr', 'tidyr', 'readr', 'dbplyr', 'backports', 'checkmate'))"

#RUN Rscript -e "install.packages('remotes')"
#RUN Rscript -e "remotes::install_github('ohdsi/SqlRender')"

RUN R CMD javareconf
RUN Rscript -e "install.packages('rJava', repos='https://rforge.net')"

RUN Rscript -e "install.packages('SqlRender') "

ENV DATABASECONNECTOR_JAR_FOLDER /usr/lib/jdbcDrivers

RUN Rscript -e "install.packages(c('DatabaseConnector','ParallelLogger','rjson', 'remotes'))"
RUN Rscript -e "remotes::install_github('OHDSI/Achilles')"
RUN Rscript -e "remotes::install_github('OHDSI/DataQualityDashboard')"
RUN Rscript -e "install.packages('/tmp/CatalogueExport.bigan',repos=NULL, type='source')"

# Copiar tus scripts R al contenedor
#COPY scripts /scripts

# Establecer el directorio de trabajo
#WORKDIR /scripts

# Establecer el comando por defecto
#CMD ["Rscript", "your_script.R"]

#CMD ["Rscript", "/your_script.R"]



