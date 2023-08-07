library(CatalogueExport.bigan)

# Configuración de la conexión a la base de datos
db_host <- "omop/omop"  # nombre del servicio de la base de datos en Docker Compose
db_port <- 5432  # puerto de PostgreSQL
db_name <- "omop"
db_user <- "omop"
db_password <- "omop"

connectionDetails <- createConnectionDetails(
  dbms="postgresql", 
  server=db_host, 
  user=db_user, 
  password=db_password, 
  port=db_port,
  pathToDriver='/usr/lib/jdbcDrivers'
)

catalogueExport(connectionDetails = connectionDetails,
         cdmDatabaseSchema = "omop",
         resultsDatabaseSchema = "results",
         vocabDatabaseSchema = "vocab",
         sourceName = "impact-wp4-PoC",
         cdmVersion = 5.3)

#catalogueExport(connectionDetails = connectionDetails,
#                cdmDatabaseSchema = "omop",
#                resultsDatabaseSchema = "results",
#                vocabDatabaseSchema = "vocab",
#                sourceName = "impact-wp4-PoC",
#                sqlOnly = TRUE,
#                cdmVersion = 5.3,
#                createTable=TRUE)
