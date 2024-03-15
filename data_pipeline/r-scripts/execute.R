# Librerías referenciadas
library(DBI)
library(RPostgreSQL)
library(Achilles)
library(CdmInspection)
library(CatalogueExport.bigan)


# Configuración de la conexión a la base de datos
db_host <- "omop-postgres" # nombre del servicio de la base de datos en Docker Compose
db_port <- 5432  # puerto de PostgreSQL
db_name <- "omop" # nombre del esquema
db_user <- "omop"
db_password <- "omop"
db_server <- paste(db_host, "/", db_name, sep = "")
authors <-"IACS"

# Establecer la conexión a la base de datos
connectionDetails <-  DatabaseConnector::createConnectionDetails(dbms = "postgresql",
	server = db_server, user = db_user, 
	password = db_password, pathToDriver ="/scripts")

# Declaraciones para ejecución
cdmSchema <- "omop"
resultsSchema <- "results"
vocabSchema <- cdmSchema
sourceName <- "BIGAN"
version <- 5.3 # Versión CDM

# ACHILLES
# Definimos los análisis
print("ACHILLES")
allAnalyses=getAnalysisDetails()$ANALYSIS_ID
longAnalyses1=c(226,1824,413,424) # Posibles análisis problemáticos
subSet1=setdiff(allAnalyses,longAnalyses1)

# Ejecutamos Achilles
achilles(connectionDetails = connectionDetails,
	cdmDatabaseSchema = cdmSchema,
	resultsDatabaseSchema = resultsSchema,
	vocabDatabaseSchema = vocabSchema,
	scratchDatabaseSchema = "temp",
	sourceName = sourceName,
	cdmVersion = 5.3,
	numThreads = 1,
	createIndices = TRUE,
	outputFolder = "/scripts/Achilles",
	createTable = TRUE,
	dropScratchTables=TRUE,
	analysisIds = subSet1)


# CDMINSPECTION
print("CDMINSPECTION")
# Prevents errors due to packages being built for other R versions:
Sys.setenv("R_REMOTES_NO_ERRORS_FROM_WARNINGS" = TRUE)
# All results smaller than this value are removed from the results.
smallCellCount <- 2
verboseMode <- TRUE
outputFolder <- "/scripts/envio/" # Folder destino
results<-cdmInspection(
  connectionDetails = connectionDetails,
  cdmDatabaseSchema = cdmSchema,
  resultsDatabaseSchema = resultsSchema,
  vocabDatabaseSchema = vocabSchema,
  databaseName = db_name,
  runVocabularyChecks = TRUE,
  runDataTablesChecks = TRUE,
  runPerformanceChecks = FALSE,
  runWebAPIChecks = TRUE,
  smallCellCount = smallCellCount,
  baseUrl="",
  sqlOnly = FALSE,
  outputFolder = outputFolder,
  verboseMode = verboseMode
)

# Guardamos resultados
generateResultsDocument(
  results,
  outputFolder = outputFolder,
  authors=authors,
  databaseId = "synpuf",
  databaseName = db_name,
  databaseDescription = authors,
  smallCellCount = smallCellCount
)

# CATALOGUEEXPORT
print("CATALOGUEEXPORT")
catalogueExport(connectionDetails = connectionDetails,
         cdmDatabaseSchema = cdmSchema,
         resultsDatabaseSchema = resultsSchema,
         vocabDatabaseSchema = cdmSchema,
         sourceName = sourceName,
         cdmVersion = version,
         outputFolder = outputFolder)