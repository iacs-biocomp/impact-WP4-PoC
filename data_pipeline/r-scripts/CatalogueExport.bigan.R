library(CatalogueExport.bigan)

# Configuración de la conexión a la base de datos
db_host <- "omop"  # nombre del servicio de la base de datos en Docker Compose
db_port <- 5432  # puerto de PostgreSQL
db_name <- "omop"
db_user <- "omop"
db_password <- "ryc1852!"

connectionDetails <- createConnectionDetails(
  dbms="postgresql", 
  server=db_host, 
  user=db_user, 
  password=db_password, 
  port=db_port,
  pathToDriver='/home/carlos/Git/lib'
)

catalogueExport(connectionDetails = connectionDetails,
         cdmDatabaseSchema = "omop",
         resultsDatabaseSchema = "results",
         vocabDatabaseSchema = "vocab",
         sourceName = "BIGAN",
         cdmVersion = 5.3)

catalogueExport(connectionDetails = connectionDetails,
                cdmDatabaseSchema = "omop",
                resultsDatabaseSchema = "results",
                vocabDatabaseSchema = "vocab",
                sourceName = "BIGAN",
                sqlOnly = TRUE,
                cdmVersion = 5.3,
                createTable=TRUE)

#                analysisIds = c(0,1,2,3,101,102,103,104,105,106,107,108,109,110,111,112,113,117,200,201,203,206,211,220,400,401,403,405,406,420,430,501,502,506,600,601,603,605,306,620,630,700,701,703,705,706,715,716,717,720,
# 730,800,801,803,805,806,815,820,830,901,920,1001,1020,1800,1801,1803,1805,1806,1815,1816,1817,1820,1830,2100,2101,2105,2120,2130,2201,5000))

