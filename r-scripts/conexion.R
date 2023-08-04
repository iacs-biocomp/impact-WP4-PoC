library(DBI)
library(RPostgreSQL)

# Configuración de la conexión a la base de datos
db_host <- "omop"  # nombre del servicio de la base de datos en Docker Compose
db_port <- 5432  # puerto de PostgreSQL
db_name <- "omop"
db_user <- "omop"
db_password <- "ryc1852!"

# Establecer la conexión a la base de datos
con <- dbConnect(
  PostgreSQL(),
  host = db_host,
  port = db_port,
  dbname = db_name,
  user = db_user,
  password = db_password
)

# Ejemplo de consulta a la base de datos
query <- "SELECT * FROM person"
result <- dbGetQuery(con, query)

# Imprimir el resultado de la consulta
print(result)

# Cerrar la conexión a la base de datos
dbDisconnect(con)
