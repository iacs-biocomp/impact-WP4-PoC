# Pasos seguidos
- Descargar el repositorio usando git
		
		git clone https://github.com/iacs-biocomp/impact-WP4-PoC
	
	
- [Descargar e instalar Docker desktop](https://www.docker.com/)

- Abrir Docker como administrador

- Abrir la consola de comandos dentro de la carpeta del repositorio y ejecutar el comando:

		docker-compose up
	
- Ahora en Docker aparece el contenedor impact-wp4-poc con dos volúmenes uno llamado omop que se encuentra en ejecución y otro parado llamado r-base

- Usando un cliente de base de datos (en mi caso, DBeaver) se puede conectar al contenedor omop que tiene una instancia de postgresql usando el usuario y contraseña omop y la dirección 127.0.0.1:5342

Llegado a este punto veo que existe un schema de postgres llamado public que está vacío, sin ninguna tabla creada