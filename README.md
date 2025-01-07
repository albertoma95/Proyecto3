# Weather prediction

## Descripción del Proyecto

Weather prediction es una herramienta desarrollada en Django que nos permite, en base a ciertas variables, predecir el clima que hace

## Características Principales

-
-
-
-

## Requisitos del Sistema

- Python 3.10 o superior.
- Django.
- Tailwind CSS para el diseño de la interfaz.

## Instalación de paquetes

- pip install -r requirements.txt

## Comandos para Iniciar el Proyecto - Desde la carpeta raíz, es decir, en la carpeta padre de prediccion_meteorologica

1. Creamos un solo csv en base a todos los demás csvs que tenemos, incluyendo tambíen la limpieza de datos
   ```bash
   python/py prediccion_meteorologica/build_dataset.py
   ```
2. Entrenamos el model en base al csv que se ha generado, creando el modelo y guardandolo
   ```bash
   python/py prediccion_meteorologica/predictor.py
   ```
3. Iniciar el servidor de Tailwind CSS para la gestión de estilos:
   ```bash
   python/py manage.py tailwind start
   ```
4. Ejecutar el servidor de desarrollo:
   ```bash
   python/py manage.py runserver --noreload
   ```
5. Visualizar la herramienta:
   ```bash
   #Entrar a tu navegador preferido
   http://127.0.0.1:8000/
   ```

## Estructura de Archivos

- `app/` - Contiene la lógica principal del proyecto.
- `static/` - Archivos estáticos como imágenes, CSS, CSV.
- `templates/` - Plantillas HTML del proyecto.
