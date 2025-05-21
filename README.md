# Práctica 3 - Clasificación con Spark (CC2425)

Este repositorio contiene la solución a la práctica 3 de la asignatura *Computación en la Nube y Big Data* (CC2425). El objetivo es aplicar distintos modelos de clasificación supervisada utilizando Spark MLlib sobre un conjunto de datos astronómicos.


## ⚙️ Requisitos

- [Docker](https://www.docker.com/)
- [Docker Compose](https://docs.docker.com/compose/)

## 🚀 Ejecución

1. Clona el repositorio:

   ```bash
   git clone https://github.com/carlotiii30/ccpractica3.git
   cd ccpractica3
   ```

2. Inicia el entorno Docker con Spark:

   ```bash
   docker-compose up -d
   ```

3. Copia el dataset a HDFS desde el contenedor del cliente:

   > ⚠️ **Importante:**  
   > El archivo `data/half_celestial.csv` no está incluido en este repositorio debido a su tamaño (mayor de 100 MB).  
   > Descárgalo manualmente y colócalo en la ruta `data/half_celestial.csv` dentro del directorio del proyecto antes de continuar.  
   > Puedes solicitar el archivo al profesor o al autor del repositorio.

   ```bash
   docker exec -it spark-client bash
   hdfs dfs -mkdir -p /user/carlota
   hdfs dfs -put /app/data/half_celestial.csv /user/carlota/
   ```

4. Ejecuta el script de clasificación:
   ```bash
   spark-submit /app/practice3.py
   ```

## 📊 Salida esperada
El script generará una tabla con el AUC de cada modelo (Logistic Regression, Random Forest, GBTClassifier), con dos configuraciones por algoritmo.