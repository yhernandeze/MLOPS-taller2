# Taller: Predicción de Especies de Pingüinos con FastAPI y Docker

En este taller se desarrolló un proyecto completo que abarca desde la obtención y procesamiento de datos hasta la creación y despliegue de una API para realizar inferencias con modelos de machine learning entrenados. Se usaron varias tecnologías y buenas prácticas para lograr un producto funcional, modular y escalable.

## Arquitectura del Proyecto

### Estructura de Archivos

La estructura del proyecto se organizó de manera modular para facilitar el mantenimiento y escalabilidad:

![Estructura de archivos del proyecto](./imagen/estructura-archivos.PNG)

```
app/
├── __init__.py
├── main.py
├── models/
│   ├── decision_tree.pkl
│   ├── knn.pkl
│   ├── logistic_regression.pkl
│   ├── modelo_logistic_regression.pkl
│   ├── scaler_decision_tree.pkl
│   ├── scaler_knn.pkl
│   ├── scaler_logistic_regression.pkl
│   └── scaler_x.pkl
├── venv/
├── docker-compose.yml
├── Dockerfile
├── modelo_final.py
└── requirements.txt
```

**Componentes principales:**
- `main.py`: Archivo principal de la API FastAPI
- `models/`: Directorio con los modelos entrenados y escaladores guardados en formato pickle
- `modelo_final.py`: Script de entrenamiento y guardado de modelos
- `Dockerfile`: Configuración para contenerización
- `docker-compose.yml`: Orquestación de servicios
- `requirements.txt`: Dependencias del proyecto

## 1. Descarga y Procesamiento de Datos

Se utilizó la librería `palmerpenguins` para descargar los datos originales de las especies de pingüinos, un dataset clásico para clasificación.

**Pasos del procesamiento:**

1. **Carga de datos**: Se cargaron los datos en un DataFrame de pandas para su manipulación y exploración inicial
2. **Limpieza**: Se verificó la existencia de datos nulos y se eliminaron las filas que contenían valores faltantes
3. **Codificación de variables categóricas**: Se identificaron las variables categóricas (sexo e isla) y se transformaron en variables dummy (one-hot encoding)
4. **Transformación de la variable objetivo**: La especie de pingüino fue convertida de valores categóricos a valores numéricos

## 2. Entrenamiento de Modelos de Machine Learning

Se definieron múltiples modelos para clasificación:

- **Regresión logística**
- **Árbol de decisión**
- **K-Nearest Neighbors (KNN)**

**Proceso de entrenamiento:**

1. **División de datos**: 70% para entrenamiento, 30% para prueba
2. **Estandarización**: Se escalaron las características numéricas
3. **Entrenamiento**: Cada modelo fue entrenado con los datos escalados
4. **Evaluación**: Se calcularon métricas de desempeño (ROC AUC)
5. **Persistencia**: Modelos y escaladores guardados en archivos `.pkl`

## 3. API REST con FastAPI

### Funcionalidades de la API

La API permite:

- **Recepción de datos**: Mediante esquema Pydantic para validación
- **Selección dinámica de modelo**: Parámetro para elegir el modelo de predicción
- **Inferencia escalada**: Procesamiento automático de datos de entrada
- **Respuesta estructurada**: Especie predicha con probabilidades por clase

### Interfaz de Usuario

La API incluye una interfaz web interactiva generada automáticamente por FastAPI:

![Interfaz de selección de modelo](./imagen/interfaz-api.jpg)


*Figura 2: Interfaz web para selección de modelo y entrada de datos*

**Selección de Modelo:**

La interfaz permite seleccionar entre los tres modelos disponibles:
- logistic_regression
- decision_tree
- knn

**Formato de Entrada:**

Los datos de entrada incluyen características del pingüino:
```json
{
  "bill_length_mm": 39.1,
  "bill_depth_mm": 18.7,
  "flipper_length_mm": 181,
  "body_mass_g": 3750,
  "year": 2007,
  "sex_Female": 0,
  "sex_Male": 1,
  "island_Biscoe": 0,
  "island_Dream": 0,
  "island_Torgersen": 1
}
```

### Ejemplo de Uso y Respuesta

![Resultado de la predicción](./imagen/resultado-prediccion.PNG)
*Figura 3: Resultado de la predicción mostrando la respuesta completa de la API*


**Request URL:**
```
http://localhost:8989/predict?model_name=logistic_regression
```

**Respuesta del Servidor:**
```json
{
  "model_used": "logistic_regression",
  "species_id": 1,
  "species_name": "Adelie",
  "probability": {
    "Adelie": 0.9998592367605084,
    "Chinstrap": 0.00009071514083751021,
    "Gentoo": 0.000050048098654133385
  }
}
```

**Detalles de la Respuesta:**
- **Código de estado**: 200 (Éxito)
- **Tipo de contenido**: application/json
- **Predicción**: Especie "Adelie" con 99.98% de probabilidad
- **Probabilidades alternativas**: Chinstrap (0.0091%) y Gentoo (0.0050%)

## 4. Contenerización con Docker

### Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Características:**
- Imagen base liviana (`python:3.11-slim`)
- Instalación optimizada de dependencias
- Montaje de modelos via volumen
- Puerto 8000 interno, mapeado a 8989 en el host
- Configuración de reinicio automático
- Healthcheck incluido

### Docker Compose

```yaml
version: '3.8'
services:
  penguin-api:
    build: .
    ports:
      - "8989:8000"
    volumes:
      - ./models:/app/models:ro
    environment:
      - PYTHONPATH=/app
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

## 5. Características Avanzadas

### Selección Dinámica de Modelo

- **Flexibilidad**: El usuario puede seleccionar el modelo en cada petición
- **Comparación**: Permite evaluar diferentes modelos sin múltiples endpoints
- **Extensibilidad**: Fácil incorporación de nuevos modelos

### Optimización de Performance

- **Carga en startup**: Modelos cargados al inicio de la aplicación
- **Reutilización**: Evita cargar modelos en cada petición
- **Escalado**: Los escaladores se aplican automáticamente

### Manejo de Errores y Logging

- **Trazabilidad**: Sistema de logging implementado
- **Robustez**: Manejo de excepciones y errores
- **Monitoreo**: Healthcheck para supervisión

## 6. Resultado Final

**API REST robusta y escalable** para predicción de especies de pingüinos

**Contenerización completa** para facilitar despliegue y distribución

**Código modular y documentado** para mantenibilidad y extensibilidad

**Selección dinámica de modelos** a través de parámetros de petición

**Interfaz web interactiva** para pruebas y desarrollo

**Pipeline completo** desde datos hasta producción

### Tecnologías Utilizadas

- **Machine Learning**: scikit-learn, pandas, numpy
- **API Framework**: FastAPI, Pydantic, Uvicorn
- **Containerización**: Docker, Docker Compose
- **Data Source**: palmerpenguins dataset
- **Serialización**: pickle para persistencia de modelos

### Casos de Uso

1. **Investigación**: Comparación de modelos para análisis científico
2. **Educación**: Demostración de pipeline ML completo
3. **Producción**: Base para sistemas de clasificación en tiempo real
4. **Desarrollo**: Prototipado rápido de APIs de ML

## 7. Instrucciones de Uso

### Requisitos Previos

- Docker y Docker Compose instalados
- Python 3.11+ (para desarrollo local)

### Ejecución con Docker

```bash
# Clonar el repositorio
git clone https://github.com/DAVID316CORDOVA/Taller-1---MLOPS.git
cd penguin-prediction-api

# Construir y ejecutar con Docker Compose
docker-compose up --build

# La API estará disponible en http://localhost:8989
```

### Ejecución Local

```bash
# Instalar dependencias
pip install -r requirements.txt

# Ejecutar la aplicación
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# La API estará disponible en http://localhost:8000
```

### Endpoints Disponibles

- **GET /**: Página de bienvenida
- **POST /predict**: Endpoint de predicción
- **GET /docs**: Documentación interactiva (Swagger UI)
- **GET /redoc**: Documentación alternativa (ReDoc)
- **GET /health**: Endpoint de health check

### Ejemplo de Uso con cURL

```bash
curl -X POST "http://localhost:8989/predict?model_name=logistic_regression" \
  -H "Content-Type: application/json" \
  -d '{
    "bill_length_mm": 39.1,
    "bill_depth_mm": 18.7,
    "flipper_length_mm": 181,
    "body_mass_g": 3750,
    "year": 2007,
    "sex_Female": 0,
    "sex_Male": 1,
    "island_Biscoe": 0,
    "island_Dream": 0,
    "island_Torgersen": 1
  }'
```

## 8. Estructura de Carpetas Recomendada para Imágenes

```
proyecto-pinguinos/
├── imagen/
│   ├── estructura-archivos.png
│   ├── interfaz-api.png
│   └── resultado-prediccion.png
├── app/
│   ├── __init__.py
│   ├── main.py
│   └── models/
├── DESCRIPCION_DEL_TALLER.md
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

El proyecto demuestra la implementación de buenas prácticas en el desarrollo de sistemas de machine learning, desde la preparación de datos hasta el despliegue en contenedores, proporcionando una base sólida para aplicaciones similares en entornos de producción.
