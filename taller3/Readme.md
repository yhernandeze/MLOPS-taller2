Taller MLflow — Guía paso a paso (README)

Nombre del archivo: reamde.md
Objetivo: Dejarte un proyecto reproducible con MinIO (S3), MySQL, MLflow (tracking + registry), JupyterLab y una API FastAPI para inferencia con el mejor modelo en Producción. Incluye ingestión de datos a MySQL, entrenamiento con GridSearch (20+ ejecuciones), registro de experimentos y despliegue de un endpoint de predicción.

Funciona en WSL (Ubuntu) o macOS. Asumimos conocimientos mínimos.

0) Prerrequisitos
0.1 Docker y Docker Compose

WSL (Ubuntu 22.04/24.04 recomendado)

Instala Docker Desktop en Windows y habilita WSL integration para tu distro Ubuntu.

Dentro de WSL:

docker version
docker compose version


si tienes permisos:

sudo usermod -aG docker $USER
exec su -l $USER  # o cierra y abre la terminal


macOS (Intel/Apple Silicon)

Instala Docker Desktop for Mac.

Verifica:
docker version
docker compose version


Si docker compose no existe, usa docker-compose (legacy) o actualiza Docker Desktop.

1) Estructura del proyecto

Crea el directorio base taller3 y su estructura:

mkdir -p ~/taller3/{api,mlflow,notebooks}
cd ~/taller3

# Script de ingesta (Python) fuera de contenedores
cat > data_ingestion.py <<'PY'
import os
import pandas as pd
from sqlalchemy import create_engine, text

# URI de BD (puedes exportarla en shell antes de ejecutar)
DATA_DB_URI = os.environ.get("DATA_DB_URI", "mysql+pymysql://mlflow_user:mlflow_pass@localhost:3306/penguins_db")

RAW_TABLE  = "penguins_raw"
PROC_TABLE = "penguins_processed"

def main():
    # Conectar al servidor MySQL
    # Asegura que la BD exista y el usuario tenga permisos
    root_uri = DATA_DB_URI.rsplit("/", 1)[0] + "/"
    eng_root = create_engine(root_uri)
    with eng_root.begin() as con:
        con.execute(text("CREATE DATABASE IF NOT EXISTS penguins_db CHARACTER SET utf8mb4"))
        con.execute(text("GRANT ALL ON penguins_db.* TO 'mlflow_user'@'%'"))
        con.execute(text("FLUSH PRIVILEGES"))

    # Conectar a la BD objetivo
    eng = create_engine(DATA_DB_URI)

    # Cargar dataset de penguins
    try:
        from palmerpenguins import load_penguins
    except Exception:
        raise SystemExit("Falta palmerpenguins. Instálalo en tu venv: pip install palmerpenguins")

    df = load_penguins()  # sin preprocesamiento

    # Persistir datos RAW
    df.to_sql(RAW_TABLE, eng, if_exists="replace", index=False)

    # Preprocesamiento mínimo para ejemplo: imputar y one-hot como columnas binarias
    # (En el notebook entrenarás de forma más completa)
    df_p = df.copy()
    # num cast
    num_cols = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g', 'year']
    for c in num_cols:
        df_p[c] = pd.to_numeric(df_p[c], errors="coerce")
    df_p['island'] = df_p['island'].astype('string').fillna('missing')
    df_p['sex']    = df_p['sex'].astype('string').fillna('missing')

    # one-hot manual a modo demo (coincidiendo con el modelo final)
    for island in ['Biscoe','Dream','Torgersen']:
        df_p[f'island_{island}'] = (df_p['island'] == island).astype(int)
    df_p['sex_female']   = (df_p['sex'] == 'female').astype(int)
    df_p['sex_male']     = (df_p['sex'] == 'male').astype(int)
    df_p['sex_missing']  = (df_p['sex'] == 'missing').astype(int)

    # quitar columnas categóricas originales (dejas la etiqueta species)
    df_p = df_p[num_cols + [f'island_{x}' for x in ['Biscoe','Dream','Torgersen']]
                + ['sex_female','sex_male','sex_missing','species']]

    df_p.to_sql(PROC_TABLE, eng, if_exists="replace", index=False)
    print("OK: penguins_raw y penguins_processed escritos en MySQL")

if __name__ == "__main__":
    main()
PY


2) Archivos de Docker
2.1 mlflow/Dockerfile

Imagen de MLflow con dependencias para MySQL y S3/MinIO:

# mlflow/Dockerfile
FROM python:3.11-slim

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc build-essential libssl-dev ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    mlflow==2.14.3 \
    boto3 botocore \
    pymysql cryptography

WORKDIR /app
EXPOSE 5000

# Nota: el comando real lo definimos en docker-compose (para usar MySQL como backend)
CMD ["mlflow", "server", "--help"]


Usamos mlflow==2.14.3 en el servidor. El cliente en Jupyter puede ser más nuevo; el protocolo REST se mantiene.

2.2 api/Dockerfile

Imagen para FastAPI (inferencia):

# api/Dockerfile
FROM python:3.11-slim

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

RUN pip install --no-cache-dir \
    fastapi uvicorn[standard] \
    mlflow boto3 botocore pandas scikit-learn

WORKDIR /app
COPY app.py /app/app.py

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]


2.3 api/app.py

(versión robusta con input-schema, recarga y coerción de tipos)

cat > api/app.py <<'EOF'
import os
import pandas as pd
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient
from mlflow.models import get_model_info

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MODEL_URI = os.getenv("MODEL_URI", "models:/PenguinClassifier/Production")

mlflow.set_tracking_uri(TRACKING_URI)

app = FastAPI(title="Penguins Inference API", version="1.1.0")

_model = None
_model_version = {"name": None, "stage": None, "version": None}
_last_load_error = None
_expected_cols = None
_expected_types = None

def _load_signature():
    global _expected_cols, _expected_types
    _expected_cols, _expected_types = None, None
    info = get_model_info(MODEL_URI)
    sig = info.signature
    if sig and sig.inputs:
        cols, types = [], {}
        for x in sig.inputs.inputs:
            cols.append(x.name)
            types[x.name] = str(x.type)
        _expected_cols = cols
        _expected_types = types

def try_load_model():
    global _model, _model_version, _last_load_error
    try:
        _last_load_error = None
        model = mlflow.pyfunc.load_model(MODEL_URI)
        ver = {"name": None, "stage": None, "version": None}
        try:
            parts = MODEL_URI.split("/")
            if len(parts) >= 3 and parts[0] == "models:":
                name, stage_or_version = parts[1], parts[2]
                client = MlflowClient()
                if stage_or_version.isdigit():
                    ver = {"name": name, "stage": None, "version": stage_or_version}
                else:
                    mv = client.get_latest_versions(name, stages=[stage_or_version])[0]
                    ver = {"name": name, "stage": stage_or_version, "version": mv.version}
        except Exception:
            pass
        _model = model
        _model_version = ver
        _load_signature()
    except Exception as e:
        _model = None
        _last_load_error = str(e)

def ensure_model_loaded():
    if _model is None:
        try_load_model()
    if _model is None:
        raise HTTPException(status_code=503,
            detail=f"Model not available yet. last_error={_last_load_error}")

@app.get("/health")
def health():
    return {
        "status": "ok",
        "tracking_uri": TRACKING_URI,
        "model_uri": MODEL_URI,
        "model_loaded": _model is not None,
        "model_version": _model_version,
        "last_load_error": _last_load_error,
        "has_signature": bool(_expected_cols),
    }

@app.post("/reload-model")
def reload_model():
    try_load_model()
    return {
        "reloaded": _model is not None,
        "model_version": _model_version,
        "last_load_error": _last_load_error,
        "input_schema_columns": _expected_cols,
        "input_schema_types": _expected_types,
    }

@app.get("/input-schema")
def input_schema():
    ensure_model_loaded()
    if not _expected_cols:
        return {"model_version": _model_version, "schema": None}
    return {
        "model_version": _model_version,
        "schema": [{"name": c, "type": _expected_types.get(c, "unknown")} for c in _expected_cols],
        "example_payload": {
            "dataframe_split": {"columns": _expected_cols, "data": [[0 for _ in _expected_cols]]}
        }
    }

@app.get("/mlflow/runs_count")
def runs_count():
    try:
        client = MlflowClient()
        exps = client.search_experiments()
        exp_ids = [e.experiment_id for e in (exps or [])]
        if not exp_ids:
            return {"experiments": 0, "runs_count": 0}
        df = mlflow.search_runs(experiment_ids=exp_ids, max_results=100000, output_format="pandas")
        return {"experiments": len(exp_ids), "runs_count": int(len(df))}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to count runs: {e}")

class DataframeSplit(BaseModel):
    columns: list
    data: list

class PredictPayload(BaseModel):
    dataframe_split: DataframeSplit | None = None
    records: list | None = None
    columns: list | None = None

def _payload_to_df(payload: PredictPayload) -> pd.DataFrame:
    if payload.dataframe_split:
        return pd.DataFrame(payload.dataframe_split.data, columns=payload.dataframe_split.columns)
    if payload.records is not None and payload.columns is not None:
        return pd.DataFrame(payload.records, columns=payload.columns)
    raise ValueError("Provide either {'dataframe_split': {...}} or {'records': [...], 'columns': [...]}")

def _validate_and_align(df: pd.DataFrame) -> pd.DataFrame:
    if not _expected_cols:
        return df
    incoming_cols = list(df.columns)
    missing = [c for c in _expected_cols if c not in incoming_cols]
    extra   = [c for c in incoming_cols if c not in _expected_cols]
    if missing or extra:
        raise HTTPException(status_code=400, detail={
            "error": "payload_columns_mismatch",
            "missing_columns": missing, "extra_columns": extra,
            "expected_columns": _expected_cols,
        })
    df = df[_expected_cols].copy()
    # Coerción estricta a float64 para cumplir 'double'
    for c in _expected_cols:
        t = _expected_types.get(c, "")
        if any(k in t for k in ("double","float","long","integer")):
            try:
                df[c] = pd.to_numeric(df[c], errors="raise").astype("float64")
            except Exception as e:
                raise HTTPException(status_code=400, detail={
                    "error": "type_coercion_failed", "column": c, "expected_type": t, "message": str(e),
                })
    return df

@app.post("/predict")
def predict(payload: PredictPayload = Body(...)):
    ensure_model_loaded()
    try:
        df = _payload_to_df(payload)
        df = _validate_and_align(df)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    try:
        preds = _model.predict(df)
        return {"predictions": [str(p) for p in preds]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")
EOF

2.4 docker-compose.yml

Stack completo: MinIO, MySQL, MLflow, JupyterLab y API.

version: "3.9"
services:
  minio:
    image: quay.io/minio/minio:latest
    container_name: Minio
    command: server /data --console-address ":9001"
    ports:
      - "9000:9000"   # API
      - "9001:9001"   # Consola web
    environment:
      - MINIO_ROOT_USER=admin
      - MINIO_ROOT_PASSWORD=supersecret
    volumes:
      - minio_data:/data
    restart: unless-stopped

  mysql:
    image: mysql:8.0
    container_name: MySQL
    ports:
      - "3306:3306"
    environment:
      - MYSQL_ROOT_PASSWORD=my-secret-pw
      - MYSQL_DATABASE=mlflow_db         # BD de metadatos MLflow
      - MYSQL_USER=mlflow_user
      - MYSQL_PASSWORD=mlflow_pass
    volumes:
      - mysql_data:/var/lib/mysql
    restart: unless-stopped

  mlflow:
    build:
      context: ./mlflow
      dockerfile: Dockerfile
    container_name: MLflow
    depends_on:
      - minio
      - mysql
    ports:
      - "5000:5000"
    environment:
      - MLFLOW_S3_ENDPOINT_URL=http://minio:9000
      - AWS_ACCESS_KEY_ID=admin
      - AWS_SECRET_ACCESS_KEY=supersecret
    command: >
      mlflow server
      --host 0.0.0.0 --port 5000
      --backend-store-uri mysql+pymysql://mlflow_user:mlflow_pass@mysql/mlflow_db
      --default-artifact-root s3://mlflows3/artifacts
      --serve-artifacts
    volumes:
      - ./mlflowdb:/app/mlflowdb
    restart: unless-stopped

  jupyter:
    image: jupyter/scipy-notebook:latest
    container_name: JupyterLab
    depends_on:
      - mlflow
    ports:
      - "8888:8888"
    environment:
      - JUPYTER_ENABLE_LAB=yes
      - MLFLOW_TRACKING_URI=http://mlflow:5000
      - MLFLOW_S3_ENDPOINT_URL=http://minio:9000
      - AWS_ACCESS_KEY_ID=admin
      - AWS_SECRET_ACCESS_KEY=supersecret
      - RESTARTABLE=yes
      - JUPYTER_TOKEN=valentasecret   # acceso sin token aleatorio
    volumes:
      - ./notebooks:/home/jovyan/work
    restart: unless-stopped

  api:
    build:
      context: ./api
    container_name: PenguinsAPI
    depends_on:
      - mlflow
    environment:
      - MLFLOW_TRACKING_URI=http://mlflow:5000
      - MODEL_URI=models:/PenguinClassifier/Production
      - MLFLOW_S3_ENDPOINT_URL=http://minio:9000
      - AWS_ACCESS_KEY_ID=admin
      - AWS_SECRET_ACCESS_KEY=supersecret
    ports:
      - "8000:8000"
    restart: unless-stopped

volumes:
  minio_data:
  mysql_data:


3) Levantar la plataforma

cd ~/taller3
docker compose up -d --build
docker compose ps


Deberías ver puertos:

MinIO: 9000 (API), 9001 (Consola web)

MySQL: 3306

MLflow: 5000

Jupyter: 8888 (token: valentasecret)

API: 8000

4) Crear el bucket S3 en MinIO

Abre http://localhost:9001

Usuario: admin
Password: supersecret

Crea un bucket llamado mlflows3.
(No hace falta crear la carpeta artifacts; MLflow la creará.)

5) Ingesta de datos a MySQL (fuera de Docker)

Tienes dos opciones para instalar dependencias del script data_ingestion.py.

5.1 Opción A — con uv (recomendada)

cd ~/taller3
uv venv
source .venv/bin/activate

uv pip install pandas SQLAlchemy pymysql palmerpenguins setuptools
export DATA_DB_URI="mysql+pymysql://mlflow_user:mlflow_pass@localhost:3306/penguins_db"
python data_ingestion.py


Deberías ver OK: penguins_raw y penguins_processed escritos en MySQL.

6) Entrenamiento en Jupyter + MLflow (20+ ejecuciones)

Abre http://localhost:8888
Crea un notebook nuevo en notebooks/ con el siguiente contenido:

import os, json
import numpy as np
import pandas as pd
import mlflow
from sqlalchemy import create_engine

from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# Conexión a MySQL con los datos procesados (creados por data_ingestion.py)
DATA_DB_URI = "mysql+pymysql://mlflow_user:mlflow_pass@mysql:3306/penguins_db"  # dentro de contenedor Jupyter apunta a 'mysql'
engine = create_engine(DATA_DB_URI)
df = pd.read_sql_table("penguins_processed", con=engine)

y = df["species"].values
X = df.drop(columns=["species"])

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42, test_size=0.2)

mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
mlflow.set_experiment("penguins_gridsearch")

# Pipeline sencillo: scaler + logreg
pipe = Pipeline([
    ("scaler", StandardScaler(with_mean=False)),  # with_mean=False por sparse/one-hot
    ("clf", LogisticRegression(max_iter=1000, multi_class="auto"))
])

# Grid >= 20 combinaciones
param_grid = {
    "clf__C": [0.01, 0.1, 1, 3, 10],
    "clf__penalty": ["l2"],
    "clf__solver": ["lbfgs", "saga", "liblinear", "newton-cg"]  # 5*1*4 = 20 runs
}

best_acc = -1
best_params = None
best_run_id = None

for params in ParameterGrid(param_grid):
    with mlflow.start_run(run_name="grid_run") as run:
        # log params
        for k, v in params.items():
            mlflow.log_param(k, v)

        # train
        model = pipe.set_params(**params)
        model.fit(X_train, y_train)

        # eval
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        mlflow.log_metric("accuracy", float(acc))

        # guarda el modelo en artifacts de este run
        mlflow.sklearn.log_model(sk_model=model, artifact_path="model")

        if acc > best_acc:
            best_acc = acc
            best_params = params
            best_run_id = run.info.run_id

print("Mejor accuracy:", best_acc, "con params:", best_params, "run:", best_run_id)

# Registrar el mejor como un Model en el Registry
model_name = "PenguinClassifier"
result = mlflow.register_model(model_uri=f"runs:/{best_run_id}/model", name=model_name)

print("Registrado como:", model_name, "version:", result.version)


En MLflow UI (http://localhost:5000
Ve a Models → PenguinClassifier.

Promociona la mejor versión a Production (Stage Production, “P” mayúscula).

Con este grid hay exactamente 20 ejecuciones. Puedes ampliar el grid si quieres más de 20.

7) Probar la API de inferencia

Asegúrate de que la API apunta a Production en docker-compose.yml:

- MODEL_URI=models:/PenguinClassifier/Production

Reinicia la API para cargar la última versión:

docker compose up -d --build api
curl -s -X POST http://localhost:8000/reload-model


7.1 Endpoints útiles

# Salud
curl -s http://localhost:8000/health

# Esquema de entrada + ejemplo de payload
curl -s http://localhost:8000/input-schema

# Contar runs en MLflow
curl -s http://localhost:8000/mlflow/runs_count

7.2 Predicción (ejemplo)

Usa las columnas que devuelve /input-schema. Con nuestro pipeline:

curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "dataframe_split": {
      "columns": ["bill_length_mm","bill_depth_mm","flipper_length_mm","body_mass_g","year",
                  "island_Biscoe","island_Dream","island_Torgersen",
                  "sex_female","sex_male","sex_missing"],
      "data": [[39.5, 17.4, 186, 3800, 2007, 0, 1, 0, 1, 0, 0]]
    }
  }'


8) Dónde se guardan tus notebooks

Por docker-compose.yml, se monta ./notebooks → /home/jovyan/work.
Todo lo que guardes en Jupyter aparece en tu host en ~/taller3/notebooks, persiste aunque detengas o reconstruyas el contenedor.

9) Solución de problemas (FAQ)

MLflow (5000) no responde / el contenedor reinicia
Revisa logs:

docker compose logs --no-color mlflow | tail -n 100

Asegúrate de tener pymysql y cryptography en la imagen MLflow (nuestro Dockerfile ya lo trae).
Verifica que MySQL y MinIO estén arriba.

MinIO sin bucket mlflows3
Crea el bucket en http://localhost:9001

Permisos MySQL / Access denied
Asegúrate de haber creado la BD penguins_db y otorgado permisos:

docker exec -it MySQL mysql -uroot -pmy-secret-pw -e \
"CREATE DATABASE IF NOT EXISTS penguins_db; GRANT ALL ON penguins_db.* TO 'mlflow_user'@'%'; FLUSH PRIVILEGES;"


La API devuelve 503: model not available

Verifica que existe una versión en Production (P mayúscula).

O fija una versión exacta: MODEL_URI=models:/PenguinClassifier/30 y docker compose up -d --build api.

Luego curl -X POST http://localhost:8000/reload-model.

/mlflow/runs_count falla
Nuestra API usa search_experiments() (compatible con mlflow 3.4+).
Si tienes un cliente muy viejo en la imagen de la API, reconstruye para actualizar dependencias.

Schema error (tipos)
Si ves errores tipo “double requerido”, envía números float. Nuestra API convierte ints a float64, pero evita strings en columnas numéricas.

Jupyter pide token
Usamos JUPYTER_TOKEN=valentasecret. Entra con ese token.

Puertos ocupados
Cambia en docker-compose.yml los puertos host si tienes conflictos (p. ej., "5002:5000").

10) Parar / limpiar

Detener pero conservar datos:docker compose down

Borrar todo (incluyendo volúmenes MinIO/MySQL):

docker compose down -v

11) Resumen rápido

docker compose up -d --build

MinIO → crea bucket mlflows3 (http://localhost:9001

cd ~/taller3
uv venv && source .venv/bin/activate
uv pip install pandas SQLAlchemy pymysql palmerpenguins setuptools
export DATA_DB_URI="mysql+pymysql://mlflow_user:mlflow_pass@localhost:3306/penguins_db"
python data_ingestion.py

Jupyter (http://localhost:8888
GridSearch (20+ runs) y registra PenguinClassifier.

MLflow UI (http://localhost:5000
API:
docker compose up -d --build api
curl -s -X POST http://localhost:8000/reload-model
curl -s http://localhost:8000/input-schema
curl -s -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{...}'

