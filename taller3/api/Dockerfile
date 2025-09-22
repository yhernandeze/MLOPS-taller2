FROM python:3.11-slim

# System deps for cryptography and some ML wheels when needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libssl-dev libffi-dev \
 && rm -rf /var/lib/apt/lists/*

# Python deps:
# - fastapi + uvicorn for API
# - mlflow client + pyfunc
# - boto3/botocore for S3/MinIO artifact access
# - pandas for payload → DataFrame
# - scikit-learn since model flavor is sklearn (needed by pyfunc to predict)
# - pymysql in case you later want to query DB from the API
RUN pip install --no-cache-dir \
    fastapi uvicorn[standard] \
    mlflow boto3 botocore \
    pandas scikit-learn pymysql

WORKDIR /app
COPY app.py /app/app.py

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

