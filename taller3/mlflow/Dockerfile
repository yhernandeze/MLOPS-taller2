# Base a lightweight Python image
FROM python:3.9-slim

# Environment variables for MLflow to find MinIO (S3) and credentials
ENV MLFLOW_S3_ENDPOINT_URL=http://minio:9000 \
    AWS_ACCESS_KEY_ID=admin \
    AWS_SECRET_ACCESS_KEY=supersecret

# Install MLflow and dependencies for S3 (boto3) and MySQL (PyMySQL)
RUN pip install --no-cache-dir mlflow boto3 pymysql cryptography

# Create a working directory (not strictly needed for mlflow server)
WORKDIR /app

# Expose default MLflow port
EXPOSE 5000

# Default command to start MLflow server (we will override some arguments via docker-compose)
CMD ["mlflow", "server", \
     "--host", "0.0.0.0", "--port", "5000", \
     "--backend-store-uri", "sqlite:///mlflowdb/mlflow.db", \
     "--default-artifact-root", "s3://mlflows3/artifacts", \
     "--serve-artifacts"]

