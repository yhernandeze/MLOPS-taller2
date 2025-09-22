import os
import pandas as pd
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient
from mlflow.models import get_model_info

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MODEL_URI = os.getenv("MODEL_URI", "models:/PenguinClassifier/Production")  # Stage or exact version

mlflow.set_tracking_uri(TRACKING_URI)

app = FastAPI(title="Penguins Inference API", version="1.1.0")

_model = None
_model_version = {"name": None, "stage": None, "version": None}
_last_load_error = None
_expected_cols: list[str] | None = None
_expected_types: dict[str, str] | None = None  # {"col": "double", ...}

def _load_signature():
    """Load the model signature (column names & types) from MLflow."""
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
    """Load model & signature; record any error instead of crashing the app."""
    global _model, _model_version, _last_load_error
    try:
        _last_load_error = None
        model = mlflow.pyfunc.load_model(MODEL_URI)
        ver = {"name": None, "stage": None, "version": None}
        try:
            parts = MODEL_URI.split("/")
            if len(parts) >= 3 and parts[0] == "models:":
                name, stage_or_version = parts[1], parts[2]
                # If a stage, this returns the latest version in that stage
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
        _expected_cols, _expected_types = None, None

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
    """Return the model's expected input columns & types (from MLflow signature)."""
    ensure_model_loaded()
    if not _expected_cols:
        return {"model_version": _model_version, "schema": None}
    return {
        "model_version": _model_version,
        "schema": [{"name": c, "type": _expected_types.get(c, "unknown")} for c in _expected_cols],
        "example_payload": {
            "dataframe_split": {
                "columns": _expected_cols,
                "data": [[0 for _ in _expected_cols]]  # placeholder zeros
            }
        }
    }

@app.get("/mlflow/runs_count")
def runs_count():
    try:
        client = MlflowClient()
        # mlflow 3.4.0 has search_experiments (list_experiments absent)
        exps = client.search_experiments()
        exp_ids = [e.experiment_id for e in (exps or [])]
        if not exp_ids:
            return {"experiments": 0, "runs_count": 0}
        df = mlflow.search_runs(
            experiment_ids=exp_ids,
            max_results=100000,
            output_format="pandas"
        )
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
    raise ValueError("Provide either {'dataframe_split': {'columns': [...], 'data': [...]}} "
                     "or {'records': [...], 'columns': [...] }")

def _validate_and_align(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure df contains exactly the expected columns in the expected order,
    and coerce numeric types to float64 to satisfy MLflow 'double' schema."""
    if not _expected_cols:
        return df

    incoming_cols = list(df.columns)
    missing = [c for c in _expected_cols if c not in incoming_cols]
    extra = [c for c in incoming_cols if c not in _expected_cols]
    if missing or extra:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "payload_columns_mismatch",
                "missing_columns": missing,
                "extra_columns": extra,
                "expected_columns": _expected_cols,
            },
        )

    # Reorder to match signature
    df = df[_expected_cols].copy()

    # Coerce numeric-like columns to float64 (MLflow 'double')
    for c in _expected_cols:
        t = _expected_types.get(c, "")
        if "double" in t or "float" in t or "long" in t or "integer" in t:
            try:
                # First convert to numeric (raises on bad strings), then enforce float64
                df[c] = pd.to_numeric(df[c], errors="raise").astype("float64")
            except Exception as e:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "type_coercion_failed",
                        "column": c,
                        "expected_type": t,
                        "message": str(e),
                    },
                )
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
        # ensure JSON-serializable
        return {"predictions": [str(p) for p in preds]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")
