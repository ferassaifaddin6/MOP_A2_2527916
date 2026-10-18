"""FastAPI service: uvicorn app:app --host 127.0.0.1 --port 8000"""
from functools import lru_cache

from fastapi import FastAPI, HTTPException

from src.inference import load_bundle, predict_record
from src.schemas import FarmRecord, PredictionResponse

app = FastAPI(title="Musanze Harvest Intelligence Service", version="1.0.0",
              description="Classroom decision-support prototype on synthetic data.")


@lru_cache(maxsize=1)
def get_bundle() -> dict:
    return load_bundle()


@app.get("/health")
def health() -> dict:
    """Liveness plus whether trained models are available."""
    try:
        get_bundle()
        return {"status": "ok", "models_loaded": True}
    except FileNotFoundError:
        return {"status": "degraded", "models_loaded": False}


@app.get("/model-info")
def model_info() -> dict:
    """Verification code, selected models and held-out metrics."""
    try:
        return get_bundle()["meta"]
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@app.post("/predict", response_model=PredictionResponse)
def predict(record: FarmRecord) -> dict:
    """Typed validation happens before this body runs (invalid input -> HTTP 422)."""
    try:
        return predict_record(record.model_dump(), get_bundle())
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
