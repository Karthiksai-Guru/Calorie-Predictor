from __future__ import annotations

import os
from functools import lru_cache

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .artifacts import load_artifacts
from .preprocessing import prepare_features
from .schemas import MetadataResponse, PredictionInput, PredictionResponse

app = FastAPI(title="Calorie Predictor API", version="1.0.0")

cors_env = os.getenv("CORS_ORIGINS", "")
if cors_env:
    allow_origins = [origin.strip() for origin in cors_env.split(",") if origin.strip()]
else:
    allow_origins = ["http://localhost:5173", "http://127.0.0.1:5173"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@lru_cache
def get_artifacts():
    return load_artifacts()


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.get("/metadata", response_model=MetadataResponse)
def get_metadata():
    _, _, metadata = get_artifacts()
    return metadata


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionInput):
    model, label_encoder, metadata = get_artifacts()

    raw_df = pd.DataFrame([payload.model_dump()])
    try:
        features = prepare_features(raw_df, label_encoder, metadata["feature_names"])
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    log_preds = model.predict(features)
    calories = float(np.expm1(log_preds)[0])

    clip_min = metadata.get("clip_min", 1)
    clip_max = metadata.get("clip_max", 314)
    calories = float(np.clip(calories, clip_min, clip_max))

    return PredictionResponse(calories=calories)
