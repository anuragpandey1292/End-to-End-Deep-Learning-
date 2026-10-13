import os
import sys

import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Ensure project root in sys.path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
)

from src.components.models import GRUModel, LSTMModel
from src.utils import load_object

app = FastAPI(
    title="Store Sales Forecasting API",
    description="Real-time multi-model deep learning inference for store sales prediction",
    version="1.0.0",
)

# Global model and preprocessor cache
MODEL_PATH = os.path.join("artifacts", "model.pth")
PREPROCESSOR_PATH = os.path.join("artifacts", "proprocessor.pkl")

preprocessor = None
model = None


class PredictionItem(BaseModel):
    store_nbr: int
    family: str
    onpromotion: float
    dcoilwtico: float | None = 50.0
    cluster: int | None = 1
    city: str | None = "Quito"
    state: str | None = "Pichincha"
    type: str | None = "D"
    year: int | None = 2017
    month: int | None = 8
    day: int | None = 16
    dayofweek: int | None = 2
    is_weekend: int | None = 0
    is_holiday: int | None = 0
    is_payday: int | None = 0
    is_earthquake_period: int | None = 0
    is_promoted: int | None = 0
    sales_lag_7: float | None = 0.0
    sales_lag_14: float | None = 0.0
    sales_rolling_mean_7: float | None = 0.0


class PredictionBatchRequest(BaseModel):
    records: list[PredictionItem]


class PredictionResponse(BaseModel):
    predictions: list[float]


@app.on_event("startup")
def load_artifacts():
    global preprocessor, model
    if os.path.exists(PREPROCESSOR_PATH):
        try:
            preprocessor = load_object(PREPROCESSOR_PATH)
        except Exception as e:
            print(f"Warning: could not load preprocessor: {e}")

    if os.path.exists(MODEL_PATH) and preprocessor is not None:
        try:
            # Infer feature dimensionality from preprocessor
            dummy_dict = {
                "store_nbr": [1],
                "onpromotion": [0.0],
                "dcoilwtico": [50.0],
                "cluster": [1],
                "is_holiday": [0],
                "is_payday": [0],
                "is_earthquake_period": [0],
                "is_promoted": [0],
                "year": [2017],
                "month": [8],
                "day": [16],
                "dayofweek": [2],
                "is_weekend": [0],
                "sales_lag_7": [0.0],
                "sales_lag_14": [0.0],
                "sales_rolling_mean_7": [0.0],
                "family": ["GROCERY I"],
                "city": ["Quito"],
                "state": ["Pichincha"],
                "type": ["A"],
            }
            dummy_df = pd.DataFrame(dummy_dict)
            transformed = preprocessor.transform(dummy_df)
            input_dim = transformed.shape[1]

            model = GRUModel(input_dim=input_dim, hidden_dim=64)
            model.load_state_dict(
                torch.load(MODEL_PATH, map_location=torch.device("cpu"))
            )
            model.eval()
        except Exception as e:
            print(f"Warning: could not load model weights: {e}")


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "preprocessor_loaded": preprocessor is not None,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict_sales(request: PredictionBatchRequest):
    global preprocessor, model
    if preprocessor is None or model is None:
        load_artifacts()

    if preprocessor is None or model is None:
        raise HTTPException(
            status_code=503,
            detail="Model or preprocessor artifacts not ready. Please ensure training pipeline has run.",
        )

    try:
        data_dicts = [item.dict() for item in request.records]
        input_df = pd.DataFrame(data_dicts)

        transformed_features = preprocessor.transform(input_df)
        if hasattr(transformed_features, "toarray"):
            transformed_features = transformed_features.toarray()

        tensor_x = torch.tensor(transformed_features, dtype=torch.float32)

        with torch.no_grad():
            preds = model(tensor_x).cpu().numpy().flatten()
            preds = np.maximum(0, preds).tolist()

        return PredictionResponse(predictions=preds)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
