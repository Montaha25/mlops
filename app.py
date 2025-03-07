from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import pickle
import uvicorn
import numpy as np
import httpx
from model_ml_pipeline import preparedata
from model_ml_pipeline import train_model
from model_ml_pipeline import save_model
# Charger le modèle `xgboost_model.pkl`
MODEL_PATH = 'xgboost_model.pkl'
try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
except Exception as e:
    raise RuntimeError(f"Erreur de chargement modèle : {e}")

app = FastAPI()

class PredictionRequest(BaseModel):
    features: List[float]

class RetrainRequest(BaseModel):
    n_estimators: int
    max_depth: int
    learning_rate: float

    class Config:
        extra = "forbid"

@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        result = model.predict([request.features])
        return {"prediction": result.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur prédiction : {e}")



@app.post("/retrain")
def retrain(request: RetrainRequest):
    """
    Réentraîne le modèle avec les nouveaux hyperparamètres.
    """
    global model

    # ✅ Préparer les données
    X_train_scaled, X_test_scaled, y_train, y_test = preparedata()

    # ✅ Réentraîner le modèle avec les hyperparamètres envoyés
    model = train_model(
        X_train_scaled, y_train,
        n_estimators=request.n_estimators,
        max_depth=request.max_depth,
        learning_rate=request.learning_rate
    )

    # ✅ Sauvegarder le modèle réentraîné
    save_model(model, MODEL_PATH)

    return {"message": "✅ Modèle réentraîné avec succès", "params": request.dict()}

