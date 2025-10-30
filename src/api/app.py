from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import uvicorn
import numpy as np

MODEL_PATH = "models/xgb_model.pkl"
PREPROCESSOR_PATH = "models/preprocessor.pkl"

app = FastAPI(title="Churn Prediction API")

model = joblib.load(MODEL_PATH)
preprocessor = joblib.load(PREPROCESSOR_PATH)

class Customer(BaseModel):
    # accept arbitrary keys, but Pydantic needs fields: we allow dynamic by dict
    __root__: dict

@app.post("/predict")
def predict_single(payload: Customer):
    data = payload.__root__
    df = pd.DataFrame([data])
    # Ensure same columns order/handling as during training: preprocessor expects raw X (original columns)
    X_trans = preprocessor.transform(df)
    proba = model.predict_proba(X_trans)[:,1][0]
    pred = int(proba >= 0.5)
    return {"churn_probability": float(proba), "prediction": int(pred)}

@app.post("/predict_batch")
def predict_batch(payload: dict):
    # expecting {"data": [ {..}, {...} ]}
    arr = payload.get("data", [])
    df = pd.DataFrame(arr)
    X_trans = preprocessor.transform(df)
    probas = model.predict_proba(X_trans)[:,1].tolist()
    preds = [int(p >= 0.5) for p in probas]
    return {"probabilities": probas, "predictions": preds}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
