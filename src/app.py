from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import os

app = FastAPI()

# HTML template directory
templates = Jinja2Templates(directory="templates")

# Load model artifacts
MODEL_DIR = "model"
model = joblib.load(os.path.join(MODEL_DIR, "xgboost_model.pkl"))
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
feature_columns = joblib.load(os.path.join(MODEL_DIR, "feature_columns.pkl"))

# Input schema
class TransactionInput(BaseModel):
    Transaction_Amount: float
    Account_Balance: float
    Avg_Transaction_Amount_7d: float
    Transaction_Distance: float
    Card_Age: int
    Transaction_Type: str
    Device_Type: str
    Location: str
    Merchant_Category: str
    Card_Type: str
    Authentication_Method: str

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
def predict_fraud(data: TransactionInput):
    try:
        df = pd.DataFrame([data.dict()])
        df["Transaction_Amount_Log"] = np.log1p(df["Transaction_Amount"])

        categorical_cols = [
            'Transaction_Type', 'Device_Type', 'Location',
            'Merchant_Category', 'Card_Type', 'Authentication_Method'
        ]
        df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

        for col in feature_columns:
            if col not in df_encoded.columns:
                df_encoded[col] = 0

        df_encoded = df_encoded[feature_columns]

        scale_cols = [
            'Transaction_Amount_Log', 'Account_Balance',
            'Avg_Transaction_Amount_7d', 'Transaction_Distance', 'Card_Age'
        ]
        df_encoded[scale_cols] = scaler.transform(df_encoded[scale_cols])

        prediction = model.predict(df_encoded)[0]
        probability = model.predict_proba(df_encoded)[0, 1]

        return {
            "prediction": int(prediction),
            "fraud_probability": round(float(probability), 6)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
