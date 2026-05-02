from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import pandas as pd
import joblib
import json

# -------------------- Load artifacts --------------------
MODEL = joblib.load("models/model.pkl")
FEATURES = json.load(open("models/feature_names.json"))

# -------------------- App --------------------
app = FastAPI(
    title="Telecom Churn Prediction API",
    description="MLOps pipeline — predict customer churn",
    version="1.0.0"
)

# -------------------- Input Schema --------------------
class CustomerData(BaseModel):
    tenure: int = Field(..., ge=0)
    MonthlyCharges: float = Field(..., ge=0)
    TotalCharges: float = Field(..., ge=0)
    SeniorCitizen: int = Field(0, ge=0, le=1)

    Partner_Yes: Optional[int] = 0
    Dependents_Yes: Optional[int] = 0
    PhoneService_Yes: Optional[int] = 0
    PaperlessBilling_Yes: Optional[int] = 0
    MultipleLines_No_phone_service: Optional[int] = 0
    MultipleLines_Yes: Optional[int] = 0
    InternetService_Fiber_optic: Optional[int] = 0
    InternetService_No: Optional[int] = 0
    OnlineSecurity_No_internet_service: Optional[int] = 0
    OnlineSecurity_Yes: Optional[int] = 0
    OnlineBackup_No_internet_service: Optional[int] = 0
    OnlineBackup_Yes: Optional[int] = 0
    DeviceProtection_No_internet_service: Optional[int] = 0
    DeviceProtection_Yes: Optional[int] = 0
    TechSupport_No_internet_service: Optional[int] = 0
    TechSupport_Yes: Optional[int] = 0
    StreamingTV_No_internet_service: Optional[int] = 0
    StreamingTV_Yes: Optional[int] = 0
    StreamingMovies_No_internet_service: Optional[int] = 0
    StreamingMovies_Yes: Optional[int] = 0
    Contract_One_year: Optional[int] = 0
    Contract_Two_year: Optional[int] = 0
    PaymentMethod_Credit_card_automatic: Optional[int] = 0
    PaymentMethod_Electronic_check: Optional[int] = 0
    PaymentMethod_Mailed_check: Optional[int] = 0


# -------------------- Output Schema --------------------
class PredictionResponse(BaseModel):
    churn: bool
    probability: float
    risk: str


# -------------------- Routes --------------------
@app.get("/")
def root():
    return {"message": "Telecom Churn API", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "ok", "version": "1.0.0"}


@app.post("/predict", response_model=PredictionResponse)
def predict_churn(data: CustomerData):
    try:
        # Convert input to DataFrame
        df = pd.DataFrame([data.model_dump()])

        # 🔥 CRITICAL FIX: align features
        df = df.reindex(columns=FEATURES, fill_value=0)

        # Predict
        pred = int(MODEL.predict(df)[0])
        prob = float(MODEL.predict_proba(df)[0][1])

        # Risk category
        if prob > 0.7:
            risk = "High"
        elif prob > 0.4:
            risk = "Medium"
        else:
            risk = "Low"

        return {
            "churn": bool(pred),
            "probability": round(prob, 3),
            "risk": risk
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model/info")
def model_info():
    metrics = json.load(open("metrics.json"))
    return {
        "n_features": len(FEATURES),
        "metrics": metrics
    }