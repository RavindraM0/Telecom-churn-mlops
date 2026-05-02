import joblib, json
import pandas as pd

_model         = None
_feature_names = None

def _load():
    global _model, _feature_names
    if _model is None:
        _model         = joblib.load("models/model.pkl")
        _feature_names = json.load(open("models/feature_names.json"))

def predict(data: dict) -> dict:
    _load()
    df    = pd.DataFrame([data]).reindex(
        columns=_feature_names, fill_value=0
    )
    pred  = int(_model.predict(df)[0])
    proba = float(_model.predict_proba(df)[0][1])
    return {
        "churn":       bool(pred),
        "probability": round(proba, 4),
        "risk": (
            "high"   if proba > 0.7 else
            "medium" if proba > 0.4 else
            "low"
        )
    }