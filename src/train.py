import pandas as pd
import yaml
import json
import os
import logging
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score

import mlflow
import mlflow.sklearn

# -------------------- Logging --------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# -------------------- Paths --------------------
TRAIN_PATH = "data/processed/train.csv"
TEST_PATH = "data/processed/test.csv"
MODEL_DIR = "models"

# -------------------- Load Params --------------------
def load_params():
    with open("params.yaml") as f:
        return yaml.safe_load(f)["train"]

# -------------------- Load Data --------------------
def load_data():
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)

    # ✅ FIX: use lowercase "churn"
    X_tr = train.drop("churn", axis=1)
    y_tr = train["churn"]

    X_te = test.drop("churn", axis=1)
    y_te = test["churn"]

    return X_tr, y_tr, X_te, y_te

# -------------------- Train --------------------
def train_model(X_tr, y_tr, params):
    logger.info("Training RandomForestClassifier...")

    model = RandomForestClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        random_state=params["random_state"],
        n_jobs=-1
    )

    model.fit(X_tr, y_tr)
    return model

# -------------------- Evaluate --------------------
def evaluate(model, X_te, y_te):
    preds = model.predict(X_te)
    probs = model.predict_proba(X_te)[:, 1]

    metrics = {
        "accuracy": round(accuracy_score(y_te, preds), 4),
        "f1": round(f1_score(y_te, preds), 4),
        "roc_auc": round(roc_auc_score(y_te, probs), 4),
        "precision": round(precision_score(y_te, preds), 4),
        "recall": round(recall_score(y_te, preds), 4),
    }

    logger.info(f"Metrics: {metrics}")
    return metrics

# -------------------- Save --------------------
def save_artifacts(model, metrics, feature_names):
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Save model
    joblib.dump(model, f"{MODEL_DIR}/model.pkl")

    # Save feature names
    with open(f"{MODEL_DIR}/feature_names.json", "w") as f:
        json.dump(feature_names, f)

    # Save metrics
    with open("metrics.json", "w") as f:
        json.dump(metrics, f)

    logger.info("Model saved to models/model.pkl")

# -------------------- Main --------------------
def main():
    params = load_params()

    # MLflow setup (local)
    mlflow.set_experiment("telecom-churn-prediction")

    X_tr, y_tr, X_te, y_te = load_data()

    with mlflow.start_run():
        model = train_model(X_tr, y_tr, params)

        metrics = evaluate(model, X_te, y_te)

        # Log to MLflow
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "model")

        # Save locally for DVC
        save_artifacts(model, metrics, X_tr.columns.tolist())

        logger.info(f"F1: {metrics['f1']}  AUC: {metrics['roc_auc']}")

# -------------------- Entry --------------------
if __name__ == "__main__":
    main()