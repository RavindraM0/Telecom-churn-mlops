import pandas as pd
import json
import logging
import joblib

from sklearn.metrics import (
    classification_report,
    accuracy_score,
    f1_score,
    roc_auc_score,
    precision_score,
    recall_score
)

# -------------------- Logging --------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# -------------------- Paths --------------------
TEST_PATH = "data/processed/test.csv"
MODEL_PATH = "models/model.pkl"
FEATURES_PATH = "models/feature_names.json"

# -------------------- Main --------------------
def main():
    # Load test data
    test = pd.read_csv(TEST_PATH)

    # ✅ FIX: use lowercase "churn"
    y_te = test["churn"]
    X_te = test.drop("churn", axis=1)

    # Load model
    model = joblib.load(MODEL_PATH)

    # Load feature names (ensure correct order)
    with open(FEATURES_PATH) as f:
        feature_names = json.load(f)

    # Align columns
    X_te = X_te.reindex(columns=feature_names, fill_value=0)

    # Predictions
    preds = model.predict(X_te)
    probs = model.predict_proba(X_te)[:, 1]

    # Metrics
    metrics = {
        "accuracy": round(accuracy_score(y_te, preds), 4),
        "f1": round(f1_score(y_te, preds), 4),
        "roc_auc": round(roc_auc_score(y_te, probs), 4),
        "precision": round(precision_score(y_te, preds), 4),
        "recall": round(recall_score(y_te, preds), 4),
    }

    # Classification report
    report = classification_report(y_te, preds, target_names=["No Churn", "Churn"])
    logger.info("\n" + report)
    logger.info(f"Metrics: {metrics}")

    # Save metrics (overwrite same file safely)
    with open("metrics.json", "w") as f:
        json.dump(metrics, f)

# -------------------- Entry --------------------
if __name__ == "__main__":
    main()