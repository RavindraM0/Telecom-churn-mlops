import pandas as pd
from sklearn.model_selection import train_test_split
import os
import yaml
import logging

# -------------------- Logging --------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# -------------------- Paths --------------------
DATA_PATH = "data/raw/churn.csv"
OUT_DIR = "data/processed"

# -------------------- Load Params --------------------
def load_params():
    with open("params.yaml") as f:
        return yaml.safe_load(f)["train"]

# -------------------- Preprocessing --------------------
def preprocess(df):
    logger.info(f"Raw shape: {df.shape}")

    # Drop ID column
    if "customerID" in df.columns:
        df = df.drop("customerID", axis=1)

    # Fix TotalCharges
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    dropped = df["TotalCharges"].isna().sum()
    df = df.dropna(subset=["TotalCharges"])
    logger.info(f"Dropped {dropped} rows with invalid TotalCharges")

    # ✅ Convert target and standardize name
    df["churn"] = (df["Churn"] == "Yes").astype(int)
    df = df.drop("Churn", axis=1)

    # One-hot encoding
    df = pd.get_dummies(df, drop_first=True)

    logger.info(f"Processed shape: {df.shape}")
    logger.info(f"Churn rate: {df['churn'].mean():.2%}")  # ✅ FIXED

    return df

# -------------------- Main Pipeline --------------------
def main():
    params = load_params()

    # Load data
    df = pd.read_csv(DATA_PATH)

    # Preprocess
    df = preprocess(df)

    # Split features & target
    X = df.drop("churn", axis=1)
    y = df["churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=params["test_size"],
        random_state=params["random_state"],
        stratify=y
    )

    # Create output dir
    os.makedirs(OUT_DIR, exist_ok=True)

    # Combine
    train = pd.concat([X_train, y_train], axis=1)
    test = pd.concat([X_test, y_test], axis=1)

    # Save outputs
    train.to_csv(f"{OUT_DIR}/train.csv", index=False)
    test.to_csv(f"{OUT_DIR}/test.csv", index=False)
    train.to_csv(f"{OUT_DIR}/reference.csv", index=False)

    logger.info(f"Train: {len(train)} rows | Test: {len(test)} rows")
    logger.info(f"Features: {X_train.shape[1]}")
    logger.info("Preprocessing completed successfully ✅")

# -------------------- Entry --------------------
if __name__ == "__main__":
    main()