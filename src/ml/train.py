import os
import pandas as pd
from preprocessing import preprocess_pipeline
from eda import full_eda
from modeling import (
    train_logistic,
    train_xgboost,
    train_isolation_forest,
    evaluate_model,
    evaluate_isolation_forest,
    cross_validate_model
)
import joblib

DATA_PATH = "../data/fraud_dataset.csv"
MODEL_DIR = "../model"

def main():
    # Ensure model directory exists
    os.makedirs(MODEL_DIR, exist_ok=True)

    # 1. Load data
    df = pd.read_csv(DATA_PATH, parse_dates=["Timestamp"])
    print("Data loaded. Shape:", df.shape)

    # 2. Run Exploratory Data Analysis
    full_eda(df, target_col="Fraud_Label")

    # 3. Preprocess data
    X_train, X_test, y_train, y_test, scaler = preprocess_pipeline(df)

    # 4. Train and evaluate Logistic Regression
    print("\nTraining Logistic Regression...")
    log_model = train_logistic(X_train, y_train)
    cross_validate_model(log_model, X_train, y_train, model_name="Logistic Regression")
    evaluate_model(log_model, X_test, y_test)
    joblib.dump(log_model, os.path.join(MODEL_DIR, "logistic_model.pkl"))

    # 5. Train and evaluate XGBoost
    print("\nTraining XGBoost...")
    xgb_model = train_xgboost(X_train, y_train)
    cross_validate_model(xgb_model, X_train, y_train, model_name="XGBoost")
    evaluate_model(xgb_model, X_test, y_test)
    joblib.dump(xgb_model, os.path.join(MODEL_DIR, "xgboost_model.pkl"))

    # 6. (Optional) Train and evaluate Isolation Forest
    # Note: Isolation Forest is unsupervised and not ideal for labeled fraud data
    print("\nTraining Isolation Forest (optional)...")
    iso_model = train_isolation_forest(X_train, contamination=0.32)
    evaluate_isolation_forest(iso_model, X_test, y_test)
    joblib.dump(iso_model, os.path.join(MODEL_DIR, "isolation_forest_model.pkl"))

    # 7. Save scaler
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))

    print("\nAll models and scaler have been saved.")

if __name__ == "__main__":
    main()
