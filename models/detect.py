"""
Phase 2 — Stage 1: Anomaly Detection with Isolation Forest (unsupervised)
Train on normal data only, predict on the full dataset.
"""

import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from pathlib import Path
from sklearn.ensemble import IsolationForest

DATA_PATH      = Path(__file__).parent.parent / "data" / "grid_data.csv"
OUTPUT_PATH    = Path(__file__).parent.parent / "data" / "grid_data_detected.csv"
MODEL_PATH     = Path(__file__).parent / "saved" / "isolation_forest.pkl"

FEATURE_EXCLUDE = ['timestamp', 'fault_type']


def get_features(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in FEATURE_EXCLUDE]


def main():
    print("Loading grid_data.csv ...")
    df = pd.read_csv(DATA_PATH, parse_dates=['timestamp'])

    features = get_features(df)
    X = df[features].values

    # Train ONLY on normal rows
    X_normal = df.loc[df['fault_type'] == 0, features].values
    print(f"Training Isolation Forest on {len(X_normal)} normal rows ...")

    model = IsolationForest(contamination=0.1, random_state=42, n_jobs=-1)
    model.fit(X_normal)

    # Predict on entire dataset
    df['anomaly_score'] = model.decision_function(X)
    df['is_anomaly']    = model.predict(X)          # -1 = anomaly, 1 = normal

    anomaly_count = (df['is_anomaly'] == -1).sum()
    detection_rate = anomaly_count / len(df)

    # --- MLflow logging ---
    mlflow.set_experiment("smart-grid-anomaly-detection")
    with mlflow.start_run(run_name="isolation_forest"):
        mlflow.log_param("contamination", 0.1)
        mlflow.log_metric("anomaly_detection_rate", detection_rate)
        mlflow.sklearn.log_model(model, "isolation_forest")

    # Save model
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved -> {MODEL_PATH}")

    # Save enriched CSV
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Detected data saved -> {OUTPUT_PATH}")
    print(f"Anomaly detection complete. Anomalies found: {anomaly_count} rows ({detection_rate:.2%})")
    print(f"is_anomaly value counts:\n{df['is_anomaly'].value_counts()}")


if __name__ == "__main__":
    main()
