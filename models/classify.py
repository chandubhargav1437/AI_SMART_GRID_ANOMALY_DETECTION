"""
Phase 3 — Stage 2: Fault Classification with XGBoost (supervised)
Runs only on anomaly rows flagged by Isolation Forest.
"""

import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.sklearn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier

DATA_PATH   = Path(__file__).parent.parent / "data" / "grid_data_detected.csv"
MODEL_PATH  = Path(__file__).parent / "saved" / "xgboost_classifier.pkl"
CM_PATH     = Path(__file__).parent / "saved" / "confusion_matrix.png"

FEATURE_EXCLUDE = ['timestamp', 'fault_type', 'is_anomaly', 'anomaly_score']
FAULT_NAMES = {1: "Overvoltage", 2: "Voltage Sag", 3: "Overload", 4: "Spike", 5: "Line Fault"}


def get_features(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in FEATURE_EXCLUDE]


def main():
    print("Loading grid_data_detected.csv ...")
    df = pd.read_csv(DATA_PATH, parse_dates=['timestamp'])

    # Keep only anomaly rows, drop normal fault_type==0
    anomaly_df = df[(df['is_anomaly'] == -1) & (df['fault_type'] != 0)].copy()
    print(f"Anomaly rows with known fault type: {len(anomaly_df)}")
    print(f"Fault type distribution:\n{anomaly_df['fault_type'].value_counts()}")

    features = get_features(df)
    X = anomaly_df[features].values

    # Remap labels to 0-based for XGBoost (1-5 → 0-4)
    y_raw = anomaly_df['fault_type'].astype(int).values
    y = y_raw - 1   # XGBoost expects 0-indexed classes

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"Train: {len(X_train)} | Test: {len(X_test)}")

    model = XGBClassifier(
        n_estimators=100,
        random_state=42,
        eval_metric='mlogloss',
        use_label_encoder=False,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_pred_raw = y_pred + 1   # back to 1-based

    # Classification report (original labels 1-5)
    target_names = [FAULT_NAMES[i] for i in sorted(FAULT_NAMES.keys())]
    report = classification_report(
        y_test + 1, y_pred_raw,
        target_names=target_names,
        output_dict=True
    )
    report_str = classification_report(y_test + 1, y_pred_raw, target_names=target_names)
    print("\n=== Classification Report ===")
    print(report_str)

    weighted = report['weighted avg']
    precision = weighted['precision']
    recall    = weighted['recall']
    f1        = weighted['f1-score']

    # --- Confusion matrix PNG ---
    cm = confusion_matrix(y_test + 1, y_pred_raw)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=target_names, yticklabels=target_names, ax=ax)
    ax.set_title("XGBoost Fault Classification — Confusion Matrix")
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    plt.tight_layout()
    CM_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(CM_PATH, dpi=150)
    plt.close()
    print(f"Confusion matrix saved -> {CM_PATH}")

    # --- MLflow logging ---
    mlflow.set_experiment("smart-grid-anomaly-detection")
    with mlflow.start_run(run_name="xgboost_classifier"):
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("test_size", 0.2)
        mlflow.log_metric("weighted_precision", precision)
        mlflow.log_metric("weighted_recall", recall)
        mlflow.log_metric("weighted_f1", f1)
        mlflow.sklearn.log_model(model, "xgboost_classifier")
        mlflow.log_artifact(str(CM_PATH))

    # Save model
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved -> {MODEL_PATH}")
    print(f"\nWeighted Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")

    if precision >= 0.90 and recall >= 0.90:
        print("[PASS] Precision and Recall are both >= 90% -- Phase 3 verified!")
    else:
        print("[WARN] Metrics below 90% -- check fault coverage.")


if __name__ == "__main__":
    main()
