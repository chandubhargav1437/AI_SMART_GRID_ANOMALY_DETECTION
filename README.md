# Smart Grid Anomaly Detection & Response System

## Section 1 — Project Overview

This system simulates real-time power grid sensor data conforming to the IEC 61850 standard and applies a two-stage ML pipeline to detect and classify electrical faults. An automated response engine maps each detected fault to a corrective grid action (relay trip, load shedding, power rerouting), and a live Streamlit dashboard streams anomaly alerts in real time. The full system is containerized with Docker and all ML experiments are tracked with MLflow.

---

## Section 2 — Architecture

```
[Data Simulator] -> [Isolation Forest] -> [XGBoost Classifier] -> [Response Engine] -> [Streamlit Dashboard]
 IEC 61850 data      anomaly flag           fault type label        corrective action      live alerts
```

**Pipeline detail:**
- **Stage 1 (Unsupervised):** Isolation Forest trained on normal-only data flags anomalies across the full 50,000-row dataset.
- **Stage 2 (Supervised):** XGBoost classifier trained only on flagged anomaly rows identifies the specific fault type (1–5).
- **Response Engine:** Confidence-aware logic triggers automated grid actions for high-confidence predictions and escalates to human review for ambiguous cases.

---

## Section 3 — How to Run Locally

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate synthetic grid data (50,000 rows)
python data/simulate.py

# 3. Run Stage 1 — Isolation Forest anomaly detection
python models/detect.py

# 4. Run Stage 2 — XGBoost fault classification
python models/classify.py

# 5. Launch the live dashboard
streamlit run dashboard/app.py
```

Dashboard opens at: **http://localhost:8501**
MLflow UI (if running locally): `mlflow ui` then visit **http://localhost:5000**

---

## Section 4 — How to Run with Docker

```bash
docker-compose up
```

| Service   | URL                        |
|-----------|----------------------------|
| Dashboard | http://localhost:8501      |
| MLflow UI | http://localhost:5000      |

> **Note:** Run `python data/simulate.py`, `python models/detect.py`, and `python models/classify.py` before `docker-compose up` to ensure data and model `.pkl` files exist in the mounted volumes.

---

## Section 5 — ML Results

### Stage 1 — Isolation Forest
| Metric                | Value  |
|-----------------------|--------|
| Contamination param   | 0.10   |
| Anomaly rows detected | 7,958  |
| Detection rate        | 15.92% |

### Stage 2 — XGBoost Classifier (on anomaly rows only)

```
              precision    recall  f1-score   support

 Overvoltage       1.00      1.00      1.00        82
 Voltage Sag       1.00      0.99      0.99        96
    Overload       1.00      1.00      1.00        34
       Spike       1.00      1.00      1.00       301
  Line Fault       0.99      1.00      1.00       198

    accuracy                           1.00       711
   macro avg       1.00      1.00      1.00       711
weighted avg       1.00      1.00      1.00       711
```

**Weighted Precision: 99.86% | Weighted Recall: 99.86% | F1: 99.86%**

### Confusion Matrix (Test Set — 711 rows)
```
             Overvoltage  Voltage Sag  Overload  Spike  Line Fault
Overvoltage       82           0          0        0        0
Voltage Sag        0          95          0        0        1
Overload           0           0         34        0        0
Spike              0           0          0      301        0
Line Fault         0           0          0        0      198
```

---

## Section 6 — Real-World Connection

This system mirrors how real Grid Automation systems work. The IEC 61850 standard governs substation communication in real SCADA systems used by companies like GE Vernova. The two-stage detection pipeline replicates how protection relays detect and classify faults in power networks. The confidence-aware response engine models the self-healing grid concept where automated actions are taken for high-confidence faults and human operators are alerted for ambiguous cases, reflecting actual safety constraints in live grid systems.

---

## Resume Bullet Points

- **Built a two-stage ML pipeline** — Isolation Forest (unsupervised) + XGBoost (supervised) — on IEC 61850-structured synthetic grid sensor data, achieving **99.86% classification accuracy** across 5 fault types including overvoltage, voltage sag, overload, spike, and line fault.
- **Implemented a confidence-aware response engine** mapping detected faults to corrective grid actions (relay trip, load shedding, power rerouting); low-confidence predictions trigger human review, reflecting real safety constraints in power grid systems.
- **Deployed a real-time Streamlit monitoring dashboard** with live time-series visualization and anomaly alerts; containerized full system using Docker and tracked all ML experiments with MLflow.

---

## Project Structure

```
smart-grid-anomaly/
├── data/
│   ├── simulate.py           # Phase 1 — IEC 61850 synthetic data generator
│   ├── grid_data.csv         # 50,000 rows of raw grid sensor data
│   └── grid_data_detected.csv# Enriched with is_anomaly + anomaly_score
├── models/
│   ├── detect.py             # Phase 2 — Isolation Forest anomaly detection
│   ├── classify.py           # Phase 3 — XGBoost fault classification
│   └── saved/
│       ├── isolation_forest.pkl
│       ├── xgboost_classifier.pkl
│       └── confusion_matrix.png
├── engine/
│   └── response.py           # Phase 4 — Confidence-aware response engine
├── dashboard/
│   └── app.py                # Phase 5 — Real-time Streamlit dashboard
├── tests/
│   └── test_pipeline.py      # Phase 6 — 4/4 pytest tests
├── Dockerfile                # Phase 7
├── docker-compose.yml        # Phase 8
├── requirements.txt          # Phase 9
└── README.md                 # Phase 10
```
