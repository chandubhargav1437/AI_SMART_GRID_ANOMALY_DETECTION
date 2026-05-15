# Smart Grid Anomaly Detection & Response System

## Overview

Smart Grid Anomaly Detection is a GE Vernova-inspired proof of concept for fault detection, classification, and automated response in electric power systems. The repository simulates IEC 61850-style grid sensor data, uses a two-stage machine learning pipeline to detect and classify anomalies, and supports a live Streamlit dashboard for real-time monitoring.

Key capabilities:
- Synthetic power grid data generation
- Unsupervised anomaly detection with Isolation Forest
- Supervised fault classification with XGBoost
- Confidence-aware response engine for corrective actions
- Streamlit dashboard for live alert monitoring
- Docker and MLflow integration for reproducibility and deployment

## Architecture

```text
[Data Simulator] -> [Isolation Forest] -> [XGBoost Classifier] -> [Response Engine] -> [Streamlit Dashboard]
    IEC 61850-style data       anomaly flag            fault type label       corrective action      live alerts
```

### Pipeline Stages
- **Stage 1:** `data/simulate.py` generates synthetic grid telemetry and fault labels.
- **Stage 2:** `models/detect.py` trains an Isolation Forest on normal data and flags anomalies in the full dataset.
- **Stage 3:** `models/classify.py` trains an XGBoost classifier on anomaly rows to predict fault type.
- **Stage 4:** `engine/response.py` maps predictions to corrective grid actions and handles low-confidence cases.
- **Stage 5:** `dashboard/app.py` displays live monitoring and alert dashboards using Streamlit.

## Repository Contents

- `data/`
  - `simulate.py` — synthetic IEC 61850-style grid data generator
  - `grid_data.csv` — raw sensor data
  - `grid_data_detected.csv` — enriched dataset after anomaly detection
- `models/`
  - `detect.py` — Isolation Forest anomaly detection flow
  - `classify.py` — XGBoost fault classification flow
  - `saved/` — saved model artifacts and confusion matrix output
- `engine/`
  - `response.py` — response logic for automated actions and human review
- `dashboard/`
  - `app.py` — Streamlit dashboard for alert visualization
- `tests/`
  - `test_pipeline.py` — pipeline validation tests
- `Dockerfile` — container image definition
- `docker-compose.yml` — local service orchestration
- `requirements.txt` — Python dependencies

## Prerequisites

- Python 3.11+ (or compatible version)
- `pip` package manager
- Docker & Docker Compose (optional, for containerized execution)

## Setup

1. Create a virtual environment (recommended):

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1  # PowerShell
# or
.\.venv\Scripts\activate.bat  # cmd
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Running Locally

1. Generate synthetic grid data:

```bash
python data/simulate.py
```

2. Run anomaly detection:

```bash
python models/detect.py
```

3. Run fault classification:

```bash
python models/classify.py
```

4. Launch the Streamlit dashboard:

```bash
streamlit run dashboard/app.py
```

5. (Optional) Start MLflow UI:

```bash
mlflow ui
```

- Dashboard: `http://localhost:8501`
- MLflow UI: `http://localhost:5000`

## Docker Usage

To start the application stack with Docker Compose:

```bash
docker-compose up --build
```

> If the repository does not already contain generated data and saved models, run `python data/simulate.py`, `python models/detect.py`, and `python models/classify.py` first.

## Model and Result Summary

### Isolation Forest
- Trained on normal-only rows from `data/grid_data.csv`
- Contamination parameter: `0.10`
- Saves model to `models/saved/isolation_forest.pkl`
- Outputs enriched dataset to `data/grid_data_detected.csv`

### XGBoost Classifier
- Trained on anomaly rows only
- Predicts 5 fault types:
  - Overvoltage
  - Voltage Sag
  - Overload
  - Spike
  - Line Fault
- Saves model to `models/saved/xgboost_classifier.pkl`
- Generates a confusion matrix image at `models/saved/confusion_matrix.png`

## Response Engine

The response engine determines whether a fault can be acted on automatically or should be escalated for manual review.
- If prediction confidence is below `0.60`, the system returns a low-confidence alert and recommends human review.
- If confidence is sufficient, it maps the fault type to a corrective action such as relay tripping, load shedding, or rerouting power.

## Testing

Run the project tests with:

```bash
pytest tests/test_pipeline.py
```

## Notes

- The project is designed as a proof of concept for grid automation and anomaly response.
- Data and models are synthetic and intended for demonstration, experimentation, and architecture validation.

## Suggested Resume Highlights

- Built a two-stage grid anomaly detection pipeline using Isolation Forest and XGBoost.
- Implemented a confidence-aware response engine for automated corrective actions and human escalation.
- Delivered a real-time Streamlit monitoring dashboard and MLflow experiment tracking.

## Contact

For questions or contributions, open an issue or submit a pull request.
