# NYC Taxi Trip Duration Predictor

An end-to-end machine learning system that predicts how long a New York City taxi trip will take, based on pickup/drop-off location, time, passenger count, and vendor. Built as a complete data science project — from problem definition through EDA, model building, and deployment as a live web application.

Given a trip's details, the system returns a predicted trip duration in seconds/minutes, using a K-Nearest Neighbours (KNN) regression model trained on ~1.4 million historical NYC taxi trips.

---

## Table of Contents

- [Overview](#overview)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Model Summary](#model-summary)
- [Model Files](#model-files)
- [Running Locally](#running-locally)
- [Running with Docker](#running-with-docker)
- [API Reference](#api-reference)
- [Deployment (Azure)](#deployment-azure)
- [Project Methodology](#project-methodology)
- [Author](#author)

---

## Overview

This project was built in six stages, following the standard model-building life cycle:

1. **Problem Definition** — can trip duration be predicted from information known before a trip starts?
2. **Hypothesis Generation** — which features (distance, time of day, day of week, vendor, passenger count) are likely to matter?
3. **Data Collection** — NYC taxi trip dataset (~1.46 million records).
4. **Data Exploration & Transformation (EDA)** — missing values, duplicates, data types, univariate/bivariate analysis, correlation heatmap.
5. **Predictive Modelling** — outlier treatment, feature engineering (Haversine distance), KNN regression (tuned via cross-validation), benchmarked against Linear Regression.
6. **Deployment** — the trained model served through a REST API, with a web front end for real-time predictions.

The full analysis and model-building process is documented in [`Untitled3.ipynb`](./Untitled3.ipynb). This README covers the deployed system (Stage 6).

---

## Tech Stack

| Layer | Technology |
|---|---|
| Model | scikit-learn (K-Nearest Neighbours Regressor) |
| Backend API | FastAPI |
| Frontend | Streamlit |
| Containerisation | Docker, Docker Compose |
| Cloud Platform | Azure ML |
| Data Analysis | Pandas, NumPy, Matplotlib, Seaborn |
| Notebook | Jupyter |

---

## Project Structure

```text
nyc-taxi-trip-duration/
│
├── backend/
│   ├── main.py              # FastAPI app: loads model + scaler, serves /predict
│   ├── requirements.txt
│   └── Dockerfile
│
├── frontend/
│   ├── app.py                # Streamlit UI: trip form + prediction display
│   ├── requirements.txt
│   ├── Dockerfile
│   ├── .streamlit/
│   │   └── config.toml
│   └── static/
│       └── taxi_banner.jpg
│
├── model/
│   ├── knn_model.pkl          # trained KNN model (K=15) — not committed, see below
│   └── scaler.pkl              # fitted StandardScaler
│
├── docker-compose.yml
├── .gitignore
├── Untitled3.ipynb           # full EDA + model-building notebook (Stages 1–5)
└── README.md
```

---

## Model Summary

The final model is a **KNN Regressor (K=15)**, selected after testing K = 1, 3, 5, 9, 15, 25, 50, 75, 100 and confirming the choice with 5-fold cross-validation.

| Metric | KNN (K=15) | Linear Regression |
|---|---|---|
| MAE | 199.90 sec | 277.96 sec |
| RMSE | 322.82 sec | 412.05 sec |
| R² | 0.754 | 0.599 |

**Features used:**

`vendor_id`, `passenger_count`, `pickup_longitude`, `pickup_latitude`, `dropoff_longitude`, `dropoff_latitude`, `store_and_fwd_flag`, `day_of_week`, `hour_of_day`, `trip_distance_km`

`trip_distance_km` is engineered at prediction time using the **Haversine formula**, calculating real-world distance (in km) between the pickup and drop-off coordinates.

Full preprocessing decisions, outlier treatment reasoning, and diagnostics are documented in the notebook and accompanying presentation materials.

---

## Model Files

`model/knn_model.pkl` is **not included** in this repository — it exceeds GitHub's 100MB file size limit. This is expected: KNN doesn't learn compressed weights like other models, it stores its entire training set internally to look up nearest neighbours at prediction time, which with ~1.15 million training rows produces a large file.

To regenerate it, run the final export cell in `Untitled3.ipynb`:

```python
import joblib
joblib.dump(final_knn, 'knn_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
```

Place the resulting `knn_model.pkl` and `scaler.pkl` into the `model/` folder before running the backend.

---

## Running Locally

### 1. Backend (FastAPI)

```bash
cd backend
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

The API will be available at `http://localhost:8000`, with interactive docs at `http://localhost:8000/docs`.

### 2. Frontend (Streamlit)

In a separate terminal:

```bash
cd frontend
python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

The app will open at `http://localhost:8501`.

> Make sure the backend is running first — the frontend sends prediction requests to it.

---

## Running with Docker

To run the full system (backend + frontend) in containers:

```bash
docker-compose up --build
```

- Backend: `http://localhost:8000`
- Frontend: `http://localhost:8501`

Stop the containers with:

```bash
docker-compose down
```

---

## API Reference

### `POST /predict`

Predicts trip duration for a given set of trip details.

**Request body:**

```json
{
  "vendor_id": 2,
  "passenger_count": 2,
  "pickup_longitude": -73.9817,
  "pickup_latitude": 40.7541,
  "dropoff_longitude": -73.9776,
  "dropoff_latitude": 40.7614,
  "store_and_fwd_flag": 0,
  "day_of_week": 2,
  "hour_of_day": 14
}
```

**Response:**

```json
{
  "predicted_duration_seconds": 744.0,
  "predicted_duration_minutes": 12.4
}
```

Distance (`trip_distance_km`) is calculated server-side from the pickup/drop-off coordinates using the Haversine formula — it doesn't need to be supplied in the request.

---

## Deployment (Azure)

This project targets **Azure ML Managed Online Endpoints** for production deployment:

1. Register the trained model (`knn_model.pkl`) and scaler as Azure ML model assets.
2. Define a scoring script (`init()` / `run()`) matching the `backend/main.py` prediction logic.
3. Build an Azure ML environment from `backend/requirements.txt`.
4. Create a Managed Online Endpoint and deployment pointing to the registered model.
5. Point the Streamlit frontend at the resulting Azure scoring URI instead of `localhost`.

This gives the project a live, shareable HTTPS endpoint rather than requiring a local server to be running for a demo.

---

## Project Methodology

The full data science process — problem definition, hypotheses, data cleaning, outlier treatment (domain-based vs. IQR), feature engineering, model tuning, and evaluation — is documented step by step in [`Untitled3.ipynb`](./Untitled3.ipynb), with accompanying written reports at each stage.

**Key methodology decisions:**
- Outliers were treated using **domain-based reasoning** rather than strict IQR cutoffs, since IQR would have discarded a large volume of valid trips (e.g. flagging any trip over 35 minutes as an outlier).
- The scaler was fit **only on training data** and applied to the test set, to prevent data leakage.
- K was chosen using an overfitting/underfitting curve **and** confirmed with 5-fold cross-validation, not a single train/test split.
- KNN was benchmarked against Linear Regression, with residual diagnostics used to explain why KNN performed better (the underlying relationship isn't fully linear).

---

## Author

**Thato Xauka**
WIL AI/ML Programme — Work-Integrated Learning Project
