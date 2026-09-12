<div align="center">

# 🚕 NYC Taxi Trip Duration Predictor

**An end-to-end machine learning system that predicts how long a New York City taxi trip will take**

Built as a complete data science project — from problem definition through EDA, model building, and deployment as a live, containerised web application.

<br/>

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-KNN%20Regressor-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Frontend-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Containerised-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![SQLite](https://img.shields.io/badge/SQLite-Database-003B57?style=for-the-badge&logo=sqlite&logoColor=white)
![Tableau](https://img.shields.io/badge/Tableau-Dashboard-E97627?style=for-the-badge&logo=tableau&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![GitHub](https://img.shields.io/badge/GitHub-Version%20Control-181717?style=for-the-badge&logo=github&logoColor=white)

</div>

---

Given a trip's details, the system returns a predicted trip duration in seconds/minutes, using a K-Nearest Neighbours (KNN) regression model trained on ~1.4 million historical NYC taxi trips.

---

## 📑 Table of Contents

- [🔎 Overview](#-overview)
- [🧰 Tech Stack](#-tech-stack)
- [📁 Project Structure](#-project-structure)
- [🤖 Model Summary](#-model-summary)
- [📦 Model Files](#-model-files)
- [💻 Running Locally](#-running-locally)
- [🐳 Running with Docker](#-running-with-docker)
- [🔌 API Reference](#-api-reference)
- [🗂️ Logging & Prediction History](#️-logging--prediction-history)
- [📊 Interactive Dashboard (Tableau)](#-interactive-dashboard-tableau)
- [📄 Reproducible Report](#-reproducible-report)
- [☁️ Deployment Status (Azure)](#️-deployment-status-azure)
- [🚧 Scope Decisions](#-scope-decisions)
- [🧪 Project Methodology](#-project-methodology)
- [👤 Author](#-author)

---

## 🔎 Overview

This project was built in six stages, following the standard model-building life cycle:

| # | Stage | Description |
|---|---|---|
| 1️⃣ | **Problem Definition** | Can trip duration be predicted from information known before a trip starts? |
| 2️⃣ | **Hypothesis Generation** | Which features (distance, time of day, day of week, vendor, passenger count) are likely to matter? |
| 3️⃣ | **Data Collection** | NYC taxi trip dataset (~1.46 million records) |
| 4️⃣ | **EDA** | Missing values, duplicates, data types, univariate/bivariate analysis, correlation heatmap |
| 5️⃣ | **Predictive Modelling** | Outlier treatment, feature engineering (Haversine distance), KNN regression tuned via cross-validation, benchmarked against Linear Regression |
| 6️⃣ | **Deployment** | Model served through a REST API, styled web front end, logging, persistent prediction history, full Docker containerisation |

📓 The full analysis is documented in [`Untitled3.ipynb`](./Untitled3.ipynb) and its standalone HTML export, [`Untitled3.html`](./Untitled3.html). This README covers the deployed system (Stage 6) and the Week 8 dashboard/reporting deliverables.

---

## 🧰 Tech Stack

| Layer | Technology | |
|---|---|---|
| 🤖 Model | scikit-learn (K-Nearest Neighbours Regressor) | ![scikit-learn](https://img.shields.io/badge/-scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white) |
| ⚡ Backend API | FastAPI, Uvicorn | ![FastAPI](https://img.shields.io/badge/-FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white) |
| 🖥️ Frontend | Streamlit | ![Streamlit](https://img.shields.io/badge/-Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white) |
| 🗄️ Database | SQLite | ![SQLite](https://img.shields.io/badge/-SQLite-003B57?style=flat-square&logo=sqlite&logoColor=white) |
| 📝 Logging | Python `logging` | ![Python](https://img.shields.io/badge/-Python-3776AB?style=flat-square&logo=python&logoColor=white) |
| 📦 Containerisation | Docker, Docker Compose | ![Docker](https://img.shields.io/badge/-Docker-2496ED?style=flat-square&logo=docker&logoColor=white) |
| 🔧 Version Control | Git, GitHub | ![Git](https://img.shields.io/badge/-Git-F05032?style=flat-square&logo=git&logoColor=white) ![GitHub](https://img.shields.io/badge/-GitHub-181717?style=flat-square&logo=github&logoColor=white) |
| 📊 Dashboard / Reporting | Tableau | ![Tableau](https://img.shields.io/badge/-Tableau-E97627?style=flat-square&logo=tableau&logoColor=white) |
| 📈 Data Analysis | Pandas, NumPy, Matplotlib, Seaborn | ![Pandas](https://img.shields.io/badge/-Pandas-150458?style=flat-square&logo=pandas&logoColor=white) ![NumPy](https://img.shields.io/badge/-NumPy-013243?style=flat-square&logo=numpy&logoColor=white) |
| 📓 Notebook | Jupyter (+ nbconvert for HTML export) | ![Jupyter](https://img.shields.io/badge/-Jupyter-F37626?style=flat-square&logo=jupyter&logoColor=white) |

---

## 📁 Project Structure

```text
nyc-taxi-trip-duration/
│
├── 🔧 backend/
│   ├── main.py              # FastAPI app: loads model + scaler, serves /predict,
│   │                        # handles logging and SQLite prediction history
│   ├── requirements.txt
│   ├── Dockerfile
│   ├── api.log              # generated at runtime — not committed
│   └── predictions.db       # generated at runtime — not committed
│
├── 🖥️ frontend/
│   ├── app.py                # Streamlit UI: trip form + prediction display
│   ├── requirements.txt
│   ├── Dockerfile
│   ├── .streamlit/
│   │   └── config.toml
│   └── static/
│       └── taxi_banner.jpg
│
├── 🤖 model/
│   ├── knn_model.pkl          # trained KNN model (K=15) — not committed, see below
│   └── scaler.pkl              # fitted StandardScaler
│
├── 🐳 docker-compose.yml
├── .gitignore
├── 📓 Untitled3.ipynb            # full EDA + model-building notebook (Stages 1–5)
├── 📄 Untitled3.html             # reproducible HTML export of the notebook
├── 📊 Tableau Packaged Workbook (.twbx).twb   # interactive dashboard
├── 🖼️ NYC_Taxi_Dashboard.png     # dashboard screenshot
└── README.md
```

---

## 🤖 Model Summary

The final model is a **KNN Regressor (K=15)**, selected after testing several values of K and confirming the choice with 5-fold cross-validation.

| Metric | KNN (K=15) ✅ | Linear Regression |
|---|:---:|:---:|
| MAE | **199.90 sec** | 277.96 sec |
| RMSE | **322.82 sec** | 412.05 sec |
| R² | **0.754** | 0.599 |

**Features used (exact training order):**

`vendor_id` · `passenger_count` · `pickup_longitude` · `pickup_latitude` · `dropoff_longitude` · `dropoff_latitude` · `store_and_fwd_flag` · `day_of_week` · `hour_of_day` · `trip_distance_km`

`trip_distance_km`, `day_of_week`, `hour_of_day`, and the numeric `store_and_fwd_flag` code are all engineered **server-side, inside the API**, from simpler raw inputs (coordinates and a timestamp) — a caller of the API never needs to compute these themselves. See [🔌 API Reference](#-api-reference) below.

Full preprocessing decisions, outlier treatment reasoning, and diagnostics are documented in the notebook and its HTML export.

---

## 📦 Model Files

> ⚠️ `model/knn_model.pkl` is **not included** in this repository — it exceeds GitHub's 100MB file size limit. This is expected: KNN doesn't learn compressed weights like other models, it stores its entire training set internally to look up nearest neighbours at prediction time, which with over a million training rows produces a large file.

To regenerate it, run the final export cell in `Untitled3.ipynb` (after running the full notebook top to bottom, so `final_knn` and `scaler` exist in memory):

```python
import joblib
joblib.dump(final_knn, 'knn_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
```

Place the resulting `knn_model.pkl` and `scaler.pkl` into the `model/` folder before running the backend.

---

## 💻 Running Locally

### 1️⃣ Backend (FastAPI)

```bash
cd backend
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

pip install -r requirements.txt
uvicorn main:app --reload
```

🌐 The API will be available at `http://localhost:8000`, with interactive docs at `http://localhost:8000/docs`.

### 2️⃣ Frontend (Streamlit)

In a **separate** terminal:

```bash
cd frontend
python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

🌐 The app will open at `http://localhost:8501`.

> 💡 Make sure the backend is running first — the frontend sends prediction requests to it.

---

## 🐳 Running with Docker

To run the full system (backend + frontend) in containers, from the project root:

```bash
docker compose up --build
```

| Service | URL |
|---|---|
| ⚡ Backend | `http://localhost:8000` |
| 🖥️ Frontend | `http://localhost:8501` |

Stop the containers with:

```bash
docker compose down
```

✅ This has been tested end-to-end: both images build successfully, both containers start cleanly, and a live prediction was confirmed working through the fully containerised stack.

> 🪟 **Note (Windows):** if Docker Desktop reports "Virtualisation support not detected" even though virtualisation is enabled in Task Manager, the underlying WSL 2 backend is likely outdated. Run `wsl --update`, restart your PC, and try again.

---

## 🔌 API Reference

### `POST /predict`

Predicts trip duration for a given trip. The API accepts raw, human-meaningful inputs — it computes distance, day of week, hour of day, and the flag's numeric code internally.

**Request body:**

```json
{
  "vendor_id": 2,
  "passenger_count": 1,
  "pickup_datetime": "2016-03-14T17:24:55",
  "pickup_longitude": -73.982,
  "pickup_latitude": 40.767,
  "dropoff_longitude": -73.964,
  "dropoff_latitude": 40.766,
  "store_and_fwd_flag": "N"
}
```

| Field | Type | Notes |
|---|---|---|
| `vendor_id` | int | 1 or 2 |
| `passenger_count` | int | 1–6 |
| `pickup_datetime` | ISO 8601 string | used to derive `day_of_week` and `hour_of_day` |
| `pickup_longitude` / `pickup_latitude` | float | constrained to the NYC bounding box |
| `dropoff_longitude` / `dropoff_latitude` | float | constrained to the NYC bounding box |
| `store_and_fwd_flag` | string | `"Y"` or `"N"` |

**Response:**

```json
{
  "predicted_trip_duration_seconds": 656.27,
  "predicted_trip_duration_minutes": 10.94
}
```

📏 `trip_distance_km` is calculated server-side from the pickup/drop-off coordinates using the Haversine formula — it is never supplied in the request.

### `GET /`

❤️ Basic health check, returns `{"status": "ok", "message": "NYC Taxi Trip Duration API is running"}`.

---

## 🗂️ Logging & Prediction History

📝 The backend writes a structured log entry for every request (timestamp, full input payload, and outcome) to both the terminal and a persistent `backend/api.log` file, using Python's built-in `logging` module. This gives a permanent, inspectable record of what the service has been asked to predict and whether it succeeded.

🗄️ In addition, every successful prediction is written to a local SQLite database (`backend/predictions.db`), recording the timestamp, all input values, and the predicted duration. This gives the system a persistent history beyond a single session, without requiring any external database server. The database write is wrapped in its own error handling, so a database failure never prevents a user from receiving their prediction — it only logs the failure separately.

> Neither `api.log` nor `predictions.db` is committed to this repository, as both are runtime-generated files (see `.gitignore`); they are created automatically the first time the backend runs.

---

## 📊 Interactive Dashboard (Tableau)

An interactive dashboard was built in **Tableau**, covering the dashboards-and-reporting-for-stakeholders module of the programme. It is included in this repository as a packaged workbook (`Tableau Packaged Workbook (.twbx).twb`), with a static screenshot (`NYC_Taxi_Dashboard.png`) for quick reference.

🧹 The dashboard was built on a **cleaned** export of the dataset (matching the same outlier removal used to train the model — trips restricted to between 1 minute and 4 hours, with geographic outliers removed) rather than the raw dataset, so that reported averages are not skewed by erroneous records.

It combines the following views:

| View | Description |
|---|---|
| 🚕 Trip Duration vs Vendor ID | Average duration compared across the two vendors |
| 🕐 Trip Duration by Hour of Day | Average duration across all 24 hours |
| 📅 Trip Duration by Day of Week | Average duration across each day |
| 🗺️ Pickup Location Map | Geographic distribution of pickups across NYC |
| 📌 KPI: Average Trip Duration | A single headline figure — **835.7 seconds** |
| 🔢 KPI: Total Trip Count | Total number of cleaned trips |

---

## 📄 Reproducible Report

The full analysis notebook is also available as a standalone HTML file, [`Untitled3.html`](./Untitled3.html), generated with:

```bash
jupyter nbconvert --to html Untitled3.ipynb
```

This lets anyone view the complete analysis — code, commentary, and charts — directly in a browser, without needing Jupyter installed.

---

## ☁️ Deployment Status (Azure)

🚧 Azure was investigated as the intended cloud deployment target, in line with the programme's specification. Initial exploration (Azure CLI, resource group planning) was carried out, but **full deployment was not completed**, as it required an active paid subscription beyond the free trial allowance.

✅ The application is fully prepared for cloud deployment — it is already Dockerised, and both containers build and run correctly locally — so deploying to Azure (or an alternative platform) remains a straightforward next step once hosting is available. At present, the application must be run locally or via Docker as described above; there is no live public URL.

---

## 🚧 Scope Decisions

This project deliberately does **not** include user authentication (login, hashed passwords) or per-user personalised data. This was a considered decision, not an oversight:

- 📘 The programme's curriculum, for the equivalent "Deployed ML Microservice" deliverable, specifies a FastAPI service, containerisation, and basic monitoring — it does not require user accounts.
- 📈 Building real authentication is a materially larger scope increase than the rest of this deliverable.
- 🔗 Per-user personalisation only makes sense once authentication exists, so it was excluded for the same reason.

---

## 🧪 Project Methodology

The full data science process — problem definition, hypotheses, data cleaning, outlier treatment (domain-based vs. IQR), feature engineering, model tuning, and evaluation — is documented step by step in [`Untitled3.ipynb`](./Untitled3.ipynb), with accompanying written reports at each stage.

**Key methodology decisions:**

- 🎯 Outliers were treated using **domain-based reasoning** rather than strict IQR cutoffs, since IQR would have discarded a large volume of valid trips (e.g. flagging any trip over 35 minutes as an outlier). Trip duration was ultimately restricted to between 1 minute and 4 hours.
- 🔒 The scaler was fit **only on training data** and applied to the test set, to prevent data leakage.
- 📉 K was chosen using an overfitting/underfitting curve **and** confirmed with 5-fold cross-validation, not a single train/test split.
- ⚖️ KNN was benchmarked against Linear Regression, with residual diagnostics used to explain why KNN performed better (the underlying relationship isn't fully linear).

---

## 👤 Author

<div align="center">

**Thato Xauka**

WIL AI/ML Programme — Work-Integrated Learning Project

</div>
