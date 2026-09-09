Here's your *PRO README* - clean, recruiter-ready, with your taxi banner. Just copy-paste this into your README.md:
*Save that banner image above as `banner.png` and upload it to your repo first, then use this README:*
![NYC Taxi Banner](banner.png)

# 🚕 NYC Taxi Trip Duration Prediction
> End-to-end data science project predicting NYC taxi trip duration using EDA, feature engineering, and KNN/Linear Regression models.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-orange)](https://scikit-learn.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-yellow)](https://jupyter.org/)
[![Status](https://img.shields.io/badge/Status-Complete-success)]()

### 📌 Overview
Can we predict how long a taxi ride will take *before* it starts? This project uses historical NYC Yellow Taxi data to build regression models that estimate trip duration in seconds/minutes.

Built as a complete Data Science lifecycle project: **Business Understanding → Data Collection → EDA → Feature Engineering → Modeling → Evaluation → Presentation.**

### 🎯 Business Problem
- Riders want accurate ETAs
- Taxi companies need better dispatching and pricing
- City planners need traffic flow insights

Goal: Predict `trip_duration` from pickup/dropoff coordinates, time, passenger count, and vendor.

### 📊 Dataset
**Source:** NYC Taxi & Limousine Commission (TLC) Trip Record Data - [Kaggle NYC Taxi Dataset](https://www.kaggle.com/c/nyc-taxi-trip-duration/data)

> **Note:** Raw `NYC.csv` is not included ( >100MB - GitHub limit). Download from Kaggle and place in `/data` folder to reproduce.

**Size:** ~1.4M trips | **Period:** Jan - June 2016 | **Target:** `trip_duration` (seconds)

### 🔧 Key Features Engineered
- **Haversine Distance** - straight-line distance between pickup/dropoff
- **Time Features** - Hour, Day of Week, Month, IsWeekend, Rush Hour
- **Geospatial Clusters** - K-Means on pickup/dropoff locations
- **Speed & Bearing** - direction of travel

### 🧠 Models Evaluated
| Model | RMSE | R2 Score | Notes |
| :--- | :--- | :--- | :--- |
| **Linear Regression** | Baseline | ~0.62 | Fast, interpretable |
| **KNN Regressor** | **Best** | **~0.78** | k=5, distance-weighted |

*KNN outperformed Linear due to non-linear geospatial patterns.*

### 📁 Repo Structure
nyc-taxi-trip-duration-prediction/
├── banner.png
├── Untitled3.ipynb  # -> rename to nyc_taxi_trip_duration.ipynb
├── NYC_Taxi_Presentation.pptx
├── README.md
└── data/ (not uploaded - add NYC.csv here locally)

### 🚀 How to Run
```bash
# Clone
git clone https://github.com/ThatoCodeXpress/nyc-taxi-trip-duration-prediction.git

# Install
pip install pandas numpy matplotlib seaborn scikit-learn jupyter

# Run notebook
jupyter notebook nyc_taxi_trip_duration.ipynb
### 📈 Key Insights from EDA
1. Trip duration is highly correlated with distance, but also spikes during 5-7 PM rush
2. Friday & Saturday nights show longest avg durations (nightlife traffic)
3. JFK trips are outliers - much longer but predictable

### 🔮 Next Steps
- [ ] Deploy as Streamlit app on `nyc-taxi-predictor.azurewebsites.net`
- [ ] Try XGBoost / LightGBM for better RMSE
- [ ] Add live map prediction UI

---
*Author:* [ThatoCodeXpress (THATO XAUKA)](https://github.com/ThatoCodeXpress) | Built with ❤️ for portfolio

**What to do now (2 mins):**

1. Save the banner image I generated above as `banner.png`
2. Go to your repo → `Add file` → Upload `banner.png`
3. Click `README.md` → pencil icon → delete everything → paste this new README → Commit

Want me to also generate the second README for your Flight Delay project with that sunset flight banner?
