"""
NYC Taxi Trip Duration — Prediction API
Stage 6: Model Deployment
"""

import logging
import sqlite3
import os
from datetime import datetime

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("api.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("nyc-taxi-api")

# ---------------------------------------------------------------------------
# Database setup
# ---------------------------------------------------------------------------
DB_PATH = "predictions.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            vendor_id INTEGER,
            passenger_count INTEGER,
            pickup_datetime TEXT,
            pickup_longitude REAL,
            pickup_latitude REAL,
            dropoff_longitude REAL,
            dropoff_latitude REAL,
            store_and_fwd_flag TEXT,
            predicted_seconds REAL
        )
    """)
    conn.commit()
    conn.close()

init_db()

# ---------------------------------------------------------------------------
# Load trained model + scaler
# ---------------------------------------------------------------------------
MODEL_PATH = os.getenv("MODEL_PATH", "../model/knn_model.pkl")
SCALER_PATH = os.getenv("SCALER_PATH", "../model/scaler.pkl")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

FEATURE_ORDER = [
    "vendor_id",
    "passenger_count",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "store_and_fwd_flag",
    "day_of_week",
    "hour_of_day",
    "trip_distance_km",
]

app = FastAPI(title="NYC Taxi Trip Duration Predictor")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class TripRequest(BaseModel):
    vendor_id: int = Field(..., ge=1, le=2)
    passenger_count: int = Field(..., ge=1, le=6)
    pickup_datetime: datetime
    pickup_longitude: float = Field(..., ge=-74.05, le=-73.75)
    pickup_latitude: float = Field(..., ge=40.60, le=40.90)
    dropoff_longitude: float = Field(..., ge=-74.05, le=-73.75)
    dropoff_latitude: float = Field(..., ge=40.60, le=40.90)
    store_and_fwd_flag: str = Field(..., pattern="^[YN]$")


class TripResponse(BaseModel):
    predicted_trip_duration_seconds: float
    predicted_trip_duration_minutes: float


def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c


def build_feature_vector(trip: TripRequest) -> np.ndarray:
    day_of_week = trip.pickup_datetime.weekday()
    hour_of_day = trip.pickup_datetime.hour
    store_and_fwd_flag_code = 1 if trip.store_and_fwd_flag.upper() == "Y" else 0
    trip_distance_km = haversine(
        trip.pickup_latitude, trip.pickup_longitude,
        trip.dropoff_latitude, trip.dropoff_longitude,
    )
    row = {
        "vendor_id": trip.vendor_id,
        "passenger_count": trip.passenger_count,
        "pickup_longitude": trip.pickup_longitude,
        "pickup_latitude": trip.pickup_latitude,
        "dropoff_longitude": trip.dropoff_longitude,
        "dropoff_latitude": trip.dropoff_latitude,
        "store_and_fwd_flag": store_and_fwd_flag_code,
        "day_of_week": day_of_week,
        "hour_of_day": hour_of_day,
        "trip_distance_km": trip_distance_km,
    }
    return np.array([[row[col] for col in FEATURE_ORDER]])


@app.get("/")
def health_check():
    return {"status": "ok", "message": "NYC Taxi Trip Duration API is running"}


@app.post("/predict", response_model=TripResponse)
def predict(trip: TripRequest):
    logger.info(f"Received prediction request: {trip.dict()}")
    try:
        X = build_feature_vector(trip)
        X_scaled = scaler.transform(X)
        prediction_seconds = float(model.predict(X_scaled)[0])
    except Exception as e:
        logger.error(f"Prediction failed for input {trip.dict()}: {e}")
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")

    logger.info(f"Prediction successful: {prediction_seconds:.2f} seconds")

    # Save this prediction to the database
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute(
            """
            INSERT INTO predictions (
                timestamp, vendor_id, passenger_count, pickup_datetime,
                pickup_longitude, pickup_latitude,
                dropoff_longitude, dropoff_latitude,
                store_and_fwd_flag, predicted_seconds
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now().isoformat(),
                trip.vendor_id,
                trip.passenger_count,
                trip.pickup_datetime.isoformat(),
                trip.pickup_longitude,
                trip.pickup_latitude,
                trip.dropoff_longitude,
                trip.dropoff_latitude,
                trip.store_and_fwd_flag,
                prediction_seconds,
            ),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Failed to save prediction to database: {e}")

    return TripResponse(
        predicted_trip_duration_seconds=round(prediction_seconds, 2),
        predicted_trip_duration_minutes=round(prediction_seconds / 60, 2),
    )