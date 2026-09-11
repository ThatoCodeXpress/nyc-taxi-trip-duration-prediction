"""
NYC Taxi Trip Duration — Streamlit Frontend
Stage 6: Model Deployment
"""

from datetime import datetime, date, time

import requests
import streamlit as st

import os
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/predict")

st.set_page_config(page_title="NYC Taxi Trip Duration", page_icon="🚕", layout="centered")

# --- Custom CSS ---
st.markdown("""
<style>
.hero {
    position: relative;
    border-radius: 14px;
    overflow: hidden;
    margin-bottom: 1.5rem;
}
.hero img {
    width: 100%;
    display: block;
    filter: brightness(55%);
}
.hero-text {
    position: absolute;
    top: 50%;
    left: 6%;
    transform: translateY(-50%);
    color: white;
}
.hero-text h1 {
    font-size: 2.2rem;
    margin: 0;
}
.hero-text p {
    font-size: 1rem;
    opacity: 0.9;
    margin-top: 0.3rem;
}
div[data-testid="stForm"] {
    background-color: #fafafa;
    border: 1px solid #eee;
    border-radius: 14px;
    padding: 1.5rem;
}
.stButton>button, div[data-testid="stFormSubmitButton"] button {
    background-color: #FFC72C;
    color: #1a1a1a;
    font-weight: 700;
    border-radius: 10px;
    border: none;
    padding: 0.7rem 0;
    width: 100%;
    font-size: 1.05rem;
}
.stButton>button:hover, div[data-testid="stFormSubmitButton"] button:hover {
    background-color: #e6b325;
    color: #000;
}
.result-box {
    background-color: #eaf1ff;
    border-radius: 12px;
    padding: 1.2rem;
    text-align: center;
    margin-top: 1rem;
}
.result-box .big {
    font-size: 2.2rem;
    font-weight: 800;
    color: #1a3d8f;
}
.result-box .small {
    color: #555;
    font-size: 0.95rem;
}
</style>
""", unsafe_allow_html=True)

# --- Hero banner ---
st.markdown("""
<div class="hero">
    <img src="app/static/taxi_banner.jpg">
    <div class="hero-text">
        <h1>🚕 NYC Taxi Trip Duration Predictor</h1>
        <p>Predict your taxi trip duration before you travel.</p>
    </div>
</div>
""", unsafe_allow_html=True)

with st.form("trip_form"):
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**📍 Pickup Location**")
        pickup_latitude = st.number_input("Latitude", value=40.767, format="%.5f", key="pl")
        pickup_longitude = st.number_input("Longitude", value=-73.982, format="%.5f", key="plo")

    with col2:
        st.markdown("**📍 Drop-off Location**")
        dropoff_latitude = st.number_input("Latitude", value=40.766, format="%.5f", key="dl")
        dropoff_longitude = st.number_input("Longitude", value=-73.964, format="%.5f", key="dlo")

    col3, col4, col5, col6 = st.columns(4)
    with col3:
        passenger_count = st.selectbox("Passenger Count", [1, 2, 3, 4, 5, 6])
    with col4:
        vendor_id = st.selectbox("Vendor", [1, 2], format_func=lambda v: f"Vendor {v}")
    with col5:
        pickup_date = st.date_input("Date", value=date(2016, 3, 14))
    with col6:
        pickup_time = st.time_input("Time", value=time(17, 24))

    store_and_fwd_flag = "N"  # default, hidden from the simplified UI

    submitted = st.form_submit_button("🚕 PREDICT TRIP DURATION")

if submitted:
    pickup_datetime = datetime.combine(pickup_date, pickup_time)

    payload = {
        "vendor_id": vendor_id,
        "passenger_count": int(passenger_count),
        "pickup_datetime": pickup_datetime.isoformat(),
        "pickup_longitude": pickup_longitude,
        "pickup_latitude": pickup_latitude,
        "dropoff_longitude": dropoff_longitude,
        "dropoff_latitude": dropoff_latitude,
        "store_and_fwd_flag": store_and_fwd_flag,
    }

    try:
        response = requests.post(API_URL, json=payload, timeout=10)
        response.raise_for_status()
        result = response.json()

        minutes = result["predicted_trip_duration_minutes"]
        seconds = result["predicted_trip_duration_seconds"]

        st.markdown(f"""
        <div class="result-box">
            <div class="small">PREDICTED TRIP DURATION</div>
            <div class="big">{minutes} MINUTES</div>
            <div class="small">≈ {seconds} seconds</div>
        </div>
        """, unsafe_allow_html=True)

    except requests.exceptions.ConnectionError:
        st.error("Could not reach the backend. Make sure the FastAPI server is running.")
    except Exception as e:
        st.error(f"Something went wrong: {e}")