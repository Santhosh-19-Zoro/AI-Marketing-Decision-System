import streamlit as st
import pandas as pd
import joblib
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -------------------------------
# Load Models (same folder)
# -------------------------------
import os

# Get base project directory
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

# Load models from /models folder
kmeans = joblib.load(os.path.join(BASE_DIR, "models/kmeans_model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "models/scaler.pkl"))

with open(os.path.join(BASE_DIR, "models/tokenizer.pkl"), "rb") as f:
    tokenizer = pickle.load(f)

sentiment_model = load_model(
    os.path.join(BASE_DIR, "models/best_sentiment_model.h5"))

max_len = 50   # MUST match training

# -------------------------------
# App UI
# -------------------------------
st.set_page_config(page_title="AI Marketing Decision System", layout="centered")

st.title("AI Marketing Decision System")
st.write("Predict Customer Segment, Sentiment & Business Action")

# -------------------------------
# User Inputs
# -------------------------------
st.subheader("Enter Customer Details")

income = st.number_input(
    "Income (Annual Income in ₹)",
    min_value=0.0,
    max_value=1000000.0,
    value=50000.0,
    help="Customer's yearly income. Typical range: ₹10,000 – ₹1,000,000")

recency = st.number_input(
    "Recency (Days since last purchase)",
    min_value=0.0,
    max_value=365.0,
    value=30.0,
    help="How recently the customer made a purchase. Lower = more recent")

days = st.number_input(
    "Customer Tenure (Days)",
    min_value=0.0,
    max_value=2000.0,
    value=180.0,
    help="Number of days since customer joined")

spending = st.number_input(
    "Total Spending (₹)",
    min_value=0.0,
    max_value=500000.0,
    value=10000.0,
    help="Total amount spent by the customer so far")

purchases = st.number_input(
    "Total Purchases",
    min_value=0.0,
    max_value=500.0,
    value=10.0,
    help="Total number of purchases made by the customer")

review = st.text_area(
    "Customer Review",
    help="Enter customer feedback (used for sentiment analysis)")

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict"):

    # --------- Segmentation ---------
    input_data = [[
        income,
        recency,
        days,
        spending,
        purchases
    ]]

    scaled_data = scaler.transform(input_data)
    cluster = kmeans.predict(scaled_data)[0]

    segment_map = {
        0: "High Value Customers",
        1: "New Customers",
        2: "At Risk Customers",
        3: "Budget Buyers"
    }

    segment = segment_map.get(cluster, "Unknown")

    # --------- Sentiment ---------
    if review.strip() != "":
        seq = tokenizer.texts_to_sequences([review])
        pad = pad_sequences(seq, maxlen=max_len)

        sentiment_score = sentiment_model.predict(pad)[0][0]
        sentiment = "Positive" if sentiment_score > 0.6 else "Negative"
    else:
        sentiment = "No Review"

    # --------- Decision Engine ---------
    if segment == "High Value Customers" and sentiment == "Negative":
        action = "Immediate Retention Offer"
        priority = "High"

    elif segment == "At Risk Customers":
        action = "Re-engagement Campaign"
        priority = "High"

    elif sentiment == "Positive":
        action = "Upsell Premium Products"
        priority = "Medium"

    else:
        action = "Standard Marketing"
        priority = "Low"

    # -------------------------------
    # Output Section
    # -------------------------------
    st.subheader("Results")

    st.success(f"Segment: {segment}")
    st.info(f"Sentiment: {sentiment}")
    st.warning(f"Recommended Action: {action}")
    st.error(f"Priority: {priority}")