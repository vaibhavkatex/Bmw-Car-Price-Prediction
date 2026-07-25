import os
import streamlit as st
import pandas as pd
import joblib

# Get absolute path of current file directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load Model, Scaler, and Column names
model = joblib.load(os.path.join(BASE_DIR, "LR_BMW.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
columns = joblib.load(os.path.join(BASE_DIR, "columns.pkl"))

st.set_page_config(
    page_title="BMW Car Price Prediction",
    page_icon="🚗",
    layout="centered"
)

st.title("🚗 BMW Car Price Prediction")
st.markdown("##### Enter Vehicle Specifications to Estimate Market Value")
st.divider()

# Inputs layout using columns
col1, col2 = st.columns(2)

with col1:
    year = st.number_input("Manufacturing Year", min_value=1990, max_value=2035, value=2018, step=1)
    mileage = st.number_input("Mileage (Miles)", min_value=0, max_value=300000, value=30000, step=1000)
    tax = st.number_input("Annual Road Tax (£)", min_value=0, max_value=1000, value=150, step=10)
    mpg = st.number_input("Miles Per Gallon (MPG)", min_value=0.0, max_value=150.0, value=55.4, step=0.1)

with col2:
    engineSize = st.number_input("Engine Size (Liters)", min_value=0.5, max_value=8.0, value=2.0, step=0.1)
    model_name = st.selectbox(
        "Select BMW Model",
        [
            "1 Series", "2 Series", "3 Series", "4 Series", "5 Series",
            "6 Series", "7 Series", "8 Series", "X1", "X2", "X3",
            "X4", "X5", "X6", "X7", "Z3", "Z4", "M2", "M3",
            "M4", "M5", "M6", "i3", "i8"
        ]
    )
    transmission = st.selectbox(
        "Transmission Type",
        ["Manual", "Automatic", "Semi-Auto"]
    )
    fuelType = st.selectbox(
        "Fuel Type",
        ["Petrol", "Diesel", "Hybrid", "Electric"]
    )

st.write("")

# Prediction Logic
if st.button("Predict Price 💰", use_container_width=True, type="primary"):
    input_df = pd.DataFrame([{
        "year": year,
        "mileage": mileage,
        "tax": tax,
        "mpg": mpg,
        "engineSize": engineSize,
        "model": model_name,
        "transmission": transmission,
        "fuelType": fuelType
    }])

    # One-hot encode categorical features
    input_df = pd.get_dummies(input_df, drop_first=True)

    # Reindex columns to match training feature set exactly
    input_df = input_df.reindex(columns=columns, fill_value=0)

    # Scale features
    input_scaled = scaler.transform(input_df)

    # Predict price
    prediction = model.predict(input_scaled)
    predicted_price = max(0, prediction[0])

    st.success(f"### Estimated Price: **£ {predicted_price:,.2f}**")