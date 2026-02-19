import streamlit as st
import pandas as pd
import joblib

# ----------------------------
# Load trained model
# ----------------------------
model = joblib.load("model.pkl")

st.set_page_config(page_title="Market Price Prediction", layout="centered")

st.title("📊 Market Food Price Prediction")
st.write("Enter market details to predict price per kg (TZS).")

# ----------------------------
# User Inputs
# ----------------------------

commodity = st.selectbox(
    "Commodity",
    ["Maize", "Rice", "Beans", "Potatoes", "Tomatoes"]
)

market = st.selectbox(
    "Market",
    ["Dodoma", "Arusha", "Mbeya", "Mwanza", "Dar es Salaam"]
)

month = st.slider("Month", 1, 12, 1)

season = st.selectbox(
    "Season",
    ["Harvest", "Mid", "Lean"]
)

supply_level = st.selectbox(
    "Supply Level",
    ["High", "Medium", "Low"]
)

# ----------------------------
# Prediction
# ----------------------------

if st.button("🔮 Predict Price"):

    # Create dataframe same structure as training data
    input_data = pd.DataFrame({
        "commodity": [commodity],
        "market": [market],
        "month": [month],
        "season": [season],
        "supply_level": [supply_level]
    })

    prediction = model.predict(input_data)

    st.success(f"💰 Predicted Price: {prediction[0]:,.0f} TZS per kg")
