import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- Load Model ---
model = joblib.load("model.pkl")

# Title
st.title("Market Price Prediction App")

st.write("Welcome! Enter features below to predict market price.")

# User Inputs (mfano kama feature1, feature2...)
feature1 = st.number_input("Feature 1")
feature2 = st.number_input("Feature 2")
feature3 = st.number_input("Feature 3")

# Button to Predict
if st.button("PREDICT"):
    # Prepare data
    X = np.array([[feature1, feature2, feature3]])
    
    prediction = model.predict(X)
    
    st.success(f"Predicted Market Price: {prediction[0]:.2f}")
