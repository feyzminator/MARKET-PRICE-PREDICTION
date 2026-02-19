import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle

# --- Load Model and Encoders ---
model = joblib.load("model.pkl")
encoders = pickle.load(open("encoder.pkl", "rb"))

# Title
st.title("Market Price Prediction App")

st.write("Welcome! Enter features below to predict market price.")

# User Inputs - All features from your training data
commodity = st.selectbox("Commodity", ["Maize", "Potatoes", "Rice", "Tomatoes", "Beans"])
market = st.selectbox("Market", ["Arusha", "Dar es Salaam", "Dodoma", "Mbeya", "Mwanza"])
month = st.number_input("Month (1-12)", min_value=1, max_value=12, value=6)
season = st.selectbox("Season", ["Harvest", "Mid", "Lean"])
supply_level = st.selectbox("Supply Level", ["High", "Medium", "Low"])

# Button to Predict
if st.button("PREDICT"):
    # Create a dataframe with the input
    input_df = pd.DataFrame({
        'commodity': [commodity],
        'market': [market],
        'month': [month],
        'season': [season],
        'supply_level': [supply_level]
    })
    
    # Apply the same encoding as training
    
    # One-hot encoding for commodity and market
    onehot_encoder = encoders['onehot_encoder']
    encoded_nominal = onehot_encoder.transform(input_df[encoders['nominal_cols']])
    
    # Get feature names
    feature_names = encoders['feature_names']
    
    # Create DataFrame with encoded columns
    encoded_df = pd.DataFrame(
        encoded_nominal, 
        columns=feature_names
    )
    
    # Drop original nominal columns and add encoded ones
    input_df = input_df.drop(columns=encoders['nominal_cols'])
    input_df = pd.concat([input_df, encoded_df], axis=1)
    
    # Label encoding for ordinal columns
    for col in encoders['ordinal_cols']:
        le = encoders['label_encoders'][col]
        input_df[col] = le.transform(input_df[col])
    
    # Ensure column order matches training data
    # Get the expected feature names from the model (if available)
    if hasattr(model, 'feature_names_in_'):
        input_df = input_df[model.feature_names_in_]
    
    # Make prediction
    prediction = model.predict(input_df)
    
    st.success(f"Predicted Market Price: {prediction[0]:.2f} TZS")
    
    # Optional: Display feature info
    st.write("Input Features Used:")
    st.write(input_df)
