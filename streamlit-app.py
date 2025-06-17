import streamlit as st
import numpy as np
import joblib
from pathlib import Path

# Load saved model and scaler once at start

MODEL_PATH = Path(__file__).resolve().parent / 'models' / 'logistic_model.pkl'
SCALER_PATH = Path(__file__).resolve().parent / 'models' / 'scaler.pkl'
# Load the model
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# Mapping from label number to iris species name
species_map = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}

st.title("Iris Species Predictor")

# Collect user inputs for features
sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, step=0.1)
sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, step=0.1)
petal_length = st.number_input("Petal Length (cm)", min_value=0.0, step=0.1)
petal_width = st.number_input("Petal Width (cm)", min_value=0.0, step=0.1)

if st.button("Predict"):
    # Prepare features in the correct order expected by the model
    user_features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    # Scale features with saved scaler (important!)
    user_features_scaled = scaler.transform(user_features)

    # Predict class label
    prediction_num = model.predict(user_features_scaled)[0]

    # Map numeric prediction to species name
    prediction_name = species_map.get(prediction_num, "Unknown")

    st.success(f"Predicted Iris Species: {prediction_name}")
