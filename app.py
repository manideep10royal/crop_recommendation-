# app.py

import streamlit as st
import pickle
import numpy as np

# Load model and label encoder
model = pickle.load(open("crop_model.pkl", "rb"))
le = pickle.load(open("label_encoder.pkl", "rb"))

st.title("🌾 Crop Recommendation System")

st.markdown("Enter the soil and climate data to get a crop recommendation:")

# Input fields
N = st.number_input("Nitrogen (N)", 0, 200, 50)
P = st.number_input("Phosphorus (P)", 0, 200, 50)
K = st.number_input("Potassium (K)", 0, 200, 50)
temperature = st.number_input("Temperature (°C)", 0.0, 50.0, 25.0)
humidity = st.number_input("Humidity (%)", 0.0, 100.0, 60.0)
ph = st.number_input("pH", 0.0, 14.0, 6.5)
rainfall = st.number_input("Rainfall (mm)", 0.0, 300.0, 100.0)

if st.button("Recommend Crop"):
    input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    prediction = model.predict(input_data)
    crop = le.inverse_transform(prediction)[0]
    st.success(f"✅ Recommended Crop: **{crop.capitalize()}**")
