import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -----------------------------
# Load Model and Encoders
# -----------------------------
model = joblib.load("logistic_bmi_model.pkl")
le_gender = joblib.load("le_gender.pkl")
le_target = joblib.load("le_target.pkl")

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="BMI Predictor",
    page_icon="⚖️",
    layout="centered"
)

st.title("⚖️ BMI Category Predictor")
st.markdown("Enter your details below to predict BMI and its category.")

# -----------------------------
# User Inputs
# -----------------------------
age = st.slider("Age", 1, 100, 25)

gender = st.selectbox("Gender", le_gender.classes_)

height_cm = st.number_input("Height (cm)", 100, 250, 170)

weight_kg = st.number_input("Weight (kg)", 30, 200, 70)

# -----------------------------
# Prediction Button
# -----------------------------
if st.button("Predict BMI"):

    # Encode gender
    gender_encoded = le_gender.transform([gender])[0]

    # Create input dataframe
    input_df = pd.DataFrame(
        [[age, gender_encoded, height_cm, weight_kg]],
        columns=['Age', 'Gender', 'Height_cm', 'Weight_kg']
    )

    # Calculate BMI manually
    height_m = height_cm / 100
    bmi_value = weight_kg / (height_m ** 2)

    # Predict category
    prediction = model.predict(input_df)
    probabilities = model.predict_proba(input_df)

    category = le_target.inverse_transform(prediction)[0]
    confidence = np.max(probabilities) * 100

    # -----------------------------
    # Display Results
    # -----------------------------
    st.success(f"### BMI: {round(bmi_value, 2)}")
    st.info(f"### Category: {category}")
    st.write(f"Confidence: {confidence:.2f}%")

    # Show probability distribution
    prob_df = pd.DataFrame({
        "Category": le_target.classes_,
        "Probability": probabilities[0]
    })

    st.bar_chart(prob_df.set_index("Category"))