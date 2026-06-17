import streamlit as st
import pandas as pd
import joblib
import os

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️")

st.title("❤️ Heart Disease Predictor")
st.write("Enter patient details to predict heart disease risk")
st.markdown("---")

# ---------------------------
# Load Model Safely
# ---------------------------
MODEL_PATH = "heart_model.pkl"   # 👈 keep file name simple

if not os.path.exists(MODEL_PATH):
    st.error("❌ Model file not found. Please upload 'heart_model.pkl' in repo.")
    st.stop()

model = joblib.load(MODEL_PATH)

# ---------------------------
# Input UI
# ---------------------------
col1, col2 = st.columns(2)

with col1:
    Age = st.number_input("Age", 20, 100, 45)
    Gender = st.selectbox("Gender", ["Male", "Female"])
    ChestPainType = st.selectbox("Chest Pain Type", ["ATA", "NAP", "ASY", "TA"])
    RestingBP = st.number_input("Resting Blood Pressure", 80, 200, 120)
    Cholesterol = st.number_input("Cholesterol", 100, 600, 200)
    FastingBS = st.selectbox("Fasting Blood Sugar", [0, 1])
    RestingECG = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])

with col2:
    MaxHR = st.number_input("Max Heart Rate", 60, 220, 150)
    ExerciseAngina = st.selectbox("Exercise Angina", ["Yes", "No"])
    Oldpeak = st.number_input("ST Depression", 0.0, 6.0, 1.0, 0.1)
    ST_Slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])
    MajorVessels = st.selectbox("Major Vessels", [0, 1, 2, 3])
    Thalassemia = st.selectbox("Thalassemia", [1, 2, 3])

# ---------------------------
# Encoding
# ---------------------------
Gender = 1 if Gender == "Male" else 0
ExerciseAngina = 1 if ExerciseAngina == "Yes" else 0

# ---------------------------
# Prediction
# ---------------------------
if st.button("Predict Heart Disease"):

    try:
        input_data = pd.DataFrame([{
            "Age": Age,
            "Gender": Gender,
            "ChestPainType": ChestPainType,
            "RestingBP": RestingBP,
            "Cholesterol": Cholesterol,
            "FastingBS": FastingBS,
            "RestingECG": RestingECG,
            "MaxHR": MaxHR,
            "ExerciseAngina": ExerciseAngina,
            "Oldpeak": Oldpeak,
            "ST_Slope": ST_Slope,
            "MajorVessels": MajorVessels,
            "Thalassemia": Thalassemia
        }])

        prediction = model.predict(input_data)[0]

        st.subheader("Result")

        if prediction == 1:
            st.error("⚠ High Risk of Heart Disease Detected")
        else:
            st.success("✅ No Heart Disease Risk Detected")

    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
