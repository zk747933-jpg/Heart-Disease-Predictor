import streamlit as st
import pandas as pd
import joblib

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️")

st.title("❤️ Heart Disease Predictor")
st.write("Enter patient details to predict heart disease risk")
st.markdown("---")

# ---------------------------
# Load Model
# ---------------------------
try:
    model = joblib.load("heart-disease-model (5).pkl")

except Exception as e:
    st.error(f"❌ Model loading failed: {e}")
    st.stop()

# ---------------------------
# Input UI
# ---------------------------
col1, col2 = st.columns(2)

with col1:
    Age = st.number_input("Age", 20, 100, 45)
    Gender = st.selectbox("Gender", ["Male", "Female"])
    ChestPainType = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
    RestingBP = st.number_input("Resting Blood Pressure", 80, 200, 120)
    Cholesterol = st.number_input("Cholesterol", 100, 600, 200)
    FastingBS = st.selectbox("Fasting Blood Sugar", [0, 1])
    RestingECG = st.selectbox("Resting ECG", [0, 1, 2])

with col2:
    MaxHR = st.number_input("Max Heart Rate", 60, 220, 150)
    ExerciseAngina = st.selectbox("Exercise Angina", [0, 1])
    Oldpeak = st.number_input("ST Depression", 0.0, 6.0, 1.0, 0.1)
    ST_Slope = st.selectbox("ST Slope", [0, 1, 2])
    MajorVessels = st.selectbox("Major Vessels", [0, 1, 2, 3])
    Thalassemia = st.selectbox("Thalassemia", [0, 1, 2, 3])

# ---------------------------
# Encoding
# ---------------------------
Gender = 1 if Gender == "Male" else 0

# ---------------------------
# Prediction
# ---------------------------
if st.button("Predict Heart Disease"):

    try:

        input_dict = {
            "Age": Age,
            "RestingBp": RestingBP,
            "Cholesterol": Cholesterol,
            "MaxHR": MaxHR,
            "ST_Depression": Oldpeak,
            "MajorVessels": MajorVessels,
            "Gender": Gender,
            "FastingBs": FastingBS,
            "ChestPainType": ChestPainType,
            "RestingECG": RestingECG,
            "ExerciseAngina": ExerciseAngina,
            "ST_Slope": ST_Slope,
            "Thalassemia": Thalassemia
        }

        input_df = pd.DataFrame([input_dict])

        prediction = model.predict(input_df)[0]

        st.subheader("Result")

        if prediction == 1:
            st.error("⚠ High Risk of Heart Disease Detected")
        else:
            st.success("✅ No Heart Disease Risk Detected")

    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
