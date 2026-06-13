import streamlit as st
import pandas as pd
import joblib

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️")

# ---------------------------
# Load Model Pipeline
# ---------------------------
try:
    data = joblib.load("heart-disease-model (3).pkl")

    model = data["model"]
    training_columns = list(data["columns"])

    if model is None or len(training_columns) == 0:
        st.error("❌ Model file is missing required keys (model, columns).")
        st.stop()

except Exception as e:
    st.error(f"❌ Model loading failed: {e}")
    st.stop()

# ---------------------------
# UI
# ---------------------------
st.title("❤️ Heart Disease Predictor")
st.write("Enter patient details to predict heart disease risk")
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    Age = st.number_input("Age", 20, 100, 45)
    Gender = st.selectbox("Gender", ["Male", "Female"])
    ChestPainType = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
    RestingBp = st.number_input("Resting Blood Pressure", 80, 200, 120)
    Cholesterol = st.number_input("Cholesterol", 100, 600, 200)
    FastingBs = st.selectbox("Fasting Blood Sugar", [0, 1])
    RestingECG = st.selectbox("Resting ECG", [0, 1, 2])

with col2:
    MaxHR = st.number_input("Max Heart Rate", 60, 220, 150)
    ExerciseAngina = st.selectbox("Exercise Angina", [0, 1])
    ST_Depression = st.number_input("ST Depression", 0.0, 6.0, 1.0, 0.1)
    ST_Slope = st.selectbox("ST Slope", [0, 1, 2])
    MajorVessels = st.selectbox("Major Vessels", [0, 1, 2, 3])
    Thalassemia = st.selectbox("Thalassemia", [0, 1, 2, 3])

# ---------------------------
# Encoding Gender
# ---------------------------
Gender = 1 if Gender == "Male" else 0

# ---------------------------
# Create Input DataFrame
# ---------------------------
input_dict = {
    "Age": Age,
    "Gender": Gender,
    "ChestPainType": ChestPainType,
    "RestingBp": RestingBp,
    "Cholesterol": Cholesterol,
    "FastingBs": FastingBs,
    "RestingECG": RestingECG,
    "MaxHR": MaxHR,
    "ExerciseAngina": ExerciseAngina,
    "ST_Depression": ST_Depression,
    "ST_Slope": ST_Slope,
    "MajorVessels": MajorVessels,
    "Thalassemia": Thalassemia
}

input_df = pd.DataFrame([input_dict])

# ---------------------------
# Prediction
# ---------------------------
if st.button("Predict Heart Disease"):
    try:
        prediction = model.predict(input_df)[0]

        # Debug Information
        st.subheader("Debug Output")
        st.write("Raw Prediction:", prediction)

        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(input_df)
            st.write("Probability:", probability)

        st.write("Input Data:")
        st.dataframe(input_df)

        # Final Result
        if prediction == 1:
            st.error("⚠ High risk of heart disease detected")
        else:
            st.success("✅ No heart disease risk detected")

    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
