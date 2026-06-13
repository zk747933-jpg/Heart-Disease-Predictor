import streamlit as st
import pandas as pd
import pickle

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(page_title="Heart Disease Predictor")

# ---------------------------
# Load Model Safely
# ---------------------------
try:
    with open("heart-disease-model.pkl", "rb") as file:
        data = pickle.load(file)

    model = data["model"]
    scaler = data["scaler"]
    training_columns = data["columns"]

    # Remove target column if present
    training_columns = [
        col for col in training_columns
        if col != "HeartDisease"
    ]

except Exception as e:
    st.error(f"❌ Model loading failed: {e}")
    st.stop()

# ---------------------------
# UI
# ---------------------------
st.title("❤️ Heart Disease Predictor")
st.write("Predict heart disease risk based on patient data")
st.markdown("---")

# ---------------------------
# Input Fields
# ---------------------------
col1, col2 = st.columns(2)

with col1:
    Age = st.number_input("Age", 20, 100, 45)
    Gender = st.selectbox("Gender", ["Male", "Female"])
    ChestPainType = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
    RestingBp = st.number_input("Resting Blood Pressure", 80, 200, 120)
    Cholesterol = st.number_input("Cholesterol", 100, 600, 200)

with col2:
    FastingBS = st.selectbox("Fasting Blood Sugar", [0, 1])
    RestingECG = st.selectbox("Resting ECG", [0, 1, 2])
    MaxHR = st.number_input("Max Heart Rate", 60, 220, 150)
    ExerciseAngina = st.selectbox("Exercise Angina", [0, 1])
    ST_Depression = st.number_input(
        "ST Depression",
        min_value=0.0,
        max_value=6.0,
        value=1.0,
        step=0.1
    )
    ST_Slope = st.selectbox("ST Slope", [0, 1, 2])
    MajorVessels = st.selectbox("Major Vessels", [0, 1, 2, 3])
    Thalassemia = st.selectbox("Thalassemia", [0, 1, 2, 3])

# ---------------------------
# Encode Gender
# ---------------------------
Gender = 1 if Gender == "Male" else 0

# ---------------------------
# Create DataFrame
# ---------------------------
input_dict = {
    "Age": Age,
    "Gender": Gender,
    "ChestPainType": ChestPainType,
    "RestingBp": RestingBp,
    "Cholesterol": Cholesterol,
    "FastingBS": FastingBS,
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
# Encoding
# ---------------------------
input_encoded = pd.get_dummies(input_df)

# ---------------------------
# Align Columns
# ---------------------------
input_encoded = input_encoded.reindex(
    columns=training_columns,
    fill_value=0
)

# ---------------------------
# Prediction
# ---------------------------
if st.button("Predict Heart Disease"):
    try:
        prediction = model.predict(input_encoded)[0]

        if prediction == 1:
            st.error("⚠ High risk of heart disease")
        else:
            st.success("✅ No heart disease risk detected")

    except Exception as e:
        st.error(f"Prediction failed: {e}")
