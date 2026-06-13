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
    data = joblib.load("heart-disease-model (4).pkl")

    model = data["model"]
    training_columns = data["columns"]

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
    Oldpeak = st.number_input("ST Depression (Oldpeak)", 0.0, 6.0, 1.0, 0.1)
    ST_Slope = st.selectbox("ST Slope", [0, 1, 2])
    MajorVessels = st.selectbox("Major Vessels", [0, 1, 2, 3])
    Thalassemia = st.selectbox("Thalassemia", [0, 1, 2, 3])

# ---------------------------
# Encoding
# ---------------------------
Gender = 1 if Gender == "Male" else 0

# ---------------------------
# Input Dictionary
# ---------------------------
input_dict = {
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
}

# ---------------------------
# Prediction Section
# ---------------------------
if st.button("Predict Heart Disease"):

    try:
        # Create DataFrame
        input_df = pd.DataFrame([input_dict])

        # Align with training columns (VERY IMPORTANT FIX)
        input_df = input_df.reindex(columns=training_columns, fill_value=0)

        # Prediction
        prediction = model.predict(input_df)[0]

        # Output
        st.subheader("Result")

        if prediction == 1:
            st.error("⚠ High risk of heart disease detected")
        else:
            st.success("✅ No heart disease risk detected")

    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
