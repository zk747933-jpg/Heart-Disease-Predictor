import streamlit as st
import pandas as pd
import pickle

# ---------------------------
# Load model
# ---------------------------
with open('heart-disease-model.pkl', 'rb') as file:
    model, scaler = pickle.load(file)   # scaler load ho raha hai, use nahi karenge

# ---------------------------
# UI
# ---------------------------
st.set_page_config(page_title="Heart Disease Predictor")

st.title("❤️ Heart Disease Predictor")
st.write("Predict heart disease risk based on patient data")
st.markdown("---")

# ---------------------------
# Input fields
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
    ST_Depression = st.number_input("ST Depression", 0.0, 6.0, 1.0, step=0.1)
    ST_Slope = st.selectbox("ST Slope", [0, 1, 2])
    MajorVessels = st.selectbox("Major Vessels", [0, 1, 2, 3])
    Thalassemia = st.selectbox("Thalassemia", [0, 1, 2, 3])

# ---------------------------
# Encode Gender
# ---------------------------
Gender = 1 if Gender == "Male" else 0

# ---------------------------
# Create dataframe
# ---------------------------
input_dict = {
    'Age': Age,
    'Gender': Gender,
    'ChestPainType': ChestPainType,
    'RestingBp': RestingBp,
    'Cholesterol': Cholesterol,
    'FastingBS': FastingBS,
    'RestingECG': RestingECG,
    'MaxHR': MaxHR,
    'ExerciseAngina': ExerciseAngina,
    'ST_Depression': ST_Depression,
    'ST_Slope': ST_Slope,
    'MajorVessels': MajorVessels,
    'Thalassemia': Thalassemia
}

input_df = pd.DataFrame([input_dict])

# ---------------------------
# Encoding
# ---------------------------
input_encoded = pd.get_dummies(input_df)

# ---------------------------
# Match training columns
# ---------------------------
training_columns = model.feature_names_in_

input_encoded = input_encoded.reindex(
    columns=training_columns,
    fill_value=0
)

# ---------------------------
# Prediction
# ---------------------------
if st.button("Predict Heart Disease"):

    prediction = model.predict(input_encoded)[0]
    probability = model.predict_proba(input_encoded)[0]

    if prediction == 1:
        st.error("⚠ High risk of heart disease")
        st.write(f"Risk Score: {probability[1]*100:.2f}%")
    else:
        st.success("✅ No heart disease risk detected")
        st.write(f"Healthy Score: {probability[0]*100:.2f}%")