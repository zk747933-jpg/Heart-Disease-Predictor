import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️")

st.title("❤️ Heart Disease Predictor")
st.write("Enter patient details to predict heart disease risk")
st.markdown("---")

# ---------------------------
# Load Model
# ---------------------------
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "heart_model.pkl")

if not os.path.exists(MODEL_PATH):
    st.error("❌ Model file not found in project folder")
    st.stop()


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


data = load_model()
model = data["model"]
COLUMNS = data["columns"]  # exact column names/order used in training

# ---------------------------
# Encodings used while TRAINING (model expects numbers, not text)
# ---------------------------
CHEST_PAIN = {"ASY": 0, "ATA": 1, "NAP": 2, "TA": 3}
REST_ECG = {"LVH": 0, "Normal": 1, "ST": 2}
SLOPE = {"Down": 0, "Flat": 1, "Up": 2}

# ---------------------------
# Input UI
# ---------------------------
col1, col2 = st.columns(2)

with col1:
    Age = st.number_input("Age", 20, 100, 45)
    Gender = st.selectbox("Gender", ["Male", "Female"])
    ChestPainType = st.selectbox(
        "Chest Pain Type",
        ["ATA", "NAP", "ASY", "TA"],
        help="TA = Typical Angina, ATA = Atypical Angina, NAP = Non-Anginal Pain, ASY = Asymptomatic",
    )
    RestingBP = st.number_input("Resting Blood Pressure", 80, 200, 120)
    Cholesterol = st.number_input("Cholesterol", 100, 600, 200)
    FastingBS = st.selectbox("Fasting Blood Sugar > 120 mg/dl (0 = No, 1 = Yes)", [0, 1])
    RestingECG = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])

with col2:
    MaxHR = st.number_input("Max Heart Rate", 60, 220, 150)
    ExerciseAngina = st.selectbox("Exercise Angina", ["Yes", "No"])
    Oldpeak = st.number_input("ST Depression", 0.0, 6.0, 1.0, 0.1)
    ST_Slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])
    MajorVessels = st.selectbox("Major Vessels", [0, 1, 2, 3])
    Thalassemia = st.selectbox(
        "Thalassemia",
        [1, 2, 3],
        format_func=lambda x: {1: "1 - Fixed defect", 2: "2 - Normal", 3: "3 - Reversible defect"}[x],
    )

# ---------------------------
# Prediction
# ---------------------------
if st.button("Predict Heart Disease"):
    try:
        row = {
            "Age": Age,
            "Gender": 1 if Gender == "Male" else 0,
            "ChestPainType": CHEST_PAIN[ChestPainType],
            "RestingBp": RestingBP,            # note: model uses 'RestingBp'
            "Cholesterol": Cholesterol,
            "FastingBs": FastingBS,            # note: model uses 'FastingBs'
            "RestingECG": REST_ECG[RestingECG],
            "MaxHR": MaxHR,
            "ExerciseAngina": 1 if ExerciseAngina == "Yes" else 0,
            "ST_Depression": Oldpeak,          # note: model uses 'ST_Depression'
            "ST_Slope": SLOPE[ST_Slope],
            "MajorVessels": MajorVessels,
            "Thalassemia": Thalassemia,
        }
        input_df = pd.DataFrame([row])[COLUMNS]

        # In this model, class 1 = NO disease, class 0 = disease (target was inverted in training data)
        proba = model.predict_proba(input_df)[0]
        classes = list(model.classes_)
        p_disease = proba[classes.index(0)]

        st.subheader("Result")
        if p_disease >= 0.5:
            st.error(f"⚠ High Risk of Heart Disease Detected ({p_disease:.0%} probability)")
        else:
            st.success(f"✅ Low Risk of Heart Disease ({p_disease:.0%} probability of disease)")
        st.caption("This is a demo ML model, not a medical diagnosis.")

    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
