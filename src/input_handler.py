"""Form input handling"""
import streamlit as st
import numpy as np
from utils.feature_maps import FEATURE_MAPS

def create_patient_form():
    """Create the input form for patient data"""
    with st.form("patient_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input("Age", min_value=20, max_value=100, value=45)
            sex = st.selectbox("Sex", ["Male", "Female"])
            chest_pain = st.selectbox(
                "Chest Pain Type",
                list(FEATURE_MAPS["chest_pain"].keys())
            )
            resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
            cholesterol = st.number_input("Cholesterol (mg/dl)", 100, 600, 200)
        
        with col2:
            fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])
            resting_ecg = st.selectbox(
                "Resting ECG",
                list(FEATURE_MAPS["ecg"].keys())
            )
            max_hr = st.number_input("Maximum Heart Rate", 60, 220, 150)
            exercise_angina = st.selectbox("Exercise-Induced Angina", ["No", "Yes"])
            st_slope = st.selectbox("ST Slope", list(FEATURE_MAPS["slope"].keys()))
        
        oldpeak = st.slider("ST Depression (Oldpeak)", 0.0, 6.0, 0.0, 0.1)
        submitted = st.form_submit_button("Predict Risk")
        
        if submitted:
            return process_input(
                age, sex, chest_pain, resting_bp, cholesterol,
                fasting_bs, resting_ecg, max_hr, exercise_angina,
                oldpeak, st_slope
            )
        return None

def process_input(age, sex, chest_pain, resting_bp, cholesterol,
                 fasting_bs, resting_ecg, max_hr, exercise_angina,
                 oldpeak, st_slope):
    """Process and prepare input data"""
    return np.array([[
        age,
        FEATURE_MAPS["sex"][sex],
        FEATURE_MAPS["chest_pain"][chest_pain],
        resting_bp,
        cholesterol,
        1 if fasting_bs == "Yes" else 0,
        FEATURE_MAPS["ecg"][resting_ecg],
        max_hr,
        1 if exercise_angina == "Yes" else 0,
        oldpeak,
        FEATURE_MAPS["slope"][st_slope]
    ]])