"""Streamlit application for heart disease prediction"""
import streamlit as st
import pandas as pd
import numpy as np
from src.model_handler import train_models, load_models, predict
import plotly.express as px
import plotly.graph_objects as go

def create_metrics_plot(results):
    """Create a bar plot comparing model metrics"""
    metrics = pd.DataFrame({
        'Model': ['Random Forest', 'Random Forest', 'XGBoost', 'XGBoost'],
        'Metric': ['Precision', 'Recall', 'Precision', 'Recall'],
        'Score': [
            float(results['rf_report'].split('\n')[2].split()[1]),
            float(results['rf_report'].split('\n')[2].split()[2]),
            float(results['xgb_report'].split('\n')[2].split()[1]),
            float(results['xgb_report'].split('\n')[2].split()[2])
        ]
    })
    
    fig = px.bar(metrics, x='Model', y='Score', color='Metric', barmode='group',
                 title='Model Performance Comparison')
    fig.update_layout(yaxis_title='Score', yaxis_range=[0, 1])
    return fig

def main():
    st.set_page_config(
        page_title="Heart Disease Prediction",
        page_icon="❤️",
        layout="wide"
    )
    
    st.title("❤️ Heart Disease Prediction Model")
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Train Model", "Make Prediction"])
    
    if page == "Train Model":
        st.header("Model Training")
        
        if st.button("Train New Models"):
            with st.spinner("Training models..."):
                results = train_models()
                
                st.success("Models trained successfully!")
                
                # Display results
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Random Forest Results")
                    st.metric("Accuracy", f"{results['rf_accuracy']:.2%}")
                    with st.expander("Detailed Metrics"):
                        st.text("Classification Report:")
                        st.text(results['rf_report'])
                
                with col2:
                    st.subheader("XGBoost Results")
                    st.metric("Accuracy", f"{results['xgb_accuracy']:.2%}")
                    with st.expander("Detailed Metrics"):
                        st.text("Classification Report:")
                        st.text(results['xgb_report'])
                
                # Plot metrics comparison
                st.plotly_chart(create_metrics_plot(results))
    
    else:
        st.header("Make Prediction")
        
        try:
            rf_model, xgb_model, scaler = load_models()
            
            # Create input form
            with st.form("prediction_form"):
                col1, col2 = st.columns(2)
                
                with col1:
                    age = st.number_input("Age", min_value=20, max_value=100, value=45)
                    sex = st.selectbox("Sex", ["Male", "Female"])
                    cp = st.selectbox("Chest Pain Type", 
                                    ["Typical Angina", "Atypical Angina", 
                                     "Non-Anginal Pain", "Asymptomatic"])
                    trestbps = st.number_input("Resting Blood Pressure", 90, 200, 120)
                    chol = st.number_input("Cholesterol", 100, 600, 200)
                    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])
                    restecg = st.selectbox("Resting ECG", 
                                         ["Normal", "ST-T Wave Abnormality", 
                                          "Left Ventricular Hypertrophy"])
                
                with col2:
                    thalach = st.number_input("Maximum Heart Rate", 60, 220, 150)
                    exang = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
                    oldpeak = st.number_input("ST Depression", 0.0, 6.0, 0.0, 0.1)
                    slope = st.selectbox("ST Slope", ["Upsloping", "Flat", "Downsloping"])
                    ca = st.number_input("Number of Major Vessels", 0, 4, 0)
                    thal = st.selectbox("Thalassemia", ["Normal", "Fixed Defect", "Reversible Defect"])
                
                submitted = st.form_submit_button("Predict")
                
                if submitted:
                    # Prepare input data
                    input_data = pd.DataFrame({
                        'age': [age],
                        'sex': [1 if sex == "Male" else 0],
                        'cp': [["Typical Angina", "Atypical Angina", 
                               "Non-Anginal Pain", "Asymptomatic"].index(cp)],
                        'trestbps': [trestbps],
                        'chol': [chol],
                        'fbs': [1 if fbs == "Yes" else 0],
                        'restecg': [["Normal", "ST-T Wave Abnormality", 
                                    "Left Ventricular Hypertrophy"].index(restecg)],
                        'thalach': [thalach],
                        'exang': [1 if exang == "Yes" else 0],
                        'oldpeak': [oldpeak],
                        'slope': [["Upsloping", "Flat", "Downsloping"].index(slope)],
                        'ca': [ca],
                        'thal': [3 if thal == "Normal" else 6 if thal == "Fixed Defect" else 7]
                    })
                    
                    # Make prediction
                    probability = predict(input_data, rf_model, xgb_model, scaler)
                    
                    # Display result
                    st.header("Prediction Result")
                    
                    # Create risk level indicator with more detailed categorization
                    if probability < 0.2:
                        risk_level = "Very Low"
                        color = "#2ecc71"
                    elif probability < 0.4:
                        risk_level = "Low"
                        color = "#27ae60"
                    elif probability < 0.6:
                        risk_level = "Moderate"
                        color = "#f1c40f"
                    elif probability < 0.8:
                        risk_level = "High"
                        color = "#e67e22"
                    else:
                        risk_level = "Very High"
                        color = "#e74c3c"
                    
                    st.markdown(
                        f"""
                        <div style="padding: 20px; background-color: {color}; 
                                  border-radius: 10px; text-align: center; color: white;">
                            <h2 style="margin: 0; font-size: 2.5em;">{probability:.1%}</h2>
                            <p style="margin: 5px 0 0 0; font-size: 1.5em;">
                                {risk_level} Risk of Heart Disease
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    
                    # Add interpretation
                    st.markdown("""
                        ### Risk Level Interpretation
                        - **Very Low** (0-20%): Minimal risk factors present
                        - **Low** (20-40%): Some risk factors, but generally healthy
                        - **Moderate** (40-60%): Several risk factors present, monitoring recommended
                        - **High** (60-80%): Significant risk factors, medical consultation advised
                        - **Very High** (80-100%): Multiple severe risk factors, immediate medical attention recommended
                    """)
                    
        except FileNotFoundError:
            st.error("Models not found. Please train the models first.")

if __name__ == "__main__":
    main()