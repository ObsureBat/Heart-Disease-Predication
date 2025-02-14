"""Streamlit application for heart disease prediction with advanced analytics"""
import streamlit as st
import pandas as pd
import numpy as np
from src.model_handler import train_models, predict
from src.data import load_and_preprocess_data
import plotly.graph_objects as go
import shap
import matplotlib.pyplot as plt
import joblib

def create_metrics_plot(results):
    """Create a comprehensive metrics comparison plot"""
    metrics_data = []
    for model, metrics in results['metrics'].items():
        for metric, value in metrics.items():
            metrics_data.append({
                'Model': model,
                'Metric': metric,
                'Score': value
            })
    
    df = pd.DataFrame(metrics_data)
    
    # Create figure with go instead of px for more control
    fig = go.Figure()
    
    # Add bars for each metric
    for metric in df['Metric'].unique():
        metric_data = df[df['Metric'] == metric]
        fig.add_trace(go.Bar(
            name=metric,
            x=metric_data['Model'],
            y=metric_data['Score'],
            text=metric_data['Score'].apply(lambda x: f'{x:.2%}'),
            textposition='auto',
        ))
    
    # Update layout
    fig.update_layout(
        title='Model Performance Comparison',
        yaxis_title='Score',
        yaxis=dict(range=[0, 1]),
        barmode='group',
        height=500
    )
    
    return fig

def plot_feature_importance(feature_importance):
    """Create feature importance plot"""
    # Sort feature importance
    feature_importance = feature_importance.sort_values('importance', ascending=True)
    
    # Create figure with go
    fig = go.Figure()
    
    # Add bar trace
    fig.add_trace(go.Bar(
        x=feature_importance['importance'],
        y=feature_importance['feature'],
        orientation='h',
        text=feature_importance['importance'].apply(lambda x: f'{x:.4f}'),
        textposition='auto',
    ))
    
    # Update layout
    fig.update_layout(
        title='Feature Importance',
        xaxis_title='Importance Score',
        yaxis=dict(autorange="reversed"),
        height=500
    )
    
    return fig

def create_shap_plot(shap_values, feature_names):
    """Create SHAP summary plot"""
    try:
        # Convert feature names to list if needed
        if isinstance(feature_names, (pd.Index, np.ndarray)):
            feature_names = feature_names.tolist()
        
        # Convert shap_values to numpy array if needed
        if isinstance(shap_values, list):
            shap_values = np.array(shap_values)
        
        # Handle different shapes of shap_values
        if len(shap_values.shape) == 1:
            # Single sample, single dimension
            feature_importance = np.abs(shap_values)
        elif len(shap_values.shape) == 2:
            # Multiple samples or already in correct format
            feature_importance = np.abs(shap_values).mean(axis=0)
        elif len(shap_values.shape) == 3:
            # Multiple samples with extra dimension
            feature_importance = np.abs(shap_values).mean(axis=(0, 1))
        else:
            raise ValueError(f"Unexpected SHAP values shape: {shap_values.shape}")
        
        # Verify dimensions match
        if len(feature_importance) != len(feature_names):
            raise ValueError(f"Mismatch between feature importance ({len(feature_importance)}) and feature names ({len(feature_names)})")
        
        # Create DataFrame for plotting
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': feature_importance.flatten()  # Ensure 1D array
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values('importance', ascending=True)
        
        # Create plot
        fig = go.Figure()
        
        # Add horizontal bar trace
        fig.add_trace(go.Bar(
            x=importance_df['importance'],
            y=importance_df['feature'],
            orientation='h',
            text=importance_df['importance'].apply(lambda x: f'{x:.4f}'),
            textposition='auto',
        ))
        
        # Update layout
        fig.update_layout(
            title='SHAP Feature Importance',
            xaxis_title='mean(|SHAP value|)',
            yaxis=dict(autorange="reversed"),
            height=500,
            showlegend=False
        )
        
        # Display plot
        st.plotly_chart(fig)
        
        # Display table
        st.write("Feature Importance Table:")
        table_df = importance_df.sort_values('importance', ascending=False).copy()
        table_df['importance'] = table_df['importance'].round(4)
        st.dataframe(table_df)
        
    except Exception as e:
        st.error(f"Could not create SHAP plot: {str(e)}")
        st.write("Debug info:")
        st.write(f"SHAP values shape: {shap_values.shape if hasattr(shap_values, 'shape') else 'no shape'}")
        st.write(f"Feature names length: {len(feature_names)}")
        st.write(f"Error: {str(e)}")

def main():
    st.set_page_config(
        page_title="Heart Disease Prediction",
        page_icon="❤️",
        layout="wide"
    )
    
    st.title("❤️ Advanced Heart Disease Prediction System")
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Train Model", "Make Prediction"])
    
    if page == "Train Model":
        st.header("Model Training and Evaluation")
        
        if st.button("Train New Models"):
            with st.spinner("Training models with advanced techniques..."):
                # Load and preprocess data
                (X_train, X_test, y_train, y_test), scaler = load_and_preprocess_data()
                
                # Train models and get results
                results = train_models(X_train, X_test, y_train, y_test, scaler)
                
                st.success("Models trained successfully!")
                
                # Display comprehensive results
                st.subheader("Model Performance Metrics")
                st.plotly_chart(create_metrics_plot(results))
                
                # Display detailed metrics for each model
                col1, col2, col3 = st.columns(3)
                models = ['Random Forest', 'XGBoost', 'Voting']
                cols = [col1, col2, col3]
                
                for model, col in zip(models, cols):
                    with col:
                        st.subheader(f"{model} Metrics")
                        metrics = results['metrics'][model]
                        for metric, value in metrics.items():
                            st.metric(metric.capitalize(), f"{value:.2%}")
                
                # Feature Importance
                st.subheader("Feature Importance Analysis")
                st.plotly_chart(plot_feature_importance(results['feature_importance']))
                
                # SHAP Values
                st.subheader("SHAP Value Analysis")
                if results['shap_values'] is not None:
                    create_shap_plot(
                        results['shap_values']['values'],
                        results['shap_values']['feature_names']
                    )
                else:
                    st.warning("SHAP values could not be calculated. Showing feature importance plot instead.")
                
                st.markdown("""
                ### Model Interpretation Guide
                - **Feature Importance**: Shows which factors most strongly influence the prediction
                - **SHAP Values**: Explains how each feature contributes to individual predictions
                - **Metrics Explanation**:
                    - Accuracy: Overall prediction accuracy
                    - Precision: Accuracy of positive predictions
                    - Recall: Ability to find all positive cases
                    - F1: Balance between precision and recall
                    - AUC-ROC: Overall ability to distinguish between classes
                """)
    
    else:
        st.header("Heart Disease Risk Assessment")
        
        try:
            # Load models
            models = {
                'rf': joblib.load('models/random_forest_model.joblib'),
                'xgb': joblib.load('models/xgboost_model.joblib'),
                'voting': joblib.load('models/voting_model.joblib')
            }
            scaler = joblib.load('models/scaler.joblib')
            
            # Create input form
            with st.form("prediction_form"):
                col1, col2 = st.columns(2)
                
                with col1:
                    age = st.number_input("Age", min_value=20, max_value=100, value=45)
                    sex = st.selectbox("Sex", ["Male", "Female"])
                    cp = st.selectbox("Chest Pain Type", 
                                    ["Typical Angina", "Atypical Angina", 
                                     "Non-Anginal Pain", "Asymptomatic"])
                    trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 90, 200, 120)
                    chol = st.number_input("Cholesterol (mg/dl)", 100, 600, 200)
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
                
                submitted = st.form_submit_button("Analyze Risk")
                
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
                    prediction = predict(input_data, models, scaler)
                    
                    # Display results
                    st.header("Risk Assessment Results")
                    
                    # Main prediction with confidence
                    risk_prob = prediction['probability']
                    confidence = prediction['confidence']
                    
                    # Determine risk level and color
                    if risk_prob < 0.2:
                        risk_level = "Very Low"
                        color = "#2ecc71"
                    elif risk_prob < 0.4:
                        risk_level = "Low"
                        color = "#27ae60"
                    elif risk_prob < 0.6:
                        risk_level = "Moderate"
                        color = "#f1c40f"
                    elif risk_prob < 0.8:
                        risk_level = "High"
                        color = "#e67e22"
                    else:
                        risk_level = "Very High"
                        color = "#e74c3c"
                    
                    # Display main prediction
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(
                            f"""
                            <div style="padding: 20px; background-color: {color}; 
                                      border-radius: 10px; text-align: center; color: white;">
                                <h2 style="margin: 0; font-size: 2.5em;">{risk_prob:.1%}</h2>
                                <p style="margin: 5px 0 0 0; font-size: 1.5em;">
                                    {risk_level} Risk of Heart Disease
                                </p>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    
                    with col2:
                        st.markdown(
                            f"""
                            <div style="padding: 20px; background-color: #34495e; 
                                      border-radius: 10px; text-align: center; color: white;">
                                <h2 style="margin: 0; font-size: 2.5em;">{confidence:.1%}</h2>
                                <p style="margin: 5px 0 0 0; font-size: 1.5em;">
                                    Prediction Confidence
                                </p>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    
                    # Model agreement
                    st.subheader("Model Agreement")
                    individual_preds = prediction['individual_predictions']
                    agreement_df = pd.DataFrame({
                        'Model': ['Random Forest', 'XGBoost', 'Ensemble'],
                        'Prediction': [
                            individual_preds['random_forest'],
                            individual_preds['xgboost'],
                            individual_preds['voting']
                        ]
                    })
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=agreement_df['Model'],
                        y=agreement_df['Prediction'],
                        text=agreement_df['Prediction'].apply(lambda x: f'{x:.2%}'),
                        textposition='auto',
                    ))
                    fig.update_layout(
                        title='Individual Model Predictions',
                        yaxis_title='Prediction',
                        yaxis=dict(range=[0, 1]),
                        height=500
                    )
                    st.plotly_chart(fig)
                    
                    # Feature contribution
                    st.subheader("Feature Contribution Analysis")
                    if prediction['shap_values'] is not None:
                        create_shap_plot(
                            prediction['shap_values'],
                            prediction['feature_names']
                        )
                    else:
                        st.warning("Feature contributions could not be calculated.")
                    
                    # Risk interpretation
                    st.markdown(f"""
                    ### Risk Level Interpretation
                    - Current Risk Level: **{risk_level}** ({risk_prob:.1%})
                    - Prediction Confidence: **{confidence:.1%}**
                    
                    #### What This Means
                    - **Risk Level**: {risk_level} risk indicates {
                        "minimal concern and healthy indicators" if risk_prob < 0.2 else
                        "some risk factors present but generally healthy" if risk_prob < 0.4 else
                        "moderate risk factors requiring attention" if risk_prob < 0.6 else
                        "significant risk factors requiring medical consultation" if risk_prob < 0.8 else
                        "multiple severe risk factors requiring immediate medical attention"
                    }
                    
                    - **Confidence Score**: {
                        "Very high confidence in the prediction" if confidence > 0.9 else
                        "High confidence in the prediction" if confidence > 0.8 else
                        "Moderate confidence in the prediction" if confidence > 0.7 else
                        "Fair confidence in the prediction" if confidence > 0.6 else
                        "Low confidence in the prediction - consider consulting a healthcare provider"
                    }
                    
                    #### Recommendations
                    {
                        "- Maintain current healthy lifestyle\n- Regular check-ups recommended" if risk_prob < 0.2 else
                        "- Consider lifestyle modifications\n- Schedule routine check-up" if risk_prob < 0.4 else
                        "- Consult healthcare provider\n- Evaluate risk factors\n- Consider lifestyle changes" if risk_prob < 0.6 else
                        "- Prompt medical consultation advised\n- Comprehensive health evaluation needed" if risk_prob < 0.8 else
                        "- Immediate medical attention recommended\n- Comprehensive cardiac evaluation needed"
                    }
                    """)
                    
        except FileNotFoundError:
            st.error("Models not found. Please train the models first.")
            st.info("Go to the 'Train Model' page and click 'Train New Models' to get started.")

if __name__ == "__main__":
    main()