"""Model training and prediction functions"""
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from .config import MODEL_CONFIG

@st.cache_resource
def train_models(X_train, y_train):
    """Train Random Forest and XGBoost models"""
    rf_model = RandomForestClassifier(
        n_estimators=MODEL_CONFIG["n_estimators"],
        random_state=MODEL_CONFIG["random_state"]
    )
    rf_model.fit(X_train, y_train)
    
    xgb_model = xgb.XGBClassifier(
        random_state=MODEL_CONFIG["random_state"],
        use_label_encoder=False,
        eval_metric='logloss'
    )
    xgb_model.fit(X_train, y_train)
    
    return rf_model, xgb_model

def get_ensemble_prediction(rf_model, xgb_model, input_scaled):
    """Get ensemble prediction from both models"""
    rf_prob = rf_model.predict_proba(input_scaled)[0][1]
    xgb_prob = xgb_model.predict_proba(input_scaled)[0][1]
    return (rf_prob + xgb_prob) / 2