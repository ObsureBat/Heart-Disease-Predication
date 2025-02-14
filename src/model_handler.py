"""Model training and prediction handling"""
import streamlit as st
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

def train_models():
    """Train models and return performance metrics"""
    from src.data import load_and_preprocess_data
    
    # Load and preprocess data
    (X_train_scaled, X_test_scaled, y_train, y_test), scaler = load_and_preprocess_data()
    
    # Convert y_train and y_test to numpy arrays and ravel them
    y_train_array = y_train.values.ravel()
    y_test_array = y_test.values.ravel()
    
    # Calculate class weights
    from sklearn.utils.class_weight import compute_class_weight
    classes = np.unique(y_train_array)
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=classes,
        y=y_train_array
    )
    class_weight_dict = dict(zip(classes, class_weights))
    
    # Train Random Forest with optimized hyperparameters
    rf_model = RandomForestClassifier(
        n_estimators=500,
        max_depth=15,
        min_samples_split=4,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=42,
        class_weight=class_weight_dict,
        bootstrap=True,
        n_jobs=-1
    )
    rf_model.fit(X_train_scaled, y_train_array)
    rf_pred = rf_model.predict(X_test_scaled)
    rf_accuracy = accuracy_score(y_test_array, rf_pred)
    rf_report = classification_report(y_test_array, rf_pred, zero_division=1)
    
    # Train XGBoost with optimized hyperparameters
    xgb_model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        gamma=0.1,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    
    # Calculate sample weight for XGBoost
    sample_weight = np.where(y_train_array == 1, 
                           len(y_train_array)/(2*np.sum(y_train_array == 1)),
                           len(y_train_array)/(2*np.sum(y_train_array == 0)))
    
    xgb_model.fit(
        X_train_scaled, 
        y_train_array,
        sample_weight=sample_weight,
        eval_set=[(X_test_scaled, y_test_array)],
        early_stopping_rounds=50,
        verbose=False
    )
    
    xgb_pred = xgb_model.predict(X_test_scaled)
    xgb_accuracy = accuracy_score(y_test_array, xgb_pred)
    xgb_report = classification_report(y_test_array, xgb_pred, zero_division=1)
    
    # Save models
    joblib.dump(rf_model, 'models/random_forest_model.joblib')
    joblib.dump(xgb_model, 'models/xgboost_model.joblib')
    joblib.dump(scaler, 'models/scaler.joblib')
    
    return {
        'rf_accuracy': rf_accuracy,
        'rf_report': rf_report,
        'xgb_accuracy': xgb_accuracy,
        'xgb_report': xgb_report
    }

def load_models():
    """Load trained models and scaler"""
    try:
        rf_model = joblib.load('models/random_forest_model.joblib')
        xgb_model = joblib.load('models/xgboost_model.joblib')
        scaler = joblib.load('models/scaler.joblib')
        return rf_model, xgb_model, scaler
    except FileNotFoundError:
        raise FileNotFoundError("Models not found. Please train the models first.")

def predict(input_data, rf_model, xgb_model, scaler):
    """Make prediction using ensemble of models"""
    # Scale input data
    input_scaled = scaler.transform(input_data)
    
    # Get predictions from both models
    rf_prob = rf_model.predict_proba(input_scaled)[0][1]
    xgb_prob = xgb_model.predict_proba(input_scaled)[0][1]
    
    # Average the predictions
    return (rf_prob + xgb_prob) / 2

def load_or_train_models():
    """Load pre-trained models or train new ones if not available"""
    try:
        return load_models()
    except FileNotFoundError:
        results = train_models()
        return load_models()

def make_prediction(input_data):
    """Make prediction using ensemble of models"""
    # Load or train models if not already loaded
    rf_model, xgb_model, scaler = load_or_train_models()
    
    # Scale input data
    input_scaled = scaler.transform(input_data)
    
    # Get predictions from both models
    rf_prob = rf_model.predict_proba(input_scaled)[0][1]
    xgb_prob = xgb_model.predict_proba(input_scaled)[0][1]
    
    # Average the predictions
    ensemble_prob = (rf_prob + xgb_prob) / 2
    
    # Get feature importance
    feature_importance = get_feature_importance(rf_model, xgb_model)
    
    return ensemble_prob, feature_importance

def get_feature_importance(rf_model, xgb_model):
    """Get combined feature importance from both models"""
    feature_names = [
        'Age', 'Sex', 'Chest Pain Type', 'Resting BP', 'Cholesterol',
        'Fasting Blood Sugar', 'Resting ECG', 'Max Heart Rate',
        'Exercise Angina', 'ST Depression', 'ST Slope'
    ]
    
    # Combine feature importance from both models
    rf_importance = rf_model.feature_importances_
    xgb_importance = xgb_model.feature_importances_
    
    # Average the feature importance
    combined_importance = (rf_importance + xgb_importance) / 2
    
    # Create list of dictionaries with feature names and importance
    importance_list = [
        {"feature": name, "importance": float(imp)}
        for name, imp in zip(feature_names, combined_importance)
    ]
    
    # Sort by importance
    importance_list.sort(key=lambda x: x["importance"], reverse=True)
    
    return importance_list