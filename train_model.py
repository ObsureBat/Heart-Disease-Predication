from src.data import load_and_preprocess_data
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import joblib
import os

def train_models():
    """Train and save the models"""
    # Load and preprocess data
    (X_train, X_test, y_train, y_test), scaler = load_and_preprocess_data()
    
    # Train Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train.values.ravel())
    
    # Train XGBoost
    xgb_model = xgb.XGBClassifier(n_estimators=100, random_state=42)
    xgb_model.fit(X_train, y_train.values.ravel())
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Save models and scaler
    joblib.dump(rf_model, 'models/random_forest_model.joblib')
    joblib.dump(xgb_model, 'models/xgboost_model.joblib')
    joblib.dump(scaler, 'models/scaler.joblib')

if __name__ == "__main__":
    train_models()