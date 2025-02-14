"""Enhanced model handler for heart disease prediction"""
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb
import joblib
from sklearn.preprocessing import StandardScaler
import shap

def get_feature_names():
    """Get feature names for the heart disease dataset"""
    return [
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
        'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
    ]

def train_models(X_train, X_test, y_train, y_test, scaler):
    """Train models with hyperparameter tuning and advanced evaluation"""
    
    # Get feature names
    feature_names = get_feature_names()
    
    # Convert data to DataFrames with feature names
    X_train_df = pd.DataFrame(X_train, columns=feature_names)
    X_test_df = pd.DataFrame(X_test, columns=feature_names)
    
    # Random Forest hyperparameter grid
    rf_param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    # XGBoost hyperparameter grid
    xgb_param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 4, 5],
        'learning_rate': [0.01, 0.1],
        'subsample': [0.8, 0.9, 1.0]
    }
    
    # Train Random Forest with GridSearchCV
    rf_grid = GridSearchCV(
        RandomForestClassifier(random_state=42),
        rf_param_grid,
        cv=5,
        scoring='precision',
        n_jobs=-1
    )
    rf_grid.fit(X_train_df, y_train.values.ravel())
    rf_model = rf_grid.best_estimator_
    
    # Train XGBoost with GridSearchCV
    xgb_grid = GridSearchCV(
        xgb.XGBClassifier(random_state=42),
        xgb_param_grid,
        cv=5,
        scoring='precision',
        n_jobs=-1
    )
    xgb_grid.fit(X_train_df, y_train.values.ravel())
    xgb_model = xgb_grid.best_estimator_
    
    # Create Voting Classifier
    voting_clf = VotingClassifier(
        estimators=[
            ('rf', rf_model),
            ('xgb', xgb_model)
        ],
        voting='soft'
    )
    voting_clf.fit(X_train_df, y_train.values.ravel())
    
    # Generate predictions
    rf_pred = rf_model.predict(X_test_df)
    xgb_pred = xgb_model.predict(X_test_df)
    voting_pred = voting_clf.predict(X_test_df)
    
    # Calculate probabilities
    rf_proba = rf_model.predict_proba(X_test_df)[:, 1]
    xgb_proba = xgb_model.predict_proba(X_test_df)[:, 1]
    voting_proba = voting_clf.predict_proba(X_test_df)[:, 1]
    
    # Calculate metrics
    results = {
        'models': {
            'rf': rf_model,
            'xgb': xgb_model,
            'voting': voting_clf
        },
        'metrics': {
            'Random Forest': {
                'accuracy': accuracy_score(y_test, rf_pred),
                'precision': precision_score(y_test, rf_pred),
                'recall': recall_score(y_test, rf_pred),
                'f1': f1_score(y_test, rf_pred),
                'auc_roc': roc_auc_score(y_test, rf_proba)
            },
            'XGBoost': {
                'accuracy': accuracy_score(y_test, xgb_pred),
                'precision': precision_score(y_test, xgb_pred),
                'recall': recall_score(y_test, xgb_pred),
                'f1': f1_score(y_test, xgb_pred),
                'auc_roc': roc_auc_score(y_test, xgb_proba)
            },
            'Voting': {
                'accuracy': accuracy_score(y_test, voting_pred),
                'precision': precision_score(y_test, voting_pred),
                'recall': recall_score(y_test, voting_pred),
                'f1': f1_score(y_test, voting_pred),
                'auc_roc': roc_auc_score(y_test, voting_proba)
            }
        }
    }
    
    # Calculate feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    results['feature_importance'] = feature_importance
    
    # Generate SHAP values for model interpretability
    try:
        explainer = shap.TreeExplainer(rf_model)
        shap_values = explainer.shap_values(X_test_df)
        
        # For binary classification, take positive class values
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Take positive class
        
        # Convert to numpy array if needed
        shap_values = np.array(shap_values)
        
        # Ensure 2D array
        if len(shap_values.shape) == 1:
            shap_values = shap_values.reshape(1, -1)
        elif len(shap_values.shape) > 2:
            shap_values = shap_values.reshape(-1, len(feature_names))
        
        # Calculate mean SHAP values
        mean_shap_values = np.abs(shap_values).mean(axis=0)
        
        # Store results
        results['shap_values'] = {
            'values': mean_shap_values,  # Store mean values for visualization
            'feature_names': feature_names
        }
    except Exception as e:
        print(f"Could not calculate SHAP values: {str(e)}")
        results['shap_values'] = None
    
    # Save models and scaler
    joblib.dump(rf_model, 'models/random_forest_model.joblib')
    joblib.dump(xgb_model, 'models/xgboost_model.joblib')
    joblib.dump(voting_clf, 'models/voting_model.joblib')
    joblib.dump(scaler, 'models/scaler.joblib')
    
    return results

def predict(input_data, models, scaler):
    """Make prediction with confidence scores and model interpretability"""
    # Get feature names
    feature_names = get_feature_names()
    
    # Scale input data
    scaled_data = scaler.transform(input_data)
    
    # Convert to DataFrame for SHAP
    scaled_df = pd.DataFrame(scaled_data, columns=feature_names)
    
    # Get predictions from all models
    rf_prob = models['rf'].predict_proba(scaled_df)[0, 1]
    xgb_prob = models['xgb'].predict_proba(scaled_df)[0, 1]
    voting_prob = models['voting'].predict_proba(scaled_df)[0, 1]
    
    # Calculate ensemble probability and confidence
    probabilities = np.array([rf_prob, xgb_prob, voting_prob])
    final_probability = np.mean(probabilities)
    confidence = 1 - np.std(probabilities)
    
    # Get feature contributions using SHAP
    try:
        explainer = shap.TreeExplainer(models['rf'])
        shap_values = explainer.shap_values(scaled_df)
        
        # For binary classification, take positive class values
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Take positive class
        
        # Convert to numpy array if needed
        shap_values = np.array(shap_values)
        
        # Ensure 2D array
        if len(shap_values.shape) == 1:
            shap_values = shap_values.reshape(1, -1)
        elif len(shap_values.shape) > 2:
            shap_values = shap_values.reshape(-1, len(feature_names))
        
        # For single prediction, take the first row
        shap_values = shap_values[0]
        
    except Exception as e:
        print(f"Could not calculate SHAP values: {str(e)}")
        shap_values = None
    
    return {
        'probability': final_probability,
        'confidence': confidence,
        'individual_predictions': {
            'random_forest': rf_prob,
            'xgboost': xgb_prob,
            'voting': voting_prob
        },
        'shap_values': shap_values,
        'feature_names': feature_names
    }