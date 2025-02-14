"""Enhanced model handler with Adaptive PSO and model stacking"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb
from lightgbm import LGBMClassifier
import joblib
import shap
from .pso_optimizer import AdaptivePSO

def get_feature_names():
    """Get feature names for the heart disease dataset"""
    return [
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
        'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
    ]

def train_models(X_train, X_test, y_train, y_test, scaler):
    """Train models with enhanced PSO optimization and model stacking"""
    
    # Get feature names
    feature_names = get_feature_names()
    
    # Convert data to DataFrames with feature names
    X_train_df = pd.DataFrame(X_train, columns=feature_names)
    X_test_df = pd.DataFrame(X_test, columns=feature_names)
    
    # Initialize Adaptive PSO optimizer
    pso = AdaptivePSO(
        n_particles=20,
        max_iter=30,
        n_iter_no_improve=5,
        tolerance=1e-4
    )
    
    # Define parameter bounds and types for all models
    model_configs = {
        'rf': {
            'class': RandomForestClassifier,
            'bounds': {
                'n_estimators': (50, 200),
                'max_depth': (3, 20),
                'min_samples_split': (2, 10),
                'min_samples_leaf': (1, 5),
                'max_features': (0.3, 0.8)
            },
            'types': {
                'n_estimators': 'int',
                'max_depth': 'int',
                'min_samples_split': 'int',
                'min_samples_leaf': 'int',
                'max_features': 'float'
            }
        },
        'xgb': {
            'class': xgb.XGBClassifier,
            'bounds': {
                'n_estimators': (50, 200),
                'max_depth': (3, 10),
                'learning_rate': (0.01, 0.3),
                'subsample': (0.6, 0.9),
                'colsample_bytree': (0.6, 0.9),
                'gamma': (0, 3)
            },
            'types': {
                'n_estimators': 'int',
                'max_depth': 'int',
                'learning_rate': 'float',
                'subsample': 'float',
                'colsample_bytree': 'float',
                'gamma': 'float'
            }
        },
        'lgb': {
            'class': LGBMClassifier,
            'bounds': {
                'n_estimators': (50, 200),
                'max_depth': (3, 10),
                'learning_rate': (0.01, 0.3),
                'subsample': (0.6, 0.9),
                'colsample_bytree': (0.6, 0.9),
                'min_child_samples': (5, 30)
            },
            'types': {
                'n_estimators': 'int',
                'max_depth': 'int',
                'learning_rate': 'float',
                'subsample': 'float',
                'colsample_bytree': 'float',
                'min_child_samples': 'int'
            }
        },
        'gb': {
            'class': GradientBoostingClassifier,
            'bounds': {
                'n_estimators': (50, 200),
                'max_depth': (3, 10),
                'learning_rate': (0.01, 0.3),
                'subsample': (0.6, 0.9),
                'min_samples_split': (2, 10),
                'min_samples_leaf': (1, 5)
            },
            'types': {
                'n_estimators': 'int',
                'max_depth': 'int',
                'learning_rate': 'float',
                'subsample': 'float',
                'min_samples_split': 'int',
                'min_samples_leaf': 'int'
            }
        }
    }
    
    # Train and optimize all models
    trained_models = {}
    optimization_results = {}
    model_predictions = {}
    
    print("Training models with PSO optimization...")
    for model_name, config in model_configs.items():
        print(f"\nOptimizing {model_name}...")
        best_params, best_score, history = pso.optimize(
            config['class'],
            config['bounds'],
            config['types'],
            X_train_df,
            y_train,
            cv=5,
            scoring='roc_auc'
        )
        
        # Train model with best parameters
        model = config['class'](**best_params, random_state=42)
        model.fit(X_train_df, y_train.values.ravel())
        
        # Store results
        trained_models[model_name] = model
        optimization_results[model_name] = {
            'best_params': best_params,
            'best_score': best_score,
            'convergence_history': history
        }
        
        # Get predictions
        model_predictions[model_name] = model.predict(X_test_df)
    
    # Create weighted voting classifier
    weights = [opt['best_score'] for opt in optimization_results.values()]
    voting_clf = VotingClassifier(
        estimators=[
            (name, model) for name, model in trained_models.items()
        ],
        voting='soft',
        weights=weights
    )
    voting_clf.fit(X_train_df, y_train.values.ravel())
    
    # Calculate metrics for all models
    results = {
        'models': trained_models | {'voting': voting_clf},
        'metrics': {},
        'optimization_results': optimization_results
    }
    
    # Calculate metrics for each model
    for name, predictions in model_predictions.items():
        proba = trained_models[name].predict_proba(X_test_df)[:, 1]
        results['metrics'][name] = {
            'accuracy': accuracy_score(y_test, predictions),
            'precision': precision_score(y_test, predictions),
            'recall': recall_score(y_test, predictions),
            'f1': f1_score(y_test, predictions),
            'auc_roc': roc_auc_score(y_test, proba)
        }
    
    # Add voting classifier metrics
    voting_pred = voting_clf.predict(X_test_df)
    voting_proba = voting_clf.predict_proba(X_test_df)[:, 1]
    results['metrics']['voting'] = {
        'accuracy': accuracy_score(y_test, voting_pred),
        'precision': precision_score(y_test, voting_pred),
        'recall': recall_score(y_test, voting_pred),
        'f1': f1_score(y_test, voting_pred),
        'auc_roc': roc_auc_score(y_test, voting_proba)
    }
    
    # Calculate feature importance using the best model
    best_model_name = max(
        results['metrics'],
        key=lambda x: results['metrics'][x]['auc_roc']
    )
    best_model = trained_models[best_model_name]
    
    if hasattr(best_model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=False)
        results['feature_importance'] = feature_importance
    
    # Generate SHAP values for model interpretability
    try:
        explainer = shap.TreeExplainer(best_model)
        shap_values = explainer.shap_values(X_test_df)
        
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        shap_values = np.array(shap_values)
        
        if len(shap_values.shape) == 1:
            shap_values = shap_values.reshape(1, -1)
        elif len(shap_values.shape) > 2:
            shap_values = shap_values.reshape(-1, len(feature_names))
        
        mean_shap_values = np.abs(shap_values).mean(axis=0)
        
        results['shap_values'] = {
            'values': mean_shap_values,
            'feature_names': feature_names
        }
    except Exception as e:
        print(f"Could not calculate SHAP values: {str(e)}")
        results['shap_values'] = None
    
    # Save models and scaler
    for name, model in trained_models.items():
        joblib.dump(model, f'models/{name}_model.joblib')
    joblib.dump(voting_clf, 'models/voting_model.joblib')
    joblib.dump(scaler, 'models/scaler.joblib')
    
    return results

def get_risk_level(probability: float) -> str:
    """Determine risk level based on probability"""
    if probability < 0.2:
        return "Very Low"
    elif probability < 0.4:
        return "Low"
    elif probability < 0.6:
        return "Moderate"
    elif probability < 0.8:
        return "High"
    else:
        return "Very High"

def predict(input_data, models, scaler):
    """Make prediction with confidence scores and model interpretability"""
    # Get feature names
    feature_names = get_feature_names()
    
    # Scale input data
    scaled_data = scaler.transform(input_data)
    
    # Convert to DataFrame for SHAP
    scaled_df = pd.DataFrame(scaled_data, columns=feature_names)
    
    # Get predictions from all models
    predictions = {}
    for name, model in models.items():
        if name != 'voting':
            predictions[name] = model.predict_proba(scaled_df)[0, 1]
    
    # Get voting classifier prediction
    voting_prob = models['voting'].predict_proba(scaled_df)[0, 1]
    
    # Calculate weighted ensemble probability
    weights = np.array([1/len(predictions)] * len(predictions))  # Equal weights
    probabilities = np.array(list(predictions.values()))
    final_probability = np.average(probabilities, weights=weights)
    
    # Calculate confidence based on prediction agreement
    confidence = 1 - np.std(probabilities)
    
    # Get feature contributions using SHAP from the best model
    best_model = models[max(predictions.keys(), key=lambda k: predictions[k])]
    try:
        explainer = shap.TreeExplainer(best_model)
        shap_values = explainer.shap_values(scaled_df)
        
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        shap_values = np.array(shap_values)
        
        if len(shap_values.shape) == 1:
            shap_values = shap_values.reshape(1, -1)
        elif len(shap_values.shape) > 2:
            shap_values = shap_values.reshape(-1, len(feature_names))
        
        shap_values = shap_values[0]
        
    except Exception as e:
        print(f"Could not calculate SHAP values: {str(e)}")
        shap_values = None
    
    return {
        'probability': final_probability,
        'confidence': confidence,
        'individual_predictions': predictions | {'voting': voting_prob},
        'shap_values': shap_values,
        'feature_names': feature_names,
        'risk_level': get_risk_level(final_probability)  # Add risk level to output
    }