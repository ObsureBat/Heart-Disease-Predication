"""Data loading and preprocessing functions"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from ucimlrepo import fetch_ucirepo

def load_and_preprocess_data():
    """Load and preprocess the heart disease dataset"""
    # Fetch dataset from UCI ML Repository
    heart_disease = fetch_ucirepo(id=45)
    
    # Get features and target
    X = pd.DataFrame(heart_disease.data.features)
    y = pd.DataFrame(heart_disease.data.targets)
    
    # Print dataset info for debugging
    print("Dataset shape:", X.shape)
    print("Target distribution:\n", y.value_counts())
    
    # Drop rows with missing values
    X = X.dropna()
    y = y.loc[X.index]
    
    # Ensure target is binary
    y = (y > 0).astype(int)  # Convert to binary classification (0: No disease, 1: Disease)
    
    # Encode categorical variables
    le = LabelEncoder()
    categorical_columns = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
    for col in categorical_columns:
        if col in X.columns:
            X[col] = le.fit_transform(X[col].astype(str))
    
    # Create scaler
    scaler = StandardScaler()
    
    # Split data with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42,
        stratify=y
    )
    
    # Fit scaler on training data and transform both sets
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Print shapes for verification
    print("Training set shape:", X_train_scaled.shape)
    print("Test set shape:", X_test_scaled.shape)
    print("Training target distribution:\n", y_train.value_counts())
    
    return (X_train_scaled, X_test_scaled, y_train, y_test), scaler