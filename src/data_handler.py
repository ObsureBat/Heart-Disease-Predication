"""Data loading and preprocessing"""
import streamlit as st
import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from utils.constants import MODEL_PARAMS

@st.cache_data
def load_dataset():
    """Load the UCI Heart Disease dataset"""
    heart_disease = fetch_ucirepo(id=45)
    return (
        pd.DataFrame(heart_disease.data.features),
        pd.DataFrame(heart_disease.data.targets)
    )

@st.cache_data
def prepare_dataset(X, y):
    """Preprocess the dataset"""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return train_test_split(
        X_scaled,
        y.values.ravel(),
        test_size=MODEL_PARAMS["test_size"],
        random_state=MODEL_PARAMS["random_state"]
    ), scaler