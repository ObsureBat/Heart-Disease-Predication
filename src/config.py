"""Configuration settings for the application"""

PAGE_CONFIG = {
    "page_title": "Heart Disease Prediction",
    "page_icon": "❤️",
    "layout": "wide"
}

CUSTOM_CSS = """
    <style>
    .main { padding: 2rem; }
    .stButton>button {
        width: 100%;
        background-color: #ff4b4b;
        color: white;
        border-radius: 0.5rem;
        padding: 0.75rem;
        margin: 1rem 0;
    }
    .stButton>button:hover {
        background-color: #ff3333;
    }
    </style>
"""

MODEL_CONFIG = {
    "random_state": 42,
    "test_size": 0.2,
    "n_estimators": 100
}

FEATURE_MAPS = {
    "sex": {"Male": 1, "Female": 0},
    "chest_pain": {
        "Typical Angina": 0,
        "Atypical Angina": 1,
        "Non-Anginal Pain": 2,
        "Asymptomatic": 3
    },
    "ecg": {
        "Normal": 0,
        "ST-T Wave Abnormality": 1,
        "Left Ventricular Hypertrophy": 2
    },
    "slope": {
        "Upsloping": 0,
        "Flat": 1,
        "Downsloping": 2
    }
}