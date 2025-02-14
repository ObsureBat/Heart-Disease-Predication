"""Feature mapping definitions"""

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