import streamlit as st
import numpy as np
import joblib

model = joblib.load('/home/aryan/Broadway_AI/models/knn_breast_cancer_model.joblib')
scaler = joblib.load('/home/aryan/Broadway_AI/models/knn_breast_cancer_scaler.joblib')
encoder = joblib.load('/home/aryan/Broadway_AI/models/knn_breast_cancer_encoder.joblib')

# model = joblib.load('../models/knn_breast_cancer_model.joblib')
# scaler = joblib.load('../models/knn_breast_cancer_scaler.joblib')
# encoder = joblib.load('../models/Knn_breast_cancer_encoder.joblib')

st.title("BREAST CANCER PREDICTION APP")
st.write("Please enter the patient's information:")

#inputs
input_features = [
    "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean",
    "compactness_mean", "concavity_mean", "concave points_mean", "symmetry_mean", "fractal_dimension_mean",
    "radius_se", "texture_se", "perimeter_se", "area_se", "smoothness_se",
    "compactness_se", "concavity_se", "concave points_se", "symmetry_se", "fractal_dimension_se",
    "radius_worst", "texture_worst", "perimeter_worst", "area_worst", "smoothness_worst",
    "compactness_worst", "concavity_worst", "concave points_worst", "symmetry_worst", "fractal_dimension_worst"
]

# Dictionary to store user inputs
user_inputs = {}

for feature in input_features:
    user_inputs[feature] = st.number_input(feature.replace("_", " ").title(), min_value=0.0)

# Predict
if st.button("Predict:"):
    input_data = np.array([list(user_inputs.values())])  # 2D array for model
    scaled = scaler.transform(input_data)
    pred = model.predict(scaled)
    prediction_label = encoder.inverse_transform(pred)[0] if encoder else pred[0]

    # Show result
    if prediction_label == 'M':
        st.warning("You are diagonised with breast cancer :(")
    else:
        st.success("Results are fine")