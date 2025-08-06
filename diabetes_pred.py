import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.title("Diabetes Prediction")
st.write("Enter health metrics to predict diabetes risk.")


try:
    model = joblib.load("logistic_regression_model.joblib")
    scaler = joblib.load("diabetes_scaler.joblib")
except Exception as e:
    st.error(f"Error loading model or scaler: {e}")
    st.stop()


@st.cache_data
def load_data():
    return pd.read_csv("./diabetes_dataset.csv")

try:
    df = load_data()
    st.subheader("Dataset Preview")
    st.dataframe(df.head())
except Exception as e:
    st.warning(f"Could not load dataset preview: {e}")


st.sidebar.header("Patient Data")
fields = {
    "Age": (0, 100, 30),
    "Pregnancy": (0, 20, 2),
    "BMI": (0.0, 50.0, 25.0),
    "Glucose": (0.0, 200.0, 100.0),
    "BloodPressure": (0, 130, 70),
    "HbA1c": (0, 15, 5),
    "LDL": (0.0, 200.0, 100.0),
    "HDL": (0.0, 100.0, 50.0),
    "Triglycerides": (0.0, 300.0, 120.0),
    "WaistCircumference": (0.0, 150.0, 90.0),
    "HipCircumference": (0.0, 150.0, 100.0),
    "WHR": (0.0, 2.0, 0.9)
}

values = []
for k, (mn, mx, dfv) in fields.items():
    if isinstance(mn, float):
        values.append(st.sidebar.number_input(k, min_value=mn, max_value=mx, value=dfv))
    else:
        values.append(st.sidebar.number_input(k, min_value=mn, max_value=mx, value=dfv, step=1))

features = np.array([values])
scaled = scaler.transform(features)

if st.sidebar.button("Predict Diabetes"):
    pred = model.predict(scaled)[0]
    prob = model.predict_proba(scaled)[0][1]
    st.subheader("Prediction Result")
    st.write("**Diabetic**" if pred == 1 else "**Non‑Diabetic**")
    st.write(f"Probability: {prob*100:.2f}%")