import streamlit as st
import pandas as pd
import joblib

# Load the trained model and the encoder
model = joblib.load("/home/aryan/Broadway_AI/models/random_forest_titanic_model.joblib")
encoder = joblib.load("/home/aryan/Broadway_AI/models/random_forest_titanic_encoder.joblib")

# Streamlit UI
st.title("Titanic Survivor Prediction")

st.markdown("""
Enter passenger details to predict if they would survive the Titanic disaster.
""")

# User input
pclass = st.selectbox("Passenger Class (Pclass)", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.number_input("Age", min_value=0.0, max_value=100.0, value=25.0)
sibsp = st.number_input("Number of Siblings/Spouses Aboard (SibSp)", min_value=0, max_value=10, value=0)
parch = st.number_input("Number of Parents/Children Aboard (Parch)", min_value=0, max_value=10, value=0)
fare = st.number_input("Fare", min_value=0.0, max_value=600.0, value=32.0)

# Prepare the data for prediction
input_data = pd.DataFrame({
    "Pclass": [pclass],
    "Sex": [sex],
    "Age": [age],
    "SibSp": [sibsp],
    "Parch": [parch],
    "Fare": [fare]
})

# Encode sex
input_data["Sex"] = encoder.transform(input_data["Sex"])

# Prediction
if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    if prediction == 1:
        st.success("This passenger would survive! 🛟")
    else:
        st.error("This passenger would not survive. ⚓️")
