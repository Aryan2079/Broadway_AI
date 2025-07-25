import streamlit as st
import joblib
from sklearn.datasets import load_iris

iris = load_iris()

print(iris.feature_names)

petal_length = st.number_input('enter petal length')
petal_width = st.number_input('enter petal width')
sepal_length = st.number_input('enter speial length')
sepal_width = st.number_input('enter sepal length')

flower_btn = st.button('predict flower')

loaded_model = joblib.load('../models/iris_classifier_knn_model.joblib')

sample = [[sepal_length,sepal_width,petal_length,petal_width]]
pred = loaded_model.predict(sample)[0]


if flower_btn:
    st.write(f'your flower is {iris.target_names[pred]}')



st.header('BMI calculator')

weight = st.number_input('enter your weight')
height = st.number_input('enter your height')


bmi_btn =st.button('show BMI')


if bmi_btn:
    bmi = weight/((height/3.28)**2)

    if bmi<16:
        st.error('Extremely Underweight')
    elif bmi>=16 and bmi<18.5:
        st.warning('Underweight')
    elif bmi>=18.5 and bmi<25:
        st.success('Healthy')
    elif bmi>=25 and bmi<30:
        st.info('Overweight')
    elif bmi>=30:
        st.error('Extremely Overweight')

