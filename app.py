import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('insurance_model.pkl')

# Function to predict charges
def predict_charges(age, sex, bmi, children, smoker, region):
    input_data = pd.DataFrame({
        'age': [age],
        'sex': [sex],
        'bmi': [bmi],
        'children': [children],
        'smoker': [smoker],
        'region': [region]
    })
    prediction = model.predict(input_data)
    return prediction[0]

# Streamlit app
st.title("Insurance Charges Prediction")

age = st.number_input("Age", min_value=0, max_value=100, value=25)
sex = st.selectbox("Sex", ["male", "female"])
bmi = st.number_input("BMI", min_value=0.0, max_value=60.0, value=25.0)
children = st.number_input("Children", min_value=0, max_value=10, value=0)
smoker = st.selectbox("Smoker", ["yes", "no"])
region = st.selectbox("Region", ["southwest", "southeast", "northwest", "northeast"])

if st.button("Predict Charges"):
    result = predict_charges(age, sex, bmi, children, smoker, region)
    st.write(f"The predicted insurance charge is ${result:.2f}")

if __name__ == "__main__":
    st.run()