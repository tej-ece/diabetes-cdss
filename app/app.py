import streamlit as st

st.set_page_config(
    page_title="Diabetes Clinical Decision Support System",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 AI Clinical Decision Support System")

st.write("Welcome to the Diabetes Risk Prediction Dashboard.")

st.header("Patient Information")

name = st.text_input("Patient Name")

age = st.number_input("Age", min_value=1, max_value=120, value=25)

glucose = st.number_input("Glucose (mg/dL)", value=100)

bmi = st.number_input("BMI", value=22.0)

if st.button("Predict"):
    st.success("Prediction module will be added soon!")