import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Diabetes CDSS",
    page_icon="🩺",
    layout="wide"
)

# --------------------------------------------------
# Load trained model
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "diabetes_model.pkl"

model = joblib.load(MODEL_PATH)

# --------------------------------------------------
# Header
# --------------------------------------------------

st.title("🩺 AI Clinical Decision Support System")

st.write(
    "Diabetes Risk Prediction Dashboard"
)

st.info(
    "This application is a research prototype and is not a substitute "
    "for professional medical diagnosis."
)

# --------------------------------------------------
# Patient Information
# --------------------------------------------------

st.header("👤 Patient Information")

patient_name = st.text_input(
    "Patient Name"
)

age = st.selectbox(
    "Age Group",
    options=list(range(1, 14)),
    index=7,
    help="CDC dataset age category: 1 = 18–24, 13 = 80+"
)

bmi = st.number_input(
    "BMI",
    min_value=10.0,
    max_value=100.0,
    value=25.0,
    step=0.1
)

# --------------------------------------------------
# Medical History
# --------------------------------------------------

st.header("🏥 Medical History")

col1, col2, col3 = st.columns(3)

with col1:
    high_bp = st.selectbox(
        "High Blood Pressure?",
        ["No", "Yes"]
    )

    high_chol = st.selectbox(
        "High Cholesterol?",
        ["No", "Yes"]
    )

    chol_check = st.selectbox(
        "Cholesterol Check in Last 5 Years?",
        ["No", "Yes"]
    )

with col2:
    smoker = st.selectbox(
        "Smoker?",
        ["No", "Yes"]
    )

    stroke = st.selectbox(
        "History of Stroke?",
        ["No", "Yes"]
    )

    heart_disease = st.selectbox(
        "Heart Disease / Heart Attack?",
        ["No", "Yes"]
    )

with col3:
    phys_activity = st.selectbox(
        "Physical Activity?",
        ["No", "Yes"]
    )

    fruits = st.selectbox(
        "Consumes Fruits Regularly?",
        ["No", "Yes"]
    )

    veggies = st.selectbox(
        "Consumes Vegetables Regularly?",
        ["No", "Yes"]
    )

# --------------------------------------------------
# Lifestyle and Healthcare
# --------------------------------------------------

st.header("🍎 Lifestyle & Healthcare")

col1, col2, col3 = st.columns(3)

with col1:
    alcohol = st.selectbox(
        "Heavy Alcohol Consumption?",
        ["No", "Yes"]
    )

with col2:
    healthcare = st.selectbox(
        "Has Healthcare Coverage?",
        ["No", "Yes"]
    )

with col3:
    no_doc_cost = st.selectbox(
        "Could Not See Doctor Due to Cost?",
        ["No", "Yes"]
    )

# --------------------------------------------------
# General Health
# --------------------------------------------------

st.header("🧠 General Health")

gen_health = st.slider(
    "General Health",
    min_value=1,
    max_value=5,
    value=3,
    help="1 = Excellent, 5 = Poor"
)

ment_health = st.slider(
    "Poor Mental Health Days (past 30 days)",
    min_value=0,
    max_value=30,
    value=0
)

phys_health = st.slider(
    "Poor Physical Health Days (past 30 days)",
    min_value=0,
    max_value=30,
    value=0
)

diff_walk = st.selectbox(
    "Difficulty Walking or Climbing Stairs?",
    ["No", "Yes"]
)

# --------------------------------------------------
# Demographics
# --------------------------------------------------

st.header("📋 Demographics")

sex = st.selectbox(
    "Sex",
    ["Female", "Male"]
)

education = st.slider(
    "Education Level",
    min_value=1,
    max_value=6,
    value=4,
    help="1 = Never attended school, 6 = College graduate"
)

income = st.slider(
    "Income Level",
    min_value=1,
    max_value=8,
    value=5,
    help="1 = Lowest income category, 8 = Highest income category"
)

# --------------------------------------------------
# Prediction
# --------------------------------------------------

st.divider()

if st.button(
    "🔍 Predict Diabetes Risk",
    type="primary",
    use_container_width=True
):

    # Convert Yes/No values into 0/1
    def binary(value):
        return 1 if value == "Yes" else 0

    # Create input dataframe
    input_data = pd.DataFrame([{
        "HighBP": binary(high_bp),
        "HighChol": binary(high_chol),
        "CholCheck": binary(chol_check),
        "BMI": bmi,
        "Smoker": binary(smoker),
        "Stroke": binary(stroke),
        "HeartDiseaseorAttack": binary(heart_disease),
        "PhysActivity": binary(phys_activity),
        "Fruits": binary(fruits),
        "Veggies": binary(veggies),
        "HvyAlcoholConsump": binary(alcohol),
        "AnyHealthcare": binary(healthcare),
        "NoDocbcCost": binary(no_doc_cost),
        "GenHlth": gen_health,
        "MentHlth": ment_health,
        "PhysHlth": phys_health,
        "DiffWalk": binary(diff_walk),
        "Sex": 1 if sex == "Male" else 0,
        "Age": age,
        "Education": education,
        "Income": income
    }])

    # Make prediction
    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)[0][1]

    st.header("📊 Prediction Result")

    if prediction == 1:

        st.error(
            "⚠️ Higher Diabetes Risk"
        )

        st.metric(
            "Estimated Risk",
            f"{probability * 100:.1f}%"
        )

        st.warning(
            "This result indicates an elevated predicted risk based on "
            "the information entered. Please consult a qualified "
            "healthcare professional for proper evaluation."
        )

    else:

        st.success(
            "✅ Lower Diabetes Risk"
        )

        st.metric(
            "Estimated Risk",
            f"{probability * 100:.1f}%"
        )

        st.info(
            "The model predicts a lower diabetes risk based on the "
            "information entered. This does not rule out diabetes."
        )