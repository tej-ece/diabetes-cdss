"""
Diabetes Clinical Decision Support System (CDSS)
--------------------------------------------------
Streamlit front-end for the SVM pipeline (StandardScaler + RBF SVC)
trained in cdss.ipynb. Reads:
    - diabetes_best_model.pkl        (required)
    - shap_feature_importance.csv    (optional, for the "key factors" panel)

Run locally:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# --------------------------------------------------------------------------
# Page config
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Diabetes CDSS",
    page_icon="\U0001FA7A",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "diabetes_best_model.pkl"
SHAP_PATH = BASE_DIR / "shap_feature_importance.csv"

# --------------------------------------------------------------------------
# Feature schema (matches X_train.columns from the notebook, 34 features)
# --------------------------------------------------------------------------
NUMERIC_FEATURES = {
    # name: (min, max, default, step, help)
    "Fasting_Blood_Glucose":      (50.0, 260.0, 100.0, 1.0, "mg/dL"),
    "Postprandial_Blood_Glucose": (70.0, 320.0, 140.0, 1.0, "mg/dL, 2-hr post-meal"),
    "HbA1c":                      (3.5, 15.0, 5.5, 0.1, "%"),
    "Random_Blood_Glucose":       (60.0, 350.0, 120.0, 1.0, "mg/dL"),
    "BMI":                        (10.0, 55.0, 24.0, 0.1, "kg/m\u00b2"),
    "Waist_Circumference":        (15.0, 65.0, 34.0, 0.5, "inches"),
    "Triglyceride_Levels":        (40.0, 320.0, 150.0, 1.0, "mg/dL"),
    "Blood_Pressure_Systolic":    (70.0, 200.0, 120.0, 1.0, "mmHg"),
    "Blood_Pressure_Diastolic":   (40.0, 130.0, 80.0, 1.0, "mmHg"),
    "LDL_Cholesterol":            (40.0, 220.0, 100.0, 1.0, "mg/dL"),
    "HDL_Cholesterol":            (15.0, 100.0, 50.0, 1.0, "mg/dL"),
    "CRP_Levels":                 (0.0, 12.0, 2.0, 0.1, "mg/L"),
    "Insulin_Levels":             (1.0, 50.0, 12.0, 0.5, "\u00b5IU/mL"),
    "HOMA_IR":                    (0.3, 6.0, 1.5, 0.1, "insulin resistance index"),
    "OGTT":                       (60.0, 320.0, 140.0, 1.0, "mg/dL, 2-hr OGTT"),
    "Creatinine_Levels":          (0.4, 2.5, 0.9, 0.01, "mg/dL"),
    "eGFR":                       (25.0, 130.0, 90.0, 1.0, "mL/min/1.73m\u00b2"),
    "Microalbuminuria":           (0.0, 300.0, 20.0, 1.0, "mg/L"),
    "Uric_Acid_Levels":           (1.5, 12.0, 5.0, 0.1, "mg/dL"),
    "Fructosamine_Levels":        (100.0, 400.0, 220.0, 1.0, "\u00b5mol/L"),
    "ALT":                        (5.0, 100.0, 25.0, 1.0, "U/L"),
    "AST":                        (5.0, 100.0, 22.0, 1.0, "U/L"),
    "C_Peptide":                  (0.1, 6.0, 1.5, 0.05, "ng/mL"),
    "Proinsulin_Levels":          (1.0, 20.0, 6.0, 0.5, "pmol/L"),
}

BINARY_FEATURES = [
    "Family_History_of_Diabetes",
    "Gestational_Diabetes",
    "PCOS",
    "Hypertension",
    "Physical_Activity",
    "Smoking",
    "Alcohol_Consumption",
    "Obesity",
    "Diet",
    "Sleep_Apnea",
]

# Final column order MUST match X_train.columns from the notebook exactly.
FEATURE_ORDER = [
    "Fasting_Blood_Glucose", "Postprandial_Blood_Glucose", "HbA1c",
    "Random_Blood_Glucose", "BMI", "Waist_Circumference", "Triglyceride_Levels",
    "Blood_Pressure_Systolic", "Blood_Pressure_Diastolic", "LDL_Cholesterol",
    "HDL_Cholesterol", "CRP_Levels", "Insulin_Levels", "HOMA_IR", "OGTT",
    "Creatinine_Levels", "eGFR", "Microalbuminuria", "Uric_Acid_Levels",
    "Fructosamine_Levels", "ALT", "AST", "C_Peptide", "Proinsulin_Levels",
    "Family_History_of_Diabetes", "Gestational_Diabetes", "PCOS", "Hypertension",
    "Physical_Activity", "Smoking", "Alcohol_Consumption", "Obesity", "Diet",
    "Sleep_Apnea",
]

# Human-readable labels for lifestyle/clinical checkboxes
BINARY_LABELS = {
    "Family_History_of_Diabetes": "Family history of diabetes",
    "Gestational_Diabetes": "History of gestational diabetes",
    "PCOS": "PCOS",
    "Hypertension": "Hypertension",
    "Physical_Activity": "Regular physical activity",
    "Smoking": "Smoker",
    "Alcohol_Consumption": "Regular alcohol consumption",
    "Obesity": "Clinically obese",
    "Diet": "Follows a controlled/diabetic diet",
    "Sleep_Apnea": "Sleep apnea",
}

# --------------------------------------------------------------------------
# Cached loaders
# --------------------------------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_shap_importance():
    if SHAP_PATH.exists():
        df = pd.read_csv(SHAP_PATH)
        return df
    return None


# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.title("\U0001FA7A Diabetes Clinical Decision Support System")
st.caption("AI-based screening support using clinical biomarkers and lifestyle factors")
st.info(
    "This tool is a research prototype for academic demonstration. "
    "It does not replace professional medical diagnosis."
)

try:
    model = load_model()
except FileNotFoundError:
    st.error(
        f"Model file not found at `{MODEL_PATH.name}`. "
        "Place `diabetes_best_model.pkl` in the same folder as this app."
    )
    st.stop()

shap_df = load_shap_importance()

# --------------------------------------------------------------------------
# Patient information (not used by the model, for the report header only)
# --------------------------------------------------------------------------
st.header("\U0001F464 Patient Information")
c1, c2, c3 = st.columns(3)
with c1:
    patient_name = st.text_input("Patient name / ID")
with c2:
    patient_age = st.number_input("Age", min_value=1, max_value=120, value=45)
with c3:
    patient_sex = st.selectbox("Sex", ["Female", "Male"])

st.divider()

# --------------------------------------------------------------------------
# Clinical / lab parameters, grouped
# --------------------------------------------------------------------------
GROUPS = {
    "\U0001FA78 Glucose Panel": [
        "Fasting_Blood_Glucose", "Postprandial_Blood_Glucose",
        "HbA1c", "Random_Blood_Glucose", "OGTT", "Fructosamine_Levels",
    ],
    "\u2696\ufe0f Body Measurements": [
        "BMI", "Waist_Circumference",
    ],
    "\U0001F493 Cardiovascular / Lipid Panel": [
        "Triglyceride_Levels", "Blood_Pressure_Systolic",
        "Blood_Pressure_Diastolic", "LDL_Cholesterol", "HDL_Cholesterol",
    ],
    "\U0001F9EA Insulin / Metabolic Markers": [
        "Insulin_Levels", "HOMA_IR", "C_Peptide", "Proinsulin_Levels", "CRP_Levels",
    ],
    "\U0001F9AA Renal / Liver Function": [
        "Creatinine_Levels", "eGFR", "Microalbuminuria",
        "Uric_Acid_Levels", "ALT", "AST",
    ],
}

st.header("\U0001F52C Clinical & Laboratory Parameters")
numeric_inputs = {}
for group_name, feats in GROUPS.items():
    with st.expander(group_name, expanded=True):
        cols = st.columns(3)
        for i, feat in enumerate(feats):
            lo, hi, default, step, help_txt = NUMERIC_FEATURES[feat]
            numeric_inputs[feat] = cols[i % 3].number_input(
                feat.replace("_", " "),
                min_value=lo, max_value=hi, value=default, step=step,
                help=help_txt, key=f"num_{feat}",
            )

st.header("\U0001FA7A Clinical & Lifestyle Factors")
binary_inputs = {}
cols = st.columns(2)
for i, feat in enumerate(BINARY_FEATURES):
    target_col = cols[i % 2]
    choice = target_col.selectbox(BINARY_LABELS[feat], ["No", "Yes"], key=f"bin_{feat}")
    binary_inputs[feat] = 1 if choice == "Yes" else 0

st.divider()

# --------------------------------------------------------------------------
# Prediction
# --------------------------------------------------------------------------
if st.button("\U0001F50D Assess Diabetes Risk", type="primary", use_container_width=True):
    patient_data = {**numeric_inputs, **binary_inputs}
    patient_df = pd.DataFrame([patient_data], columns=FEATURE_ORDER)

    prediction = model.predict(patient_df)[0]
    probability = model.predict_proba(patient_df)[0]
    risk_prob = probability[1] * 100  # P(Positive)

    st.header("\U0001F4CA AI Risk Assessment")

    result_col, gauge_col = st.columns([1, 1])

    with result_col:
        if prediction == 1:
            st.error("**Prediction: POSITIVE (elevated diabetes risk)**")
        else:
            st.success("**Prediction: NEGATIVE (low diabetes risk)**")
        st.metric("Predicted risk probability", f"{risk_prob:.1f}%")
        if patient_name:
            st.caption(f"Patient: {patient_name} \u00b7 Age {patient_age} \u00b7 {patient_sex}")

    with gauge_col:
        st.progress(min(int(risk_prob), 100))
        st.caption("Risk probability (0-100%)")

    # ---------------- Key contributing factors (from saved SHAP CSV) -----
    st.subheader("\U0001F9E9 Key Contributing Factors (model-level)")
    if shap_df is not None:
        top_feats = shap_df.sort_values("Mean_Absolute_SHAP", ascending=False).head(8).copy()
        top_feats["Feature"] = top_feats["Feature"].str.replace("_", " ", regex=False)
        top_feats = top_feats.sort_values("Mean_Absolute_SHAP", ascending=True)

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.barh(top_feats["Feature"], top_feats["Mean_Absolute_SHAP"])
        ax.set_xlabel("Mean |SHAP value|")
        fig.tight_layout()
        st.pyplot(fig)

        st.caption(
            "These are the features the model relies on most heavily overall "
            "(from SHAP analysis on the test set), not necessarily this patient specifically."
        )
    else:
        st.caption(
            "Upload `shap_feature_importance.csv` alongside this app to show "
            "global feature importance here."
        )

    # ---------------------------- Recommendations -------------------------
    st.subheader("\U0001FA7A CDSS Recommendations")
    recs = []
    if numeric_inputs["HbA1c"] >= 6.5:
        recs.append("HbA1c is in the diabetic range (\u2265 6.5%) \u2014 recommend clinical confirmation and follow-up.")
    elif numeric_inputs["HbA1c"] >= 5.7:
        recs.append("HbA1c is in the prediabetic range (5.7\u20136.4%) \u2014 recommend lifestyle counseling and re-testing.")
    if numeric_inputs["Fasting_Blood_Glucose"] >= 126:
        recs.append("Fasting glucose is elevated (\u2265 126 mg/dL) \u2014 recommend repeat fasting glucose test.")
    if numeric_inputs["BMI"] >= 30:
        recs.append("BMI indicates obesity \u2014 recommend weight management and dietary counseling.")
    if numeric_inputs["Blood_Pressure_Systolic"] >= 140 or numeric_inputs["Blood_Pressure_Diastolic"] >= 90:
        recs.append("Blood pressure is elevated \u2014 recommend cardiovascular risk evaluation.")
    if numeric_inputs["HOMA_IR"] >= 2.5:
        recs.append("HOMA-IR suggests insulin resistance \u2014 recommend further metabolic work-up.")
    if binary_inputs["Physical_Activity"] == 0:
        recs.append("Patient reports low physical activity \u2014 recommend structured exercise plan.")
    if binary_inputs["Smoking"] == 1:
        recs.append("Patient is a smoker \u2014 recommend smoking cessation counseling.")

    if not recs:
        recs.append("No major red-flag markers detected. Recommend routine annual screening.")

    for r in recs:
        st.write(f"- {r}")

    st.caption(
        "This assessment is intended for decision support and screening assistance only "
        "and does not replace professional medical diagnosis."
    )