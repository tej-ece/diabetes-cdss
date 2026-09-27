import os
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(
    page_title="AI-Based Two-Stage Diabetes Risk CDSS",
    page_icon="🩺",
    layout="wide",
)

# ---------------------------------------------------------
# LOAD TRAINED MODELS & ARTIFACTS
# ---------------------------------------------------------
@st.cache_resource
def load_all_models():
    # Load Stage 1: Non-Lab Screening Model
    xgb_model, screening_features = None, None
    if os.path.exists("diabetes_screening_xgb_pipeline.pkl") and os.path.exists(
        "screening_features.pkl"
    ):
        xgb_model = joblib.load("diabetes_screening_xgb_pipeline.pkl")
        screening_features = joblib.load("screening_features.pkl")

    # Load Stage 2: Diagnostic Model
    rf_model, lab_features = None, None
    rf_model_path = os.path.join("models", "diabetes_model.pkl")
    rf_feat_path = os.path.join("models", "model_features.pkl")

    if os.path.exists(rf_model_path) and os.path.exists(rf_feat_path):
        rf_model = joblib.load(rf_model_path)
        lab_features = joblib.load(rf_feat_path)

    return xgb_model, screening_features, rf_model, lab_features


xgb_model, screening_features, rf_model, lab_features = load_all_models()

# ---------------------------------------------------------
# HEADER & NAVIGATION TABS
# ---------------------------------------------------------
st.title("🩺 Two-Stage Explainable AI Diabetes Risk CDSS")

tab1, tab2 = st.tabs(
    [
        "📋 STAGE 1: Non-Lab Community Screening",
        "🔬 STAGE 2: Laboratory Clinical Diagnostics",
    ]
)


# =========================================================
# TAB 1: STAGE 1 - NON-LAB COMMUNITY SCREENING
# =========================================================
with tab1:
    st.markdown("### Explainable Clinical Decision Support Prototype")
    st.write(
        "This research prototype uses machine learning to estimate diabetes screening risk "
        "from non-laboratory health and lifestyle indicators."
    )
    st.info("Research Prototype | Model: XGBoost | Explainability: SHAP")

    if xgb_model is None or screening_features is None:
        st.error(
            "⚠️ **Missing Artifacts:** Stage 1 model files (`diabetes_screening_xgb_pipeline.pkl`) "
            "were not found in the project root folder."
        )
    else:
        st.header("Patient Information")

        col1, col2, col3 = st.columns(3)

        with col1:
            bmi = st.number_input(
                "BMI",
                min_value=10.0,
                max_value=60.0,
                value=25.0,
                step=0.1,
                key="s1_bmi",
            )
            waist = st.number_input(
                "Waist Circumference (cm)",
                min_value=40.0,
                max_value=180.0,
                value=85.0,
                step=1.0,
                key="s1_waist",
            )
            systolic = st.number_input(
                "Systolic Blood Pressure (mmHg)",
                min_value=70.0,
                max_value=250.0,
                value=120.0,
                step=1.0,
                key="s1_sys",
            )
            diastolic = st.number_input(
                "Diastolic Blood Pressure (mmHg)",
                min_value=40.0,
                max_value=150.0,
                value=80.0,
                step=1.0,
                key="s1_dia",
            )

        with col2:
            family_history = st.selectbox(
                "Family History of Diabetes", ["No", "Yes"], key="s1_fam"
            )
            hypertension = st.selectbox(
                "Hypertension", ["No", "Yes"], key="s1_hyp"
            )
            physical_activity = st.selectbox(
                "Physical Activity", ["Low", "Moderate", "High"], key="s1_act"
            )
            smoking = st.selectbox("Smoking", ["No", "Yes"], key="s1_smoke")

        with col3:
            alcohol = st.selectbox(
                "Alcohol Consumption", ["No", "Yes"], key="s1_alc"
            )
            obesity = st.selectbox("Obesity", ["No", "Yes"], key="s1_obs")
            pcos = st.selectbox("PCOS", ["No", "Yes"], key="s1_pcos")
            gestational_diabetes = st.selectbox(
                "Gestational Diabetes", ["No", "Yes"], key="s1_gest"
            )

        st.divider()

        predict_button = st.button(
            "🔍 Predict Diabetes Risk",
            type="primary",
            use_container_width=True,
            key="s1_btn",
        )

        if predict_button:
            input_data = pd.DataFrame(
                [
                    {
                        "BMI": bmi,
                        "Waist_Circumference": waist,
                        "Blood_Pressure_Systolic": systolic,
                        "Blood_Pressure_Diastolic": diastolic,
                        "Family_History_of_Diabetes": family_history,
                        "Hypertension": hypertension,
                        "Physical_Activity": physical_activity,
                        "Smoking": smoking,
                        "Alcohol_Consumption": alcohol,
                        "Obesity": obesity,
                        "PCOS": pcos,
                        "Gestational_Diabetes": gestational_diabetes,
                    }
                ]
            )

            # Ensure exact feature order
            input_data = input_data[screening_features]

            # Prediction
            prediction = xgb_model.predict(input_data)[0]
            probability = xgb_model.predict_proba(input_data)[0][1]

            # Result Section
            st.header("Prediction Result")
            result_col1, result_col2 = st.columns(2)

            with result_col1:
                if prediction == 1:
                    st.error("⚠️ Screening Result: Positive")
                else:
                    st.success("✅ Screening Result: Negative")

            with result_col2:
                st.metric(
                    "Model-Estimated Probability", f"{probability * 100:.2f}%"
                )

            # Risk Stratification
            if probability < 0.30:
                risk_level = "Low"
            elif probability < 0.85:
                risk_level = "Moderate"
            else:
                risk_level = "High"

            st.subheader("Risk Stratification")

            if risk_level == "Low":
                st.success("🟢 Low model-estimated screening risk")
            elif risk_level == "Moderate":
                st.warning("🟡 Moderate model-estimated screening risk")
            else:
                st.error("🔴 High model-estimated screening risk")

            # Suggested Next Step
            st.subheader("Suggested Next Step")
            if prediction == 1 or risk_level in ["Moderate", "High"]:
                st.write(
                    "The screening model indicates an elevated likelihood of diabetes. "
                    "**Further clinical evaluation and Stage 2 laboratory testing (HbA1c & Fasting Glucose)** "
                    "should be conducted."
                )
            else:
                st.write(
                    "The screening model does not indicate elevated diabetes risk based on the entered "
                    "screening features. Routine health monitoring and healthy lifestyle practices are recommended."
                )

            # SHAP Explanation
            st.divider()
            st.header("🔎 Why did the model make this prediction?")
            st.write(
                "SHAP (SHapley Additive exPlanations) is used to examine how individual input features "
                "contributed to the model's prediction."
            )

            try:
                preprocessor = xgb_model.named_steps["preprocessor"]
                estimator = xgb_model.named_steps["model"]

                transformed_input = preprocessor.transform(input_data)
                if hasattr(transformed_input, "toarray"):
                    transformed_input = transformed_input.toarray()

                feature_names = preprocessor.get_feature_names_out()
                input_shap = pd.DataFrame(
                    transformed_input, columns=feature_names
                )

                explainer = shap.TreeExplainer(estimator)
                shap_values = explainer.shap_values(input_shap)

                if isinstance(shap_values, list):
                    shap_values_plot = shap_values[1]
                else:
                    shap_values_plot = shap_values
                    if len(np.shape(shap_values_plot)) == 3:
                        shap_values_plot = shap_values_plot[:, :, 1]

                fig, ax = plt.subplots(figsize=(10, 5))
                shap.summary_plot(
                    shap_values_plot,
                    input_shap,
                    plot_type="bar",
                    show=False,
                )
                plt.tight_layout()
                st.pyplot(fig)

                st.caption(
                    "Higher absolute SHAP values indicate a stronger contribution to the model prediction. "
                    "SHAP values describe model behavior and should not be interpreted as causal medical effects."
                )
            except Exception as e:
                st.warning(
                    "SHAP explanation could not be generated for this prediction."
                )


# =========================================================
# TAB 2: STAGE 2 - CLINICAL DIAGNOSTIC CDSS (LAB)
# =========================================================
with tab2:
    st.markdown("### 🔬 Stage 2: Diagnostic Clinical Decision Support System")
    st.write(
        "Incorporate laboratory biomarkers (HbA1c, Fasting Glucose, Postprandial Glucose) "
        "for high-precision clinical evaluation (**Random Forest - 97.90% Recall**)."
    )

    if rf_model is None or lab_features is None:
        st.error(
            "⚠️ **Missing Artifacts:** Diagnostic model files (`models/diabetes_model.pkl`) "
            "were not detected."
        )
    else:
        st.sidebar.markdown("### 🔬 Stage 2 Lab Parameters")
        hba1c = st.sidebar.number_input(
            "HbA1c Level (%)", 4.0, 15.0, 6.5, 0.1, key="s2_hba1c"
        )
        fbg = st.sidebar.number_input(
            "Fasting Blood Glucose (mg/dL)",
            50.0,
            300.0,
            110.0,
            key="s2_fbg",
        )
        pbg = st.sidebar.number_input(
            "Postprandial Blood Glucose (mg/dL)",
            70.0,
            400.0,
            140.0,
            key="s2_pbg",
        )
        bmi_lab = st.sidebar.number_input(
            "BMI (kg/m²)", 10.0, 60.0, 26.5, 0.5, key="s2_bmi"
        )
        sys_bp_lab = st.sidebar.number_input(
            "Systolic BP (mmHg)", 80, 220, 120, key="s2_sys"
        )

        fam_hist_lab = st.sidebar.selectbox(
            "Family History of Diabetes",
            ["No", "Yes"],
            index=1,
            key="s2_fam",
        )
        hypertension_lab = st.sidebar.selectbox(
            "Hypertension", ["No", "Yes"], index=0, key="s2_hyp"
        )
        physical_act_lab = st.sidebar.selectbox(
            "Physical Activity", ["Yes", "No"], index=0, key="s2_act"
        )

        input_dict = {feat: 0 for feat in lab_features}
        if "HbA1c" in input_dict:
            input_dict["HbA1c"] = hba1c
        if "Fasting_Blood_Glucose" in input_dict:
            input_dict["Fasting_Blood_Glucose"] = fbg
        if "Postprandial_Blood_Glucose" in input_dict:
            input_dict["Postprandial_Blood_Glucose"] = pbg
        if "BMI" in input_dict:
            input_dict["BMI"] = bmi_lab
        if "Blood_Pressure_Systolic" in input_dict:
            input_dict["Blood_Pressure_Systolic"] = sys_bp_lab
        if "Family_History_of_Diabetes_Yes" in input_dict:
            input_dict["Family_History_of_Diabetes_Yes"] = (
                1 if fam_hist_lab == "Yes" else 0
            )
        if "Hypertension_Yes" in input_dict:
            input_dict["Hypertension_Yes"] = (
                1 if hypertension_lab == "Yes" else 0
            )
        if "Physical_Activity_Yes" in input_dict:
            input_dict["Physical_Activity_Yes"] = (
                1 if physical_act_lab == "Yes" else 0
            )

        input_df = pd.DataFrame([input_dict])

        st.subheader("Active Patient Biomarkers")
        p_col1, p_col2, p_col3, p_col4 = st.columns(4)
        p_col1.metric("HbA1c", f"{hba1c}%")
        p_col2.metric("Fasting Glucose", f"{fbg} mg/dL")
        p_col3.metric("Postprandial Glucose", f"{pbg} mg/dL")
        p_col4.metric("BMI", f"{bmi_lab} kg/m²")

        st.divider()

        if st.button(
            "🔍 Run Stage 2 Diagnostic Evaluation",
            type="primary",
            use_container_width=True,
            key="s2_btn",
        ):
            probability_lab = rf_model.predict_proba(input_df)[0][1]
            prediction_lab = rf_model.predict(input_df)[0]

            st.header("Diagnostic Verdict")
            d_col1, d_col2, d_col3 = st.columns(3)

            with d_col1:
                if prediction_lab == 1 or probability_lab > 0.45:
                    st.error("🚨 Diagnostic Result: HIGH RISK / POSITIVE")
                else:
                    st.success("✅ Diagnostic Result: LOW RISK / NORMAL")

            with d_col2:
                st.metric(
                    "Diabetic Probability", f"{probability_lab * 100:.2f}%"
                )

            with d_col3:
                st.metric("Model Recall (Clinical Safety)", "97.90%")

            # SHAP Waterfall Plot for Tab 2
            st.divider()
            st.header("🔎 Patient-Specific Diagnostic Breakdown")

            explainer_lab = shap.TreeExplainer(rf_model)
            shap_vals_lab = explainer_lab.shap_values(input_df)

            if isinstance(shap_vals_lab, list):
                vals = shap_vals_lab[1][0]
                base_val = explainer_lab.expected_value[1]
            else:
                vals = (
                    shap_vals_lab[0, :, 1]
                    if len(shap_vals_lab.shape) == 3
                    else shap_vals_lab[0]
                )
                base_val = (
                    explainer_lab.expected_value[1]
                    if isinstance(
                        explainer_lab.expected_value, (list, np.ndarray)
                    )
                    else explainer_lab.expected_value
                )

            fig_waterfall, ax = plt.subplots(figsize=(8, 4))
            shap.plots.waterfall(
                shap.Explanation(
                    values=vals,
                    base_values=base_val,
                    data=input_df.iloc[0],
                    feature_names=lab_features,
                ),
                show=False,
            )
            st.pyplot(fig_waterfall)

# ---------------------------------------------------------
# GLOBAL DISCLAIMER
# ---------------------------------------------------------
st.divider()
st.caption(
    "⚠️ This application is a research prototype and is not a substitute for professional medical diagnosis. "
    "Model outputs should not be used as the sole basis for medical decisions."
)