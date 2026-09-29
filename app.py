
import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Diabetes CDSS",
    page_icon="🩺",
    layout="wide"
)


# ============================================================
# MODEL PATHS
# ============================================================

STAGE1_MODEL_PATH = "diabetes_screening_xgb_pipeline.pkl"
STAGE1_FEATURES_PATH = "screening_features.pkl"

STAGE2_MODEL_PATH = "models/stage2_random_forest.pkl"
STAGE2_FEATURES_PATH = "models/stage2_lab_features.pkl"


# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_models():

    stage1_model = joblib.load(STAGE1_MODEL_PATH)
    stage1_features = joblib.load(STAGE1_FEATURES_PATH)

    stage2_model = joblib.load(STAGE2_MODEL_PATH)
    stage2_features = joblib.load(STAGE2_FEATURES_PATH)

    return (
        stage1_model,
        stage1_features,
        stage2_model,
        stage2_features
    )


try:

    (
        stage1_model,
        stage1_features,
        stage2_model,
        stage2_features
    ) = load_models()

except Exception as e:

    st.error(f"Model loading error: {e}")
    st.stop()


# ============================================================
# HEADER
# ============================================================

st.title(
    "🩺 Explainable AI-Based Clinical Decision Support System"
)

st.markdown(
    """
    ### Diabetes Screening and Personalized Management

    Select the assessment pathway according to the information
    available.
    """
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("System Information")

    st.markdown(
        """
        **Non-Laboratory Assessment**

        XGBoost-based assessment using anthropometric,
        blood-pressure, family-history and lifestyle indicators.

        **Laboratory Assessment**

        Random Forest-based assessment using six laboratory
        biomarkers.

        **Explainability**

        SHAP-based feature contribution analysis.
        """
    )

    st.divider()

    st.caption(
        "Academic research prototype — not clinically validated."
    )


# ============================================================
# ASSESSMENT SELECTION
# ============================================================

assessment_mode = st.radio(
    "Choose the assessment pathway based on the information available:",
    [
        "🩺 Non-Laboratory-Based Assessment",
        "🧪 Laboratory-Based Assessment"
    ],
    horizontal=True
)


# ============================================================
# NON-LABORATORY ASSESSMENT
# ============================================================

if assessment_mode == "🩺 Non-Laboratory-Based Assessment":

    st.header("🩺 Non-Laboratory-Based Assessment")

    st.info(
        "Assessment using non-laboratory anthropometric, "
        "blood-pressure, family-history and lifestyle parameters."
    )

    # --------------------------------------------------------
    # NUMERICAL PARAMETERS
    # --------------------------------------------------------

    st.subheader("Anthropometric and Blood Pressure Parameters")

    col1, col2 = st.columns(2)

    with col1:

        bmi = st.number_input(
            "BMI",
            min_value=10.0,
            max_value=60.0,
            value=25.0,
            step=0.001
        )

        waist = st.number_input(
            "Waist Circumference (cm)",
            min_value=20.0,
            max_value=150.0,
            value=80.0,
            step=0.001
        )

    with col2:

        systolic_bp = st.number_input(
            "Systolic Blood Pressure (mmHg)",
            min_value=70.0,
            max_value=220.0,
            value=120.0,
            step=0.001
        )

        diastolic_bp = st.number_input(
            "Diastolic Blood Pressure (mmHg)",
            min_value=40.0,
            max_value=140.0,
            value=80.0,
            step=0.001
        )

    # --------------------------------------------------------
    # CATEGORICAL PARAMETERS
    # --------------------------------------------------------

    st.subheader("Family History and Lifestyle Parameters")

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        family_history = st.selectbox(
            "Family History of Diabetes",
            ["No", "Yes"]
        )

    with col2:

        hypertension = st.selectbox(
            "Hypertension",
            ["No", "Yes"]
        )

    with col3:

        physical_activity = st.selectbox(
            "Physical Activity",
            ["No", "Yes"]
        )

    with col4:

        smoking = st.selectbox(
            "Smoking",
            ["No", "Yes"]
        )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        alcohol_consumption = st.selectbox(
            "Alcohol Consumption",
            ["No", "Yes"]
        )

    with col2:

        obesity = st.selectbox(
            "Obesity",
            ["No", "Yes"]
        )

    with col3:

        pcos = st.selectbox(
            "PCOS",
            ["No", "Yes"]
        )

    with col4:

        gestational_diabetes = st.selectbox(
            "Gestational Diabetes",
            ["No", "Yes"]
        )

    st.divider()

    # --------------------------------------------------------
    # PREDICTION BUTTON
    # --------------------------------------------------------

    if st.button(
        "🔍 Perform Non-Laboratory Assessment",
        type="primary",
        use_container_width=True
    ):

        # IMPORTANT:
        # Pass Yes/No strings directly because the saved
        # pipeline contains the OneHotEncoder.

        input_data = {
            "BMI": bmi,
            "Waist_Circumference": waist,
            "Blood_Pressure_Systolic": systolic_bp,
            "Blood_Pressure_Diastolic": diastolic_bp,
            "Family_History_of_Diabetes": family_history,
            "Hypertension": hypertension,
            "Physical_Activity": physical_activity,
            "Smoking": smoking,
            "Alcohol_Consumption": alcohol_consumption,
            "Obesity": obesity,
            "PCOS": pcos,
            "Gestational_Diabetes": gestational_diabetes
        }

        input_df = pd.DataFrame([input_data])

        # Exact training order
        input_df = input_df[stage1_features]

        # ----------------------------------------------------
        # PREDICTION
        # ----------------------------------------------------

        try:

            probability = stage1_model.predict_proba(
                input_df
            )[0, 1]

        except Exception as e:

            st.error(
                f"Non-laboratory prediction error: {e}"
            )
            st.stop()

        probability_percent = probability * 100

        # ----------------------------------------------------
        # RISK CLASSIFICATION
        # ----------------------------------------------------

        if probability < 0.30:

            risk_category = "Low Risk"
            risk_message = "Lower Risk Signal"

        elif probability < 0.85:

            risk_category = "Moderate Risk"
            risk_message = "Moderate Risk Signal"

        else:

            risk_category = "High Risk"
            risk_message = "Positive Risk Signal"

        # ----------------------------------------------------
        # RESULT
        # ----------------------------------------------------

        st.subheader("Assessment Result")

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Estimated Probability",
                f"{probability_percent:.2f}%"
            )

        with col2:

            st.metric(
                "Risk Category",
                risk_category
            )

        if risk_category == "Low Risk":

            st.success(
                f"🟢 {risk_category} — {risk_message}"
            )

        elif risk_category == "Moderate Risk":

            st.warning(
                f"🟡 {risk_category} — {risk_message}"
            )

        else:

            st.error(
                f"🔴 {risk_category} — {risk_message}"
            )

        # ----------------------------------------------------
        # INPUT SUMMARY
        # ----------------------------------------------------

        st.subheader("Input Summary")

        st.dataframe(
            input_df,
            use_container_width=True
        )

        # ----------------------------------------------------
        # SHAP
        # ----------------------------------------------------

        st.subheader("Explainable AI Analysis")

        try:

            # The Stage-1 model is a preprocessing pipeline.
            # Use the transformed data and final estimator.

            if hasattr(stage1_model, "named_steps"):

                final_estimator = stage1_model.steps[-1][1]

                preprocessing_pipeline = stage1_model[
                    :-1
                ]

                transformed_input = (
                    preprocessing_pipeline.transform(input_df)
                )

                explainer = shap.TreeExplainer(
                    final_estimator,
                    feature_perturbation="tree_path_dependent"
                )

                shap_result = explainer(
                    transformed_input
                )

                shap_values = shap_result.values

                # Handle binary classification output
                if len(shap_values.shape) == 3:

                    shap_values = shap_values[0, :, 1]

                else:

                    shap_values = shap_values[0]

                # Get transformed feature names
                try:

                    transformed_names = (
                        preprocessing_pipeline
                        .get_feature_names_out()
                    )

                except Exception:

                    transformed_names = [
                        f"Feature {i + 1}"
                        for i in range(len(shap_values))
                    ]

            else:

                explainer = shap.TreeExplainer(
                    stage1_model,
                    feature_perturbation="tree_path_dependent"
                )

                shap_result = explainer(input_df)

                shap_values = shap_result.values[0]

                transformed_names = stage1_features


            shap_df = pd.DataFrame({
                "Feature": transformed_names,
                "SHAP Contribution": shap_values
            })

            shap_df["Absolute Contribution"] = (
                shap_df["SHAP Contribution"].abs()
            )

            shap_df = shap_df.sort_values(
                "Absolute Contribution",
                ascending=False
            )

            st.dataframe(
                shap_df[
                    [
                        "Feature",
                        "SHAP Contribution"
                    ]
                ],
                use_container_width=True
            )

            fig, ax = plt.subplots(figsize=(9, 5))

            plot_df = shap_df.sort_values(
                "SHAP Contribution"
            )

            ax.barh(
                plot_df["Feature"],
                plot_df["SHAP Contribution"]
            )

            ax.set_xlabel("SHAP Contribution")

            ax.set_title(
                "Feature Contributions to the Prediction"
            )

            plt.tight_layout()

            st.pyplot(fig)

            plt.close(fig)

        except Exception as e:

            st.warning(
                f"SHAP explanation could not be generated: {e}"
            )


# ============================================================
# LABORATORY ASSESSMENT
# ============================================================

else:

    st.header("🧪 Laboratory-Based Assessment")

    st.info(
        "Assessment using six laboratory biomarkers."
    )

    st.subheader("Laboratory Parameters")

    col1, col2 = st.columns(2)

    with col1:

        hba1c = st.number_input(
            "HbA1c (%)",
            min_value=0.0,
            max_value=20.0,
            value=5.5,
            step=0.001
        )

        fasting_glucose = st.number_input(
            "Fasting Blood Glucose (mg/dL)",
            min_value=40.0,
            max_value=400.0,
            value=100.0,
            step=0.001
        )

        postprandial_glucose = st.number_input(
            "Postprandial Blood Glucose (mg/dL)",
            min_value=40.0,
            max_value=500.0,
            value=140.0,
            step=0.001
        )

    with col2:

        insulin = st.number_input(
            "Insulin Levels",
            min_value=0.0,
            max_value=100.0,
            value=10.0,
            step=0.001
        )

        homa_ir = st.number_input(
            "HOMA-IR",
            min_value=0.0,
            max_value=20.0,
            value=1.5,
            step=0.001
        )

        c_peptide = st.number_input(
            "C-Peptide",
            min_value=0.0,
            max_value=15.0,
            value=2.0,
            step=0.001
        )

    st.caption(
        "Laboratory values are entered manually in this "
        "academic prototype."
    )

    st.divider()

    if st.button(
        "🧪 Perform Laboratory Assessment",
        type="primary",
        use_container_width=True
    ):

        # ----------------------------------------------------
        # CREATE INPUT
        # ----------------------------------------------------

        lab_data = {
            "HbA1c": hba1c,
            "Fasting_Blood_Glucose": fasting_glucose,
            "Postprandial_Blood_Glucose": postprandial_glucose,
            "Insulin_Levels": insulin,
            "HOMA_IR": homa_ir,
            "C_Peptide": c_peptide
        }

        lab_df = pd.DataFrame([lab_data])

        # Exact training order
        lab_df = lab_df[stage2_features]

        # ----------------------------------------------------
        # PREDICTION
        # ----------------------------------------------------

        try:

            probability = stage2_model.predict_proba(
                lab_df
            )[0, 1]

        except Exception as e:

            st.error(
                f"Laboratory prediction error: {e}"
            )
            st.stop()

        probability_percent = probability * 100

        # ----------------------------------------------------
        # RISK CATEGORY
        # ----------------------------------------------------

        if probability < 0.30:

            risk_category = "Lower Risk"
            risk_message = "Lower Risk Signal"

        elif probability < 0.70:

            risk_category = "Intermediate Risk"
            risk_message = "Intermediate Risk Signal"

        else:

            risk_category = "Higher Risk"
            risk_message = "Higher Risk Signal"

        # ----------------------------------------------------
        # RESULT
        # ----------------------------------------------------

        st.subheader("Laboratory Assessment Result")

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Estimated Probability",
                f"{probability_percent:.2f}%"
            )

        with col2:

            st.metric(
                "Risk Category",
                risk_category
            )

        if risk_category == "Lower Risk":

            st.success(
                f"🟢 {risk_category} — {risk_message}"
            )

        elif risk_category == "Intermediate Risk":

            st.warning(
                f"🟡 {risk_category} — {risk_message}"
            )

        else:

            st.error(
                f"🔴 {risk_category} — {risk_message}"
            )

        # ----------------------------------------------------
        # INPUT SUMMARY
        # ----------------------------------------------------

        st.subheader("Laboratory Input Summary")

        st.dataframe(
            lab_df,
            use_container_width=True
        )

        # ----------------------------------------------------
        # SHAP
        # ----------------------------------------------------

        st.subheader("Explainable AI Analysis")

        try:

            explainer = shap.TreeExplainer(
                stage2_model
            )

            shap_result = explainer(lab_df)

            shap_values = shap_result.values

            if len(shap_values.shape) == 3:

                shap_values = shap_values[0, :, 1]

            else:

                shap_values = shap_values[0]

            shap_df = pd.DataFrame({
                "Feature": stage2_features,
                "SHAP Contribution": shap_values
            })

            shap_df["Absolute Contribution"] = (
                shap_df["SHAP Contribution"].abs()
            )

            shap_df = shap_df.sort_values(
                "Absolute Contribution",
                ascending=False
            )

            st.dataframe(
                shap_df[
                    [
                        "Feature",
                        "SHAP Contribution"
                    ]
                ],
                use_container_width=True
            )

            fig, ax = plt.subplots(figsize=(9, 5))

            plot_df = shap_df.sort_values(
                "SHAP Contribution"
            )

            ax.barh(
                plot_df["Feature"],
                plot_df["SHAP Contribution"]
            )

            ax.set_xlabel("SHAP Contribution")

            ax.set_title(
                "Laboratory Feature Contributions"
            )

            plt.tight_layout()

            st.pyplot(fig)

            plt.close(fig)

        except Exception as e:

            st.warning(
                f"SHAP explanation could not be generated: {e}"
            )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Explainable AI-Based Clinical Decision Support System "
    "for Diabetes Screening and Personalized Management | "
    "Academic Research Prototype — Not Clinically Validated"
)
