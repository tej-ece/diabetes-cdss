import os
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st


st.set_page_config(
    page_title="AI Diabetes Risk Screening",
    page_icon="🩺",
    layout="wide",
)

APP_DIR = Path(__file__).resolve().parent


@st.cache_resource
def load_models():
    """Load the XGBoost and Random Forest artifacts from the app directory."""
    xgb_model = screening_features = rf_model = rf_features = None

    xgb_path = APP_DIR / "diabetes_screening_xgb_pipeline.pkl"
    xgb_features_path = APP_DIR / "screening_features.pkl"
    rf_path = APP_DIR / "models" / "diabetes_model.pkl"
    rf_features_path = APP_DIR / "models" / "model_features.pkl"

    if xgb_path.exists() and xgb_features_path.exists():
        xgb_model = joblib.load(xgb_path)
        screening_features = list(joblib.load(xgb_features_path))

    if rf_path.exists() and rf_features_path.exists():
        rf_model = joblib.load(rf_path)
        rf_features = list(joblib.load(rf_features_path))

    return xgb_model, screening_features, rf_model, rf_features


xgb_model, screening_features, rf_model, rf_features = load_models()


def positive_probability(model, input_df):
    """Get predicted label and probability for the positive class (1 when available)."""
    classes = list(model.classes_)
    positive_label = 1 if 1 in classes else classes[-1]
    probability_index = classes.index(positive_label)
    prediction = model.predict(input_df)[0]
    probability = float(model.predict_proba(input_df)[0][probability_index])
    return prediction, probability, positive_label


def show_result(model, input_df, heading):
    _, probability, _ = positive_probability(model, input_df)

    # Prototype display bands matching the example screening UI. Validate these
    # separately for each model before clinical interpretation.
    if probability < 0.30:
        risk_level = "Low"
    elif probability < 0.85:
        risk_level = "Moderate"
    else:
        risk_level = "High"

    st.header("Prediction Result")
    result_col, probability_col = st.columns([1.35, 1])
    with result_col:
        if risk_level == "Low":
            st.success("🟢 Low model-estimated screening risk")
        elif risk_level == "Moderate":
            st.warning("🟡 Moderate model-estimated screening risk")
        else:
            st.error("🔴 High model-estimated screening risk")
    with probability_col:
        st.metric("Model-Estimated Probability", f"{probability * 100:.2f}%")

    st.header("Suggested Next Step")
    if risk_level in ["Moderate", "High"]:
        st.write(
            "Consider discussing this screening estimate with a qualified health professional. "
            "This result is not a diagnosis."
        )
    else:
        st.write(
            "This score is low under the prototype's display thresholds. It does not rule out "
            "diabetes; seek professional advice if you have concerns."
        )

    return probability

def class_shap_values(explainer, input_df, class_index):
    """Normalize common SHAP binary-classification output formats."""
    values = explainer.shap_values(input_df)
    expected = explainer.expected_value

    if isinstance(values, list):
        selected_values = np.asarray(values[class_index])[0]
    else:
        values = np.asarray(values)
        if values.ndim == 3:
            selected_values = values[0, :, class_index]
        elif values.ndim == 2:
            selected_values = values[0]
        else:
            raise ValueError("Unsupported SHAP output shape")

    if isinstance(expected, (list, tuple, np.ndarray)):
        base_value = np.asarray(expected).reshape(-1)[class_index]
    else:
        base_value = expected

    return np.asarray(selected_values), float(base_value)


def show_waterfall(values, base_value, row, feature_names):
    explanation = shap.Explanation(
        values=np.asarray(values),
        base_values=base_value,
        data=np.asarray(row).reshape(-1),
        feature_names=list(feature_names),
    )
    shap.plots.waterfall(explanation, show=False)
    fig = plt.gcf()
    st.pyplot(fig, clear_figure=True, use_container_width=True)
    plt.close(fig)


def binary_choice(label, key):
    value = st.selectbox(label, ["No", "Yes"], key=key)
    return int(value == "Yes")


# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------
st.title("🩺 AI-Based Diabetes Risk Screening")
st.subheader("Explainable Clinical Decision Support Prototype")
st.write(
    "Choose the screening pathway that matches the information you have. "
    "Each pathway uses a separately trained model."
)
st.info(
    "Research Prototype | XGBoost + Random Forest | Explainability: SHAP"
)

with st.sidebar:
    st.header("System Information")
    st.write("**Application:** Diabetes Risk Screening")
    st.write("**Models:** XGBoost + Random Forest")
    st.write("**Explainability:** SHAP")
    st.write("**Mode:** Screening")
    st.write("**Status:** Research Prototype")
    st.divider()
    st.caption(
        "XGBoost uses non-laboratory screening inputs. The Random Forest uses "
        "health-indicator features and does not currently use HbA1c or glucose."
    )

with st.expander("About this prototype"):
    st.write(
        "The model outputs are estimates based on the data used to train each model. "
        "They do not confirm or exclude diabetes and must not be used alone for treatment decisions."
    )

stage1, stage2 = st.tabs(
    ["📋 Non-lab screening · XGBoost", "🔎 Health indicators · Random Forest"]
)


# =========================================================
# STAGE 1: XGBOOST NON-LAB FORM
# =========================================================
with stage1:
    st.markdown("### Non-laboratory screening")
    st.caption("Use this pathway when you do not have laboratory results.")

    if xgb_model is None or screening_features is None:
        st.error(
            "XGBoost artifacts not found. Place `diabetes_screening_xgb_pipeline.pkl` "
            "and `screening_features.pkl` beside `app.py`."
        )
    else:
        with st.form("xgb_screening_form"):
            st.markdown("#### Patient information")
            c1, c2, c3 = st.columns(3)

            with c1:
                bmi = st.number_input("BMI", 10.0, 60.0, 25.0, 0.1, key="s1_bmi")
                waist = st.number_input(
                    "Waist circumference (cm)", 40.0, 180.0, 85.0, 1.0, key="s1_waist"
                )
                systolic = st.number_input(
                    "Systolic blood pressure (mmHg)", 70.0, 250.0, 120.0, 1.0, key="s1_sys"
                )
                diastolic = st.number_input(
                    "Diastolic blood pressure (mmHg)", 40.0, 150.0, 80.0, 1.0, key="s1_dia"
                )

            with c2:
                family_history = st.selectbox(
                    "Family history of diabetes", ["No", "Yes"], key="s1_fam"
                )
                hypertension = st.selectbox(
                    "Hypertension", ["No", "Yes"], key="s1_hyp"
                )
                physical_activity = st.selectbox(
                    "Physical activity", ["Low", "Moderate", "High"], key="s1_activity"
                )
                smoking = st.selectbox("Smoking", ["No", "Yes"], key="s1_smoking")

            with c3:
                alcohol = st.selectbox(
                    "Alcohol consumption", ["No", "Yes"], key="s1_alcohol"
                )
                obesity = st.selectbox("Obesity", ["No", "Yes"], key="s1_obesity")
                pcos = st.selectbox("PCOS", ["No", "Yes"], key="s1_pcos")
                gestational_diabetes = st.selectbox(
                    "Gestational diabetes", ["No", "Yes"], key="s1_gestational"
                )

            run_xgb = st.form_submit_button(
                "🔍 Predict screening risk", type="primary", use_container_width=True
            )

        if run_xgb:
            xgb_raw = pd.DataFrame([{
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
            }])

            missing = [name for name in screening_features if name not in xgb_raw.columns]
            if missing:
                st.error(
                    "The XGBoost feature list does not match the form. Missing fields: "
                    + ", ".join(missing)
                )
            else:
                xgb_input = xgb_raw[screening_features]
                try:
                    show_result(xgb_model, xgb_input, "Screening result")
                    st.markdown("#### Why did the model make this prediction?")
                    try:
                        preprocessor = xgb_model.named_steps["preprocessor"]
                        estimator = xgb_model.named_steps["model"]
                        transformed = preprocessor.transform(xgb_input)
                        if hasattr(transformed, "toarray"):
                            transformed = transformed.toarray()
                        names = preprocessor.get_feature_names_out()
                        transformed_df = pd.DataFrame(transformed, columns=names)
                        class_index = list(xgb_model.classes_).index(
                            1 if 1 in xgb_model.classes_ else xgb_model.classes_[-1]
                        )
                        explainer = shap.TreeExplainer(estimator)
                        shap_vals, base = class_shap_values(
                            explainer, transformed_df, class_index
                        )
                        show_waterfall(shap_vals, base, transformed_df.iloc[0], names)
                        st.caption(
                            "SHAP explains model behavior for this input; it does not identify medical causes."
                        )
                    except Exception:
                        st.caption("The prediction was generated, but its SHAP plot was unavailable.")
                except Exception as exc:
                    st.error(f"Could not run the XGBoost model: {exc}")


# =========================================================
# STAGE 2: RANDOM FOREST HEALTH-INDICATOR FORM
# =========================================================
with stage2:
    st.markdown("### Health-indicator screening")
    st.caption(
        "This Random Forest was trained on health and lifestyle indicators, not HbA1c or glucose values."
    )

    if rf_model is None or rf_features is None:
        st.error(
            "Random Forest artifacts not found. Place `diabetes_model.pkl` and "
            "`model_features.pkl` inside the `models` folder beside `app.py`."
        )
    else:
        with st.form("rf_health_form"):
            st.markdown("#### Health conditions")
            c1, c2, c3 = st.columns(3)
            with c1:
                high_bp = binary_choice("High blood pressure", "rf_highbp")
                high_chol = binary_choice("High cholesterol", "rf_highchol")
                chol_check = binary_choice("Cholesterol checked", "rf_cholcheck")
                smoker = binary_choice("Smoker", "rf_smoker")
                stroke = binary_choice("History of stroke", "rf_stroke")
                heart_disease = binary_choice(
                    "Heart disease or heart attack history", "rf_heartdisease"
                )
                diff_walk = binary_choice(
                    "Difficulty walking or climbing stairs", "rf_diffwalk"
                )
            with c2:
                phys_activity = binary_choice(
                    "Physical activity", "rf_physactivity"
                )
                fruits = binary_choice("Eats fruit regularly", "rf_fruits")
                veggies = binary_choice("Eats vegetables regularly", "rf_veggies")
                heavy_alcohol = binary_choice(
                    "Heavy alcohol consumption", "rf_alcohol"
                )
                healthcare = binary_choice(
                    "Has healthcare coverage", "rf_healthcare"
                )
                no_doc_cost = binary_choice(
                    "Could not see a doctor because of cost", "rf_nodoccost"
                )
            with c3:
                general_health_label = st.selectbox(
                    "General health",
                    ["Excellent", "Very good", "Good", "Fair", "Poor"],
                    key="rf_genhlth",
                )
                ment_hlth = st.number_input(
                    "Days of poor mental health (past 30 days)",
                    min_value=0, max_value=30, value=0, step=1, key="rf_menthlth"
                )
                phys_hlth = st.number_input(
                    "Days of poor physical health (past 30 days)",
                    min_value=0, max_value=30, value=0, step=1, key="rf_physhlth"
                )
                bmi_rf = st.number_input(
                    "BMI", min_value=12, max_value=98, value=25, step=1, key="rf_bmi"
                )
                sex_label = st.selectbox(
                    "Sex", ["Female", "Male"], key="rf_sex"
                )

            st.markdown("#### Demographic categories")
            d1, d2, d3 = st.columns(3)
            age_labels = [
                "18–24", "25–29", "30–34", "35–39", "40–44", "45–49", "50–54",
                "55–59", "60–64", "65–69", "70–74", "75–79", "80 or older",
            ]
            education_labels = [
                "Never attended / kindergarten only", "Grades 1–8",
                "Grades 9–11", "Grade 12 / GED", "College 1–3 years",
                "College 4+ years",
            ]
            income_labels = [
                "Under $10,000", "$10,000–$14,999", "$15,000–$19,999",
                "$20,000–$24,999", "$25,000–$34,999", "$35,000–$49,999",
                "$50,000–$74,999", "$75,000 or more",
            ]
            with d1:
                age_label = st.selectbox("Age group", age_labels, key="rf_age")
            with d2:
                education_label = st.selectbox(
                    "Education", education_labels, key="rf_education"
                )
            with d3:
                income_label = st.selectbox("Income", income_labels, key="rf_income")

            run_rf = st.form_submit_button(
                "🔍 Run health-indicator screening",
                type="primary",
                use_container_width=True,
            )

        if run_rf:
            # These category codes match the common BRFSS Diabetes Health Indicators layout.
            rf_values = {
                "HighBP": high_bp,
                "HighChol": high_chol,
                "CholCheck": chol_check,
                "BMI": bmi_rf,
                "Smoker": smoker,
                "Stroke": stroke,
                "HeartDiseaseorAttack": heart_disease,
                "PhysActivity": phys_activity,
                "Fruits": fruits,
                "Veggies": veggies,
                "HvyAlcoholConsump": heavy_alcohol,
                "AnyHealthcare": healthcare,
                "NoDocbcCost": no_doc_cost,
                "GenHlth": ["Excellent", "Very good", "Good", "Fair", "Poor"].index(
                    general_health_label
                ) + 1,
                "MentHlth": ment_hlth,
                "PhysHlth": phys_hlth,
                "DiffWalk": diff_walk,
                "Sex": 1 if sex_label == "Male" else 0,
                "Age": age_labels.index(age_label) + 1,
                "Education": education_labels.index(education_label) + 1,
                "Income": income_labels.index(income_label) + 1,
            }

            unknown = [name for name in rf_features if name not in rf_values]
            if unknown:
                st.error(
                    "The Random Forest expects features this form does not map: "
                    + ", ".join(unknown)
                )
            else:
                rf_input = pd.DataFrame(
                    [{name: rf_values[name] for name in rf_features}],
                    columns=rf_features,
                )
                try:
                    show_result(rf_model, rf_input, "Health-indicator result")
                    st.markdown("#### Why did the model make this prediction?")
                    try:
                        classes = list(rf_model.classes_)
                        positive_label = 1 if 1 in classes else classes[-1]
                        class_index = classes.index(positive_label)
                        explainer = shap.TreeExplainer(rf_model)
                        shap_vals, base = class_shap_values(
                            explainer, rf_input, class_index
                        )
                        show_waterfall(shap_vals, base, rf_input.iloc[0], rf_features)
                        st.caption(
                            "SHAP explains model behavior for this input; it does not identify medical causes."
                        )
                    except Exception:
                        st.caption("The prediction was generated, but its SHAP plot was unavailable.")
                except Exception as exc:
                    st.error(f"Could not run the Random Forest model: {exc}")


st.divider()
st.caption(
    "⚠️ Research prototype only. This app does not diagnose diabetes or recommend medication. "
    "Discuss health concerns and results with a qualified health professional."
)



