import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# STAGE 2: LABORATORY PARAMETER MODEL
# Model: Random Forest
# ============================================================


# ------------------------------------------------------------
# 1. LOAD DATASET
# ------------------------------------------------------------

DATA_PATH = "diabetes_data.csv"

df = pd.read_csv(DATA_PATH)

print("=" * 70)
print("STAGE 2 - LABORATORY PARAMETER MODEL")
print("=" * 70)

print("\nDataset shape:", df.shape)


# ------------------------------------------------------------
# 2. DEFINE LABORATORY FEATURES
# ------------------------------------------------------------
# These are the six core biomarkers specified in the project
# Phase-I report.

LAB_FEATURES = [
    "HbA1c",
    "Fasting_Blood_Glucose",
    "Postprandial_Blood_Glucose",
    "Insulin_Levels",
    "HOMA_IR",
    "C_Peptide"
]


# ------------------------------------------------------------
# 3. CHECK FEATURES
# ------------------------------------------------------------

print("\nChecking laboratory features...")

missing_features = [
    feature for feature in LAB_FEATURES
    if feature not in df.columns
]

if missing_features:
    print("\nERROR - Missing laboratory features:")
    for feature in missing_features:
        print("-", feature)

    raise SystemExit(
        "\nPlease check the column names in diabetes_data.csv."
    )

print("All six laboratory features found successfully.")


# ------------------------------------------------------------
# 4. TARGET VARIABLE
# ------------------------------------------------------------

print("\nTarget distribution:")

print(
    df["Diabetes_Status"].value_counts()
)


# Convert target:
# Negative = 0
# Positive = 1

y = df["Diabetes_Status"].map({
    "Negative": 0,
    "Positive": 1
})


# Check for unexpected target values

if y.isna().any():
    print("\nERROR: Unexpected values found in Diabetes_Status.")
    print(df["Diabetes_Status"].unique())
    raise SystemExit()


# ------------------------------------------------------------
# 5. INPUT FEATURES
# ------------------------------------------------------------

X = df[LAB_FEATURES].copy()

print("\nStage-2 laboratory features:")

for feature in LAB_FEATURES:
    print("-", feature)

print("\nFeature matrix:", X.shape)
print("Target:", y.shape)


# ------------------------------------------------------------
# 6. CHECK MISSING VALUES
# ------------------------------------------------------------

print("\nMissing values:")

print(X.isnull().sum())

if X.isnull().sum().sum() > 0:

    print("\nMissing values detected.")
    print("Filling missing values using median.")

    X = X.fillna(X.median())

else:

    print("No missing values found.")


# ------------------------------------------------------------
# 7. TRAIN / TEST SPLIT
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\n" + "=" * 70)
print("TRAIN / TEST SPLIT")
print("=" * 70)

print("Training samples:", len(X_train))
print("Testing samples :", len(X_test))


# ------------------------------------------------------------
# 8. RANDOM FOREST MODEL
# ------------------------------------------------------------

print("\nTraining Random Forest...")


rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    random_state=42,
    class_weight="balanced"
)


rf_model.fit(
    X_train,
    y_train
)


print("Random Forest training completed.")


# ------------------------------------------------------------
# 9. PREDICTION
# ------------------------------------------------------------

y_pred = rf_model.predict(X_test)

y_prob = rf_model.predict_proba(X_test)[:, 1]


# ------------------------------------------------------------
# 10. PERFORMANCE METRICS
# ------------------------------------------------------------

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred
)

recall = recall_score(
    y_test,
    y_pred
)

f1 = f1_score(
    y_test,
    y_pred
)

roc_auc = roc_auc_score(
    y_test,
    y_prob
)


# ------------------------------------------------------------
# 11. CONFUSION MATRIX
# ------------------------------------------------------------

cm = confusion_matrix(
    y_test,
    y_pred
)


print("\n" + "=" * 70)
print("STAGE-2 RANDOM FOREST RESULTS")
print("=" * 70)

print("\nAccuracy :", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall   :", round(recall, 4))
print("F1 Score :", round(f1, 4))
print("ROC-AUC  :", round(roc_auc, 4))

print("\nConfusion Matrix:")
print(cm)


# ------------------------------------------------------------
# 12. CLASSIFICATION REPORT
# ------------------------------------------------------------

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "Negative",
            "Positive"
        ]
    )
)


# ------------------------------------------------------------
# 13. FEATURE IMPORTANCE
# ------------------------------------------------------------

feature_importance = pd.DataFrame({
    "Feature": LAB_FEATURES,
    "Importance": rf_model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)


print("\n" + "=" * 70)
print("RANDOM FOREST FEATURE IMPORTANCE")
print("=" * 70)

print(
    feature_importance.to_string(index=False)
)


# ------------------------------------------------------------
# 14. SAVE MODEL
# ------------------------------------------------------------

os.makedirs(
    "models",
    exist_ok=True
)


MODEL_PATH = "models/stage2_random_forest.pkl"

FEATURE_PATH = "models/stage2_lab_features.pkl"

RESULT_PATH = "models/stage2_random_forest_results.csv"

IMPORTANCE_PATH = "models/stage2_lab_feature_importance.csv"


joblib.dump(
    rf_model,
    MODEL_PATH
)


joblib.dump(
    LAB_FEATURES,
    FEATURE_PATH
)


# ------------------------------------------------------------
# 15. SAVE RESULTS
# ------------------------------------------------------------

results = pd.DataFrame({
    "Metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "ROC-AUC"
    ],
    "Value": [
        accuracy,
        precision,
        recall,
        f1,
        roc_auc
    ]
})


results.to_csv(
    RESULT_PATH,
    index=False
)


feature_importance.to_csv(
    IMPORTANCE_PATH,
    index=False
)


# ------------------------------------------------------------
# 16. FINAL OUTPUT
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("FILES SAVED")
print("=" * 70)

print("\nModel:")
print(MODEL_PATH)

print("\nFeatures:")
print(FEATURE_PATH)

print("\nResults:")
print(RESULT_PATH)

print("\nFeature importance:")
print(IMPORTANCE_PATH)


print("\n" + "=" * 70)
print("STAGE-2 RANDOM FOREST TRAINING COMPLETED")
print("=" * 70) 
# ============================================================
# SHAP ANALYSIS – RANDOM FOREST LABORATORY MODEL
# Add this AFTER the Random Forest model has been trained
# ============================================================

import shap
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


print("\n" + "=" * 60)
print("SHAP ANALYSIS")
print("=" * 60)


# ------------------------------------------------------------
# 1. CREATE SHAP EXPLAINER
# ------------------------------------------------------------

explainer = shap.TreeExplainer(
    rf_model
)


# ------------------------------------------------------------
# 2. CALCULATE SHAP VALUES
# ------------------------------------------------------------

shap_values = explainer.shap_values(
    X_test
)


# ------------------------------------------------------------
# 3. HANDLE BINARY CLASSIFICATION OUTPUT
# ------------------------------------------------------------

if isinstance(shap_values, list):

    # Class 1 = Diabetes Positive
    shap_positive = shap_values[1]

else:

    shap_values = np.asarray(shap_values)

    if shap_values.ndim == 3:

        # New SHAP format:
        # samples × features × classes

        shap_positive = shap_values[:, :, 1]

    else:

        shap_positive = shap_values


print(
    "SHAP values calculated successfully."
)

print(
    "SHAP shape:",
    shap_positive.shape
)


# ------------------------------------------------------------
# 4. SHAP FEATURE IMPORTANCE
# ------------------------------------------------------------

mean_abs_shap = np.abs(
    shap_positive
).mean(axis=0)


shap_importance = pd.DataFrame({

    "Feature": X_test.columns,

    "Mean_Absolute_SHAP":
        mean_abs_shap

})


shap_importance = shap_importance.sort_values(
    "Mean_Absolute_SHAP",
    ascending=False
)


print("\nSHAP Feature Importance:")
print(
    shap_importance.to_string(index=False)
)


# ------------------------------------------------------------
# 5. SAVE SHAP FEATURE IMPORTANCE
# ------------------------------------------------------------

shap_importance.to_csv(
    "models/stage2_shap_feature_importance.csv",
    index=False
)


# ------------------------------------------------------------
# 6. SHAP BAR PLOT
# ------------------------------------------------------------

plt.figure(
    figsize=(10, 6)
)

shap.summary_plot(
    shap_positive,
    X_test,
    plot_type="bar",
    show=False
)

plt.title(
    "Stage 2 – SHAP Feature Importance"
)

plt.tight_layout()

plt.savefig(
    "models/stage2_shap_bar.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ------------------------------------------------------------
# 7. SHAP SUMMARY / BEESWARM PLOT
# ------------------------------------------------------------

plt.figure(
    figsize=(10, 6)
)

shap.summary_plot(
    shap_positive,
    X_test,
    show=False
)

plt.title(
    "Stage 2 – SHAP Summary Plot"
)

plt.tight_layout()

plt.savefig(
    "models/stage2_shap_summary.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ------------------------------------------------------------
# 8. PRINT TOP FEATURES
# ------------------------------------------------------------

print("\nTop SHAP Features:")

for _, row in shap_importance.iterrows():

    print(
        f"{row['Feature']}: "
        f"{row['Mean_Absolute_SHAP']:.6f}"
    )


print("\nSHAP analysis completed successfully.")

print("\nSaved files:")

print(
    "models/stage2_shap_feature_importance.csv"
)

print(
    "models/stage2_shap_bar.png"
)

print(
    "models/stage2_shap_summary.png"
)