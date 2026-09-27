import os
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# 1. Locate and load your local dataset
data_path = (
    "data/diabetes_data.csv"
    if os.path.exists("data/diabetes_data.csv")
    else "diabetes_data.csv"
)
df = pd.read_csv(data_path)

# 2. Select the non-lab screening features
screening_features = [
    "BMI",
    "Waist_Circumference",
    "Blood_Pressure_Systolic",
    "Blood_Pressure_Diastolic",
    "Family_History_of_Diabetes",
    "Hypertension",
    "Physical_Activity",
    "Smoking",
    "Alcohol_Consumption",
    "Obesity",
    "PCOS",
    "Gestational_Diabetes",
]

X = df[screening_features]
y = (df["Diabetes_Status"] == "Positive").astype(int)

# 3. Handle categories and numbers automatically
categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
numeric_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", numeric_cols),
        (
            "cat",
            OneHotEncoder(
                drop="first", handle_unknown="ignore", sparse_output=False
            ),
            categorical_cols,
        ),
    ]
)

pipeline = Pipeline(
    [
        ("preprocessor", preprocessor),
        (
            "model",
            GradientBoostingClassifier(
                n_estimators=100, max_depth=4, random_state=42
            ),
        ),
    ]
)

# 4. Train the model
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
pipeline.fit(X_train, y_train)

# 5. Save the two .pkl files into your local folder
joblib.dump(pipeline, "diabetes_screening_xgb_pipeline.pkl")
joblib.dump(screening_features, "screening_features.pkl")

print("\n SUCCESS! The following two files were created in your folder:")
print(" 1. diabetes_screening_xgb_pipeline.pkl")
print(" 2. screening_features.pkl\n")