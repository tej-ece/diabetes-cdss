import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("Loading dataset...")

# Load dataset
data = pd.read_csv("data/diabetes_health_indicators.csv")

# Separate features and target
X = data.drop("Diabetes_binary", axis=1)
y = data["Diabetes_binary"]

# Remove ID because it is only an identifier
if "ID" in X.columns:
    X = X.drop("ID", axis=1)

print("Dataset loaded!")
print("Samples:", X.shape[0])
print("Features:", X.shape[1])

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# Lightweight model pipeline
model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42
    ))
])

print("\nTraining Logistic Regression model...")

model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)

print("\n===== MODEL RESULTS =====")
print("Accuracy:", round(accuracy * 100, 2), "%")

print("\n===== CLASSIFICATION REPORT =====")
print(classification_report(y_test, y_pred))

print("\n===== CONFUSION MATRIX =====")
print(confusion_matrix(y_test, y_pred))

# Save compact model
joblib.dump(model, "models/diabetes_model.pkl")

print("\nModel saved successfully!")
print("Location: models/diabetes_model.pkl")
