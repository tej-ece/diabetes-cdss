import pandas as pd

# Load dataset
data = pd.read_csv("data/diabetes_health_indicators.csv")

print("===== DATASET SHAPE =====")
print(data.shape)

print("\n===== COLUMN NAMES =====")
print(data.columns.tolist())

print("\n===== MISSING VALUES =====")
print(data.isnull().sum().sum())

print("\n===== TARGET DISTRIBUTION =====")
print(data["Diabetes_binary"].value_counts())

print("\n===== TARGET PERCENTAGE =====")
print(data["Diabetes_binary"].value_counts(normalize=True) * 100)

print("\n===== DATA TYPES =====")
print(data.dtypes)

print("\n===== BASIC STATISTICS =====")
print(data.describe())