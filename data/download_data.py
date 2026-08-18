from ucimlrepo import fetch_ucirepo

print("Downloading dataset...")

# Fetch CDC Diabetes Health Indicators dataset
dataset = fetch_ucirepo(id=891)

# Get features and target
X = dataset.data.features
y = dataset.data.targets

# Combine features and target
data = X.copy()
data["Diabetes_binary"] = y["Diabetes_binary"]

# Save dataset
data.to_csv("data/diabetes_health_indicators.csv", index=False)

print("Dataset downloaded successfully!")
print("Shape:", data.shape)

print("\nColumns:")
print(data.columns.tolist())

print("\nFirst 5 rows:")
print(data.head())