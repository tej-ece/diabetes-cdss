from ucimlrepo import fetch_ucirepo

print("Downloading dataset...")

dataset = fetch_ucirepo(id=891)

X = dataset.data.features
y = dataset.data.targets

data = X.copy()
data["Diabetes_binary"] = y["Diabetes_binary"]

data.to_csv("data/diabetes_health_indicators.csv", index=False)

print("Dataset downloaded successfully!")
print("Shape:", data.shape)

print("\nColumns:")
print(data.columns.tolist())

print("\nFirst 5 rows:")
print(data.head())
