
import pandas as pd

# Step 1: Load cleaned dataset
df = pd.read_csv("customer_churn_cleaned.csv")

# Step 2: Separate features and target
X = df.drop(columns=["Churn", "CustomerID"], errors="ignore")
y = df["Churn"]

# Step 3: Display shapes
print("\n========== FEATURE AND TARGET SHAPES ==========")
print("Features (X):", X.shape)
print("Target (y):", y.shape)

# Step 4: Display feature names
print("\n========== INPUT FEATURES ==========")
print(X.columns.tolist())

# Step 5: Display target distribution
print("\n========== TARGET DISTRIBUTION ==========")
print(y.value_counts().sort_index())

# Step 6: Validate separation
print("\n========== VALIDATION ==========")
print("Churn present in X:", "Churn" in X.columns)
print("CustomerID present in X:", "CustomerID" in X.columns)
print("Missing target values:", y.isnull().sum())

assert "Churn" not in X.columns
assert "CustomerID" not in X.columns
assert len(X) == len(y)

print("\nFeature-target separation completed successfully!")
