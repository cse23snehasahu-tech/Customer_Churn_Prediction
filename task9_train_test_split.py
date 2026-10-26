
import pandas as pd
from sklearn.model_selection import train_test_split

# Step 1: Load cleaned dataset
df = pd.read_csv("customer_churn_cleaned.csv")

# Step 2: Separate features and target
X = df.drop(columns=["Churn", "CustomerID"], errors="ignore")
y = df["Churn"]

# Step 3: Split into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Step 4: Display dataset sizes
print("\n========== TRAIN-TEST SPLIT ==========")
print("Original dataset:", X.shape)
print("Training features:", X_train.shape)
print("Testing features:", X_test.shape)
print("Training targets:", y_train.shape)
print("Testing targets:", y_test.shape)

# Step 5: Compare churn distributions
print("\n========== TRAINING CHURN DISTRIBUTION (%) ==========")
print((y_train.value_counts(normalize=True) * 100).round(2))

print("\n========== TESTING CHURN DISTRIBUTION (%) ==========")
print((y_test.value_counts(normalize=True) * 100).round(2))

# Step 6: Verify split sizes
print("\n========== VALIDATION ==========")
print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))
print("Total rows:", len(X_train) + len(X_test))

assert len(X_train) + len(X_test) == len(df)
assert len(X_test) == round(len(df) * 0.20)

print("\nTrain-test split completed successfully!")
