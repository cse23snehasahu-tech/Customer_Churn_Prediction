
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Step 1: Load cleaned dataset
df = pd.read_csv("customer_churn_cleaned.csv")

# Step 2: Separate features and target
X = df.drop(columns=["Churn", "CustomerID"], errors="ignore")
y = df["Churn"]

# Step 3: Identify numerical and categorical columns
numerical_features = X.select_dtypes(
    include=np.number
).columns.tolist()

categorical_features = X.select_dtypes(
    exclude=np.number
).columns.tolist()

print("\n========== FEATURE TYPES ==========")
print("Numerical features:", numerical_features)
print("Categorical features:", categorical_features)

# Step 4: Split data before fitting preprocessing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Step 5: Create preprocessing pipelines
preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            StandardScaler(),
            numerical_features
        ),
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ]
)

# Step 6: Fit only on training data
X_train_processed = preprocessor.fit_transform(X_train)

# Transform test data using the same fitted preprocessor
X_test_processed = preprocessor.transform(X_test)

# Step 7: Display results
print("\n========== PREPROCESSING RESULTS ==========")
print("Original training shape:", X_train.shape)
print("Original testing shape:", X_test.shape)

print("Processed training shape:", X_train_processed.shape)
print("Processed testing shape:", X_test_processed.shape)

# Step 8: Display encoded feature names
feature_names = preprocessor.get_feature_names_out()

print("\n========== PROCESSED FEATURE NAMES ==========")
for feature in feature_names:
    print(feature)

# Step 9: Validate preprocessing
print("\n========== VALIDATION ==========")
print("Training rows preserved:", 
      X_train_processed.shape[0] == len(X_train))
print("Testing rows preserved:",
      X_test_processed.shape[0] == len(X_test))
print("All processed values finite:",
      np.isfinite(X_train_processed).all()
      and np.isfinite(X_test_processed).all())

print("\nEncoding and scaling completed successfully!")
