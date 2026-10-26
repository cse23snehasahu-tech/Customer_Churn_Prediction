
import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

print("=" * 65)
print("TASK 14 - FEATURE IMPORTANCE ANALYSIS")
print("=" * 65)

# 1. Load dataset
df = pd.read_csv("customer_churn_cleaned.csv")

X = df.drop(columns=["Churn", "CustomerID"], errors="ignore")
y = df["Churn"].astype(int)

# 2. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 3. Identify feature types
numeric_features = X_train.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_features = X_train.select_dtypes(
    exclude=["number"]
).columns.tolist()

# 4. Preprocessing
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features)
])

# 5. Train Random Forest
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ))
])

model.fit(X_train, y_train)

# 6. Calculate permutation importance on unseen test data
importance_result = permutation_importance(
    model,
    X_test,
    y_test,
    scoring="f1",
    n_repeats=5,
    random_state=42,
    n_jobs=-1
)

importance_df = pd.DataFrame({
    "Feature": X_test.columns,
    "Importance": importance_result.importances_mean,
    "Importance_Std": importance_result.importances_std
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
).reset_index(drop=True)

print("\nFeature Importance Ranking:")
print(importance_df.to_string(index=False))

# 7. Save results
output_dir = "Feature_Importance_Results"
os.makedirs(output_dir, exist_ok=True)

importance_df.to_csv(
    os.path.join(output_dir, "feature_importance.csv"),
    index=False
)

# 8. Plot top 10 features
top_features = importance_df.head(10).sort_values(
    by="Importance",
    ascending=True
)

plt.figure(figsize=(10, 6))
plt.barh(
    top_features["Feature"],
    top_features["Importance"],
    xerr=top_features["Importance_Std"],
    capsize=3
)

plt.title("Top 10 Features for Customer Churn Prediction")
plt.xlabel("Decrease in F1 Score When Feature Is Permuted")
plt.ylabel("Feature")
plt.tight_layout()

plt.savefig(
    os.path.join(output_dir, "top_10_feature_importance.png"),
    dpi=300
)
plt.show()

print(f"\nResults saved in: {output_dir}")
print("\nTASK 14 COMPLETED")
