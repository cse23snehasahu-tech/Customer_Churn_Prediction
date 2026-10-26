
import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

print("=" * 65)
print("TASK 18 - CHURN RISK ANALYSIS")
print("=" * 65)

# 1. Load cleaned dataset
df = pd.read_csv("customer_churn_cleaned.csv")

X = df.drop(columns=["Churn", "CustomerID"], errors="ignore")
y = df["Churn"].astype(int)

# 2. Train-test split
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

# 5. Train model
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    ))
])

model.fit(X_train, y_train)

# 6. Predict churn probabilities on held-out test data
churn_class_index = list(
    model.named_steps["classifier"].classes_
).index(1)

churn_probability = model.predict_proba(X_test)[
    :, churn_class_index
] * 100

# 7. Assign illustrative risk levels
def assign_risk(probability):
    if probability >= 70:
        return "High"
    elif probability >= 40:
        return "Medium"
    return "Low"

risk_results = X_test.copy()
risk_results["Actual_Churn"] = y_test.values
risk_results["Churn_Probability_Percent"] = churn_probability
risk_results["Risk_Level"] = [
    assign_risk(p) for p in churn_probability
]

# 8. Display risk distribution
risk_counts = risk_results["Risk_Level"].value_counts().reindex(
    ["High", "Medium", "Low"], fill_value=0
)

print("\nCustomer Risk Distribution:")
print(risk_counts.to_string())

print("\nRisk-wise actual churn rate:")
print(
    risk_results.groupby("Risk_Level", observed=False)[
        "Actual_Churn"
    ].agg(["count", "mean"]).rename(
        columns={
            "count": "Customers",
            "mean": "Actual_Churn_Rate"
        }
    ).to_string()
)

# 9. Examine numerical features across actual churn classes
analysis_df = X_test.copy()
analysis_df["Actual_Churn"] = y_test.values

numeric_to_analyze = [
    feature for feature in
    ["Age", "Tenure", "Usage Frequency", "Support Calls",
     "Payment Delay", "Total Spend", "Last Interaction"]
    if feature in numeric_features
]

if numeric_to_analyze:
    print("\nAverage numerical feature values by actual churn:")
    print(
        analysis_df.groupby("Actual_Churn")[numeric_to_analyze]
        .mean()
        .T.to_string()
    )

# 10. Save results
output_dir = "Churn_Risk_Results"
os.makedirs(output_dir, exist_ok=True)

risk_results.to_csv(
    os.path.join(output_dir, "customer_risk_analysis.csv"),
    index=False
)

risk_counts.rename_axis("Risk_Level").reset_index(
    name="Customer_Count"
).to_csv(
    os.path.join(output_dir, "risk_distribution.csv"),
    index=False
)

# 11. Plot risk distribution
plt.figure(figsize=(7, 5))
risk_counts.plot(kind="bar")
plt.title("Customer Churn Risk Distribution")
plt.xlabel("Risk Level")
plt.ylabel("Number of Customers")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig(
    os.path.join(output_dir, "risk_distribution.png"),
    dpi=300
)
plt.show()

print(f"\nResults saved in: {output_dir}")
print("\nTASK 18 COMPLETED")
