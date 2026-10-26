
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

print("=" * 65)
print("TASK 13 - CONFUSION MATRIX ANALYSIS")
print("=" * 65)

# 1. Load cleaned dataset
df = pd.read_csv("customer_churn_cleaned.csv")

X = df.drop(columns=["Churn", "CustomerID"], errors="ignore")
y = df["Churn"].astype(int)

# 2. Use the same split as earlier tasks
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 3. Prepare numeric and categorical columns
numeric_features = X_train.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_features = X_train.select_dtypes(
    exclude=["number"]
).columns.tolist()

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

# 4. Select the best model based on Task 11 F1 score
results = pd.read_csv("task11_model_results.csv")
best_row = results.loc[results["F1 Score"].idxmax()]
best_model_name = str(best_row["Model"]).strip()

print(f"\nSelected model: {best_model_name}")
print(f"Task 11 F1 Score: {best_row['F1 Score']}")

model_options = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ),
    "SVM": SVC(),
    "KNN": KNeighborsClassifier(n_neighbors=5)
}

if best_model_name not in model_options:
    raise ValueError(
        f"Model name '{best_model_name}' was not recognized. "
        "Check the Model column in task11_model_results.csv."
    )

# 5. Train selected model and predict
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", model_options[best_model_name])
])

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 6. Confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
tn, fp, fn, tp = cm.ravel()

print("\nConfusion Matrix:")
print(cm)

print("\nPrediction breakdown:")
print(f"True Negatives  (TN): {tn}")
print(f"False Positives (FP): {fp}")
print(f"False Negatives (FN): {fn}")
print(f"True Positives  (TP): {tp}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# 7. Save confusion matrix chart
os.makedirs("Confusion_Matrix_Results", exist_ok=True)

plt.figure(figsize=(7, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Stayed (0)", "Churned (1)"],
    yticklabels=["Stayed (0)", "Churned (1)"]
)

plt.title(f"Confusion Matrix - {best_model_name}")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.tight_layout()

chart_path = os.path.join(
    "Confusion_Matrix_Results",
    "confusion_matrix.png"
)
plt.savefig(chart_path, dpi=300)
plt.show()

# 8. Save numerical results
summary = pd.DataFrame({
    "Metric": [
        "True Negatives",
        "False Positives",
        "False Negatives",
        "True Positives"
    ],
    "Count": [tn, fp, fn, tp]
})

summary.to_csv(
    os.path.join(
        "Confusion_Matrix_Results",
        "confusion_matrix_counts.csv"
    ),
    index=False
)

print(f"\nResults saved in: Confusion_Matrix_Results")
print("TASK 13 COMPLETED")
