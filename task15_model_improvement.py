
import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier

print("=" * 65)
print("TASK 15 - MODEL IMPROVEMENT")
print("=" * 65)

# 1. Load cleaned dataset
df = pd.read_csv("customer_churn_cleaned.csv")

X = df.drop(columns=["Churn", "CustomerID"], errors="ignore")
y = df["Churn"].astype(int)

# 2. Split data consistently
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

numeric_features = X_train.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_features = X_train.select_dtypes(
    exclude=["number"]
).columns.tolist()

# 3. Preprocessing
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

# 4. Models and improvement variants
models = {
    "Logistic Regression - Baseline":
        LogisticRegression(max_iter=1000),

    "Logistic Regression - Balanced":
        LogisticRegression(max_iter=1000, class_weight="balanced"),

    "Random Forest - Baseline":
        RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        ),

    "Random Forest - Balanced":
        RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced"
        ),

"SVM - Baseline":
    LinearSVC(random_state=42, max_iter=3000),

    "SVM - Balanced":
    LinearSVC(
        class_weight="balanced",
        random_state=42,
        max_iter=3000
    ),

    "KNN - Baseline":
        KNeighborsClassifier(n_neighbors=5)
}

results = []

# 5. Train and evaluate models
for name, classifier in models.items():
    print(f"\nTraining: {name}")

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ])

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "F1 Score": f1_score(
            y_test, predictions, zero_division=0
        )
    })

# 6. Compare results
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(
    by="F1 Score", ascending=False
).reset_index(drop=True)

print("\nMODEL IMPROVEMENT COMPARISON")
print(results_df.to_string(index=False))

# 7. Save results
output_dir = "Model_Improvement_Results"
os.makedirs(output_dir, exist_ok=True)

results_df.to_csv(
    os.path.join(output_dir, "task15_improvement_results.csv"),
    index=False
)

best = results_df.iloc[0]

print("\nBest model by F1 Score:")
print(best.to_string())

print(f"\nResults saved in: {output_dir}")
print("\nTASK 15 COMPLETED")
