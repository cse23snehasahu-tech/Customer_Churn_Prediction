
import os
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

print("=" * 65)
print("TASK 16 - HYPERPARAMETER TUNING")
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

# 3. Identify column types
numeric_features = X_train.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_features = X_train.select_dtypes(
    exclude=["number"]
).columns.tolist()

# 4. Preprocessing pipelines
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

# 5. Logistic Regression pipeline
lr_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

lr_params = {
    "classifier__C": [0.1, 1, 10],
    "classifier__class_weight": [None, "balanced"]
}

print("\nTuning Logistic Regression...")
lr_search = GridSearchCV(
    lr_pipeline,
    lr_params,
    scoring="f1",
    cv=3,
    n_jobs=-1,
    verbose=1
)
lr_search.fit(X_train, y_train)

# 6. Random Forest pipeline
rf_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        random_state=42,
        n_jobs=1
    ))
])

rf_params = {
    "classifier__n_estimators": [50, 100],
    "classifier__max_depth": [15],
    "classifier__min_samples_leaf": [3],
    "classifier__class_weight": [None, "balanced"]
}

print("\nTuning Random Forest...")
rf_search = GridSearchCV(
    rf_pipeline,
    rf_params,
    scoring="f1",
    cv=3,
    n_jobs=-1,
    verbose=1
)
rf_search.fit(X_train, y_train)

# 7. Evaluate tuned models on the held-out test set
results = []

for name, search in [
    ("Tuned Logistic Regression", lr_search),
    ("Tuned Random Forest", rf_search)
]:
    predictions = search.predict(X_test)

    results.append({
        "Model": name,
        "Best CV F1": search.best_score_,
        "Test Accuracy": accuracy_score(y_test, predictions),
        "Test Precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "Test Recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "Test F1": f1_score(
            y_test, predictions, zero_division=0
        ),
        "Best Parameters": str(search.best_params_)
    })

    print(f"\n{name}")
    print("Best parameters:", search.best_params_)
    print(f"Best cross-validation F1: {search.best_score_:.4f}")

results_df = pd.DataFrame(results)

output_dir = "Hyperparameter_Tuning_Results"
os.makedirs(output_dir, exist_ok=True)

results_df.to_csv(
    os.path.join(output_dir, "tuning_comparison.csv"),
    index=False
)

print("\nFinal tuning comparison:")
print(results_df.to_string(index=False))

print(f"\nResults saved in: {output_dir}")
print("\nTASK 16 COMPLETED")
