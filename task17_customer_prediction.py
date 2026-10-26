
import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

print("=" * 65)
print("TASK 17 - CUSTOMER CHURN PREDICTION SYSTEM")
print("=" * 65)

# 1. Load cleaned dataset
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

# 3. Identify column types
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

# 5. Train the prediction model
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

print("\nModel trained successfully!")

# 6. Take new customer details
print("\nEnter new customer details:")

customer_data = {}

for feature in X.columns:
    if feature in numeric_features:
        while True:
            try:
                value = float(input(f"{feature}: "))
                customer_data[feature] = value
                break
            except ValueError:
                print("Please enter a valid number.")
    else:
        customer_data[feature] = input(
            f"{feature} {list(df[feature].dropna().unique())}: "
        )

# 7. Predict churn
new_customer = pd.DataFrame([customer_data], columns=X.columns)

prediction = int(model.predict(new_customer)[0])
probabilities = model.predict_proba(new_customer)[0]
classes = list(model.named_steps["classifier"].classes_)

churn_probability = float(
    probabilities[classes.index(1)] * 100
)

print("\n" + "=" * 50)
print("CUSTOMER CHURN PREDICTION RESULT")
print("=" * 50)

if prediction == 1:
    print("Prediction: Customer is predicted to CHURN")
else:
    print("Prediction: Customer is predicted to STAY")

print(f"Predicted churn probability: {churn_probability:.2f}%")

# 8. Illustrative risk categories
if churn_probability >= 70:
    risk = "High"
elif churn_probability >= 40:
    risk = "Medium"
else:
    risk = "Low"

print(f"Churn Risk Level: {risk}")
print("=" * 50)

# 9. Save prediction
os.makedirs("Customer_Prediction_Results", exist_ok=True)

result = new_customer.copy()
result["Predicted_Churn"] = prediction
result["Churn_Probability_Percent"] = round(churn_probability, 2)
result["Risk_Level"] = risk

result.to_csv(
    "Customer_Prediction_Results/new_customer_prediction.csv",
    index=False
)

print("\nPrediction saved successfully!")
print("TASK 17 COMPLETED")
