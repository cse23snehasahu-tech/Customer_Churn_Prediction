
import os
import pandas as pd
from datetime import datetime

print("=" * 65)
print("TASK 21 - CUSTOMER CHURN PREDICTION PROJECT CONCLUSION")
print("=" * 65)

# 1. Load final model comparison
comparison_path = os.path.join(
    "Final_Model_Analysis",
    "final_model_comparison.csv"
)

if not os.path.exists(comparison_path):
    raise FileNotFoundError(
        "Final comparison file not found. "
        "Please run Task 20 first."
    )

comparison = pd.read_csv(comparison_path)

required_columns = [
    "Model", "Type", "Accuracy", "Precision", "Recall", "F1"
]

for column in required_columns:
    if column not in comparison.columns:
        raise ValueError(
            f"Required column '{column}' is missing."
        )

# 2. Select best result by F1
comparison = comparison.dropna(subset=["F1"])

if comparison.empty:
    raise ValueError("No valid F1 scores are available.")

best_model = comparison.loc[comparison["F1"].idxmax()]

# 3. Load feature importance if available
importance_path = os.path.join(
    "Feature_Importance_Results",
    "feature_importance.csv"
)

top_features = []

if os.path.exists(importance_path):
    importance = pd.read_csv(importance_path)

    if {"Feature", "Importance"}.issubset(importance.columns):
        importance = importance.sort_values(
            "Importance", ascending=False
        )
        top_features = importance.head(5)["Feature"].tolist()

# 4. Prepare conclusion using actual results
conclusion = f"""
CUSTOMER CHURN PREDICTION SYSTEM
FINAL PROJECT CONCLUSION
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}

1. PROJECT OBJECTIVE
The objective of this project was to develop a machine learning
system that predicts whether a customer is likely to churn or stay.

2. DATA PREPARATION
The dataset was cleaned and explored. CustomerID was excluded
from the model features because it is an identifier rather than
a behavioral predictor. Numerical and categorical features were
preprocessed using imputation, scaling, and encoding as appropriate.

3. MODEL DEVELOPMENT
Multiple classification algorithms were evaluated. These included
Logistic Regression, Random Forest, Support Vector Machine, and
K-Nearest Neighbors. Model improvement and hyperparameter tuning
were also explored.

4. BEST RESULT IN THE FINAL COMPARISON
Model: {best_model['Model']}
Model type: {best_model['Type']}
Accuracy: {best_model['Accuracy']:.4f}
Precision: {best_model['Precision']:.4f}
Recall: {best_model['Recall']:.4f}
F1 Score: {best_model['F1']:.4f}

This model has the highest F1 score among the entries in the
final comparison file. Its practical suitability should also
be assessed using recall, precision, business costs, and validation.

5. IMPORTANT FEATURES
"""

if top_features:
    conclusion += "\n".join(
        f"- {feature}" for feature in top_features
    )
else:
    conclusion += (
        "Feature importance results were not available. "
        "Run Task 14 to include the feature ranking."
    )

conclusion += """

6. CUSTOMER RETENTION
Predictions can help prioritize customer outreach, investigate
support issues, improve billing communication, and encourage
customer engagement. These actions should be tested to measure
their actual retention impact.

7. LIMITATIONS
Predictions are estimates and are not guarantees of customer
behavior. Results depend on data quality, class distribution,
feature relevance, and model validation. Feature associations
do not necessarily imply causation.

8. CONCLUSION
This project demonstrates an end-to-end customer churn prediction
workflow, including data preparation, exploratory analysis,
classification, evaluation, risk estimation, and retention
recommendations. The final model should be selected based on
validated performance and the business objective.
"""

# 5. Save the conclusion
output_dir = "Final_Project_Report"
os.makedirs(output_dir, exist_ok=True)

report_path = os.path.join(
    output_dir,
    "customer_churn_project_conclusion.txt"
)

with open(report_path, "w", encoding="utf-8") as file:
    file.write(conclusion)

print(conclusion)
print("\nConclusion saved to:", report_path)
print("\nTASK 21 COMPLETED")
print("ALL 21 PROJECT TASKS HAVE BEEN PROCESSED!")
