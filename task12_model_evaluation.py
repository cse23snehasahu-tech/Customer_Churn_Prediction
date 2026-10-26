
import pandas as pd
import matplotlib.pyplot as plt
import os

# Step 1: Load results from Task 11
results = pd.read_csv("task11_model_results.csv")

# Step 2: Validate required columns
required_columns = [
    "Model", "Accuracy", "Precision", "Recall", "F1 Score"
]

missing_columns = [
    column for column in required_columns
    if column not in results.columns
]

if missing_columns:
    raise ValueError(f"Missing columns: {missing_columns}")

if results.empty:
    raise ValueError("Model results file is empty.")

# Step 3: Sort models by F1 score
results = results.sort_values(
    by="F1 Score",
    ascending=False
).reset_index(drop=True)

# Step 4: Display comparison table
print("\n========== MODEL PERFORMANCE COMPARISON ==========")
print(results.round(4).to_string(index=False))

# Step 5: Identify best model by F1 score
best_f1_model = results.loc[results["F1 Score"].idxmax()]

print("\n========== BEST MODEL BY F1 SCORE ==========")
print("Model:", best_f1_model["Model"])
print("F1 Score:", round(best_f1_model["F1 Score"], 4))

# Step 6: Identify best model for detecting churn
best_recall_model = results.loc[results["Recall"].idxmax()]

print("\n========== BEST MODEL BY RECALL ==========")
print("Model:", best_recall_model["Model"])
print("Recall:", round(best_recall_model["Recall"], 4))

# Step 7: Plot metric comparison
os.makedirs("Model_Evaluation", exist_ok=True)

metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]

ax = results.set_index("Model")[metrics].plot(
    kind="bar",
    figsize=(11, 6)
)

ax.set_title("Customer Churn Model Comparison")
ax.set_ylabel("Score")
ax.set_xlabel("Machine Learning Model")
ax.set_ylim(0, 1)
plt.xticks(rotation=20, ha="right")
plt.legend(loc="best")
plt.tight_layout()

plt.savefig("Model_Evaluation/model_comparison.png")
plt.close()

# Step 8: Save the sorted comparison
results.to_csv(
    "Model_Evaluation/model_comparison.csv",
    index=False
)

print("\n========== EVALUATION SUMMARY ==========")
print("Models evaluated:", len(results))
print("Comparison saved in Model_Evaluation folder.")
print("Task 12 completed successfully!")
