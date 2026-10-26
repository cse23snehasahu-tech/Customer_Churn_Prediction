
import os
import pandas as pd
import matplotlib.pyplot as plt

print("=" * 65)
print("TASK 20 - FINAL MODEL ANALYSIS")
print("=" * 65)

# 1. Load baseline results
baseline_path = "task11_model_results.csv"

# 2. Load tuned-model results
tuned_path = os.path.join(
    "Hyperparameter_Tuning_Results",
    "tuning_comparison.csv"
)

if not os.path.exists(baseline_path):
    raise FileNotFoundError(
        f"Baseline results not found: {baseline_path}"
    )

if not os.path.exists(tuned_path):
    raise FileNotFoundError(
        f"Tuned results not found: {tuned_path}"
    )

baseline = pd.read_csv(baseline_path)
tuned = pd.read_csv(tuned_path)

# 3. Standardize metric names
baseline_required = ["Model", "Accuracy", "Precision", "Recall", "F1 Score"]

for column in baseline_required:
    if column not in baseline.columns:
        raise ValueError(
            f"Missing '{column}' in {baseline_path}"
        )

tuned_required = [
    "Model",
    "Test Accuracy",
    "Test Precision",
    "Test Recall",
    "Test F1"
]

for column in tuned_required:
    if column not in tuned.columns:
        raise ValueError(
            f"Missing '{column}' in {tuned_path}"
        )

baseline_comparison = baseline[
    ["Model", "Accuracy", "Precision", "Recall", "F1 Score"]
].copy()

baseline_comparison["Type"] = "Baseline"
baseline_comparison.rename(
    columns={
        "F1 Score": "F1"
    },
    inplace=True
)

tuned_comparison = tuned[
    ["Model", "Test Accuracy", "Test Precision", "Test Recall", "Test F1"]
].copy()

tuned_comparison.rename(
    columns={
        "Test Accuracy": "Accuracy",
        "Test Precision": "Precision",
        "Test Recall": "Recall",
        "Test F1": "F1"
    },
    inplace=True
)
tuned_comparison["Type"] = "Tuned"

# 4. Combine results
comparison = pd.concat(
    [baseline_comparison, tuned_comparison],
    ignore_index=True
)

comparison = comparison[
    ["Model", "Type", "Accuracy", "Precision", "Recall", "F1"]
]

comparison = comparison.sort_values(
    by="F1",
    ascending=False
).reset_index(drop=True)

print("\nFINAL MODEL COMPARISON")
print(comparison.to_string(index=False))

# 5. Identify best models
best_f1 = comparison.loc[comparison["F1"].idxmax()]
best_recall = comparison.loc[comparison["Recall"].idxmax()]

print("\nBest result by F1 Score:")
print(best_f1.to_string())

print("\nBest result by Recall:")
print(best_recall.to_string())

# 6. Save final comparison
output_dir = "Final_Model_Analysis"
os.makedirs(output_dir, exist_ok=True)

comparison.to_csv(
    os.path.join(output_dir, "final_model_comparison.csv"),
    index=False
)

# 7. Plot F1 scores
plot_data = comparison.sort_values("F1", ascending=True)

plt.figure(figsize=(10, 7))
plt.barh(
    plot_data["Model"] + " (" + plot_data["Type"] + ")",
    plot_data["F1"]
)
plt.xlabel("Test F1 Score")
plt.title("Baseline vs Tuned Model Comparison")
plt.tight_layout()

plt.savefig(
    os.path.join(output_dir, "final_model_comparison.png"),
    dpi=300
)
plt.close()

# 8. Save a concise summary
summary = pd.DataFrame([
    {
        "Selection_Criterion": "Highest F1 Score",
        "Model": best_f1["Model"],
        "Type": best_f1["Type"],
        "Score": best_f1["F1"]
    },
    {
        "Selection_Criterion": "Highest Recall",
        "Model": best_recall["Model"],
        "Type": best_recall["Type"],
        "Score": best_recall["Recall"]
    }
])

summary.to_csv(
    os.path.join(output_dir, "best_model_summary.csv"),
    index=False
)

print(f"\nResults saved in: {output_dir}")
print("\nTASK 20 COMPLETED")
