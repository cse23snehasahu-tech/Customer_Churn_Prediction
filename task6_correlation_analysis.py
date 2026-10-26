
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load cleaned dataset
df = pd.read_csv("customer_churn_cleaned.csv")

os.makedirs("Correlation_Results", exist_ok=True)

# Select numerical columns only
numerical_df = df.select_dtypes(include="number")

# Calculate correlation matrix
correlation_matrix = numerical_df.corr()

print("\n========== CORRELATION WITH CHURN ==========")
print(
    correlation_matrix["Churn"]
    .drop("Churn")
    .sort_values(ascending=False)
    .round(4)
)

# Plot complete correlation matrix
plt.figure(figsize=(12, 9))

sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5
)

plt.title("Customer Churn Correlation Heatmap")
plt.tight_layout()
plt.savefig("Correlation_Results/correlation_heatmap.png")
plt.close()

# Save correlation matrix as CSV
correlation_matrix.to_csv(
    "Correlation_Results/correlation_matrix.csv"
)

# Identify strongly correlated feature pairs
print("\n========== STRONGLY CORRELATED FEATURES ==========")

columns = correlation_matrix.columns

for i in range(len(columns)):
    for j in range(i + 1, len(columns)):
        value = correlation_matrix.iloc[i, j]

        if abs(value) >= 0.70:
            print(
                f"{columns[i]} and {columns[j]}: {value:.3f}"
            )

print("\nCorrelation analysis completed!")
print("Results saved in Correlation_Results folder.")
