
import pandas as pd
import matplotlib.pyplot as plt
plt.switch_backend("Agg")
import seaborn as sns

# Load cleaned dataset
df = pd.read_csv("customer_churn_cleaned.csv")

# Create output folder for charts
import os
os.makedirs("EDA_Charts", exist_ok=True)

sns.set_theme(style="whitegrid")

# Numerical features
numerical_features = [
    "Age",
    "Tenure",
    "Usage Frequency",
    "Support Calls",
    "Payment Delay",
    "Total Spend",
    "Last Interaction"
]

# Plot churn rate across numerical feature ranges
for column in numerical_features:
    plt.figure(figsize=(8, 5))

    sns.boxplot(
        data=df,
        x="Churn",
        y=column
    )

    plt.title(f"{column} vs Customer Churn")
    plt.xlabel("Churn (0 = Stayed, 1 = Churned)")
    plt.ylabel(column)
    plt.tight_layout()

    plt.savefig(f"EDA_Charts/{column.replace(' ', '_')}_vs_Churn.png")
    plt.close()

# Categorical feature analysis
categorical_features = [
    "Gender",
    "Subscription Type",
    "Contract Length"
]

for column in categorical_features:
    plt.figure(figsize=(8, 5))

    sns.barplot(
        data=df,
        x=column,
        y="Churn",
        errorbar=None
    )

    plt.title(f"Churn Rate by {column}")
    plt.xlabel(column)
    plt.ylabel("Average Churn Rate")
    plt.xticks(rotation=20)
    plt.tight_layout()

    plt.savefig(f"EDA_Charts/Churn_Rate_by_{column.replace(' ', '_')}.png")
    plt.close()

# Print average churn rates by category
print("\n========== CHURN RATE BY CATEGORY ==========")

for column in categorical_features:
    print(f"\n{column}:")
    print(
        (df.groupby(column, observed=True)["Churn"].mean() * 100)
        .round(2)
        .sort_values(ascending=False)
    )

print("\nEDA completed successfully!")
print("Charts saved in the EDA_Charts folder.")
