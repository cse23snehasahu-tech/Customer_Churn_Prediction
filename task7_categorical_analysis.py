
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load cleaned dataset
df = pd.read_csv("customer_churn_cleaned.csv")

os.makedirs("Categorical_Analysis", exist_ok=True)

categorical_features = [
    "Gender",
    "Subscription Type",
    "Contract Length"
]

print("\n========== CATEGORY COUNTS ==========")

for column in categorical_features:
    print(f"\n{column}:")
    print(df[column].value_counts(dropna=False))

print("\n========== CHURN RATE BY CATEGORY ==========")

for column in categorical_features:
    churn_rates = (
        df.groupby(column, observed=True)["Churn"]
        .agg(["mean", "count"])
    )

    churn_rates["Churn Rate (%)"] = churn_rates["mean"] * 100

    print(f"\n{column}:")
    print(churn_rates[["count", "Churn Rate (%)"]].round(2))

    plt.figure(figsize=(8, 5))

    sns.barplot(
        data=df,
        x=column,
        y="Churn",
        errorbar=None
    )

    plt.title(f"Churn Rate by {column}")
    plt.xlabel(column)
    plt.ylabel("Churn Rate")
    plt.xticks(rotation=20)
    plt.tight_layout()

    filename = column.replace(" ", "_")
    plt.savefig(f"Categorical_Analysis/{filename}_churn_rate.png")
    plt.close()

print("\nCategorical analysis completed!")
print("Charts saved in Categorical_Analysis folder.")
