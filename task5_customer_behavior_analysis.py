
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

df = pd.read_csv("customer_churn_cleaned.csv")

os.makedirs("Behavior_Charts", exist_ok=True)
sns.set_theme(style="whitegrid")

# 1. Support Calls vs Churn
# 2. Payment Delay vs Churn
# 3. Usage Frequency vs Churn
# 4. Tenure vs Churn
# 5. Last Interaction vs Churn

features = [
    "Support Calls",
    "Payment Delay",
    "Usage Frequency",
    "Tenure",
    "Last Interaction"
]

for feature in features:
    plt.figure(figsize=(9, 5))

    sns.histplot(
        data=df,
        x=feature,
        hue="Churn",
        bins=30,
        stat="density",
        common_norm=False,
        element="step",
        fill=False
    )

    plt.title(f"{feature} Distribution by Churn")
    plt.tight_layout()
    plt.savefig(
        f"Behavior_Charts/{feature.replace(' ', '_')}_distribution.png"
    )
    plt.close()

# Analyze average values for churned vs retained customers
print("\n========== CUSTOMER BEHAVIOR COMPARISON ==========")

comparison = df.groupby("Churn")[
    features + ["Total Spend"]
].mean().round(2)

comparison.index = comparison.index.map({
    0: "Stayed",
    1: "Churned"
})

print(comparison)

# Churn rates by contract length
print("\n========== CHURN RATE BY CONTRACT LENGTH ==========")
print(
    (df.groupby("Contract Length", observed=True)["Churn"].mean() * 100)
    .round(2)
    .sort_values(ascending=False)
)

# Churn rates by subscription type
print("\n========== CHURN RATE BY SUBSCRIPTION TYPE ==========")
print(
    (df.groupby("Subscription Type", observed=True)["Churn"].mean() * 100)
    .round(2)
    .sort_values(ascending=False)
)

print("\nBehavior analysis completed!")
print("Charts saved in Behavior_Charts folder.")
