
import os
import pandas as pd

print("=" * 65)
print("TASK 19 - CUSTOMER RETENTION RECOMMENDATIONS")
print("=" * 65)

# 1. Load risk analysis results from Task 18
file_path = os.path.join(
    "Churn_Risk_Results",
    "customer_risk_analysis.csv"
)

df = pd.read_csv(file_path)

# 2. Define business recommendations
def recommend_action(row):
    actions = []
    risk = row["Risk_Level"]

    if risk == "High":
        actions.append(
            "Prioritize customer outreach and offer personalized retention support."
        )
    elif risk == "Medium":
        actions.append(
            "Monitor customer engagement and offer targeted incentives if appropriate."
        )
    else:
        actions.append(
            "Maintain service quality and encourage continued engagement."
        )

    # Behavior-based recommendations
    if "Support Calls" in df.columns:
        if row["Support Calls"] >= 5:
            actions.append(
                "Review unresolved support issues and improve service assistance."
            )

    if "Payment Delay" in df.columns:
        if row["Payment Delay"] >= 15:
            actions.append(
                "Offer payment reminders and clarify billing difficulties."
            )

    if "Usage Frequency" in df.columns:
        if row["Usage Frequency"] <= 5:
            actions.append(
                "Encourage product usage with onboarding tips and relevant features."
            )

    if "Tenure" in df.columns:
        if row["Tenure"] <= 6:
            actions.append(
                "Provide onboarding support and early-stage customer guidance."
            )

    return " ".join(actions)

# 3. Generate recommendations
df["Retention_Recommendation"] = df.apply(
    recommend_action,
    axis=1
)

# 4. Display examples by risk category
for risk in ["High", "Medium", "Low"]:
    subset = df[df["Risk_Level"] == risk]

    print(f"\n{'=' * 20} {risk.upper()} RISK {'=' * 20}")
    print(f"Number of customers: {len(subset)}")

    if not subset.empty:
        columns_to_show = [
            col for col in [
                "Churn_Probability_Percent",
                "Risk_Level",
                "Support Calls",
                "Payment Delay",
                "Usage Frequency",
                "Tenure",
                "Retention_Recommendation"
            ]
            if col in subset.columns
        ]

        print(subset[columns_to_show].head(5).to_string(index=False))

# 5. Save recommendations
output_dir = "Retention_Recommendations"
os.makedirs(output_dir, exist_ok=True)

df.to_csv(
    os.path.join(output_dir, "customer_retention_recommendations.csv"),
    index=False
)

summary = (
    df.groupby("Risk_Level")
    .agg(
        Customer_Count=("Risk_Level", "size"),
        Average_Churn_Probability=("Churn_Probability_Percent", "mean")
    )
    .reset_index()
)

summary.to_csv(
    os.path.join(output_dir, "retention_summary.csv"),
    index=False
)

print(f"\nResults saved in: {output_dir}")
print("\nTASK 19 COMPLETED")
