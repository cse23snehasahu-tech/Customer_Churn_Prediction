
import pandas as pd

# Step 1: Load original dataset
df = pd.read_csv("customer_churn_dataset-testing-master.csv")

print("========== ORIGINAL DATA ==========")
print("Original shape:", df.shape)

# Step 2: Remove CustomerID
if "CustomerID" in df.columns:
    df = df.drop(columns=["CustomerID"])
    print("CustomerID removed successfully.")
else:
    print("CustomerID not found.")

# Step 3: Check missing values
print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

# Step 4: Remove duplicate records
duplicate_count = df.duplicated().sum()
print("\nDuplicate records found:", duplicate_count)

if duplicate_count > 0:
    df = df.drop_duplicates()
    print("Duplicate records removed.")

# Step 5: Check data types
print("\n========== DATA TYPES ==========")
print(df.dtypes)

# Step 6: Check categorical values
categorical_columns = df.select_dtypes(
    include=["object", "category"]
).columns

print("\n========== CATEGORICAL VALUES ==========")
for column in categorical_columns:
    print(f"\n{column}:")
    print(df[column].value_counts(dropna=False))

# Step 7: Check numerical ranges
print("\n========== NUMERICAL RANGES ==========")
print(df.select_dtypes(include="number").describe().T)

# Step 8: Verify target variable
print("\n========== TARGET VALIDATION ==========")
if "Churn" in df.columns:
    print(df["Churn"].value_counts(dropna=False))
    print("Valid target values:", df["Churn"].dropna().isin([0, 1]).all())

# Step 9: Save cleaned dataset
output_file = "customer_churn_cleaned.csv"
df.to_csv(output_file, index=False)

print("\n========== CLEANING SUMMARY ==========")
print("Final shape:", df.shape)
print("Missing values remaining:", df.isnull().sum().sum())
print("Duplicate records remaining:", df.duplicated().sum())
print("Saved cleaned dataset as:", output_file)
