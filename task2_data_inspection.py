
import pandas as pd

# Step 1: Load dataset
df = pd.read_csv("customer_churn_dataset-testing-master.csv")

# Step 2: Dataset dimensions
print("\n========== DATASET SHAPE ==========")
print("Rows and columns:", df.shape)

# Step 3: Column names
print("\n========== COLUMN NAMES ==========")
print(df.columns.tolist())

# Step 4: Data types and non-null counts
print("\n========== DATASET INFO ==========")
df.info()

# Step 5: Unique values in each column
print("\n========== UNIQUE VALUES ==========")
for column in df.columns:
    print(f"{column}: {df[column].nunique()} unique values")

# Step 6: Missing values
print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())
print("Total missing values:", df.isnull().sum().sum())

# Step 7: Duplicate records
print("\n========== DUPLICATE RECORDS ==========")
print("Total duplicates:", df.duplicated().sum())

# Step 8: Numerical summary
print("\n========== NUMERICAL SUMMARY ==========")
print(df.describe())

# Step 9: Churn distribution
print("\n========== CHURN DISTRIBUTION ==========")
print(df["Churn"].value_counts().sort_index())

# Step 10: Churn percentages
print("\n========== CHURN PERCENTAGES ==========")
print((df["Churn"].value_counts(normalize=True).sort_index() * 100).round(2))

# Step 11: Check target classes
print("\n========== TARGET CLASS CHECK ==========")
print("Target classes:", sorted(df["Churn"].dropna().unique()))
