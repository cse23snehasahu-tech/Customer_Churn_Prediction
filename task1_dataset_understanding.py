
import pandas as pd

# Step 1: Load the dataset
df = pd.read_csv("customer_churn_dataset-testing-master.csv")

# Step 2: Display the first five records
print("\n========== FIRST 5 RECORDS ==========")
print(df.head())

# Step 3: Display the last five records
print("\n========== LAST 5 RECORDS ==========")
print(df.tail())

# Step 4: Display five random records
print("\n========== RANDOM 5 RECORDS ==========")
print(df.sample(5, random_state=42))

# Step 5: Display dataset dimensions
print("\n========== DATASET SHAPE ==========")
print("Total customers:", df.shape[0])
print("Total columns:", df.shape[1])

# Step 6: Display column names
print("\n========== COLUMN NAMES ==========")
print(df.columns.tolist())

# Step 7: Display data types
print("\n========== DATA TYPES ==========")
print(df.dtypes)

# Step 8: Identify the target variable
print("\n========== TARGET VARIABLE ==========")
print("Target column: Churn")
print("Target meaning: 1 = Customer Churned, 0 = Customer Stayed")

# Step 9: Display a concise dataset summary
print("\n========== DATASET INFORMATION ==========")
df.info()
