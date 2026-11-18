
import pandas as pd
import numpy as np
from pathlib import Path

# 1. Set the path to your downloaded CSV

file_path = Path(
    "data/raw/MachineLearningCSV/MachineLearningCVE/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
)


if not file_path.exists():
    raise FileNotFoundError(
        f"Dataset not found: {file_path}\n"
        "Check the filename and folder."
    )

# 2. Load a manageable sample for initial exploration
df = pd.read_csv(file_path, nrows=10000, low_memory=False)

# 3. Clean accidental spaces in column names and labels
df.columns = df.columns.str.strip()

if "Label" not in df.columns:
    raise ValueError(
        "Could not find 'Label'. "
        f"Available columns: {df.columns.tolist()}"
    )

df["Label"] = df["Label"].astype("string").str.strip()

print("\n--- DATASET OVERVIEW ---")
print(f"Rows loaded: {len(df):,}")
print(f"Number of columns: {len(df.columns)}")

print("\n--- FIRST FIVE ROWS ---")
print(df.head())

print("\n--- COLUMN NAMES ---")
for column in df.columns:
    print(column)

print("\n--- DATA TYPES ---")
print(df.dtypes.value_counts())

print("\n--- LABEL COUNTS ---")
print(df["Label"].value_counts(dropna=False))

print("\n--- MISSING VALUES: TOP 15 COLUMNS ---")
print(df.isnull().sum().sort_values(ascending=False).head(15))

print("\n--- DUPLICATE ROWS ---")
print(df.duplicated().sum())

print("\n--- NUMERIC SUMMARY ---")
print(df.describe().T.head(15))
