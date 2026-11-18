
import pandas as pd
import numpy as np

file_path = "data/live/live_features.csv"

df = pd.read_csv(file_path)

print("\n========== LIVE FEATURE VALIDATION ==========")

print("Dataset shape:", df.shape)

numeric_df = df.select_dtypes(include=[np.number])

print("\nMissing values:")
print(df.isnull().sum())

print("\nInfinite values:")
print(
    np.isinf(numeric_df).sum()
)

print("\nNegative values:")
print(
    (numeric_df < 0).sum()
)

print("\nZero-duration records:")
print(
    (df["Flow Duration"] == 0).sum()
)

print("\nFeature statistics:")
print(numeric_df.describe().to_string())

print("\n========== VALIDATION COMPLETE ==========")
