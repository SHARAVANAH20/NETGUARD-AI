
import pandas as pd
import numpy as np

file_path = "data/live/live_flows_v2.csv"

df = pd.read_csv(file_path)

print("\n========== LIVE V2 DATA INSPECTION ==========")

print("\nDataset shape:")
print(df.shape)

print("\nAvailable columns:")
for column in df.columns:
    print("-", column)

print("\nMissing values:")
print(df.isnull().sum())

print("\nInfinite values:")
print(
    np.isinf(df.select_dtypes(include=[np.number])).sum()
)

print("\nFirst 5 records:")
print(df.head().to_string(index=False))

print("\n========== INSPECTION COMPLETE ==========")
