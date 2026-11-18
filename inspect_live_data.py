
import pandas as pd

file_path = "data/live/live_flows.csv"

df = pd.read_csv(file_path)

print("\n========== LIVE FLOW DATA INSPECTION ==========")

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
for column in df.columns:
    print("-", column)

print("\nFirst 5 Live Records:")
print(df.head().to_string(index=False))

print("\nMissing Values:")
print(df.isnull().sum())

print("\n========== INSPECTION COMPLETE ==========")
