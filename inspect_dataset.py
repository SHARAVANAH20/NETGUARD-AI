
import pandas as pd
from pathlib import Path

data_path = Path("data/processed/X_train.csv")

df = pd.read_csv(data_path)

print("\n========== TRAINING DATA INSPECTION ==========")

print("\nDataset Shape:")
print(df.shape)

print("\nFirst 5 Rows:")
print(df.head())

print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum().sum())

print("\nInfinite Values:")
print(
    df.select_dtypes(include="number")
    .isin([float("inf"), float("-inf")])
    .sum()
    .sum()
)

print("\n========== INSPECTION COMPLETE ==========")
