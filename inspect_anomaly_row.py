
import pandas as pd

df = pd.read_csv("data/processed/X_train.csv")

# Find records containing negative values
features = [
    "Flow Duration",
    "Flow Bytes/s",
    "Flow Packets/s"
]

negative_rows = df[
    (df[features] < 0).any(axis=1)
]

print("\n========== UNUSUAL TRAINING RECORD ==========")

print("Number of unusual records:", len(negative_rows))

if not negative_rows.empty:
    for index, row in negative_rows.iterrows():
        print(f"\nRecord Index: {index}")

        print("\nNegative feature values:")
        for feature in features:
            if row[feature] < 0:
                print(f"{feature}: {row[feature]}")

        print("\nComplete record:")
        print(row.to_string())

print("\n========== INSPECTION COMPLETE ==========")
