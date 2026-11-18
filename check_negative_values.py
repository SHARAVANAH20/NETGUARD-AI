
import pandas as pd

df = pd.read_csv("data/processed/X_train.csv")

features = [
    "Flow Duration",
    "Flow Bytes/s",
    "Flow Packets/s"
]

print("\n========== NEGATIVE VALUE CHECK ==========")

for feature in features:
    negative_count = (df[feature] < 0).sum()

    print(f"\nFeature: {feature}")
    print(f"Negative records: {negative_count}")
    print(f"Percentage: {negative_count / len(df) * 100:.4f}%")

print("\n========== CHECK COMPLETE ==========")
