
import pandas as pd

train = pd.read_csv("data/processed/X_train.csv")
live = pd.read_csv("data/live/live_features.csv")

features = [
    "Destination Port",
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Average Packet Size",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Down/Up Ratio"
]

print("\n========== LIVE VS TRAINING COMPARISON ==========")

for feature in features:

    if feature not in train.columns or feature not in live.columns:
        print(f"\n{feature}: Not available in both datasets")
        continue

    print(f"\nFeature: {feature}")

    print(
        "Training range:",
        train[feature].min(),
        "to",
        train[feature].max()
    )

    print(
        "Live range:",
        live[feature].min(),
        "to",
        live[feature].max()
    )

print("\n========== COMPARISON COMPLETE ==========")
