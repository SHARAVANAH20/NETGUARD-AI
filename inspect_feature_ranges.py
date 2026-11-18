
import pandas as pd
from pathlib import Path

df = pd.read_csv("data/processed/X_train.csv")

selected_features = [
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Total Length of Fwd Packets",
    "Total Length of Bwd Packets",
    "Fwd Packet Length Mean",
    "Bwd Packet Length Mean",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Flow IAT Mean",
    "Fwd IAT Mean",
    "Bwd IAT Mean",
    "SYN Flag Count",
    "ACK Flag Count",
    "RST Flag Count",
    "Average Packet Size",
    "Active Mean",
    "Idle Mean"
]

print("\n========== FEATURE RANGE INSPECTION ==========")

print(
    df[selected_features].describe().T[
        ["mean", "std", "min", "50%", "max"]
    ].to_string()
)

print("\n========== INSPECTION COMPLETE ==========")
