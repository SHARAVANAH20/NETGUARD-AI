
import pandas as pd
import numpy as np
from pathlib import Path


INPUT_FILE = Path("data/live/live_flows.csv")
OUTPUT_FILE = Path("data/live/live_features.csv")


def extract_features():

    if not INPUT_FILE.exists():
        print("Live flow CSV not found.")
        return

    df = pd.read_csv(INPUT_FILE)

    if df.empty:
        print("No live flow records available.")
        return

    features = pd.DataFrame()

    # Basic flow information
    features["Destination Port"] = df["Destination Port"]
    features["Flow Duration"] = (
        df["Duration Seconds"] * 1_000_000
    )

    # Packet counts
    features["Total Fwd Packets"] = df["Forward Packets"]
    features["Total Backward Packets"] = df["Backward Packets"]

    # Average packet length
    features["Average Packet Size"] = df["Average Packet Size"]

    # Total bytes
    features["Total Bytes"] = df["Total Bytes"]

    # Flow rates
    duration_seconds = df["Duration Seconds"].replace(0, np.nan)

    features["Flow Bytes/s"] = (
        df["Total Bytes"] / duration_seconds
    ).fillna(0)

    features["Flow Packets/s"] = (
        df["Total Packets"] / duration_seconds
    ).fillna(0)

    # Packet direction ratios
    features["Down/Up Ratio"] = (
        df["Backward Packets"] /
        df["Forward Packets"].replace(0, np.nan)
    ).fillna(0)

    # Source and destination information
    features["Source IP"] = df["Source IP"]
    features["Destination IP"] = df["Destination IP"]
    features["Source Port"] = df["Source Port"]
    features["Protocol"] = df["Protocol"]

    features.replace(
        [np.inf, -np.inf],
        0,
        inplace=True
    )

    features.to_csv(OUTPUT_FILE, index=False)

    print("\n========== LIVE FEATURE EXTRACTION ==========")

    print("Input flow records:", len(df))
    print("Extracted feature columns:", len(features.columns))
    print("Output file:", OUTPUT_FILE)

    print("\nFirst 5 extracted records:")
    print(features.head().to_string(index=False))

    print("\n========== EXTRACTION COMPLETE ==========")


if __name__ == "__main__":
    extract_features()
