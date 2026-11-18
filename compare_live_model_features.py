import pandas as pd
import joblib

model = joblib.load("models/random_forest_model.pkl")

live_data = pd.read_csv("data/live/live_flows_v2.csv")

model_features = list(model.feature_names_in_)
live_columns = list(live_data.columns)

available = [
    feature for feature in model_features
    if feature in live_columns
]

missing = [
    feature for feature in model_features
    if feature not in live_columns
]

print("\n========== FEATURE COMPARISON ==========")
print("Model requires:", len(model_features))
print("Live data contains:", len(live_columns))
print("Matching model features:", len(available))
print("Missing model features:", len(missing))

print("\n---------- AVAILABLE FEATURES ----------")
for feature in available:
    print(feature)

print("\n---------- MISSING FEATURES ----------")
for feature in missing:
    print(feature)