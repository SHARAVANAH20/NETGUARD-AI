import pandas as pd
import joblib

from pathlib import Path
from sklearn.ensemble import RandomForestClassifier


print("Loading training data...")

X_train = pd.read_csv("data/processed/X_train.csv")
y_train = pd.read_csv("data/processed/y_train.csv")

# Load live network traffic
live_data = pd.read_csv("data/live/live_flows_v2.csv")

# Identify features available in both datasets
live_features = [
    column for column in X_train.columns
    if column in live_data.columns
]

print("Available matching features:", len(live_features))

if len(live_features) != 41:
    raise ValueError(
        f"Expected 41 matching features, found {len(live_features)}"
    )

# Train a separate model using only matching features
X_live_train = X_train[live_features]

print("Training live Random Forest model...")

live_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)

live_model.fit(X_live_train, y_train.values.ravel())

# Save model and feature names
Path("models").mkdir(exist_ok=True)

joblib.dump(
    {
        "model": live_model,
        "features": live_features
    },
    "models/live_random_forest_model.pkl"
)

print("\nLive AI model trained successfully!")
print("Features:", len(live_features))
print("Classes:", live_model.classes_)
print("Saved to: models/live_random_forest_model.pkl")
