
import joblib
import pandas as pd
from pathlib import Path

# LOAD TRAINED MODEL
model_path = Path("models/random_forest_model.pkl")

model = joblib.load(model_path)

# LOAD TRAINING DATA
train_path = Path("data/processed/X_train.csv")

X_train = pd.read_csv(train_path)

print("\n========== NETGUARD AI MODEL INSPECTION ==========")

print("\nModel Type:")
print(type(model).__name__)

print("\nNumber of Expected Features:")
print(model.n_features_in_)

print("\nNumber of Training Features:")
print(len(X_train.columns))

print("\nFeature Names Expected by Model:")

for index, feature in enumerate(model.feature_names_in_, start=1):
    print(f"{index}. {feature}")

print("\nDo Training Features Match Model?")

print(list(X_train.columns) == list(model.feature_names_in_))

print("\n========== INSPECTION COMPLETE ==========")
print("\n========== MODEL CLASS CHECK ==========")

print("Model classes:", model.classes_)

print("Number of classes:", len(model.classes_))

print("\n========== CLASS CHECK COMPLETE ==========")