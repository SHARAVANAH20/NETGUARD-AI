
import pandas as pd
import numpy as np
import joblib

from pathlib import Path

# -------------------------------------------------
# 1. LOAD TRAINED RANDOM FOREST MODEL
# -------------------------------------------------

model_path = Path("models")

model = joblib.load(
    model_path / "random_forest_model.pkl"
)

feature_columns = joblib.load(
    model_path / "feature_columns.pkl"
)

print("Random Forest model loaded successfully!")

# -------------------------------------------------
# 2. LOAD TEST DATA
# -------------------------------------------------

data_path = Path("data/processed")

X_test = pd.read_csv(data_path / "X_test.csv")
y_test = pd.read_csv(data_path / "y_test.csv").iloc[:, 0]

# Maintain the exact feature order used during training
X_test = X_test[feature_columns]

print("\nTest dataset loaded successfully!")
print("Number of test records:", len(X_test))
print("Number of features:", len(feature_columns))

# -------------------------------------------------
# 3. CHECK DATA
# -------------------------------------------------

if X_test.isnull().any().any():
    raise ValueError("Missing values found in test data.")

if not np.isfinite(X_test.to_numpy()).all():
    raise ValueError("Invalid numerical values found.")

# -------------------------------------------------
# 4. MAKE PREDICTIONS
# -------------------------------------------------

predictions = model.predict(X_test)

attack_class_index = list(model.classes_).index(1)

attack_probabilities = model.predict_proba(X_test)[
    :, attack_class_index
]

print("\nPredictions generated successfully!")

# -------------------------------------------------
# 5. CREATE RESULTS TABLE
# -------------------------------------------------

results = pd.DataFrame({
    "Actual": y_test.map({
        0: "BENIGN",
        1: "ATTACK"
    }).to_numpy(),

    "Predicted": pd.Series(predictions).map({
        0: "BENIGN",
        1: "ATTACK"
    }).to_numpy(),

    "Attack_Probability": np.round(
        attack_probabilities, 4
    )
})

# -------------------------------------------------
# 6. DISPLAY SAMPLE PREDICTIONS
# -------------------------------------------------

print("\n--- SAMPLE NETWORK TRAFFIC PREDICTIONS ---")

print(results.head(10).to_string(index=False))

# -------------------------------------------------
# 7. PREDICTION SUMMARY
# -------------------------------------------------

benign_count = int(np.sum(predictions == 0))
attack_count = int(np.sum(predictions == 1))

print("\n--- PREDICTION SUMMARY ---")

print("Total records analysed:", len(predictions))
print("Benign traffic:", benign_count)
print("Attack traffic:", attack_count)

# -------------------------------------------------
# 8. SAVE PREDICTIONS
# -------------------------------------------------

output_path = data_path / "day6_predictions.csv"

results.to_csv(output_path, index=False)

print("\nPredictions saved successfully!")
print(output_path)

print("\nDAY 6 COMPLETED!")
