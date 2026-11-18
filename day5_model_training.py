
import pandas as pd
import numpy as np
import joblib

from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# -------------------------------------------------
# 1. LOAD PREPARED DATA
# -------------------------------------------------

data_path = Path("data/processed")

X_train = pd.read_csv(data_path / "X_train.csv")
X_test = pd.read_csv(data_path / "X_test.csv")

y_train = pd.read_csv(data_path / "y_train.csv").iloc[:, 0]
y_test = pd.read_csv(data_path / "y_test.csv").iloc[:, 0]

# Check for invalid data
if X_train.isnull().any().any() or X_test.isnull().any().any():
    raise ValueError("Missing values found in features.")

if not np.isfinite(X_train.to_numpy()).all():
    raise ValueError("Non-finite values found in training features.")

if not np.isfinite(X_test.to_numpy()).all():
    raise ValueError("Non-finite values found in testing features.")

if y_train.nunique() < 2 or y_test.nunique() < 2:
    raise ValueError(
        "Both BENIGN and ATTACK classes must be present "
        "in training and testing data."
    )

print("\n--- DATASET INFORMATION ---")
print("Training shape:", X_train.shape)
print("Testing shape:", X_test.shape)
print("Number of features:", X_train.shape[1])

print("\nTraining labels:")
print(y_train.value_counts())

# -------------------------------------------------
# 2. CREATE RANDOM FOREST MODEL
# -------------------------------------------------

print("\n--- TRAINING RANDOM FOREST ---")

random_forest = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

random_forest.fit(X_train, y_train)

rf_predictions = random_forest.predict(X_test)

# -------------------------------------------------
# 3. CREATE LOGISTIC REGRESSION MODEL
# -------------------------------------------------

print("\n--- TRAINING LOGISTIC REGRESSION ---")

logistic_model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    ))
])

logistic_model.fit(X_train, y_train)

lr_predictions = logistic_model.predict(X_test)

# -------------------------------------------------
# 4. EVALUATION FUNCTION
# -------------------------------------------------

def evaluate_model(name, actual, predicted):

    print(f"\n{'=' * 45}")
    print(f"MODEL: {name}")
    print(f"{'=' * 45}")

    print("Accuracy:", round(
        accuracy_score(actual, predicted), 4
    ))

    print("Precision:", round(
        precision_score(actual, predicted, zero_division=0), 4
    ))

    print("Recall:", round(
        recall_score(actual, predicted, zero_division=0), 4
    ))

    print("F1 Score:", round(
        f1_score(actual, predicted, zero_division=0), 4
    ))

    print("\nConfusion Matrix:")
    print(confusion_matrix(actual, predicted, labels=[0, 1]))

    print("\nClassification Report:")
    print(classification_report(
        actual,
        predicted,
        labels=[0, 1],
        target_names=["BENIGN", "ATTACK"],
        zero_division=0
    ))

# -------------------------------------------------
# 5. EVALUATE BOTH MODELS
# -------------------------------------------------

evaluate_model(
    "RANDOM FOREST",
    y_test,
    rf_predictions
)

evaluate_model(
    "LOGISTIC REGRESSION",
    y_test,
    lr_predictions
)

# -------------------------------------------------
# 6. SAVE TRAINED MODELS
# -------------------------------------------------

model_path = Path("models")
model_path.mkdir(parents=True, exist_ok=True)

joblib.dump(
    random_forest,
    model_path / "random_forest_model.pkl"
)

joblib.dump(
    logistic_model,
    model_path / "logistic_regression_model.pkl"
)

# Save feature names in the exact order used during training
joblib.dump(
    list(X_train.columns),
    model_path / "feature_columns.pkl"
)

print("\n--- MODELS SAVED SUCCESSFULLY ---")
print("models/random_forest_model.pkl")
print("models/logistic_regression_model.pkl")
print("models/feature_columns.pkl")
print("\n--- TRAINING LABEL COUNTS ---")
print(y_train.value_counts())

print("\n--- TESTING LABEL COUNTS ---")
print(y_test.value_counts())

print("\n--- UNIQUE TRAINING LABELS ---")
print(y_train.unique())

print("\n--- UNIQUE TESTING LABELS ---")
print(y_test.unique())


print("\nDAY 5 COMPLETED!")
