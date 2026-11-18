
import pandas as pd
import numpy as np
import joblib

from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

import matplotlib.pyplot as plt

# -------------------------------------------------
# 1. LOAD PREDICTIONS
# -------------------------------------------------

data_path = Path("data/processed")

results_path = data_path / "day6_predictions.csv"

results = pd.read_csv(results_path)

print("\n--- PREDICTION DATA LOADED ---")
print("Total records:", len(results))

# -------------------------------------------------
# 2. CONVERT LABELS INTO NUMBERS
# -------------------------------------------------

actual = results["Actual"].map({
    "BENIGN": 0,
    "ATTACK": 1
})

predicted = results["Predicted"].map({
    "BENIGN": 0,
    "ATTACK": 1
})

if actual.isnull().any() or predicted.isnull().any():
    raise ValueError("Unexpected labels found in prediction file.")

# -------------------------------------------------
# 3. CALCULATE PERFORMANCE METRICS
# -------------------------------------------------

accuracy = accuracy_score(actual, predicted)

precision = precision_score(
    actual, predicted, zero_division=0
)

recall = recall_score(
    actual, predicted, zero_division=0
)

f1 = f1_score(
    actual, predicted, zero_division=0
)

print("\n--- MODEL PERFORMANCE ---")

print("Accuracy:", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall:", round(recall, 4))
print("F1 Score:", round(f1, 4))

# -------------------------------------------------
# 4. GENERATE CONFUSION MATRIX
# -------------------------------------------------

cm = confusion_matrix(actual, predicted, labels=[0, 1])

tn, fp, fn, tp = cm.ravel()

print("\n--- CONFUSION MATRIX ---")
print(cm)

print("\nTrue Negatives:", tn)
print("False Positives:", fp)
print("False Negatives:", fn)
print("True Positives:", tp)

# -------------------------------------------------
# 5. IDENTIFY INCORRECT PREDICTIONS
# -------------------------------------------------

incorrect = results[
    results["Actual"] != results["Predicted"]
].copy()

print("\n--- INCORRECT PREDICTIONS ---")

print("Total incorrect predictions:", len(incorrect))

print("\nIncorrect prediction details:")
print(incorrect.head(10).to_string(index=False))

# -------------------------------------------------
# 6. SEPARATE FALSE POSITIVES AND FALSE NEGATIVES
# -------------------------------------------------

false_positives = results[
    (results["Actual"] == "BENIGN") &
    (results["Predicted"] == "ATTACK")
].copy()

false_negatives = results[
    (results["Actual"] == "ATTACK") &
    (results["Predicted"] == "BENIGN")
].copy()

print("\nFalse Positives:", len(false_positives))
print("False Negatives:", len(false_negatives))

# -------------------------------------------------
# 7. SAVE INCORRECT PREDICTIONS
# -------------------------------------------------

incorrect_path = data_path / "incorrect_predictions.csv"

incorrect.to_csv(incorrect_path, index=False)

false_positives.to_csv(
    data_path / "false_positives.csv",
    index=False
)

false_negatives.to_csv(
    data_path / "false_negatives.csv",
    index=False
)

print("\nIncorrect predictions saved successfully!")

# -------------------------------------------------
# 8. CLASSIFICATION REPORT
# -------------------------------------------------

print("\n--- DETAILED CLASSIFICATION REPORT ---")

print(classification_report(
    actual,
    predicted,
    labels=[0, 1],
    target_names=["BENIGN", "ATTACK"],
    zero_division=0
))

# -------------------------------------------------
# 9. GENERATE CONFUSION MATRIX GRAPH
# -------------------------------------------------

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["BENIGN", "ATTACK"]
)

display.plot(values_format="d")

plt.title("Random Forest - Confusion Matrix")
plt.tight_layout()

plt.savefig(
    data_path / "confusion_matrix.png",
    dpi=300
)

plt.show()

# -------------------------------------------------
# 10. FINAL STATUS
# -------------------------------------------------

print("\n--- DAY 7 SUMMARY ---")

print("Total records analysed:", len(results))
print("Correct predictions:", len(results) - len(incorrect))
print("Incorrect predictions:", len(incorrect))

print("\nAll analysis files saved in:", data_path)

print("\nDAY 7 COMPLETED!")
