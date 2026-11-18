
import pandas as pd
import joblib
import matplotlib.pyplot as plt

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

print("\nRandom Forest model loaded successfully!")

# -------------------------------------------------
# 2. EXTRACT FEATURE IMPORTANCE
# -------------------------------------------------

importances = model.feature_importances_

if len(importances) != len(feature_columns):
    raise ValueError(
        "Feature count does not match model importance values."
    )

feature_importance = pd.DataFrame({
    "Feature": feature_columns,
    "Importance": importances
})

# -------------------------------------------------
# 3. SORT FEATURES BY IMPORTANCE
# -------------------------------------------------

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\n--- TOP 20 IMPORTANT FEATURES ---")

print(
    feature_importance.head(20).to_string(index=False)
)

# -------------------------------------------------
# 4. SAVE FEATURE IMPORTANCE RESULTS
# -------------------------------------------------

output_path = Path("data/processed")

output_path.mkdir(parents=True, exist_ok=True)

csv_path = output_path / "feature_importance.csv"

feature_importance.to_csv(
    csv_path,
    index=False
)

print("\nFeature importance saved:")
print(csv_path)

# -------------------------------------------------
# 5. CREATE TOP 20 FEATURE GRAPH
# -------------------------------------------------

top_features = feature_importance.head(20).sort_values(
    by="Importance",
    ascending=True
)

plt.figure(figsize=(12, 8))

plt.barh(
    top_features["Feature"],
    top_features["Importance"]
)

plt.xlabel("Feature Importance Score")
plt.ylabel("Network Traffic Features")

plt.title(
    "Top 20 Important Features - Random Forest"
)

plt.tight_layout()

# -------------------------------------------------
# 6. SAVE GRAPH
# -------------------------------------------------

graph_path = output_path / "feature_importance.png"

plt.savefig(
    graph_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nGraph saved successfully:")
print(graph_path)

# -------------------------------------------------
# 7. DISPLAY MOST IMPORTANT FEATURE
# -------------------------------------------------

most_important = feature_importance.iloc[0]

print("\n--- MOST IMPORTANT FEATURE ---")

print("Feature:", most_important["Feature"])

print(
    "Importance:",
    round(most_important["Importance"], 6)
)

print("\nDAY 8 COMPLETED!")
