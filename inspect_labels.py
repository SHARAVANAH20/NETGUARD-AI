
import pandas as pd

# Load the original training labels
y_train = pd.read_csv("data/processed/y_train.csv")

print("\n========== TRAINING LABEL INSPECTION ==========")

print("Label column names:")
print(y_train.columns.tolist())

print("\nLabel distribution:")
print(y_train.iloc[:, 0].value_counts())

print("\nUnique labels:")
print(y_train.iloc[:, 0].unique())

print("\n========== INSPECTION COMPLETE ==========")
