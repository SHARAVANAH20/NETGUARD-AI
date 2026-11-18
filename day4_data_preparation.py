
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

# -------------------------------------------------
# 1. LOAD THE DATASET
# -------------------------------------------------

file_path = Path(
    "data/raw/MachineLearningCSV/MachineLearningCVE/"
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
)

if not file_path.exists():
    raise FileNotFoundError(f"Dataset not found: {file_path}")

df = pd.read_csv(file_path, low_memory=False)
df.columns = df.columns.str.strip()

df.columns = df.columns.str.strip()
print("\nRAW LABEL DISTRIBUTION:")
print(df["Label"].value_counts())

df["Target"] = np.where(
    df["Label"].str.upper() == "BENIGN", 0, 1
)

print("\nBINARY TARGET DISTRIBUTION:")
print(df["Target"].value_counts())


if "Label" not in df.columns:
    raise ValueError("Label column not found in the dataset.")

df["Label"] = df["Label"].astype("string").str.strip()

print("\n--- ORIGINAL DATASET ---")
print("Shape:", df.shape)

# -------------------------------------------------
# 2. REMOVE ROWS WITHOUT LABELS AND DUPLICATES
# -------------------------------------------------

df = df.dropna(subset=["Label"]).copy()

before = len(df)
df = df.drop_duplicates().copy()

print("Duplicate rows removed:", before - len(df))

# -------------------------------------------------
# 3. CREATE BINARY TARGET
# -------------------------------------------------

df["Target"] = np.where(
    df["Label"].str.upper() == "BENIGN",
    0,
    1
)

print("\n--- TARGET DISTRIBUTION ---")
print(df["Target"].value_counts())
print("0 = BENIGN | 1 = ATTACK")

# -------------------------------------------------
# 4. REMOVE IDENTIFIER COLUMNS
# -------------------------------------------------

identifier_columns = [
    col for col in df.columns
    if any(term in col.lower() for term in [
        "flow id",
        "source ip",
        "src ip",
        "destination ip",
        "dst ip",
        "timestamp"
    ])
]

print("\n--- IDENTIFIER COLUMNS REMOVED ---")
print(identifier_columns)

df = df.drop(
    columns=identifier_columns + ["Label"],
    errors="ignore"
)

# -------------------------------------------------
# 5. CONVERT FEATURES TO NUMERIC
# -------------------------------------------------

for col in df.columns:
    if col != "Target":
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Replace infinite values with missing values
feature_columns = [
    col for col in df.columns if col != "Target"
]

df[feature_columns] = df[feature_columns].replace(
    [np.inf, -np.inf],
    np.nan
)

# Remove columns that contain no usable numeric values
empty_columns = [
    col for col in feature_columns
    if df[col].isna().all()
]

print("\n--- EMPTY COLUMNS REMOVED ---")
print(empty_columns)

df = df.drop(columns=empty_columns)

feature_columns = [
    col for col in df.columns if col != "Target"
]

# -------------------------------------------------
# 6. SEPARATE FEATURES AND TARGET
# -------------------------------------------------

X = df[feature_columns]
y = df["Target"]

# Remove rows with no usable feature values at all
valid_rows = X.notna().any(axis=1)
X = X.loc[valid_rows].copy()
y = y.loc[valid_rows].copy()

print("\n--- FEATURE INFORMATION ---")
print("Number of features:", X.shape[1])
print("Number of usable rows:", X.shape[0])

# -------------------------------------------------
# 7. SPLIT INTO TRAINING AND TESTING DATA
# -------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n--- DATA SPLIT ---")
print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))

# -------------------------------------------------
# 8. HANDLE MISSING VALUES
# -------------------------------------------------

imputer = SimpleImputer(strategy="median")

X_train = pd.DataFrame(
    imputer.fit_transform(X_train),
    columns=X.columns,
    index=X_train.index
)

X_test = pd.DataFrame(
    imputer.transform(X_test),
    columns=X.columns,
    index=X_test.index
)

print("\n--- MISSING VALUES AFTER IMPUTATION ---")
print("Training missing values:", X_train.isnull().sum().sum())
print("Testing missing values:", X_test.isnull().sum().sum())

# -------------------------------------------------
# 9. CHECK CLASS DISTRIBUTION
# -------------------------------------------------

print("\n--- TRAINING CLASS DISTRIBUTION ---")
print(y_train.value_counts())

print("\n--- TESTING CLASS DISTRIBUTION ---")
print(y_test.value_counts())

# -------------------------------------------------
# 10. SAVE PREPARED DATA
# -------------------------------------------------

output_folder = Path("data/processed")
output_folder.mkdir(parents=True, exist_ok=True)

X_train.to_csv(output_folder / "X_train.csv", index=False)
X_test.to_csv(output_folder / "X_test.csv", index=False)

y_train.to_csv(output_folder / "y_train.csv", index=False)
y_test.to_csv(output_folder / "y_test.csv", index=False)

print("\n--- DAY 4 COMPLETED ---")
print("Prepared datasets saved in:", output_folder)


print("\n--- BINARY TARGET DISTRIBUTION ---")
print(df["Target"].value_counts())