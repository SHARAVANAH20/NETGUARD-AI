import joblib
from pathlib import Path

model_path = Path("models/random_forest_model.pkl")

if not model_path.exists():
    print("Model file not found!")
else:
    model = joblib.load(model_path)

    print("Model loaded successfully!")
    print("Model type:", type(model).__name__)
    print("Expected features:", model.n_features_in_)
    print("Classes:", model.classes_)