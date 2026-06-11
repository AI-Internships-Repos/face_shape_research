from pathlib import Path
import joblib

base = Path("d:/Thoriq/Work/MadeByHumans/Research Face Shape/deployments/models")
model = base / "xgboost_model.joblib"
label = base / "label_encoder.joblib"

print("Model exists:", model.exists())
print("Label exists:", label.exists())

loaded_model = joblib.load(model)
loaded_label = joblib.load(label)

print("Loaded model type:", type(loaded_model))
print("Loaded label type:", type(loaded_label))
