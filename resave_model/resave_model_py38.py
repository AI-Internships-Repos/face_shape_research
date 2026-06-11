from pathlib import Path
import joblib
import numpy as np
import xgboost as xgb

print("Python:", __import__("sys").version)
print("numpy:", np.__version__)
print("xgboost:", xgb.__version__)

base_dir = Path(__file__).resolve().parent.parent
model_in = base_dir / "models" / "xgb_baseline_pipeline.joblib"
model_out = base_dir / "models" / "xgboost_model_py38.joblib"

if not model_in.exists():
    raise FileNotFoundError(f"Model not found: {model_in}")

model = joblib.load(model_in)
joblib.dump(model, model_out)
print("Saved:", model_out)
