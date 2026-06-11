import os
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

RANDOM_STATE = 42

base_dir = Path(__file__).resolve().parent
project_root = base_dir.parent
output_path = project_root / "output"
train_section = "training_set"
test_section = "testing_set"


def load_metrics(section: str) -> pd.DataFrame:
    csv_files = list((output_path / section).glob("**/metrics.csv"))
    if not csv_files:
        raise FileNotFoundError(f"Tidak ada metrics.csv ditemukan di: {output_path / section}")

    df = pd.concat((pd.read_csv(f) for f in csv_files), ignore_index=True)
    df["section"] = section
    return df


train_df = load_metrics(train_section)
test_df = load_metrics(test_section)

id_columns = ["class_name", "image_file", "relative_path", "section"]
target_col = "class_name"

candidate_features = [
    col for col in train_df.columns if col not in id_columns and col in test_df.columns
]

X_train_raw = train_df[candidate_features].copy()
X_test_raw = test_df[candidate_features].copy()
y_train_raw = train_df[target_col].astype(str).copy()
y_test_raw = test_df[target_col].astype(str).copy()

for col in candidate_features:
    X_train_raw[col] = pd.to_numeric(X_train_raw[col], errors="coerce")
    X_test_raw[col] = pd.to_numeric(X_test_raw[col], errors="coerce")

valid_feature_columns = [
    col
    for col in candidate_features
    if not (X_train_raw[col].isna().all() and X_test_raw[col].isna().all())
]

X_train_raw = X_train_raw[valid_feature_columns]
X_test_raw = X_test_raw[valid_feature_columns]

angle_columns = [col for col in valid_feature_columns if "angle" in col.lower()]
degree_columns = [col for col in valid_feature_columns if "degree" in col.lower()]

for col in angle_columns:
    X_train_raw[col] = X_train_raw[col].abs()
    X_test_raw[col] = X_test_raw[col].abs()

for col in degree_columns:
    X_train_raw[col] = X_train_raw[col] / 360
    X_test_raw[col] = X_test_raw[col] / 360

if not valid_feature_columns:
    raise ValueError("Tidak ada fitur valid setelah preprocessing.")

transformers = [
    (
        "impute_all_numeric",
        Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]),
        valid_feature_columns,
    )
]

preprocessor = ColumnTransformer(
    transformers=transformers,
    remainder="drop",
    verbose_feature_names_out=False,
    sparse_threshold=0,
)

label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train_raw)

unseen_labels = sorted(set(y_test_raw.unique()) - set(label_encoder.classes_))
if unseen_labels:
    raise ValueError(f"Ada label testing yang tidak ada di training: {unseen_labels}")

xgb_baseline = xgb.XGBClassifier(
    objective="multi:softprob",
    eval_metric="mlogloss",
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=RANDOM_STATE,
    use_label_encoder=False,
    n_jobs=-1,
)

pipeline_xgb = Pipeline(steps=[("prep", preprocessor), ("model", xgb_baseline)])

print("Training XGBoost baseline...")
pipeline_xgb.fit(X_train_raw, y_train)

print("Evaluating on test set...")
y_pred = pipeline_xgb.predict(X_test_raw)
acc = accuracy_score(y_test_raw, label_encoder.inverse_transform(y_pred))
f1w = f1_score(y_test_raw, label_encoder.inverse_transform(y_pred), average="weighted")
print(f"Test Accuracy: {acc:.4f} | Test F1 (weighted): {f1w:.4f}")
print("\nClassification report:")
print(classification_report(y_test_raw, label_encoder.inverse_transform(y_pred), digits=4))

models_dir = project_root / "deployments" / "models"
models_dir.mkdir(parents=True, exist_ok=True)

model_path = models_dir / "xgboost_model_py38.joblib"
label_path = models_dir / "label_encoder.joblib"

import joblib

joblib.dump(pipeline_xgb, model_path)
joblib.dump(label_encoder, label_path)

print(f"Saved model pipeline to: {model_path}")
print(f"Saved label encoder to: {label_path}")
