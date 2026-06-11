import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from dataclasses import dataclass
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

transformers = [
    (
        "impute_all_numeric",
        Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
    )
]

xgb_baseline = xgb.XGBClassifier(
    objective="multi:softprob",
    eval_metric="mlogloss",
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42,
    use_label_encoder=False,
    n_jobs=-1,
)

preprocessor = ColumnTransformer(
    transformers=transformers,
    remainder="drop",
    verbose_feature_names_out=False,
    sparse_threshold=0,
 )

pipeline_xgb = Pipeline(steps=[("prep", preprocessor), ("model", xgb_baseline)])

@dataclass
class Metric:
    face_height: float
    face_width: float
    forehead_height: float
    forehead_width: float
    jaw_width: float
    skeleton_angle_l: float
    skeleton_angle_r: float
    chin_angle: float
    chin_width: float
    forehead_angle: float

    def __post_init__(self):
        self.skeleton_angle_l = abs(self.skeleton_angle_l)
        self.skeleton_angle_r = abs(self.skeleton_angle_r)
        self.chin_angle = self.chin_angle / 360
        self.forehead_angle = self.forehead_angle / 360


class XGBoostInference:
    def __init__(self, xgboost_path, class_loader_path):
        if not Path(xgboost_path).exists() or not Path(class_loader_path).exists():
            raise FileNotFoundError("One or both of the specified files not found")
        
        self.model = joblib.load(xgboost_path)
        self.class_leader = joblib.load(class_loader_path)

    def _feature_dict(self, metrics: Metric) -> dict:
        return {
            "face_height": metrics.face_height,
            "face_width": metrics.face_width,
            "forehead_height": metrics.forehead_height,
            "forehead_width": metrics.forehead_width,
            "jaw_width": metrics.jaw_width,
            "skeleton_angle_l": metrics.skeleton_angle_l,
            "skeleton_angle_r": metrics.skeleton_angle_r,
            "chin_angle": metrics.chin_angle,
            "chin_width": metrics.chin_width,
            "forehead_angle": metrics.forehead_angle,
        }

    def inference(self, metrics: Metric):
        feature_dict = self._feature_dict(metrics)

        if hasattr(self.model, "feature_names_in_"):
            columns = list(self.model.feature_names_in_)
            row = [feature_dict.get(col, np.nan) for col in columns]
        else:
            columns = list(feature_dict.keys())
            row = [feature_dict[col] for col in columns]

        X = pd.DataFrame([row], columns=columns)

        if isinstance(self.model, xgb.Booster):
            prediction = self.model.predict(xgb.DMatrix(X))
        else:
            prediction = self.model.predict(X)

        if prediction.ndim > 1:
            prediction = np.argmax(prediction, axis=1)

        predicted_class = self.class_leader.inverse_transform([int(prediction[0])])
        return predicted_class[0]