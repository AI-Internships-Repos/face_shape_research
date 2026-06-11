import os
import sys
import platform
import numpy as np
import cv2
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass, field
from typing import Optional
import threading

os.environ["TF_USE_LEGACY_KERAS"] = "1"
platform.processor = lambda: ""

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PREPROCESSING_DIR = PROJECT_ROOT / "dataset_preprocessing_and_extract3d"

if str(PREPROCESSING_DIR) not in sys.path:
    sys.path.insert(0, str(PREPROCESSING_DIR))

import dlib
from retinaface import RetinaFace
from face_parsing.face_parsing import FaceParsingExtractor
from forehead_dense.forehead_dense import forehead_points_dense
try:
    from xgboost_inference import Metric, XGBoostInference
except Exception:
    Metric = None
    XGBoostInference = None

logger = logging.getLogger(__name__)

PREDICTOR_PATH = str(PROJECT_ROOT / "models" / "shape_predictor_68_face_landmarks.dat")
XGB_MODEL_PATH = Path(__file__).resolve().parent / "models" / "xgboost_model.joblib"
XGB_LABEL_PATH = Path(__file__).resolve().parent / "models" / "label_encoder.joblib"

METRIC_NAMES = [
    "face_height", "face_width", "forehead_height", "jaw_width", "chin_width",
    "chin_angle_degree", "forehead_width", "skeleton_angle_left",
    "skeleton_angle_right", "jidat_angle_downward"
]

METRIC_LABELS = {
    "face_height": "Face Height",
    "face_width": "Face Width",
    "forehead_height": "Forehead Height",
    "jaw_width": "Jaw Width",
    "chin_width": "Chin Width",
    "chin_angle_degree": "Chin Angle (°)",
    "forehead_width": "Forehead Width",
    "skeleton_angle_left": "Skeleton Angle L",
    "skeleton_angle_right": "Skeleton Angle R",
    "jidat_angle_downward": "Forehead Angle (↓)",
}


@dataclass
class ExtractionResult:
    index: int
    filename: str
    raw_face: Optional[np.ndarray] = None
    masked_face: Optional[np.ndarray] = None
    metrics: Optional[dict] = None
    predicted_class: Optional[str] = None
    error: Optional[str] = None
    success: bool = False
    used_fallback: bool = False


class ModelCache:
    """Singleton-ish holder for heavy ML models."""
    _instance = None
    _lock = threading.Lock()

    def __init__(self):
        self.hog_detector = dlib.get_frontal_face_detector()
        self.landmark_predictor = dlib.shape_predictor(PREDICTOR_PATH)
        self.face_parser = FaceParsingExtractor()
        self._xgb_infer: Optional[XGBoostInference] = None
        logger.info("Models loaded successfully.")

    @classmethod
    def get(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def get_xgb_infer(self) -> Optional[XGBoostInference]:
        if XGBoostInference is None or Metric is None:
            return None
        if self._xgb_infer is not None:
            return self._xgb_infer
        if not XGB_MODEL_PATH.exists() or not XGB_LABEL_PATH.exists():
            logger.warning("XGBoost model files not found; prediction disabled.")
            return None
        try:
            self._xgb_infer = XGBoostInference(str(XGB_MODEL_PATH), str(XGB_LABEL_PATH))
            return self._xgb_infer
        except Exception as exc:
            logger.warning(f"Failed to load XGBoost model: {exc}")
            return None


# ── Core extraction functions ──

def extract_face(image_path: str) -> Optional[np.ndarray]:
    faces = RetinaFace.extract_faces(img_path=image_path, align=True, expand_face_area=15)
    if faces:
        # RetinaFace already returns RGB faces; keep as-is for consistent pipeline
        return faces[0]
    return None


def extract_face_dlib_only(image_path: str, models: "ModelCache",
                           expand=(0.7, 0.25, 0.25, 0.25)) -> Optional[np.ndarray]:
    """Fallback face crop when RetinaFace fails but dlib can still find a face.

    Detects the face with dlib on the original image, then crops a generously
    expanded region around it (extra on top so the forehead/hairline survives
    parsing) and returns an RGB crop. Cropping — instead of using the full image —
    keeps metrics normalized by a face-sized ``max_dim``, comparable to the
    RetinaFace path.

    Args:
        expand: (top, bottom, left, right) margins as fractions of the dlib box.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = models.hog_detector(gray)
    if len(faces) == 0:
        return None
    face = faces[0]
    h_img, w_img = img.shape[:2]
    bw = face.right() - face.left()
    bh = face.bottom() - face.top()
    et, eb, el, er = expand
    x0 = max(0, int(face.left() - el * bw))
    x1 = min(w_img, int(face.right() + er * bw))
    y0 = max(0, int(face.top() - et * bh))
    y1 = min(h_img, int(face.bottom() + eb * bh))
    crop = img[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)


def process_dlib_landmarks(image_cv2: np.ndarray, models: ModelCache) -> Optional[np.ndarray]:
    gray = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2GRAY)
    faces = models.hog_detector(gray)
    if len(faces) == 0:
        return None
    face = faces[0]
    landmarks = models.landmark_predictor(gray, face)
    if landmarks.num_parts != 68:
        return None
    return np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(68)])


def face_skin_parsing(aligned_face: np.ndarray, models: ModelCache):
    parsing_mask = models.face_parser.extract(aligned_face)
    face_masked = models.face_parser.apply_mask(parsing_mask, aligned_face)
    return parsing_mask, face_masked


# ── Metrics computation (copied from pipeline_dlib_dataset.py) ──

def compute_chin_angle_degree(p8, p4, p12, eps=1e-8):
    a = np.array(p4, dtype=np.float32) - np.array(p8, dtype=np.float32)
    b = np.array(p12, dtype=np.float32) - np.array(p8, dtype=np.float32)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + eps
    cos_theta = np.clip(np.dot(a, b) / denom, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


def compute_skeleton_angle(landmark_pt, angle_pt):
    vec = np.array(angle_pt, dtype=np.float32) - np.array(landmark_pt, dtype=np.float32)
    return float(np.degrees(np.arctan2(vec[0], -vec[1])))


def angle_ccw_deg(u, v):
    cross = u[0]*v[1] - u[1]*v[0]
    dot = u[0]*v[0] + u[1]*v[1]
    ang = np.degrees(np.arctan2(cross, dot))
    return (ang + 360.0) % 360.0


def jidat_angle_downward(p90, p70, p110):
    a = np.array(p70, dtype=np.float32) - np.array(p90, dtype=np.float32)
    b = np.array(p110, dtype=np.float32) - np.array(p90, dtype=np.float32)
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na < 1e-8 or nb < 1e-8:
        return None
    a /= na; b /= nb
    down = np.array([0.0, 1.0], dtype=np.float32)
    ang_ab = angle_ccw_deg(a, b)
    ang_ad = angle_ccw_deg(a, down)
    return float(ang_ab) if ang_ad <= ang_ab + 1e-6 else float(360.0 - ang_ab)


def compute_relative_forehead_width(angle_points, parsing_mask, p27, p90,
                                     angle_candidates=None, normalize_by=None):
    if angle_candidates is None:
        angle_candidates = [30, 40, 50, 60, 120, 130, 140, 150]
    p27_vec = np.array(p27, dtype=np.float32)
    p90_vec = np.array(p90, dtype=np.float32)
    center_dir = p90_vec - p27_vec
    center_norm = float(np.linalg.norm(center_dir))
    if center_norm < 1e-8:
        return 0.0
    center_dir /= center_norm
    normal_dir = np.array([-center_dir[1], center_dir[0]], dtype=np.float32)

    mask = parsing_mask
    if mask.ndim == 3 and mask.shape[2] == 1:
        mask = mask[:, :, 0]
    elif mask.ndim == 3 and mask.shape[2] >= 3:
        mask = cv2.cvtColor(mask.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    mask_bin = (mask > 0).astype(np.uint8)
    h, w = mask_bin.shape
    if normalize_by is None:
        normalize_by = max(h, w)
    max_steps = int(np.hypot(h, w))

    def _trace(start_xy, dir_xy):
        last = None
        for step in range(max_steps + 1):
            x = int(round(float(start_xy[0] + dir_xy[0] * step)))
            y = int(round(float(start_xy[1] + dir_xy[1] * step)))
            if x < 0 or x >= w or y < 0 or y >= h:
                break
            if mask_bin[y, x] > 0:
                last = (x, y)
            elif step > 0 and last is not None:
                break
        return last

    max_fw = 0.0
    for a in angle_candidates:
        if a not in angle_points:
            continue
        start_pt = np.array(angle_points[a], dtype=np.float32)
        end_a = _trace(start_pt, normal_dir)
        end_b = _trace(start_pt, -normal_dir)
        if end_a is None or end_b is None:
            continue
        width = float(np.linalg.norm(np.array(end_a, dtype=np.float32) - np.array(end_b, dtype=np.float32)))
        max_fw = max(max_fw, width)

    return float(max_fw / float(normalize_by)) if max_fw > 0 else 0.0


def extract_metrics(face_skin, landmarks, angle_points):
    h, w = face_skin.shape[:2]
    max_dim = max(w, h)
    return {
        "face_height": float(np.linalg.norm(landmarks[8] - angle_points[90]) / max_dim),
        "face_width": float(np.linalg.norm(landmarks[1] - landmarks[15]) / max_dim),
        "forehead_height": float(np.linalg.norm(landmarks[27] - angle_points[90]) / max_dim),
        "jaw_width": float(np.linalg.norm(landmarks[4] - landmarks[12]) / max_dim),
        "chin_width": float(np.linalg.norm(landmarks[6] - landmarks[10]) / max_dim),
        "chin_angle_degree": compute_chin_angle_degree(landmarks[8], landmarks[4], landmarks[12]),
        "forehead_width": compute_relative_forehead_width(
            angle_points, face_skin, landmarks[27], angle_points[90], normalize_by=max_dim),
        "skeleton_angle_left": compute_skeleton_angle(landmarks[4], angle_points[0]),
        "skeleton_angle_right": compute_skeleton_angle(landmarks[12], angle_points[180]),
        "jidat_angle_downward": jidat_angle_downward(angle_points[90], angle_points[70], angle_points[110]),
    }


def extract_single_image(index: int, image_path: Path, models: ModelCache) -> ExtractionResult:
    """Full pipeline for one image: extract face → landmarks → mask → metrics."""
    result = ExtractionResult(index=index, filename=image_path.name)
    try:
        raw_face = extract_face(str(image_path))
        if raw_face is None:
            # RetinaFace failed — bypass it and try dlib directly on the image.
            raw_face = extract_face_dlib_only(str(image_path), models)
            if raw_face is None:
                result.error = "No face detected (RetinaFace + dlib fallback)"
                return result
            result.used_fallback = True
        result.raw_face = raw_face

        landmarks = process_dlib_landmarks(raw_face, models)
        if landmarks is None:
            result.error = "No landmarks detected (dlib)"
            return result

        parsing_mask, face_skin = face_skin_parsing(raw_face, models)
        if face_skin is None:
            result.error = "Face parsing failed"
            return result
        result.masked_face = face_skin

        angle_points = forehead_points_dense(parsing_mask, landmarks, angle_step=10)
        metrics = extract_metrics(face_skin, landmarks, angle_points)
        result.metrics = metrics
        xgb_infer = models.get_xgb_infer()
        if xgb_infer and metrics:
            metric_input = Metric(
            face_height=metrics.get("face_height", np.nan),
                face_width=metrics.get("face_width", np.nan),
                forehead_height=metrics.get("forehead_height", np.nan),
                forehead_width=metrics.get("forehead_width", np.nan),
                jaw_width=metrics.get("jaw_width", np.nan),
                skeleton_angle_l=metrics.get("skeleton_angle_left", np.nan),
                skeleton_angle_r=metrics.get("skeleton_angle_right", np.nan),
                chin_angle=metrics.get("chin_angle_degree", np.nan),
                chin_width=metrics.get("chin_width", np.nan),
                forehead_angle=metrics.get("jidat_angle_downward", np.nan),
            )
            result.predicted_class = xgb_infer.inference(metric_input)
        result.success = True

    except Exception as e:
        result.error = str(e)
        logger.error(f"Extraction failed for {image_path.name}: {e}")

    return result


# ── Batch extractor with background threading ──
class BatchExtractor:
    """
    This class is used to extract images in batches of N, with background pre-fetching.
    """
    def __init__(self, image_paths: list[Path], batch_size: int = 10):
        """
        Args:
            image_paths (list[Path]): List of image paths to extract.
            batch_size (int): Number of images to extract in each batch.
        """
        self.image_paths = image_paths
        self.batch_size = batch_size
        self.cache: dict[int, ExtractionResult] = {}
        self._models: Optional[ModelCache] = None
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._pending_future: Optional[Future] = None
        self._lock = threading.Lock()
        self._extracted_up_to = -1

    def _get_models(self):
        if self._models is None:
            self._models = ModelCache.get()
        return self._models

    def _extract_batch(self, start: int, end: int):
        models = self._get_models()
        for i in range(start, min(end, len(self.image_paths))):
            if i in self.cache:
                continue
            result = extract_single_image(i, self.image_paths[i], models)
            with self._lock:
                self.cache[i] = result
        with self._lock:
            self._extracted_up_to = max(self._extracted_up_to, min(end, len(self.image_paths)) - 1)

    def ensure_extracted(self, index: int):
        if index in self.cache:
            return
        if self._pending_future and not self._pending_future.done():
            self._pending_future.result()
        # Still not cached? Extract synchronously
        if index not in self.cache:
            batch_start = index
            batch_end = min(index + self.batch_size, len(self.image_paths))
            self._extract_batch(batch_start, batch_end)

    def maybe_prefetch(self, current_index: int):
        threshold = self._extracted_up_to - 4
        if current_index >= threshold:
            next_start = self._extracted_up_to + 1
            if next_start < len(self.image_paths):
                if self._pending_future is None or self._pending_future.done():
                    next_end = next_start + self.batch_size
                    self._pending_future = self._executor.submit(
                        self._extract_batch, next_start, next_end
                    )

    def get(self, index: int) -> Optional[ExtractionResult]:
        with self._lock:
            return self.cache.get(index)

    def initial_extract(self, count: int = 10):
        self._extract_batch(0, min(count, len(self.image_paths)))

    def shutdown(self):
        self._executor.shutdown(wait=False)
