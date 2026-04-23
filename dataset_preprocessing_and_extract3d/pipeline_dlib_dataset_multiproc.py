import os

from .face_parsing.face_parsing import FaceParsingExtractor
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from retinaface import RetinaFace
import cv2
import dlib
import numpy as np
import csv
import logging
import argparse
from pathlib import Path
from tqdm.contrib.concurrent import process_map
from .forehead_dense import forehead_points_dense

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset" / "face_shape"
OUTPUT_DIR = PROJECT_ROOT / "output"
CLASS_CSV_FILENAME = "metrics.csv"
LOWER_FACE_LABELS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16]
SHAPE_PREDICTOR_PATH = PROJECT_ROOT / "models" / "shape_predictor_68_face_landmarks.dat"


def _resolve_shape_predictor_path(shape_predictor_path):
    path = Path(shape_predictor_path).expanduser()
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    return path


def _initialize_dlib_predictor(shape_predictor_path):
    predictor_path = _resolve_shape_predictor_path(shape_predictor_path)
    if not predictor_path.exists():
        raise FileNotFoundError(f"shape predictor file not found: {predictor_path}")
    return predictor_path
## Metrics Utils Function
def compute_relative_forehead_width(angle_points, parsing_mask, p27, p90, angle_candidates=None, normalize_by=None):
    if angle_candidates is None:
        angle_candidates = [30, 40, 50, 60, 120, 130, 140, 150]

    p27_vec = np.array(p27, dtype=np.float32)
    p90_vec = np.array(p90, dtype=np.float32)

    center_dir = p90_vec - p27_vec
    center_norm = float(np.linalg.norm(center_dir))
    if center_norm < 1e-8:
        raise ValueError("Garis center (27->90) tidak valid")
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
    if normalize_by <= 0:
        raise ValueError("normalize_by harus > 0")

    max_steps = int(np.hypot(h, w))

    def _trace_last_inside(start_xy, dir_xy):
        last_inside = None
        for step in range(max_steps + 1):
            x = int(round(float(start_xy[0] + dir_xy[0] * step)))
            y = int(round(float(start_xy[1] + dir_xy[1] * step)))
            if x < 0 or x >= w or y < 0 or y >= h:
                break
            if mask_bin[y, x] > 0:
                last_inside = (x, y)
            elif step > 0 and last_inside is not None:
                break
        return last_inside

    max_forehead_width_px = 0.0

    for a in angle_candidates:
        if a not in angle_points:
            continue

        start_pt = np.array(angle_points[a], dtype=np.float32)
        end_a = _trace_last_inside(start_pt, normal_dir)
        end_b = _trace_last_inside(start_pt, -normal_dir)
        if end_a is None or end_b is None:
            continue

        width_px = float(np.linalg.norm(np.array(end_a, dtype=np.float32) - np.array(end_b, dtype=np.float32)))
        max_forehead_width_px = max(max_forehead_width_px, width_px)

    if max_forehead_width_px <= 0:
        raise ValueError("Tidak menemukan forehead width line yang valid")

    return float(max_forehead_width_px / float(normalize_by))


def compute_skeleton_angle(landmark_pt, angle_pt):
    vec = np.array(angle_pt, dtype=np.float32) - np.array(landmark_pt, dtype=np.float32)
    angle_deg = np.degrees(np.arctan2(vec[0], -vec[1]))
    return angle_deg


def compute_chin_angle_degree(p8, p4, p12, eps=1e-8):
    chin_vec_a = np.array(p4, dtype=np.float32) - np.array(p8, dtype=np.float32)
    chin_vec_b = np.array(p12, dtype=np.float32) - np.array(p8, dtype=np.float32)

    denom = (np.linalg.norm(chin_vec_a) * np.linalg.norm(chin_vec_b)) + eps
    cos_theta = np.clip(np.dot(chin_vec_a, chin_vec_b) / denom, -1.0, 1.0)
    chin_angle_rad = np.arccos(cos_theta)
    return float(np.degrees(chin_angle_rad))


def angle_ccw_deg(u, v):
    # angle dari u ke v berlawanan jarum jam, hasil 0..360
    cross = u[0] * v[1] - u[1] * v[0]
    dot = u[0] * v[0] + u[1] * v[1]
    ang = np.degrees(np.arctan2(cross, dot))
    return (ang + 360.0) % 360.0


def jidat_angle_downward(p90, p70, p110):
    a = np.array(p70, dtype=np.float32) - np.array(p90, dtype=np.float32)
    b = np.array(p110, dtype=np.float32) - np.array(p90, dtype=np.float32)

    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-8 or nb < 1e-8:
        return None

    a /= na
    b /= nb

    down = np.array([0.0, 1.0], dtype=np.float32)

    ang_ab = angle_ccw_deg(a, b)
    ang_ad = angle_ccw_deg(a, down)

    # kalau "down" ada di dalam wedge a->b (CCW), pakai ang_ab
    # kalau tidak, berarti wedge yang mengarah ke bawah adalah reflex: 360 - ang_ab
    if ang_ad <= ang_ab + 1e-6:
        return float(ang_ab)
    else:
        return float(360.0 - ang_ab)


def extract_metrics(face_masked, landmarks, angle_points):
    pic_height, pic_width = face_masked.shape[:2]
    max_dim = max(pic_width, pic_height)
    metrics = {
        "face_height": np.linalg.norm(landmarks[8] - angle_points[90]) / max_dim,
        "face_width": np.linalg.norm(landmarks[1] - landmarks[15]) / max_dim,
        "forehead_height": np.linalg.norm(landmarks[27] - angle_points[90]) / max_dim,
        "jaw_width": np.linalg.norm(landmarks[4] - landmarks[12]) / max_dim,
        "chin_width": np.linalg.norm(landmarks[6] - landmarks[10]) / max_dim,
        "chin_angle_degree": compute_chin_angle_degree(landmarks[8], landmarks[4], landmarks[12]),
        "forehead_width": compute_relative_forehead_width(
            angle_points,
            face_masked,
            landmarks[27],
            angle_points[90],
            normalize_by=max_dim,
        ),
        "skeleton_angle_left": compute_skeleton_angle(landmarks[4], angle_points[0]),
        "skeleton_angle_right": compute_skeleton_angle(landmarks[12], angle_points[180]),
        "jidat_angle_downward": jidat_angle_downward(angle_points[90], angle_points[70], angle_points[110]),
    }

    return metrics


class DatasetPreprocessor:
    def __init__(self, dataset_dir, output_dir, class_csv_filename, shape_predictor_path):
        self.dataset_dir = Path(dataset_dir)
        self.output_dir = Path(output_dir)
        self.class_csv_filename = class_csv_filename
        self.shape_predictor_path = _initialize_dlib_predictor(shape_predictor_path)
        self.hog_face_detector = dlib.get_frontal_face_detector()
        self.dlib_facelandmark = dlib.shape_predictor(str(self.shape_predictor_path))
        self.face_parser = FaceParsingExtractor()

    def _extract_face(self, image_file):
        faces = RetinaFace.extract_faces(img_path=str(image_file), align=True, expand_face_area=15)
        if faces:
            return cv2.cvtColor(faces[0], cv2.COLOR_BGR2RGB)

        logger.warning(f"[EXTRACT FACE]:No face detected in {image_file}")
        return None

    def _process_dlib_landmarks(self, image_cv2):
        gray = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2GRAY)
        faces = self.hog_face_detector(gray)

        if len(faces) == 0:
            logger.warning("[DLIB]: No face detected in the image.")
            return None

        landmarks = self.dlib_facelandmark(gray, faces[0])
        if landmarks.num_parts != 68:
            logger.warning(f"[DLIB]: Expected 68 landmarks, but detected {landmarks.num_parts}.")
            return None

        return np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(68)])

    def _face_skin_parsing(self, aligned_face):
        parsing_mask = self.face_parser.extract(aligned_face)
        face_masked = self.face_parser.apply_mask(parsing_mask, aligned_face)
        return parsing_mask, face_masked

    def _process_image(self, image_file, output_class_dir, csv_writer, dataset_path, class_name=None):
        logger.info(f"[MAIN]:  Start processing image {image_file.name} for class {class_name}")
        face_skin_dir = output_class_dir / "face_skin"
        face_skin_dir.mkdir(parents=True, exist_ok=True)
        face_masked_dir = output_class_dir / "face_masked"
        face_masked_dir.mkdir(parents=True, exist_ok=True)

        probe_face = self._extract_face(image_file)
        if probe_face is None:
            raise ValueError(f"[MAIN]:  No face detected in probe image {image_file} for class {class_name}. Skipping class.")

        probe_landmarks = self._process_dlib_landmarks(probe_face)
        if probe_landmarks is None:
            raise ValueError(f"[MAIN]:  No landmarks detected in probe image {image_file} for class {class_name}. Skipping class.")

        probe_face_masked, probe_face_skin = self._face_skin_parsing(probe_face)
        if probe_face_masked is None or probe_face_skin is None:
            raise ValueError(f"[MAIN]:  Face skin parsing failed for probe image {image_file} for class {class_name}. Skipping class.")

        cv2.imwrite(str(face_skin_dir / image_file.name), probe_face_skin)
        cv2.imwrite(str(face_masked_dir / image_file.name), probe_face_masked)

        probe_face_angle_points = forehead_points_dense(
            probe_face_masked,
            probe_landmarks,
            angle_step=10,
        )
        try:
            relative_path = image_file.relative_to(dataset_path).as_posix()
        except ValueError:
            logger.warning(f"[MAIN]:  Failed to resolve relative path for {image_file} against {dataset_path}. Falling back to class/name path.")
            fallback_class_name = class_name or "unknown"
            relative_path = (Path(fallback_class_name) / image_file.name).as_posix()

        metric_list = [class_name, image_file.name, relative_path]
        metrics = extract_metrics(probe_face_skin, probe_landmarks, probe_face_angle_points)
        metric_list.extend(metrics.values())
        csv_writer.writerow(metric_list)
        logger.info(f"[MAIN]:  Processed {image_file} for class {class_name} Successfully")

    def _process_class(self, dataset_name, dataset_path, class_name, class_path):
        logger.info(f"[MAIN]:  Processing Class: {class_name}")
        output_class_dir = self.output_dir / dataset_name / class_name
        output_class_dir.mkdir(parents=True, exist_ok=True)

        image_files = [
            file_path for file_path in class_path.glob("*")
            if file_path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ]
        if not image_files:
            logger.warning(f"[MAIN]:  No images found for class {class_name} in {class_path}")
            return

        csv_path = output_class_dir / self.class_csv_filename

        with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow([
                "class_name",
                "image_file",
                "relative_path",
                "face_height",
                "face_width",
                "forehead_height",
                "jaw_width",
                "chin_width",
                "chin_angle_degree",
                "forehead_width",
                "skeleton_angle_left",
                "skeleton_angle_right",
                "jidat_angle_downward",
            ])

            for image_file in image_files:
                try:
                    self._process_image(
                        image_file=image_file,
                        output_class_dir=output_class_dir,
                        csv_writer=csv_writer,
                        dataset_path=dataset_path,
                        class_name=class_name,
                    )
                except Exception as e:
                    logger.error(f"[MAIN]:  Error processing probe image {image_file} for class {class_name}: {e}")
                    continue

        logger.info(f"[MAIN]:  Saved CSV for class {class_name}: {csv_path}")

    def process_dataset(self, dataset_name, dataset_path):
        logger.info(f"[MAIN]:Processing Dataset: {dataset_name}")
        (self.output_dir / dataset_name).mkdir(parents=True, exist_ok=True)

        class_names = os.listdir(dataset_path)
        class_paths = {
            cls: dataset_path / cls
            for cls in class_names
            if (dataset_path / cls).is_dir()
        }
        for cls, cls_path in class_paths.items():
            self._process_class(dataset_name, dataset_path, cls, cls_path)


def get_dataset_paths(dataset_dir):
    dataset_names = os.listdir(dataset_dir)
    return {
        name: dataset_dir / name
        for name in dataset_names
        if (dataset_dir / name).is_dir()
    }


def parse_arguments():
    parser = argparse.ArgumentParser(description="Preprocess face shape datasets and extract 3D landmarks.")
    parser.add_argument("-s", "--dataset_dir", type=str, default=str(DATASET_DIR), help="Path to the input dataset directory.")
    parser.add_argument("-o", "--output_dir", type=str, default=str(OUTPUT_DIR), help="Path to the output directory for processed data.")
    parser.add_argument("-c", "--class_csv_filename", type=str, default=CLASS_CSV_FILENAME, help="Filename for the landmarks CSV within each class directory.")
    parser.add_argument("-w", "--max_workers", type=int, default=5, help="Number of worker processes for class-level multiprocessing.")
    parser.add_argument("-p", "--shape_predictor_path", type=str, default=str(SHAPE_PREDICTOR_PATH), help="Path to dlib's shape predictor model file.")
    return parser.parse_args()


def _process_class_wrapper(args):
    dataset_name, class_name, dataset_dir, output_dir, class_csv_filename, shape_predictor_path = args
    preprocessor = DatasetPreprocessor(dataset_dir, output_dir, class_csv_filename, shape_predictor_path)
    dataset_path = Path(dataset_dir) / dataset_name
    class_path = dataset_path / class_name
    preprocessor._process_class(dataset_name, dataset_path, class_name, class_path)


def main():
    args = parse_arguments()
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    dlib_facelandmark_path = args.shape_predictor_path
    predictor_path = _initialize_dlib_predictor(dlib_facelandmark_path)
    logger.info(f"[MAIN]:Using shape predictor: {predictor_path}")
    class_csv_filename = args.class_csv_filename
    max_workers = args.max_workers

    try:
        for dataset_name, dataset_path in get_dataset_paths(dataset_dir).items():
            logger.info(f"[MAIN]:Processing Dataset: {dataset_name}")
            (output_dir / dataset_name).mkdir(parents=True, exist_ok=True)

            class_names = os.listdir(dataset_path)
            class_dirs = [cls for cls in class_names if (dataset_path / cls).is_dir()]

            if not class_dirs:
                logger.warning(f"[MAIN]:No classes found for dataset {dataset_name}")
                continue

            process_map(
                _process_class_wrapper,
                [
                    (dataset_name, class_name, dataset_dir, output_dir, class_csv_filename, str(predictor_path))
                    for class_name in class_dirs
                ],
                max_workers=max_workers,
                desc=f"Processing Classes in {dataset_name}",
            )
    except Exception as e:
        logger.error(f"[MAIN]:  Error processing dataset: {e}")


if __name__ == "__main__":
    main()