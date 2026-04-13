import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import csv
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from urllib.request import urlretrieve
from retinaface import RetinaFace
import logging
import argparse
from pathlib import Path

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
CLASS_CSV_FILENAME = "landmarks.csv"

def create_face_landmarker():
    model_path = "face_landmarker.task"
    if not os.path.exists(model_path):
        urlretrieve(
            "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
            model_path,
        )

    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    return vision.FaceLandmarker.create_from_options(options)

def extract_face(image_file):
    faces = RetinaFace.extract_faces(img_path=str(image_file), align=True, expand_face_area=8)
    if faces:
        return cv2.cvtColor(faces[0], cv2.COLOR_BGR2RGB)
    else:
        logger.warning(f"[EXTRACT FACE]:No face detected in {image_file}")
        return None


def build_landmark_header(landmark_count):
    header = ["image_name", "image_path"]
    for i in range(landmark_count):
        header.extend([f"x_{i}", f"y_{i}", f"z_{i}"])
    return header


def flatten_landmarks(landmarks):
    flat = []
    for lm in landmarks:
        flat.extend([lm.x, lm.y, lm.z])
    return flat


def process_image(image_file, output_class_dir, face_mesh, csv_writer, expected_landmark_count):
    logger.info(f"[MAIN]:    Processing Image: {image_file.name}")
    face = extract_face(image_file)
    if face is not None:
        output_path = output_class_dir / image_file.name
        cv2.imwrite(str(output_path), face)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=face)
        results = face_mesh.detect(mp_image)
        if results.face_landmarks:
            landmarks = results.face_landmarks[0]
            if len(landmarks) != expected_landmark_count:
                logger.warning(
                    f"[MAIN]:      Landmark count mismatch for {image_file}. "
                    f"Expected {expected_landmark_count}, got {len(landmarks)}. Row skipped."
                )
                return

            csv_writer.writerow([
                output_path.name,
                (Path(output_class_dir.parents[1].name) / output_path.relative_to(output_class_dir.parents[1])).as_posix(),
                *flatten_landmarks(landmarks),
            ])
            logger.info(f"[MAIN]:    Landmarks detected: {len(landmarks)} points")
        else:
            logger.warning(f"[MAIN]:    Warning: No landmarks detected in {image_file}")
    else:
        logger.info(f"[MAIN]:    Skipping landmark extraction for {image_file} due to no face detected.")


def process_class(dataset_name, class_name, class_path, face_mesh, output_dir, class_csv_filename):
    logger.info(f"[MAIN]:  Processing Class: {class_name}")
    output_class_dir = output_dir / dataset_name / class_name
    output_class_dir.mkdir(parents=True, exist_ok=True)

    image_files = [
        file_path for file_path in class_path.glob("*")
        if file_path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]
    if not image_files:
        logger.warning(f"[MAIN]:  No images found for class {class_name} in {class_path}")
        return

    csv_path = output_class_dir / class_csv_filename

    # Probe one image to lock a stable schema for model training.
    expected_landmark_count = None
    for image_file in image_files:
        probe_face = extract_face(image_file)
        if probe_face is None:
            continue
        probe_mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=probe_face)
        probe_results = face_mesh.detect(probe_mp_image)
        if probe_results.face_landmarks:
            expected_landmark_count = len(probe_results.face_landmarks[0])
            break

    if expected_landmark_count is None:
        logger.warning(f"[MAIN]:  No valid landmarks for class {class_name}. CSV not created.")
        return

    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(build_landmark_header(expected_landmark_count))

        for image_file in image_files:
            process_image(
                image_file=image_file,
                output_class_dir=output_class_dir,
                face_mesh=face_mesh,
                csv_writer=csv_writer,
                expected_landmark_count=expected_landmark_count,
            )

    logger.info(f"[MAIN]:  Saved CSV for class {class_name}: {csv_path}")


def process_dataset(dataset_name, dataset_path, face_mesh, output_dir, class_csv_filename):
    logger.info(f"[MAIN]:Processing Dataset: {dataset_name}")
    (output_dir / dataset_name).mkdir(parents=True, exist_ok=True)

    class_names = os.listdir(dataset_path)
    class_paths = {
        cls: dataset_path / cls
        for cls in class_names
        if (dataset_path / cls).is_dir()
    }
    for cls, cls_path in class_paths.items():
        process_class(dataset_name, cls, cls_path, face_mesh, output_dir, class_csv_filename)


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
    return parser.parse_args()


def main():
    face_mesh = None
    args = parse_arguments()
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    class_csv_filename = args.class_csv_filename
    try:
        face_mesh = create_face_landmarker()
        datasets = get_dataset_paths(dataset_dir)
        for name, path in datasets.items():
            process_dataset(name, path, face_mesh, output_dir, class_csv_filename)
    finally:
        if face_mesh is not None:
            face_mesh.close()


if __name__ == "__main__":
    main()