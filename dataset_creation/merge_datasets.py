"""Merge two (or more) exported face-shape datasets into a new export.

Each exported dataset has the layout produced by main_streamlit.py:

    dataset-<n>-kelas-<uid>/
        <ClassName>/
            metrics.csv                  # class_name,image_file,relative_path,<10 metrics>
            face_skin/<image_file>       # masked face image

This tool combines the per-class samples from every input dataset and caps each
class to a maximum count (default 50). If a class has MORE than the cap across
all inputs, it randomly samples `max` of them; if it has fewer, every sample is
kept (3-5 per class is fine). The output is written in the exact same layout.

Usage:
    python merge_datasets.py \
        --inputs output/dataset-5-kelas-c49 output/dataset-5-kelas-0ee \
        --max 50

    # custom output name / reproducible sampling
    python merge_datasets.py -i <a> <b> -m 50 -o output/dataset-5-kelas-mix --seed 7
"""
import argparse
import csv
import random
import shutil
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Canonical CSV header — kept identical to the Streamlit exporter output.
CSV_HEADER = [
    "class_name", "image_file", "relative_path",
    "face_height", "face_width", "forehead_height", "jaw_width", "chin_width",
    "chin_angle_degree", "forehead_width", "skeleton_angle_left",
    "skeleton_angle_right", "jidat_angle_downward",
]
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def discover_classes(input_dirs):
    """Union of class names across all inputs (subdirs holding a metrics.csv)."""
    classes = []
    seen = set()
    for d in input_dirs:
        for child in sorted(d.iterdir()):
            if child.is_dir() and (child / "metrics.csv").exists() and child.name not in seen:
                seen.add(child.name)
                classes.append(child.name)
    return classes


def collect_samples(input_dirs, class_name):
    """Gather every (row, source_image_path) pair for a class across inputs.

    Drives off metrics.csv rows; only keeps a row whose face_skin image exists.
    """
    samples = []
    for d in input_dirs:
        csv_path = d / class_name / "metrics.csv"
        if not csv_path.exists():
            continue
        with csv_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_name = row.get("image_file")
                if not img_name:
                    continue
                src_img = d / class_name / "face_skin" / img_name
                if not src_img.exists():
                    print(f"  ! skip (image missing): {class_name}/{img_name} in {d.name}")
                    continue
                samples.append((row, src_img))
    return samples


def merge(input_dirs, output_root, max_per_class, rng):
    classes = discover_classes(input_dirs)
    if not classes:
        raise SystemExit("No classes with metrics.csv found in the given inputs.")

    output_root.mkdir(parents=True, exist_ok=True)
    summary = []

    for cls in classes:
        samples = collect_samples(input_dirs, cls)
        total = len(samples)

        if total > max_per_class:
            chosen = rng.sample(samples, max_per_class)
        else:
            chosen = samples

        out_cls_dir = output_root / cls
        out_skin_dir = out_cls_dir / "face_skin"
        out_skin_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_cls_dir / "metrics.csv"

        used_names = set()
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(CSV_HEADER)

            for row, src_img in chosen:
                # Resolve filename collisions across source datasets.
                name = src_img.name
                if name in used_names:
                    stem, suf = Path(name).stem, Path(name).suffix
                    i = 1
                    while f"{stem}_{i}{suf}" in used_names:
                        i += 1
                    name = f"{stem}_{i}{suf}"
                used_names.add(name)

                shutil.copy2(src_img, out_skin_dir / name)

                row = dict(row)
                row["class_name"] = cls
                row["image_file"] = name
                row["relative_path"] = f"{cls}/{name}"
                writer.writerow([row.get(col, "") for col in CSV_HEADER])

        summary.append((cls, total, len(chosen)))

    return summary


def parse_args():
    p = argparse.ArgumentParser(description="Merge exported face-shape datasets with a per-class cap.")
    p.add_argument(
        "-i", "--inputs", nargs="+", type=str,
        default=[
            str(PROJECT_ROOT / "output" / "dataset-5-kelas-c49"),
            str(PROJECT_ROOT / "output" / "dataset-5-kelas-0ee"),
        ],
        help="Input dataset folders to merge.",
    )
    p.add_argument("-m", "--max", type=int, default=50, help="Max samples per class (default 50).")
    p.add_argument("-o", "--output", type=str, default=None, help="Output folder (default output/dataset-<n>-kelas-<uid>).")
    p.add_argument("--seed", type=int, default=42, help="Random seed for sampling (default 42).")
    return p.parse_args()


def main():
    args = parse_args()
    input_dirs = [Path(p).expanduser().resolve() for p in args.inputs]
    for d in input_dirs:
        if not d.exists():
            raise SystemExit(f"Input not found: {d}")

    rng = random.Random(args.seed)

    if args.output:
        output_root = Path(args.output).expanduser().resolve()
    else:
        n_classes = len(discover_classes(input_dirs))
        uid = uuid.uuid4().hex[:3]
        output_root = PROJECT_ROOT / "output" / f"dataset-{n_classes}-kelas-{uid}"

    print(f"Merging {len(input_dirs)} datasets -> {output_root}")
    print(f"Max per class: {args.max} | seed: {args.seed}\n")

    summary = merge(input_dirs, output_root, args.max, rng)

    print(f"\n{'class':12} {'combined':>9} {'exported':>9}")
    print("-" * 32)
    grand = 0
    for cls, total, kept in summary:
        flag = "  (sampled)" if total > kept else ""
        print(f"{cls:12} {total:>9} {kept:>9}{flag}")
        grand += kept
    print("-" * 32)
    print(f"{'TOTAL':12} {'':>9} {grand:>9}")
    print(f"\nDone -> {output_root}")


if __name__ == "__main__":
    main()
