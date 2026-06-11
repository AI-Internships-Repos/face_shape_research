"""Face Shape Dataset Creation App — Streamlit"""
import streamlit as st
import os, sys, csv, uuid, json, time
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from glob import glob
from concurrent.futures import ThreadPoolExecutor
import threading

st.set_page_config(page_title="Face Shape Dataset Creator", layout="wide", page_icon="🧬")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CLASSES = ["Heart", "Oblong", "Oval", "Round", "Square"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
ANALYSIS_DIRS = [
    PROJECT_ROOT / "output" / "training_set" / "_analysis",
    PROJECT_ROOT / "output" / "testing_set" / "_analysis",
]
XGB_MODEL_PATH = PROJECT_ROOT / "dataset_creation" / "models" / "xgboost_model.joblib"
XGB_LABEL_PATH = PROJECT_ROOT / "dataset_creation" / "models" / "label_encoder.joblib"
CHECKPOINT_ROOT = PROJECT_ROOT / "output" / "_checkpoint"
CHECKPOINT_VERSION = 1

# ── CSS ──
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
* { font-family: 'Inter', sans-serif; }
.stApp { background: radial-gradient(1200px 800px at 10% 0%, #0b1f3a 0%, #0b1220 45%, #090c12 100%); }
h1, h2, h3 { color: #f9fafb !important; }
.metric-card {
    background: #0f172a; border: 1px solid #1f2937;
    border-radius: 12px; padding: 12px 16px; margin: 4px 0;
    transition: box-shadow 0.2s;
}
.metric-card:hover { box-shadow: 0 6px 18px rgba(0,0,0,0.25); }
.metric-label { font-size: 11px; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.4px; font-weight: 600; }
.metric-value { font-size: 20px; font-weight: 700; color: #e5e7eb; margin-top: 2px; }
.metric-bar { height: 4px; border-radius: 2px; margin-top: 6px; background: #1f2937; }
.metric-bar-fill { height: 100%; border-radius: 2px; transition: width 0.4s ease; }
.class-badge {
    display: inline-block; padding: 4px 12px; border-radius: 16px; font-weight: 600;
    font-size: 13px; color: white; margin: 2px;
}
.skip-badge { background: #64748b; }
.status-card {
    background: #0f172a; border: 1px solid #1f2937;
    border-radius: 12px; padding: 16px; margin: 10px 0;
}
.img-container {
    background: #0b1220; border-radius: 10px; padding: 8px;
    border: 1px solid #1f2937;
}
.nav-info { text-align: center; color: #9ca3af; font-size: 14px; padding: 8px; }
.shortcut-hint {
    font-size: 10px; color: #9ca3af; background: #111827;
    padding: 2px 6px; border-radius: 4px; margin-left: 4px;
}
div[data-testid="stHorizontalBlock"] button {
    border-radius: 8px !important; font-weight: 600 !important; transition: all 0.2s !important;
}
.review-stat {
    background: #0f172a; border-radius: 10px; padding: 16px;
    border: 1px solid #1f2937; margin: 6px 0;
}
</style>""", unsafe_allow_html=True)

CLASS_COLORS = ["#6366f1","#f43f5e","#10b981","#f59e0b","#8b5cf6","#06b6d4","#ec4899","#14b8a6","#f97316"]

def get_class_color(i):
    return CLASS_COLORS[i % len(CLASS_COLORS)]

# ── Session State Init ──
def init_state():
    defaults = {
        "step": 1, "classes": list(DEFAULT_CLASSES), "input_folder": "",
        "image_paths": [], "current_index": 0, "assignments": {},
        "skipped": set(), "extractor": None, "new_class_name": "",
        "extraction_started": False, "export_done": False,
        "export_uid": uuid.uuid4().hex[:3],
        "checkpoint_id": None, "checkpoint_created_at": None,
        "checkpoint_manager": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ── Helpers ──
def scan_images(folder):
    p = Path(folder)
    if not p.exists():
        return []
    files = sorted([f for f in p.rglob("*") if f.is_file() and f.suffix.lower() in IMAGE_EXTS])
    return files


def _serialize_assignments(assignments: dict) -> dict:
    return {str(k): v for k, v in assignments.items()}


def _deserialize_assignments(assignments: dict) -> dict:
    out = {}
    for k, v in assignments.items():
        try:
            out[int(k)] = v
        except Exception:
            continue
    return out


def get_checkpoint_manager():
    cache_id = st.session_state.get("checkpoint_id")
    if not cache_id:
        return None
    mgr = st.session_state.get("checkpoint_manager")
    if mgr is None or mgr.cache_id != cache_id:
        mgr = CheckpointManager(cache_id)
        st.session_state.checkpoint_manager = mgr
    return mgr


def build_checkpoint_metadata(cache_id: str) -> dict:
    created_at = st.session_state.checkpoint_created_at or time.time()
    return {
        "version": CHECKPOINT_VERSION,
        "cache_id": cache_id,
        "created_at": created_at,
        "updated_at": time.time(),
        "input_folder": st.session_state.input_folder,
        "classes": list(st.session_state.classes),
        "image_paths": [str(p) for p in st.session_state.image_paths],
        "assignments": _serialize_assignments(st.session_state.assignments),
        "skipped": sorted(list(st.session_state.skipped)),
        "current_index": int(st.session_state.current_index),
        "export_uid": st.session_state.export_uid,
    }


def save_checkpoint_state():
    mgr = get_checkpoint_manager()
    if mgr is None:
        return
    metadata = build_checkpoint_metadata(mgr.cache_id)
    st.session_state.checkpoint_created_at = metadata["created_at"]
    mgr.save_metadata_async(metadata)


def save_checkpoint_record(index: int, image_path: Path, result, assigned_class: str):
    mgr = get_checkpoint_manager()
    if mgr is None or result is None:
        return
    if result.metrics is None and result.masked_face is None:
        return
    mgr.update_record(
        index,
        image_path,
        result.masked_face,
        result.metrics,
        result.predicted_class,
        assigned_class,
    )


def list_checkpoints():
    if not CHECKPOINT_ROOT.exists():
        return []
    items = []
    for meta_path in CHECKPOINT_ROOT.glob("*/metadata.json"):
        try:
            with meta_path.open("r", encoding="utf-8") as f:
                meta = json.load(f)
            items.append(meta)
        except Exception:
            continue
    return sorted(items, key=lambda m: m.get("updated_at", 0), reverse=True)


def load_checkpoint(cache_id: str):
    meta_path = CHECKPOINT_ROOT / cache_id / "metadata.json"
    if not meta_path.exists():
        st.error("Checkpoint metadata not found.")
        return
    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)

    st.session_state.checkpoint_id = meta.get("cache_id", cache_id)
    st.session_state.checkpoint_created_at = meta.get("created_at")
    st.session_state.classes = list(meta.get("classes", DEFAULT_CLASSES))
    st.session_state.input_folder = meta.get("input_folder", "")
    st.session_state.image_paths = [Path(p) for p in meta.get("image_paths", [])]
    st.session_state.assignments = _deserialize_assignments(meta.get("assignments", {}))
    st.session_state.skipped = set(meta.get("skipped", []))
    st.session_state.current_index = int(meta.get("current_index", 0))
    st.session_state.export_uid = meta.get("export_uid", uuid.uuid4().hex[:3])
    st.session_state.step = 2
    st.session_state.extractor = None
    st.session_state.extraction_started = False
    st.session_state.export_done = False

    get_checkpoint_manager()

@st.cache_data(show_spinner=False)
def load_metric_thresholds():
    import pandas as pd

    quartile_path = None
    for base in ANALYSIS_DIRS:
        candidate = base / "training_set_quartile_summary.csv"
        if candidate.exists():
            quartile_path = candidate
            break
        candidate = base / "testing_set_quartile_summary.csv"
        if candidate.exists():
            quartile_path = candidate
            break

    if quartile_path is None:
        return {}

    df = pd.read_csv(quartile_path)
    if "quantile" not in df.columns:
        return {}

    metric_cols = [c for c in df.columns if c not in {"class_name", "quantile"}]
    thresholds = {}
    for col in metric_cols:
        q1 = df.loc[df["quantile"] == "q1", col].median()
        med = df.loc[df["quantile"] == "median", col].median()
        q3 = df.loc[df["quantile"] == "q3", col].median()
        if np.isfinite(q1) and np.isfinite(med) and np.isfinite(q3):
            thresholds[col] = {"q1": float(q1), "median": float(med), "q3": float(q3)}

    return thresholds


def _metric_status(value, thresholds):
    if thresholds is None or value is None:
        return "OK"
    q1 = thresholds.get("q1")
    q3 = thresholds.get("q3")
    if q1 is None or q3 is None:
        return "OK"
    if value < q1:
        return "LOW"
    if value > q3:
        return "HIGH"
    return "OK"


def render_metric_card(label, value, color="#6366f1", max_val=1.0, thresholds=None):
    if value is None:
        value = 0.0
    status = _metric_status(value, thresholds)
    if thresholds and thresholds.get("q1") is not None and thresholds.get("q3") is not None:
        span = max(float(thresholds["q3"]) - float(thresholds["q1"]), 1e-6)
        pct = (float(value) - float(thresholds["q1"])) / span * 100
    else:
        pct = abs(float(value)) / max(abs(max_val), 1e-6) * 100
    pct = float(min(max(pct, 0.0), 100.0))
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value" style="color:{color}">{value:.4f} <span style="font-size:11px;color:#9ca3af">{status}</span></div>
        <div class="metric-bar"><div class="metric-bar-fill" style="width:{pct}%;background:linear-gradient(90deg,{color},{color}88)"></div></div>
    </div>""", unsafe_allow_html=True)

def render_radar_chart(metrics_dict, thresholds_map=None):
    labels = list(metrics_dict.keys())
    values = [float(v) if v is not None else 0.0 for v in metrics_dict.values()]

    norm = []
    for name, val in zip(labels, values):
        th = thresholds_map.get(name) if thresholds_map else None
        if th and th.get("q1") is not None and th.get("q3") is not None:
            span = max(float(th["q3"]) - float(th["q1"]), 1e-6)
            v = (val - float(th["q1"])) / span
            norm.append(float(min(max(v, 0.0), 1.0)))
        else:
            # Fallback normalization
            m = max(abs(val), 0.01)
            norm.append(float(val) / float(m))
    N = len(labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    norm += norm[:1]
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True))
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    ax.fill(angles, norm, alpha=0.15, color="#6366f1")
    ax.plot(angles, norm, color="#6366f1", linewidth=2)
    ax.set_xticks(angles[:-1])
    short = [l.replace("_"," ").title()[:10] for l in labels]
    ax.set_xticklabels(short, size=7, color="#9ca3af")
    ax.tick_params(axis='y', labelsize=6, colors="#4b5563")
    ax.spines['polar'].set_color('#333')
    ax.grid(color="#333", linewidth=0.5)
    plt.tight_layout()
    return fig


class CheckpointManager:
    def __init__(self, cache_id: str):
        self.cache_id = cache_id
        self.cache_dir = CHECKPOINT_ROOT / cache_id
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir = self.cache_dir / "images"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.meta_path = self.cache_dir / "metadata.json"
        self.records_path = self.cache_dir / "records.json"
        self.records_csv_path = self.cache_dir / "records.csv"
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._lock = threading.Lock()
        self.records = self._load_records()
        self._records_csv = self.records_csv_path.open("a", newline="", encoding="utf-8")
        self._records_writer = csv.writer(self._records_csv)
        if self.records_csv_path.stat().st_size == 0:
            self._records_writer.writerow([
                "index",
                "class_name",
                "image_file",
                "image_path",
                "masked_path",
                "predicted_class",
                "updated_at",
            ])
            self._records_csv.flush()

    def _load_records(self):
        if not self.records_path.exists():
            return {}
        try:
            with self.records_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _write_json(self, path: Path, data: dict):
        tmp_path = path.with_suffix(".tmp")
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=True, indent=2)
        tmp_path.replace(path)

    def save_metadata_async(self, metadata: dict):
        self._executor.submit(self._write_json, self.meta_path, metadata)

    def save_records_async(self):
        with self._lock:
            snapshot = dict(self.records)
        self._executor.submit(self._write_json, self.records_path, snapshot)

    def update_record(self, index: int, image_path: Path, masked_face, metrics, predicted_class, assigned_class):
        record = {
            "index": int(index),
            "class_name": assigned_class,
            "image_file": image_path.name,
            "image_path": str(image_path),
            "metrics": metrics or {},
            "predicted_class": predicted_class,
            "updated_at": time.time(),
        }

        if masked_face is not None:
            out_path = self.images_dir / f"{index:06d}_{image_path.name}"
            image_copy = masked_face.copy()
            record["masked_path"] = str(out_path.relative_to(self.cache_dir))
            self._executor.submit(self._write_image, out_path, image_copy)

        with self._lock:
            self.records[str(index)] = record
        self.save_records_async()
        self._append_csv_async(record)

    @staticmethod
    def _write_image(path: Path, image_rgb):
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))

    def _append_csv_async(self, record: dict):
        self._executor.submit(self._append_csv_row, record)

    def _append_csv_row(self, record: dict):
        with self._lock:
            self._records_writer.writerow([
                record.get("index"),
                record.get("class_name"),
                record.get("image_file"),
                record.get("image_path"),
                record.get("masked_path", ""),
                record.get("predicted_class", ""),
                record.get("updated_at"),
            ])
            self._records_csv.flush()


@st.cache_resource(show_spinner=False)
def load_xgb_infer():
    from xgboost_inference import XGBoostInference

    if not XGB_MODEL_PATH.exists() or not XGB_LABEL_PATH.exists():
        return None
    return XGBoostInference(str(XGB_MODEL_PATH), str(XGB_LABEL_PATH))


def resolve_result(idx, ext, mgr, want_image=False):
    """Return an extraction-like object for an index.

    Prefers the live in-memory extractor cache; falls back to the persisted
    checkpoint records on disk so that Review/Export see every annotated image
    even after a resume (when ext.cache starts empty).
    """
    from types import SimpleNamespace

    r = ext.get(idx) if ext else None
    if r and r.metrics:
        return r

    if mgr is not None:
        rec = mgr.records.get(str(idx))
        if rec and rec.get("metrics"):
            masked = None
            mp = rec.get("masked_path")
            if want_image and mp:
                p = mgr.cache_dir / mp
                if p.exists():
                    img = cv2.imread(str(p))
                    if img is not None:
                        masked = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return SimpleNamespace(
                metrics=rec.get("metrics"),
                predicted_class=rec.get("predicted_class"),
                masked_face=masked,
            )

    return r


def _build_xgb_metric(metrics_dict):
    from xgboost_inference import Metric

    if not metrics_dict:
        return None

    required = [
        "face_height",
        "face_width",
        "forehead_height",
        "forehead_width",
        "jaw_width",
        "skeleton_angle_left",
        "skeleton_angle_right",
        "chin_angle_degree",
        "chin_width",
        "jidat_angle_downward",
    ]
    if any(metrics_dict.get(k) is None for k in required):
        return None

    return Metric(
        face_height=float(metrics_dict["face_height"]),
        face_width=float(metrics_dict["face_width"]),
        forehead_height=float(metrics_dict["forehead_height"]),
        forehead_width=float(metrics_dict["forehead_width"]),
        jaw_width=float(metrics_dict["jaw_width"]),
        skeleton_angle_l=float(metrics_dict["skeleton_angle_left"]),
        skeleton_angle_r=float(metrics_dict["skeleton_angle_right"]),
        chin_angle=float(metrics_dict["chin_angle_degree"]),
        chin_width=float(metrics_dict["chin_width"]),
        forehead_angle=float(metrics_dict["jidat_angle_downward"]),
    )

# ── STEP 1: Setup ──
def render_setup():
    st.markdown("# 🧬 Face Shape Dataset Creator")
    st.markdown("##### Configure your dataset classes and select the input image folder")

    checkpoints = list_checkpoints()
    if checkpoints:
        st.markdown("### ♻️ Resume Checkpoint")
        for meta in checkpoints:
            cache_id = meta.get("cache_id", "")
            total = len(meta.get("image_paths", []))
            assigned = len(meta.get("assignments", {}))
            skipped = len(meta.get("skipped", []))
            updated_at = meta.get("updated_at", 0)
            updated_str = time.strftime("%Y-%m-%d %H:%M", time.localtime(updated_at)) if updated_at else "-"
            c1, c2 = st.columns([4, 1])
            with c1:
                st.markdown(
                    f"**{cache_id}** — {assigned} classified, {skipped} skipped, {total} total · Updated {updated_str}"
                )
            with c2:
                if st.button("Resume", key=f"resume_{cache_id}"):
                    load_checkpoint(cache_id)
                    st.rerun()
        st.divider()

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("### 📂 Input Folder")
        folder = st.text_input("Path to folder with raw face images:",
                               value=st.session_state.input_folder, key="folder_input",
                               placeholder="e.g. D:/images/faces")
        st.session_state.input_folder = folder
        if folder:
            imgs = scan_images(folder)
            if imgs:
                st.success(f"✅ Found **{len(imgs)}** images")
            else:
                st.error("No images found (.jpg/.jpeg/.png)")

    with col2:
        st.markdown("### 🏷️ Classes")
        for i, cls in enumerate(st.session_state.classes):
            c1, c2 = st.columns([5, 1])
            with c1:
                st.markdown(f'<span class="class-badge" style="background:{get_class_color(i)}">{i+1}. {cls}</span>', unsafe_allow_html=True)
            with c2:
                if st.button("✕", key=f"del_{i}", help=f"Remove {cls}"):
                    st.session_state.classes.pop(i)
                    st.rerun()

        nc1, nc2 = st.columns([3, 1])
        with nc1:
            new_name = st.text_input("Add class:", key="new_class_input", placeholder="e.g. Diamond", label_visibility="collapsed")
        with nc2:
            if st.button("➕ Add", key="add_class"):
                if new_name and new_name.strip() not in st.session_state.classes:
                    st.session_state.classes.append(new_name.strip())
                    st.rerun()

    st.divider()
    if st.session_state.input_folder and scan_images(st.session_state.input_folder):
        if st.button("🚀 Start Classification", type="primary", use_container_width=True):
            st.session_state.image_paths = scan_images(st.session_state.input_folder)
            st.session_state.current_index = 0
            st.session_state.assignments = {}
            st.session_state.skipped = set()
            st.session_state.extractor = None
            st.session_state.extraction_started = False
            st.session_state.export_done = False
            cache_id = f"cache_{uuid.uuid4().hex[:8]}"
            st.session_state.checkpoint_id = cache_id
            st.session_state.checkpoint_created_at = time.time()
            st.session_state.checkpoint_manager = CheckpointManager(cache_id)
            save_checkpoint_state()
            st.session_state.step = 2
            st.rerun()
    else:
        st.info("👆 Please select a valid input folder with images to continue.")

# ── STEP 2: Classify ──
def render_classify():
    from extraction_helpers import BatchExtractor, METRIC_LABELS, METRIC_NAMES

    paths = st.session_state.image_paths
    total = len(paths)
    idx = st.session_state.current_index
    classes = st.session_state.classes

    # Init extractor
    if st.session_state.extractor is None:
        with st.spinner("🔄 Loading models & extracting first batch..."):
            ext = BatchExtractor(paths, batch_size=10)
            ext.initial_extract(10)
            st.session_state.extractor = ext
            st.session_state.extraction_started = True

    ext = st.session_state.extractor

    # Ensure current is extracted
    if idx not in ext.cache:
        with st.spinner(f"🔄 Extracting image {idx+1}..."):
            ext.ensure_extracted(idx)
    ext.maybe_prefetch(idx)
    result = ext.get(idx)

    # ── Header ──
    classified = len(st.session_state.assignments)
    skipped = len(st.session_state.skipped)
    remaining = total - classified - skipped

    hc1, hc2, hc3, hc4, hc5 = st.columns(5)
    hc1.metric("📷 Total", total)
    hc2.metric("✅ Classified", classified)
    hc3.metric("⏭️ Skipped", skipped)
    hc4.metric("📋 Remaining", remaining)
    current_class = st.session_state.assignments.get(idx, None)
    status_text = current_class if current_class else ("Skipped" if idx in st.session_state.skipped else "—")
    hc5.metric("🏷️ Current", status_text)

    st.progress(max((classified + skipped) / max(total, 1), 0.0))

    # ── Main Content ──
    if result is None:
        st.error("Extraction result not available.")
        return

    img_col, mask_col, metric_col = st.columns([1, 1, 1.2])

    with img_col:
        st.markdown("#### 📸 Raw Face")
        if result.raw_face is not None:
            st.markdown('<div class="img-container">', unsafe_allow_html=True)
            st.image(result.raw_face, use_container_width=True, channels="RGB")
            st.markdown('</div>', unsafe_allow_html=True)
        elif result.error:
            st.warning(f"⚠️ {result.error}")
            # Show original image as fallback
            orig = cv2.imread(str(paths[idx]))
            if orig is not None:
                st.image(cv2.cvtColor(orig, cv2.COLOR_BGR2RGB), use_container_width=True)
        if getattr(result, "used_fallback", False):
            st.caption("🔁 dlib fallback (RetinaFace gagal deteksi)")
        st.caption(f"📁 {paths[idx].name}")

    with mask_col:
        st.markdown("#### 🎭 Masked Face (RGB)")
        if result.masked_face is not None:
            st.markdown('<div class="img-container">', unsafe_allow_html=True)
            st.image(result.masked_face, use_container_width=True, channels="RGB")
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("No mask available for this image")

    with metric_col:
        st.markdown("#### 🤖 XGBoost Prediction")
        if result.predicted_class:
            pred = result.predicted_class
            pred_color = get_class_color(classes.index(pred)) if pred in classes else "#64748b"
            st.markdown(
                f'<span class="class-badge" style="background:{pred_color}">XGBoost: {pred}</span>',
                unsafe_allow_html=True,
            )
        else:
            st.info("Prediction not available")

        st.markdown("#### 📊 Face Metrics")
        if result.metrics:
            thresholds_map = load_metric_thresholds()
            # Split metrics into two sub-columns
            mc1, mc2 = st.columns(2)
            names = list(result.metrics.keys())
            half = (len(names) + 1) // 2
            max_vals = {"chin_angle_degree": 180, "skeleton_angle_left": 90,
                        "skeleton_angle_right": 90, "jidat_angle_downward": 360}
            for i, name in enumerate(names):
                col_target = mc1 if i < half else mc2
                with col_target:
                    label = METRIC_LABELS.get(name, name)
                    mv = max_vals.get(name, 1.0)
                    color = get_class_color(i)
                    render_metric_card(label, result.metrics[name], color, mv, thresholds_map.get(name))
            # Radar chart
            fig = render_radar_chart(result.metrics, thresholds_map)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

            xgb_infer = load_xgb_infer()
            if xgb_infer is not None:
                if "xgb_predictions" not in st.session_state:
                    st.session_state.xgb_predictions = {}
                if idx not in st.session_state.xgb_predictions:
                    metric_obj = _build_xgb_metric(result.metrics)
                    if metric_obj is not None:
                        try:
                            st.session_state.xgb_predictions[idx] = xgb_infer.inference(metric_obj)
                        except Exception:
                            st.session_state.xgb_predictions[idx] = None
                    else:
                        st.session_state.xgb_predictions[idx] = None

                pred = st.session_state.xgb_predictions.get(idx)
                if pred:
                    if pred in classes:
                        pred_color = get_class_color(classes.index(pred))
                    else:
                        pred_color = "#64748b"
                    st.markdown(
                        f'<span class="class-badge" style="background:{pred_color}">XGBoost: {pred}</span>',
                        unsafe_allow_html=True,
                    )
        else:
            st.info("No metrics — extraction failed for this image")

    # ── Navigation + Class Assignment ──
    st.divider()

    nav_c1, nav_c2, nav_c3 = st.columns([1, 2, 1])
    with nav_c1:
        if st.button("◀ Previous [Left]", disabled=(idx <= 0), use_container_width=True, key="btn_prev"):
            st.session_state.current_index = max(0, idx - 1)
            st.rerun()
    with nav_c2:
        st.markdown(f'<div class="nav-info">Image <b>{idx+1}</b> of <b>{total}</b></div>', unsafe_allow_html=True)
    with nav_c3:
        if st.button("[Right] Next ▶", disabled=(idx >= total - 1), use_container_width=True, key="btn_next"):
            st.session_state.current_index = min(total - 1, idx + 1)
            st.rerun()

    # Class buttons
    st.markdown("##### 🏷️ Assign Class:")
    btn_cols = st.columns(len(classes) + 1)
    for i, cls in enumerate(classes):
        with btn_cols[i]:
            is_active = (st.session_state.assignments.get(idx) == cls)
            label = f"{'✓ ' if is_active else ''}[{i+1}] {cls}"
            if st.button(label, key=f"cls_{i}", use_container_width=True,
                         type="primary" if is_active else "secondary"):
                st.session_state.assignments[idx] = cls
                st.session_state.skipped.discard(idx)
                save_checkpoint_record(idx, paths[idx], result, cls)
                save_checkpoint_state()
                # Auto-advance
                if idx < total - 1:
                    st.session_state.current_index = idx + 1
                st.rerun()
    with btn_cols[-1]:
        is_skipped = idx in st.session_state.skipped
        skip_label = "✓ [S] Skip" if is_skipped else "[S] Skip"
        if st.button(skip_label, key="btn_skip", use_container_width=True):
            if is_skipped:
                st.session_state.skipped.discard(idx)
            else:
                st.session_state.skipped.add(idx)
                st.session_state.assignments.pop(idx, None)
            save_checkpoint_state()
            if idx < total - 1:
                st.session_state.current_index = idx + 1
            st.rerun()

    # Finish button
    st.divider()
    fc1, fc2 = st.columns([3, 1])
    with fc2:
        if st.button("📊 Finish & Review", type="primary", use_container_width=True):
            st.session_state.step = 3
            st.rerun()

    # Quick stats (collapsible)
    with st.expander("📊 Statistik (Per Kelas)", expanded=False):
        class_counts = {cls: 0 for cls in classes}
        for cls_name in st.session_state.assignments.values():
            if cls_name in class_counts:
                class_counts[cls_name] += 1

        rc1, rc2, rc3 = st.columns(3)
        rc1.metric("✅ Classified", classified)
        rc2.metric("⏭️ Skipped", skipped)
        rc3.metric("❓ Unclassified", remaining)

        if current_class:
            st.caption(f"Target saat ini: {current_class}")

        for i, cls in enumerate(classes):
            count = class_counts.get(cls, 0)
            pct = (count / max(total, 1)) * 100
            color = get_class_color(i)
            st.markdown(
                f'<span class="class-badge" style="background:{color}">{cls}: {count} ({pct:.1f}%)</span>',
                unsafe_allow_html=True,
            )

    # ── Keyboard Shortcuts (JS injection) ──
    shortcut_js = """<script>
    const doc = window.parent.document;
    doc.addEventListener('keydown', function(e) {
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
        const key = e.key;
        if (key === 'ArrowLeft') {
            const btn = Array.from(doc.querySelectorAll('button')).find(
                b => b.textContent.includes('Previous') && b.textContent.includes('[Left]')
            );
            if (btn && !btn.disabled) btn.click();
        } else if (key === 'ArrowRight') {
            const btn = Array.from(doc.querySelectorAll('button')).find(
                b => b.textContent.includes('Next') && b.textContent.includes('[Right]')
            );
            if (btn && !btn.disabled) btn.click();
        } else if (key === 's' || key === 'S') {
            const btn = Array.from(doc.querySelectorAll('button')).find(b => b.textContent.includes('[S]'));
            if (btn) btn.click();
        } else if (key >= '1' && key <= '9') {
            const idx = parseInt(key) - 1;
            const btns = Array.from(doc.querySelectorAll('button')).filter(b => {
                const t = b.textContent;
                return t.includes('[' + key + ']') && !t.includes('Previous') && !t.includes('Next');
            });
            if (btns.length > 0) btns[0].click();
        }
    });
    </script>"""
    st.components.v1.html(shortcut_js, height=0)

# ── STEP 3: Review ──
def render_review():
    from extraction_helpers import METRIC_NAMES, METRIC_LABELS

    st.markdown("# 📊 Dataset Review")
    classes = st.session_state.classes
    assignments = st.session_state.assignments
    ext = st.session_state.extractor
    mgr = get_checkpoint_manager()
    total = len(st.session_state.image_paths)
    classified = len(assignments)
    skipped = len(st.session_state.skipped)
    unclassified = total - classified - skipped

    # Summary header
    sc1, sc2, sc3, sc4 = st.columns(4)
    sc1.metric("📷 Total Images", total)
    sc2.metric("✅ Classified", classified)
    sc3.metric("⏭️ Skipped", skipped)
    sc4.metric("❓ Unclassified", unclassified)

    if unclassified > 0:
        st.warning(f"⚠️ {unclassified} images are unclassified and will NOT be exported.")

    # Per-class statistics
    st.markdown("### 📈 Per-Class Statistics")

    class_data = {cls: [] for cls in classes}
    class_xgb = {cls: [] for cls in classes}
    xgb_infer = load_xgb_infer()
    for img_idx, cls_name in assignments.items():
        if cls_name in class_data:
            r = resolve_result(img_idx, ext, mgr)
            if r and r.metrics:
                class_data[cls_name].append(r.metrics)

    # XGBoost prediction summary
    pred_counts = {cls: 0 for cls in classes}
    pred_extra = {}
    pred_total = 0
    pred_match = 0
    for img_idx, cls_name in assignments.items():
        r = resolve_result(img_idx, ext, mgr)
        pred = r.predicted_class if r else None
        if pred:
            pred_total += 1
            if pred in pred_counts:
                pred_counts[pred] += 1
            else:
                pred_extra[pred] = pred_extra.get(pred, 0) + 1
            if pred == cls_name:
                pred_match += 1

    if pred_total > 0:
        st.markdown("### 🤖 XGBoost Prediction Summary")
        pc1, pc2, pc3 = st.columns(3)
        pc1.metric("Predicted", pred_total)
        pc2.metric("Matches", pred_match)
        pc3.metric("Match Rate", f"{(pred_match / max(pred_total, 1)) * 100:.1f}%")
        for cls in classes:
            color = get_class_color(classes.index(cls))
            st.markdown(
                f'<span class="class-badge" style="background:{color}">{cls}: {pred_counts[cls]}</span>',
                unsafe_allow_html=True,
            )
        for cls, count in pred_extra.items():
            st.markdown(
                f'<span class="class-badge" style="background:#64748b">{cls}: {count}</span>',
                unsafe_allow_html=True,
            )

    if xgb_infer is not None:
        if "xgb_predictions" not in st.session_state:
            st.session_state.xgb_predictions = {}
        for img_idx, cls_name in assignments.items():
            if cls_name not in class_xgb:
                continue
            r = resolve_result(img_idx, ext, mgr)
            if not r or not r.metrics:
                continue
            if img_idx not in st.session_state.xgb_predictions:
                metric_obj = _build_xgb_metric(r.metrics)
                if metric_obj is not None:
                    try:
                        st.session_state.xgb_predictions[img_idx] = xgb_infer.inference(metric_obj)
                    except Exception:
                        st.session_state.xgb_predictions[img_idx] = None
                else:
                    st.session_state.xgb_predictions[img_idx] = None
            pred = st.session_state.xgb_predictions.get(img_idx)
            if pred:
                class_xgb[cls_name].append(pred)

    tabs = st.tabs(classes)
    for tab, cls in zip(tabs, classes):
        with tab:
            data = class_data[cls]
            color = get_class_color(classes.index(cls))
            st.markdown(f'<span class="class-badge" style="background:{color};font-size:16px">{cls}: {len(data)} images</span>', unsafe_allow_html=True)

            if not data:
                st.info("No images assigned to this class yet. Empty folder will still be created.")
                continue

            # Stats table
            import pandas as pd
            rows = []
            for m in METRIC_NAMES:
                vals = [d[m] for d in data if d.get(m) is not None]
                if vals:
                    rows.append({
                        "Metric": METRIC_LABELS.get(m, m),
                        "Count": len(vals),
                        "Mean": f"{np.mean(vals):.4f}",
                        "Std": f"{np.std(vals):.4f}",
                        "Min": f"{np.min(vals):.4f}",
                        "Max": f"{np.max(vals):.4f}",
                    })
            if rows:
                df = pd.DataFrame(rows)
                st.dataframe(df, use_container_width=True, hide_index=True)

            if class_xgb.get(cls):
                pred_counts = pd.Series(class_xgb[cls]).value_counts().rename_axis("XGBoost")
                pred_df = pred_counts.reset_index(name="Count")
                st.markdown("#### 🤖 XGBoost Prediction Breakdown")
                st.dataframe(pred_df, use_container_width=True, hide_index=True)

            # Box plot
            if len(data) >= 2:
                fig, axes = plt.subplots(2, 5, figsize=(16, 6))
                fig.patch.set_facecolor("#0f0c29")
                fig.suptitle(f"{cls} — Metric Distributions", color="white", fontsize=14)
                for i, m in enumerate(METRIC_NAMES):
                    ax = axes[i // 5][i % 5]
                    vals = [d[m] for d in data if d.get(m) is not None]
                    if vals:
                        bp = ax.boxplot(vals, patch_artist=True, widths=0.6)
                        bp['boxes'][0].set_facecolor(color + "44")
                        bp['boxes'][0].set_edgecolor(color)
                        bp['medians'][0].set_color("white")
                        for w in bp['whiskers'] + bp['caps']:
                            w.set_color("#666")
                    ax.set_title(METRIC_LABELS.get(m, m)[:12], fontsize=8, color="#ccc")
                    ax.set_facecolor("#1a1333")
                    ax.tick_params(colors="#888", labelsize=6)
                    for spine in ax.spines.values():
                        spine.set_color("#333")
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

    # Navigation
    st.divider()
    bc1, bc2, bc3 = st.columns([1, 1, 1])
    with bc1:
        if st.button("◀ Back to Classify", use_container_width=True):
            st.session_state.step = 2
            st.rerun()
    with bc3:
        if st.button("💾 Export Dataset", type="primary", use_container_width=True):
            st.session_state.step = 4
            st.rerun()

# ── STEP 4: Export ──
def render_export():
    from extraction_helpers import METRIC_NAMES, METRIC_LABELS

    st.markdown("# 💾 Export Dataset")
    classes = st.session_state.classes
    assignments = st.session_state.assignments
    ext = st.session_state.extractor
    mgr = get_checkpoint_manager()
    paths = st.session_state.image_paths
    n_classes = len(classes)
    uid = st.session_state.export_uid
    folder_name = f"dataset-{n_classes}-kelas-{uid}"
    output_root = PROJECT_ROOT / "output" / folder_name

    # Confirmation
    st.markdown(f"""<div class="status-card">
        <h3>📁 Output: <code>output/{folder_name}/</code></h3>
        <p>Classes: {', '.join(classes)} | Images to export: {len(assignments)}</p>
    </div>""", unsafe_allow_html=True)

    # Show breakdown
    for cls in classes:
        count = sum(1 for v in assignments.values() if v == cls)
        color = get_class_color(classes.index(cls))
        st.markdown(f'<span class="class-badge" style="background:{color}">{cls}: {count} images</span>', unsafe_allow_html=True)

    st.divider()

    if st.session_state.export_done:
        st.success(f"✅ Dataset exported to `output/{folder_name}/`")
        if st.button("🏠 Start New Dataset", use_container_width=True):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()
        return

    col1, col2 = st.columns(2)
    with col1:
        if st.button("◀ Back to Review", use_container_width=True):
            st.session_state.step = 3
            st.rerun()
    with col2:
        confirm = st.button("✅ Confirm Export", type="primary", use_container_width=True)

    if confirm:
        progress = st.progress(0, "Exporting...")
        status = st.empty()

        # Create class dirs
        for cls in classes:
            (output_root / cls / "face_skin").mkdir(parents=True, exist_ok=True)

        # Group by class
        class_images = {cls: [] for cls in classes}
        for img_idx, cls_name in assignments.items():
            if cls_name in class_images:
                class_images[cls_name].append(img_idx)

        total_export = len(assignments)
        done = 0

        for cls in classes:
            indices = class_images[cls]
            csv_path = output_root / cls / "metrics.csv"

            with csv_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["class_name", "image_file", "relative_path"] + METRIC_NAMES)

                for img_idx in indices:
                    r = resolve_result(img_idx, ext, mgr, want_image=True)
                    img_path = paths[img_idx]

                    if r and r.masked_face is not None:
                        out_img = output_root / cls / "face_skin" / img_path.name
                        cv2.imwrite(str(out_img), cv2.cvtColor(r.masked_face, cv2.COLOR_RGB2BGR))

                    if r and r.metrics:
                        rel_path = f"{cls}/{img_path.name}"
                        row = [cls, img_path.name, rel_path] + [r.metrics.get(m, "") for m in METRIC_NAMES]
                        writer.writerow(row)

                    done += 1
                    progress.progress(done / max(total_export, 1), f"Exporting {cls}... ({done}/{total_export})")
                    status.text(f"Saved: {img_path.name} → {cls}/")

        progress.progress(1.0, "✅ Export complete!")
        st.session_state.export_done = True
        st.balloons()
        st.rerun()

# ── Main Router ──
step = st.session_state.step
if step == 1:
    render_setup()
elif step == 2:
    render_classify()
elif step == 3:
    render_review()
elif step == 4:
    render_export()
