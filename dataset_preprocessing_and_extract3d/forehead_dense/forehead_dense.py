import numpy as np
import cv2

def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-8:
        return np.array([0.0, -1.0], dtype=np.float32)
    return (v / n).astype(np.float32)

def _rotate(v: np.ndarray, deg: float) -> np.ndarray:
    rad = np.deg2rad(deg)
    c, s = np.cos(rad), np.sin(rad)
    return np.array([v[0] * c - v[1] * s, v[0] * s + v[1] * c], dtype=np.float32)

def _clip_point(pt_xy: np.ndarray, h: int, w: int) -> tuple[int, int]:
    x = int(np.clip(round(float(pt_xy[0])), 0, w - 1))
    y = int(np.clip(round(float(pt_xy[1])), 0, h - 1))
    return (x, y)

def _point_at_length(
    length_val: float,
    dir_xy: np.ndarray,
    center: np.ndarray,
    h: int,
    w: int,
) -> tuple[int, int]:
    pt = center + dir_xy * float(length_val)
    return _clip_point(pt, h, w)

def _ray_segments(dir_xy: np.ndarray, center: np.ndarray, mask_bin: np.ndarray, h: int, w: int):
    max_steps = int(np.hypot(h, w))
    segments = []
    in_run = False
    run_start = 0
    run_len = 0

    for t in range(1, max_steps + 1):
        x = int(round(center[0] + dir_xy[0] * t))
        y = int(round(center[1] + dir_xy[1] * t))

        if x < 0 or x >= w or y < 0 or y >= h:
            if in_run:
                segments.append((run_start, t - 1, run_len))
            break

        if mask_bin[y, x] > 0:
            if not in_run:
                in_run = True
                run_start = t
                run_len = 1
            else:
                run_len += 1
        else:
            if in_run:
                segments.append((run_start, t - 1, run_len))
                in_run = False
                run_len = 0

    return segments

def _farthest_white_hit(dir_xy: np.ndarray, center: np.ndarray, mask_bin: np.ndarray, h: int, w: int):
    segments = _ray_segments(dir_xy, center, mask_bin, h, w)
    if not segments:
        return None
    best = None
    for s, e, run_len in segments:
        score = e + 0.05 * run_len
        if (best is None) or (score > best[0]):
            best = (score, s, e, run_len)
    assert best is not None
    _, s, e, run_len = best
    return e, _point_at_length(float(e), dir_xy, center, h, w), run_len

def _snap_near_target(
    dir_xy: np.ndarray,
    target_len: float,
    center: np.ndarray,
    mask_bin: np.ndarray,
    h: int,
    w: int,
    window: int = 40,
) -> tuple[int, int]:
    target_len = float(target_len)
    candidates = [target_len]
    for d in range(1, window + 1):
        candidates.extend([target_len + d, target_len - d])

    for L in candidates:
        if L <= 0:
            continue
        x, y = _point_at_length(L, dir_xy, center, h, w)
        if mask_bin[y, x] > 0:
            return (x, y)

    return _point_at_length(target_len, dir_xy, center, h, w)

def _ellipse_len(dir_xy: np.ndarray, rx: float, ry: float) -> float:
    dx, dy = float(dir_xy[0]), float(dir_xy[1])
    denom = (dx * dx) / (rx * rx) + (dy * dy) / (ry * ry)
    return float(1.0 / np.sqrt(max(denom, 1e-8)))


def _predict_len(
    a: int,
    prev_len: float | None,
    prev_prev_len: float | None,
    geo_len: dict,
    face_w: float,
) -> float:
    g = geo_len[a]
    if prev_len is None:
        return float(g)
    if prev_prev_len is None:
        return float(0.70 * prev_len + 0.30 * g)
    slope = float(np.clip(prev_len - prev_prev_len, -0.18 * face_w, 0.24 * face_w))
    pred = prev_len + slope
    return float(0.75 * pred + 0.25 * g)


def _sweep_side(
    side_angles,
    side_name: str,
    geo_len: dict,
    face_w: float,
    face_h: float,
    center_y: float,
    center_x: float,
    center: np.ndarray,
    h: int,
    w: int,
    dirs: dict,
    raw_valid: dict,
    raw_len: dict,
    raw_run: dict,
    final_len: dict,
    source: dict,
) -> None:
    prev_len = None
    prev_prev_len = None
    prev_y = None

    for a in side_angles:
        pred = _predict_len(a, prev_len, prev_prev_len, geo_len, face_w)
        g = geo_len[a]

        if raw_valid[a]:
            cand = float(raw_len[a])
            src = "mask-farthest"
        else:
            cand = float(pred)
            src = "fallback-none"

        band_low = max(0.11 * face_w, 0.22 * g)
        band_high = max(0.12 * face_w, 0.26 * g)
        min_allow = max(0.55 * g, pred - band_low)
        max_allow = min(1.55 * g, pred + band_high)

        if prev_len is not None:
            ratio_floor = 0.84
            if abs(a - 90) <= 50:
                ratio_floor = 0.88
            min_allow = max(min_allow, prev_len * ratio_floor)
            max_allow = min(max_allow, prev_len * 1.22 + 0.08 * face_w)

        if (cand < min_allow) or (cand > max_allow):
            cand = float(np.clip(pred, min_allow, max_allow))
            src = "fallback-trend"

        if raw_valid[a] and raw_run[a] <= 2 and src.startswith("mask"):
            cand = float(np.clip(0.60 * cand + 0.40 * pred, min_allow, max_allow))
            src = "fallback-collision"

        probe = _point_at_length(cand, dirs[a], center, h, w)
        if prev_y is not None:
            y_drop_limit = max(6.0, 0.08 * face_h)
            if probe[1] > prev_y + y_drop_limit:
                dy = float(dirs[a][1])
                target_y = prev_y + y_drop_limit
                if dy < -1e-6:
                    len_needed = (target_y - center_y) / dy
                    cand = max(cand, float(len_needed))
                cand = float(np.clip(cand, min_allow, max_allow))
                probe = _point_at_length(cand, dirs[a], center, h, w)
                src = "fallback-y"

        if side_name == "left" and probe[0] > center_x + 2:
            cand = float(np.clip(pred, min_allow, max_allow))
            probe = _point_at_length(cand, dirs[a], center, h, w)
            src = "fallback-side"
        if side_name == "right" and probe[0] < center_x - 2:
            cand = float(np.clip(pred, min_allow, max_allow))
            probe = _point_at_length(cand, dirs[a], center, h, w)
            src = "fallback-side"

        final_len[a] = cand
        source[a] = src
        prev_prev_len = prev_len
        prev_len = cand
        prev_y = float(probe[1])


def _radius_floor(
    side_angles,
    angle_points: dict,
    center: np.ndarray,
    final_len: dict,
    geo_len: dict,
    cap: dict,
    dirs: dict,
    source: dict,
    mask_bin: np.ndarray,
    h: int,
    w: int,
) -> None:
    prev_r = None
    for a in side_angles:
        pt = np.array(angle_points[a], dtype=np.float32)
        r = float(np.linalg.norm(pt - center))
        if prev_r is not None:
            floor_r = prev_r * (0.88 if abs(a - 90) <= 50 else 0.84)
            if r < floor_r:
                cand = max(final_len[a], floor_r)
                cand = float(np.clip(cand, 0.55 * geo_len[a], min(1.55 * geo_len[a], cap[a])))
                angle_points[a] = _snap_near_target(dirs[a], cand, center, mask_bin, h, w, window=44)
                final_len[a] = cand
                source[a] = "fallback-trend"
                pt = np.array(angle_points[a], dtype=np.float32)
                r = float(np.linalg.norm(pt - center))
        prev_r = r

def forehead_points_dense(face_mask: np.ndarray,
                        landmarks: np.ndarray,
                        angle_start: int = 0,
                        angle_end: int = 180,
                        angle_step: int = 10
                        ) -> dict[int, tuple[int, int]]:
    """
    Dense forehead boundary ala Cell 17, dengan aturan:
    - Ambil kandidat dari titik putih TERJAUH dari titik 27 pada tiap sudut.
    - Jika kandidat jelek (collision/rambut/putus), fallback berurutan dari tren panjang sudut sebelumnya.
    - Sudut 90 diproses terakhir dan tetap vertikal dari titik 27.
    """
    mask = np.asarray(face_mask)

    if mask.ndim == 3 and mask.shape[2] == 1:
        mask = mask[:, :, 0]
    elif mask.ndim == 3 and mask.shape[2] >= 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)

    if mask.ndim != 2:
        raise ValueError(f"face_mask harus 2D/3D, dapat shape={mask.shape}")

    mask_bin = (mask > 0).astype(np.uint8)
    h, w = mask_bin.shape

    lm = np.asarray(landmarks).reshape(-1, 2).astype(np.float32)
    if lm.shape[0] < 68:
        raise ValueError("landmarks minimal harus 68 titik")

    p0 = lm[0]
    p8 = lm[8]
    p16 = lm[16]
    p17 = lm[17]
    p19 = lm[19]
    p24 = lm[24]
    p26 = lm[26]
    p27 = lm[27]

    center = p27.astype(np.float32)
    center_x = float(center[0])
    center_y = float(center[1])

    angles = list(range(int(angle_start), int(angle_end) + 1, int(angle_step)))
    if 90 not in angles:
        angles.append(90)
    angles = sorted(set([a for a in angles if 0 <= a <= 180]))

    up_vec = np.array([0.0, -1.0], dtype=np.float32)
    dirs = {a: _normalize(_rotate(up_vec, float(a - 90))) for a in angles}

    face_w = float(np.linalg.norm(p16 - p0))
    face_h = float(np.linalg.norm(p8 - p27))

    rx = 0.49 * face_w
    ry = 0.70 * face_h
    ry = float(np.clip(ry, 0.38 * face_w, 1.08 * face_h))
    rx = float(max(rx, 1e-6))
    ry = float(max(ry, 1e-6))

    geo_len = {a: _ellipse_len(dirs[a], rx, ry) for a in angles}

    cap_left = min(
        float(np.linalg.norm(p27 - p0)) * 0.98,
        float(np.linalg.norm(p27 - p17)) + 0.90 * float(np.linalg.norm(p17 - p0)),
    )
    cap_right = min(
        float(np.linalg.norm(p27 - p16)) * 0.98,
        float(np.linalg.norm(p27 - p26)) + 0.90 * float(np.linalg.norm(p26 - p16)),
    )
    cap_mid = 1.22 * face_h

    cap = {}
    for a in angles:
        t = a / 180.0
        cap_lr = (1.0 - t) * cap_left + t * cap_right
        cap_top = cap_mid * np.sin(np.deg2rad(a))
        cap[a] = float(max(0.40 * geo_len[a], min(1.55 * geo_len[a], cap_lr + 0.22 * cap_top)))

    raw_len = {}
    raw_point = {}
    raw_run = {}
    raw_valid = {}

    for a in angles:
        hit = _farthest_white_hit(dirs[a], center, mask_bin, h, w)
        if hit is None:
            raw_len[a] = np.nan
            raw_point[a] = None
            raw_run[a] = 0
            raw_valid[a] = False
            continue

        e, pt, run_len = hit
        raw_len[a] = float(e)
        raw_point[a] = pt
        raw_run[a] = int(run_len)

        raw_valid[a] = bool(
            (raw_len[a] >= 0.20 * geo_len[a])
            and (raw_len[a] <= 1.95 * geo_len[a])
            and (raw_run[a] >= 2)
        )

    final_len = {}
    source = {}

    left_angles = sorted([a for a in angles if a < 90])
    right_angles = sorted([a for a in angles if a > 90], reverse=True)

    _sweep_side(
        left_angles,
        "left",
        geo_len,
        face_w,
        face_h,
        center_y,
        center_x,
        center,
        h,
        w,
        dirs,
        raw_valid,
        raw_len,
        raw_run,
        final_len,
        source,
    )
    _sweep_side(
        right_angles,
        "right",
        geo_len,
        face_w,
        face_h,
        center_y,
        center_x,
        center,
        h,
        w,
        dirs,
        raw_valid,
        raw_len,
        raw_run,
        final_len,
        source,
    )

    if 90 in angles:
        neighbor_vals = []
        if 80 in final_len:
            neighbor_vals.append(final_len[80])
        if 100 in final_len:
            neighbor_vals.append(final_len[100])
        if not neighbor_vals:
            if left_angles:
                neighbor_vals.append(final_len[left_angles[-1]])
            if right_angles:
                neighbor_vals.append(final_len[right_angles[-1]])

        neighbor_ref = max(neighbor_vals) if neighbor_vals else geo_len[90]
        pred90 = float(max(0.98 * neighbor_ref, geo_len[90]))

        if raw_valid.get(90, False) and raw_run.get(90, 0) >= 3:
            cand90 = float(raw_len[90])
            src90 = "mask-farthest"
        else:
            cand90 = pred90
            src90 = "fallback-none"

        min90 = max(0.72 * neighbor_ref, 0.45 * geo_len[90])
        max90 = min(1.18 * neighbor_ref + 0.05 * face_w, 1.45 * geo_len[90], cap[90])

        if (cand90 < min90) or (cand90 > max90):
            cand90 = float(np.clip(pred90, min90, max90))
            src90 = "fallback-trend"

        if len(neighbor_vals) >= 2:
            neigh_mean = float(np.mean(neighbor_vals))
            clipped = float(np.clip(cand90, 0.92 * neigh_mean, 1.08 * neigh_mean))
            if abs(clipped - cand90) > 1e-6:
                cand90 = clipped
                src90 = "fallback-center"

        final_len[90] = cand90
        source[90] = src90

    # Global anti-spike smoothing di domain panjang sudut.
    for _ in range(2):
        for i in range(1, len(angles)):
            a0 = angles[i - 1]
            a1 = angles[i]
            max_delta = max(0.16 * face_w, 0.34 * geo_len[a1])
            upper = final_len[a0] + max_delta
            lower = final_len[a0] - max_delta
            old = final_len[a1]
            final_len[a1] = float(np.clip(final_len[a1], lower, upper))
            if abs(final_len[a1] - old) > 1e-6:
                source[a1] = "fallback-smooth"

        for i in range(len(angles) - 2, -1, -1):
            a0 = angles[i + 1]
            a1 = angles[i]
            max_delta = max(0.16 * face_w, 0.34 * geo_len[a1])
            upper = final_len[a0] + max_delta
            lower = final_len[a0] - max_delta
            old = final_len[a1]
            final_len[a1] = float(np.clip(final_len[a1], lower, upper))
            if abs(final_len[a1] - old) > 1e-6:
                source[a1] = "fallback-smooth"

    # Center-band clamp untuk meredam notch/spike sekitar 90..120.
    center_band = [a for a in angles if 70 <= a <= 120]
    for _ in range(2):
        for i in range(1, len(center_band)):
            a0 = center_band[i - 1]
            a1 = center_band[i]
            max_delta = max(0.10 * face_w, 0.18 * geo_len[a1])
            old = final_len[a1]
            final_len[a1] = float(np.clip(final_len[a1], final_len[a0] - max_delta, final_len[a0] + max_delta))
            if abs(final_len[a1] - old) > 1e-6:
                source[a1] = "fallback-center"

        for i in range(len(center_band) - 2, -1, -1):
            a0 = center_band[i + 1]
            a1 = center_band[i]
            max_delta = max(0.10 * face_w, 0.18 * geo_len[a1])
            old = final_len[a1]
            final_len[a1] = float(np.clip(final_len[a1], final_len[a0] - max_delta, final_len[a0] + max_delta))
            if abs(final_len[a1] - old) > 1e-6:
                source[a1] = "fallback-center"

    for i in range(1, len(angles) - 1):
        a_prev = angles[i - 1]
        a = angles[i]
        a_next = angles[i + 1]
        if a == 90:
            continue
        med = float(np.median([final_len[a_prev], final_len[a], final_len[a_next]]))
        tol = max(0.12 * face_w, 0.20 * geo_len[a])
        if abs(final_len[a] - med) > tol:
            final_len[a] = float(0.65 * final_len[a] + 0.35 * med)
            source[a] = "fallback-smooth"

    angle_points = {}
    for a in angles:
        use_raw = (
            source.get(a, "") == "mask-farthest"
            and raw_point[a] is not None
            and np.isfinite(raw_len[a])
            and abs(raw_len[a] - final_len[a]) <= 0.09 * geo_len[a]
        )
        if use_raw:
            angle_points[a] = raw_point[a]
        else:
            angle_points[a] = _snap_near_target(dirs[a], final_len[a], center, mask_bin, h, w, window=40)
            if source.get(a, "").startswith("mask"):
                source[a] = "fallback-trend"

    _radius_floor(left_angles, angle_points, center, final_len, geo_len, cap, dirs, source, mask_bin, h, w)
    _radius_floor(right_angles, angle_points, center, final_len, geo_len, cap, dirs, source, mask_bin, h, w)

    for a in left_angles:
        x, y = angle_points[a]
        x = min(x, int(np.floor(center_x)) - 1)
        angle_points[a] = (int(np.clip(x, 0, w - 1)), y)

    right_sorted = sorted([a for a in angles if a > 90])

    if left_angles:
        last_x = -10**9
        for a in left_angles:
            x, y = angle_points[a]
            if x < last_x:
                x = last_x
            angle_points[a] = (int(np.clip(x, 0, w - 1)), y)
            last_x = x

    if right_sorted:
        last_x = int(np.ceil(center_x)) + 1
        for a in right_sorted:
            x, y = angle_points[a]
            x = max(x, int(np.ceil(center_x)) + 1)
            if x < last_x:
                x = last_x
            angle_points[a] = (int(np.clip(x, 0, w - 1)), y)
            last_x = x

    if 90 in angle_points:
        p90 = _snap_near_target(dirs[90], final_len[90], center, mask_bin, h, w, window=48)
        angle_points[90] = (int(np.clip(round(center_x), 0, w - 1)), int(np.clip(p90[1], 0, h - 1)))

    return angle_points

if __name__ == "__main__":
    def build_face_boundary_dense(landmarks: np.ndarray, angle_points: dict) -> list:
        lm = np.asarray(landmarks).reshape(-1, 2).astype(np.int32)
        jaw = [tuple(map(int, lm[i])) for i in range(17)]
        top_angles = sorted(angle_points.keys(), reverse=True)
        top_chain = [tuple(map(int, angle_points[a])) for a in top_angles]
        return jaw + top_chain + [jaw[0]]
    
    def _pick_one_test_image(test_root):
        patterns = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
        class_dirs = sorted([d for d in test_root.iterdir() if d.is_dir()])
        for cls_dir in class_dirs:
            files = []
            for p in patterns:
                files.extend(sorted(cls_dir.glob(p)))
            if files:
                return cls_dir.name, files[0]
        raise FileNotFoundError(f"Tidak ada gambar di {test_root}")


    def _debug_one_image_notebook_flow() -> None:
        import os
        import sys
        import platform
        from pathlib import Path
        from typing import Any, Callable, cast

        # Samakan setup dengan notebook sebelum import RetinaFace/TensorFlow.
        os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
        # Workaround untuk beberapa environment Windows/Python baru yang freeze di platform.processor().
        platform.processor = (lambda: "")

        try:
            import dlib
            from retinaface import RetinaFace
        except Exception as exc:
            raise RuntimeError(
                "Gagal import dlib/retinaface. Pastikan dependency ter-install dan gunakan Python 3.10/3.11 untuk stack TensorFlow/RetinaFace."
            ) from exc

        this_file = Path(__file__).resolve()
        project_root = this_file.parents[2]
        preprocessing_root = this_file.parents[1]
        dataset_root = project_root / "dataset" / "face_shape" / "testing_set"
        predictor_path = project_root / "models" / "shape_predictor_68_face_landmarks.dat"

        if not dataset_root.exists():
            raise FileNotFoundError(f"Dataset tidak ditemukan: {dataset_root}")
        if not predictor_path.exists():
            raise FileNotFoundError(f"Model dlib tidak ditemukan: {predictor_path}")

        if str(preprocessing_root) not in sys.path:
            sys.path.insert(0, str(preprocessing_root))

        try:
            from face_parsing.face_parsing import FaceParsingExtractor
        except Exception as exc:
            raise RuntimeError(
                "Gagal import FaceParsingExtractor. Cek path/module face_parsing."
            ) from exc

        cls_name, img_path = _pick_one_test_image(dataset_root)
        print("Debug flow (1 image) mengikuti notebook:")
        print("1) RetinaFace align=True, expand_face_area=15")
        print("2) Dlib 68 landmarks")
        print("3) Face parsing mask")
        print("4) Forehead dense angle 0..180 step 10")
        print(f"Class: {cls_name}")
        print(f"Image: {img_path}")

        img_path_str = str(img_path)
        faces = RetinaFace.extract_faces(img_path=img_path_str, align=True, expand_face_area=15)
        if len(faces) <= 0:
            raise RuntimeError("RetinaFace tidak mendeteksi wajah pada gambar debug.")

        aligned_face = faces[0]
        if aligned_face.dtype != np.uint8:
            scale = 255.0 if aligned_face.max() <= 1.0 else 1.0
            aligned_face = np.clip(aligned_face * scale, 0, 255).astype(np.uint8)
        if aligned_face.ndim == 2:
            aligned_face = cv2.cvtColor(aligned_face, cv2.COLOR_GRAY2RGB)

        detector_factory = getattr(dlib, "get_frontal_face_detector", None)
        predictor_factory = getattr(dlib, "shape_predictor", None)
        if not callable(detector_factory) or not callable(predictor_factory):
            raise RuntimeError("Versi dlib tidak menyediakan detector/predictor API yang dibutuhkan.")

        detector_factory = cast(Callable[[], Any], detector_factory)
        predictor_factory = cast(Callable[[str], Any], predictor_factory)
        hog_face_detector: Any = detector_factory()
        dlib_facelandmark: Any = predictor_factory(str(predictor_path))

        face_bgr = cv2.cvtColor(aligned_face, cv2.COLOR_RGB2BGR)
        gray_img = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        detected_faces = hog_face_detector(gray_img)
        if len(detected_faces) <= 0:
            raise RuntimeError("Dlib tidak mendeteksi wajah pada aligned face debug.")

        shape = dlib_facelandmark(gray_img, detected_faces[0])
        landmarks = np.array([(shape.part(i).x, shape.part(i).y) for i in range(68)], dtype=np.int32)

        face_parser = FaceParsingExtractor()
        parsing_mask = face_parser.extract(aligned_face)

        angle_points = forehead_points_dense(
            parsing_mask,
            landmarks,
            angle_start=0,
            angle_end=180,
            angle_step=10,
        )
        boundary_points = build_face_boundary_dense(landmarks, angle_points)

        if parsing_mask.ndim == 2:
            vis = cv2.cvtColor(parsing_mask.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        elif parsing_mask.ndim == 3 and parsing_mask.shape[2] == 1:
            vis = cv2.cvtColor(parsing_mask[:, :, 0].astype(np.uint8), cv2.COLOR_GRAY2RGB)
        else:
            vis = parsing_mask.copy().astype(np.uint8)

        if vis.max() <= 1:
            vis = (vis * 255).astype(np.uint8)

        overlay = vis.copy()
        cv2.fillPoly(overlay, [np.array(boundary_points, dtype=np.int32)], (70, 70, 255))
        vis = cv2.addWeighted(overlay, 0.16, vis, 0.84, 0)

        jaw_pts = [tuple(map(int, landmarks[idx])) for idx in range(17)]
        for j in range(16):
            cv2.line(vis, jaw_pts[j], jaw_pts[j + 1], (0, 220, 255), 1)

        top_angles_desc = sorted(angle_points.keys(), reverse=True)
        top_chain = [angle_points[a] for a in top_angles_desc]
        for j in range(len(top_chain) - 1):
            cv2.line(vis, top_chain[j], top_chain[j + 1], (0, 255, 0), 1)

        if 180 in angle_points:
            cv2.line(vis, jaw_pts[16], angle_points[180], (255, 0, 255), 2)
        if 0 in angle_points:
            cv2.line(vis, angle_points[0], jaw_pts[0], (255, 0, 255), 2)
        cv2.polylines(vis, [np.array(boundary_points, dtype=np.int32)], True, (255, 0, 255), 2)

        for a in sorted(angle_points.keys()):
            pt = tuple(map(int, angle_points[a]))
            cv2.circle(vis, pt, 2, (0, 255, 0), -1)
            if a % 20 == 0 or a == 90:
                cv2.putText(
                    vis,
                    str(a),
                    (pt[0] + 2, pt[1] - 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    (255, 60, 60),
                    1,
                    cv2.LINE_AA,
                )

        if 90 in angle_points:
            p27 = tuple(map(int, landmarks[27]))
            cv2.circle(vis, p27, 4, (255, 0, 0), -1)
            cv2.line(vis, p27, tuple(map(int, angle_points[90])), (255, 255, 0), 1)

        title = f"Dense Debug 1 Image - {cls_name}"
        vis_bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
        cv2.imshow(title, vis_bgr)
        print("Tampilan debug sudah dibuka. Tekan tombol apa saja untuk menutup jendela.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    _debug_one_image_notebook_flow()
