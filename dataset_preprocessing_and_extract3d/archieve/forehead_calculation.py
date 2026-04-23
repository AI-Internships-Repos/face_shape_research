import numpy as np
import cv2


def forehead_5_boundary(face_mask: np.ndarray,
                        landmarks: np.ndarray,
                        return_debug: bool = False):
    """
    Forehead 5 boundary (mask-first + fallback):
    - Prioritas hasil dari mask putih jika kualitasnya bagus.
    - Fallback geometri hanya dipakai saat titik mask tidak masuk akal.
    - Sudut 90 selalu vertikal dari titik 27 (x konstan, arah ke atas).
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

    def _normalize(v: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(v)
        if n < 1e-8:
            return np.array([0.0, -1.0], dtype=np.float32)
        return (v / n).astype(np.float32)

    def _rotate(v: np.ndarray, deg: float) -> np.ndarray:
        rad = np.deg2rad(deg)
        c, s = np.cos(rad), np.sin(rad)
        return np.array([v[0] * c - v[1] * s, v[0] * s + v[1] * c], dtype=np.float32)

    # Sudut 90 dipatok vertikal lurus ke atas dari titik 27.
    up_vec = np.array([0.0, -1.0], dtype=np.float32)

    angle_to_offset = {
        30: -60.0,
        60: -30.0,
        90: 0.0,
        120: 30.0,
        150: 60.0,
    }
    directions = {a: _normalize(_rotate(up_vec, off)) for a, off in angle_to_offset.items()}
    center = p27.astype(np.float32)
    max_steps = int(np.hypot(h, w))

    def _trace_last_white(start_xy: np.ndarray, dir_xy: np.ndarray):
        sx, sy = float(start_xy[0]), float(start_xy[1])
        last_white = None
        seen_white = False
        miss_after_white = 0

        for t in range(1, max_steps + 1):
            x = int(round(sx + dir_xy[0] * t))
            y = int(round(sy + dir_xy[1] * t))

            if x < 0 or x >= w or y < 0 or y >= h:
                break

            if mask_bin[y, x] > 0:
                last_white = (x, y)
                seen_white = True
                miss_after_white = 0
            elif seen_white:
                miss_after_white += 1
                if miss_after_white >= 3:
                    break

        return last_white

    def _clip_point(pt_xy: np.ndarray) -> tuple[int, int]:
        x = int(np.clip(round(float(pt_xy[0])), 0, w - 1))
        y = int(np.clip(round(float(pt_xy[1])), 0, h - 1))
        return (x, y)

    def _point_at_length(length_val: float, dir_xy: np.ndarray) -> tuple[int, int]:
        pt = center + dir_xy * float(length_val)
        return _clip_point(pt)

    def _snap_near_target(dir_xy: np.ndarray, target_len: float, window: int = 40) -> tuple[int, int]:
        candidates = [target_len]
        for d in range(1, window + 1):
            candidates.extend([target_len - d, target_len + d])

        for L in candidates:
            if L <= 0:
                continue
            x, y = _point_at_length(L, dir_xy)
            if mask_bin[y, x] > 0:
                return (x, y)

        return _point_at_length(target_len, dir_xy)

    face_w = float(np.linalg.norm(p16 - p0))
    face_h = float(np.linalg.norm(p8 - p27))

    # Geometri umum dipakai hanya sebagai fallback / guard rails.
    rx = 0.49 * face_w
    ry = 0.70 * face_h
    ry = float(np.clip(ry, 0.38 * face_w, 1.08 * face_h))
    rx = float(max(rx, 1e-6))
    ry = float(max(ry, 1e-6))

    def _ellipse_len(dir_xy: np.ndarray) -> float:
        dx, dy = float(dir_xy[0]), float(dir_xy[1])
        denom = (dx * dx) / (rx * rx) + (dy * dy) / (ry * ry)
        return float(1.0 / np.sqrt(max(denom, 1e-8)))

    geo_len = {a: _ellipse_len(directions[a]) for a in [30, 60, 90, 120, 150]}

    # Batas jarak agar titik tidak terlalu liar relatif terhadap landmark 0 dan 16.
    cap = {
        30: min(float(np.linalg.norm(p27 - p0)) * 0.98, float(np.linalg.norm(p27 - p17)) + 0.90 * float(np.linalg.norm(p17 - p0))),
        60: float(np.linalg.norm(p27 - p19)) + 1.00 * float(np.linalg.norm(p19 - p0)),
        90: 1.22 * face_h,
        120: float(np.linalg.norm(p27 - p24)) + 1.00 * float(np.linalg.norm(p24 - p16)),
        150: min(float(np.linalg.norm(p27 - p16)) * 0.98, float(np.linalg.norm(p27 - p26)) + 0.90 * float(np.linalg.norm(p26 - p16))),
    }

    raw_points = {}
    raw_len = {}
    raw_valid = {}

    for a in [30, 60, 90, 120, 150]:
        hit = _trace_last_white(center, directions[a])
        if hit is None:
            raw_points[a] = None
            raw_len[a] = 0.0
            raw_valid[a] = False
            continue

        L = float(np.linalg.norm(np.array(hit, dtype=np.float32) - center))
        raw_points[a] = hit
        raw_len[a] = L
        raw_valid[a] = (L >= 0.20 * geo_len[a]) and (L <= 1.70 * geo_len[a])

    # Quality gate mask: jika bagus, pakai mask langsung (dengan clip minimal).
    mask_available = sum(1 for a in [30, 60, 90, 120, 150] if raw_points[a] is not None) >= 4

    if mask_available:
        x0, x16 = float(p0[0]), float(p16[0])
        x_left, x_right = min(x0, x16), max(x0, x16)
        margin_x = 0.10 * face_w

        # Isi titik yang hilang pakai geometri sementara agar pengecekan tetap lengkap.
        tmp_points = {}
        for a in [30, 60, 90, 120, 150]:
            if raw_points[a] is not None:
                tmp_points[a] = raw_points[a]
            else:
                tmp_points[a] = _point_at_length(geo_len[a], directions[a])

        x30, y30 = tmp_points[30]
        x60, y60 = tmp_points[60]
        x90, y90 = tmp_points[90]
        x120, y120 = tmp_points[120]
        x150, y150 = tmp_points[150]

        # Urutan x harus naik dari kiri ke kanan agar tidak miring aneh.
        order_ok = (x30 < x60 < x90 < x120 < x150)

        # 30 dan 150 harus tetap masuk akal relatif ke titik pojok 0 dan 16.
        bound_ok = (
            (x30 >= x_left - margin_x) and
            (x150 <= x_right + margin_x) and
            (x30 <= x90 - 3) and
            (x150 >= x90 + 3)
        )

        # Titik 90 harus cukup tinggi (ke atas) dibanding 60/120.
        vertical_ok = y90 <= min(y60, y120) + int(0.12 * face_h)

        # Simetri kasar pasangan kiri-kanan.
        pair_30_150 = abs(raw_len[30] - raw_len[150]) / max((raw_len[30] + raw_len[150]) / 2.0, 1e-6)
        pair_60_120 = abs(raw_len[60] - raw_len[120]) / max((raw_len[60] + raw_len[120]) / 2.0, 1e-6)
        symmetry_ok = (pair_30_150 <= 0.55) and (pair_60_120 <= 0.50)

        raw_mask_good = order_ok and bound_ok and vertical_ok and symmetry_ok
    else:
        raw_mask_good = False

    if raw_mask_good:
        # Mask bagus: ikutin mask, hanya clip kewajaran terhadap cap.
        final_len = {}
        for a in [30, 60, 90, 120, 150]:
            L = raw_len[a] if raw_valid[a] else geo_len[a]
            final_len[a] = float(np.clip(L, 0.35 * geo_len[a], min(1.35 * geo_len[a], cap[a])))

        # Jaga 90 tetap vertikal kuat agar tidak turun/miring.
        side_mid_max = max(final_len[60], final_len[120])
        if final_len[90] < 0.90 * side_mid_max:
            final_len[90] = min(max(0.95 * side_mid_max, 0.90 * geo_len[90]), min(1.25 * geo_len[90], cap[90]))

        mode = "mask"
    else:
        # Fallback: gunakan geometri + simetri bila mask jelek.
        final_len = {}
        for a in [30, 60, 90, 120, 150]:
            if raw_valid[a]:
                # Tetap pakai sedikit info mask supaya tidak terlalu kaku.
                L = 0.30 * raw_len[a] + 0.70 * geo_len[a]
            else:
                L = geo_len[a]
            final_len[a] = float(np.clip(L, 0.45 * geo_len[a], min(1.35 * geo_len[a], cap[a])))

        for a_left, a_right in [(30, 150), (60, 120)]:
            l1, l2 = final_len[a_left], final_len[a_right]
            pair_avg = (l1 + l2) / 2.0
            diff_ratio = abs(l1 - l2) / max(pair_avg, 1e-6)
            if diff_ratio > 0.20:
                target = max(pair_avg, 0.92 * ((geo_len[a_left] + geo_len[a_right]) / 2.0))
                final_len[a_left] = float(np.clip(target, 0.45 * geo_len[a_left], min(1.35 * geo_len[a_left], cap[a_left])))
                final_len[a_right] = float(np.clip(target, 0.45 * geo_len[a_right], min(1.35 * geo_len[a_right], cap[a_right])))

        if final_len[60] < final_len[30]:
            final_len[60] = min(max(final_len[30] * 1.04, final_len[60]), min(1.35 * geo_len[60], cap[60]))
        if final_len[120] < final_len[150]:
            final_len[120] = min(max(final_len[150] * 1.04, final_len[120]), min(1.35 * geo_len[120], cap[120]))

        side_mid_max = max(final_len[60], final_len[120])
        if final_len[90] < 0.92 * side_mid_max:
            final_len[90] = min(max(0.98 * side_mid_max, 0.90 * geo_len[90]), min(1.28 * geo_len[90], cap[90]))

        mode = "fallback"

    forehead_points_dict = {}
    for a in [30, 60, 90, 120, 150]:
        forehead_points_dict[a] = _snap_near_target(directions[a], final_len[a], window=36)

    # Enforce x ordering tipis agar tidak zig-zag/miring.
    if not (
        forehead_points_dict[30][0] < forehead_points_dict[60][0] <
        forehead_points_dict[90][0] < forehead_points_dict[120][0] <
        forehead_points_dict[150][0]
    ):
        x_sorted = sorted([
            forehead_points_dict[30][0],
            forehead_points_dict[60][0],
            forehead_points_dict[90][0],
            forehead_points_dict[120][0],
            forehead_points_dict[150][0],
        ])
        y_map = {
            30: forehead_points_dict[30][1],
            60: forehead_points_dict[60][1],
            90: forehead_points_dict[90][1],
            120: forehead_points_dict[120][1],
            150: forehead_points_dict[150][1],
        }
        forehead_points_dict[30] = (x_sorted[0], y_map[30])
        forehead_points_dict[60] = (x_sorted[1], y_map[60])
        forehead_points_dict[90] = (x_sorted[2], y_map[90])
        forehead_points_dict[120] = (x_sorted[3], y_map[120])
        forehead_points_dict[150] = (x_sorted[4], y_map[150])

    # 90 harus tepat di atas x titik 27 (vertikal lurus).
    x90 = int(round(float(p27[0])))
    y90 = forehead_points_dict[90][1]
    y90 = int(np.clip(y90, 0, h - 1))
    forehead_points_dict[90] = (int(np.clip(x90, 0, w - 1)), y90)

    forehead_points = [
        forehead_points_dict[30],
        forehead_points_dict[60],
        forehead_points_dict[90],
        forehead_points_dict[120],
        forehead_points_dict[150],
    ]

    debug_info = {
        "mode": mode,
        "raw_mask_good": raw_mask_good,
        "raw_len": raw_len,
        "geo_len": geo_len,
        "final_len": final_len,
        "cap": cap,
    }

    if return_debug:
        return forehead_points, debug_info
    return forehead_points


def build_face_boundary(landmarks: np.ndarray, forehead_points: list) -> list:
    """
    Boundary tertutup:
    0 -> 1 -> ... -> 16 -> 150 -> 120 -> 90 -> 60 -> 30 -> 0
    """
    lm = np.asarray(landmarks).reshape(-1, 2).astype(np.int32)
    jaw = [tuple(map(int, lm[i])) for i in range(17)]

    fp = {
        30: tuple(map(int, forehead_points[0])),
        60: tuple(map(int, forehead_points[1])),
        90: tuple(map(int, forehead_points[2])),
        120: tuple(map(int, forehead_points[3])),
        150: tuple(map(int, forehead_points[4])),
    }

    boundary_points = jaw + [fp[150], fp[120], fp[90], fp[60], fp[30], jaw[0]]
    return boundary_points


forehead_results = {}

rows = len(class_names)
fig, axes = plt.subplots(rows, 2, figsize=(11, 4 * rows), squeeze=False)

for i, cls in enumerate(class_names):
    parsing_info = face_parser_results.get(cls, {})
    parsing_mask = parsing_info.get("parsing_mask", None)
    masked_face = parsing_info.get("face_masked", None)
    dlib_info = dlib_results.get(cls, {})

    if parsing_mask is None or dlib_info.get("num_faces", 0) <= 0:
        axes[i, 0].text(0.5, 0.5, "Data parsing/dlib belum tersedia", ha="center", va="center")
        axes[i, 0].set_title(f"{cls} - Input")
        axes[i, 0].axis("off")

        axes[i, 1].text(0.5, 0.5, "Lewati forehead boundary", ha="center", va="center")
        axes[i, 1].set_title(f"{cls} - Forehead")
        axes[i, 1].axis("off")
        continue

    landmarks = np.asarray(dlib_info["landmarks"][0], dtype=np.int32)
    forehead_points, debug_info = forehead_5_boundary(parsing_mask, landmarks, return_debug=True)
    boundary_points = build_face_boundary(landmarks, forehead_points)

    p27 = tuple(map(int, landmarks[27]))

    if parsing_mask.ndim == 2:
        vis = cv2.cvtColor(parsing_mask.astype(np.uint8), cv2.COLOR_GRAY2RGB)
    elif parsing_mask.ndim == 3 and parsing_mask.shape[2] == 1:
        vis = cv2.cvtColor(parsing_mask[:, :, 0].astype(np.uint8), cv2.COLOR_GRAY2RGB)
    else:
        vis = parsing_mask.copy().astype(np.uint8)

    if vis.max() <= 1:
        vis = (vis * 255).astype(np.uint8)

    # Overlay boundary area agar terlihat seperti penutup muka.
    overlay = vis.copy()
    cv2.fillPoly(overlay, [np.array(boundary_points, dtype=np.int32)], (70, 70, 255))
    vis = cv2.addWeighted(overlay, 0.18, vis, 0.82, 0)

    # Titik pusat 27 + garis 90 vertikal (tanpa garis turun ke bawah).
    cv2.circle(vis, p27, 4, (255, 0, 0), -1)
    p90 = tuple(map(int, forehead_points[2]))
    cv2.line(vis, p27, p90, (255, 255, 0), 1)

    # Garis landmark bawah wajah 0..16
    jaw_pts = [tuple(map(int, landmarks[idx])) for idx in range(17)]
    for j in range(16):
        cv2.line(vis, jaw_pts[j], jaw_pts[j + 1], (0, 220, 255), 1)

    # Garis titik dahi 150 -> 120 -> 90 -> 60 -> 30
    fp = {
        30: tuple(map(int, forehead_points[0])),
        60: tuple(map(int, forehead_points[1])),
        90: tuple(map(int, forehead_points[2])),
        120: tuple(map(int, forehead_points[3])),
        150: tuple(map(int, forehead_points[4])),
    }
    top_chain = [fp[150], fp[120], fp[90], fp[60], fp[30]]
    for j in range(len(top_chain) - 1):
        cv2.line(vis, top_chain[j], top_chain[j + 1], (0, 255, 0), 2)

    # Tutup boundary: 16 -> 150, 30 -> 0
    cv2.line(vis, jaw_pts[16], fp[150], (255, 0, 255), 2)
    cv2.line(vis, fp[30], jaw_pts[0], (255, 0, 255), 2)
    cv2.polylines(vis, [np.array(boundary_points, dtype=np.int32)], True, (255, 0, 255), 2)

    for angle, pt in fp.items():
        cv2.circle(vis, pt, 4, (0, 255, 0), -1)
        cv2.putText(vis, str(angle), (pt[0] + 3, pt[1] - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 50, 50), 1, cv2.LINE_AA)

    if masked_face is not None:
        axes[i, 0].imshow(masked_face)
    else:
        aligned_face = flow_results[cls].get("aligned_face", None)
        if aligned_face is not None:
            axes[i, 0].imshow(aligned_face)
        else:
            axes[i, 0].imshow(np.zeros((vis.shape[0], vis.shape[1], 3), dtype=np.uint8))
    axes[i, 0].set_title(f"{cls} - Face Masked")
    axes[i, 0].axis("off")

    axes[i, 1].imshow(vis)
    axes[i, 1].set_title(f"{cls} - Forehead + Closed Boundary ({debug_info['mode']})")
    axes[i, 1].axis("off")

    forehead_results[cls] = {
        "forehead_points": forehead_points,
        "boundary_points": boundary_points,
        "landmark_27": p27,
        "debug": debug_info,
    }

plt.tight_layout()
plt.show()

for cls in class_names:
    if cls in forehead_results:
        dbg = forehead_results[cls]["debug"]
        print(
            cls,
            "mode=", dbg["mode"],
            "raw_ok=", dbg["raw_mask_good"],
            "forehead=", forehead_results[cls]["forehead_points"],
            "boundary_n=", len(forehead_results[cls]["boundary_points"]),
        )