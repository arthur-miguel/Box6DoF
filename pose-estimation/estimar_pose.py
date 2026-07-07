"""
estimar_pose.py
===============
Núcleo do pipeline de estimação de pose 6-DoF de caixas: detectores de
keypoints (ArUco, HSV, segmentação por aprendizado e arestas), solucionador
PnP, filtro temporal e cadeia cinemática até a base do robô.
"""

import math
import os
import cv2
import numpy as np

_UR_DH = {
    "UR3":   [( 0.0,      0.1519,  math.pi/2), (-0.24365, 0.0,      0.0     ),
              (-0.21325,  0.0,      0.0      ), ( 0.0,     0.11235,  math.pi/2),
              ( 0.0,      0.08535,-math.pi/2 ), ( 0.0,     0.0819,   0.0     )],
    "UR5":   [( 0.0,      0.089159,math.pi/2), (-0.425,   0.0,       0.0    ),
              (-0.39225,  0.0,      0.0      ), ( 0.0,     0.10915,  math.pi/2),
              ( 0.0,      0.09465,-math.pi/2 ), ( 0.0,     0.0823,   0.0    )],
    "UR5e":  [( 0.0,      0.1625,  math.pi/2), (-0.425,   0.0,       0.0    ),
              (-0.3922,   0.0,      0.0      ), ( 0.0,     0.1333,   math.pi/2),
              ( 0.0,      0.0997, -math.pi/2 ), ( 0.0,     0.0996,   0.0    )],
    "UR10":  [( 0.0,      0.1273,  math.pi/2), (-0.612,   0.0,       0.0    ),
              (-0.5723,   0.0,      0.0      ), ( 0.0,     0.1639,   math.pi/2),
              ( 0.0,      0.1157, -math.pi/2 ), ( 0.0,     0.0922,   0.0    )],
    "UR10e": [( 0.0,      0.1807,  math.pi/2), (-0.6127,  0.0,       0.0    ),
              (-0.57155,  0.0,      0.0      ), ( 0.0,     0.17415,  math.pi/2),
              ( 0.0,      0.12,   -math.pi/2 ), ( 0.0,     0.1156,   0.0    )],
    "UR16e": [( 0.0,      0.1807,  math.pi/2), (-0.4784,  0.0,       0.0    ),
              (-0.36,     0.0,      0.0      ), ( 0.0,     0.17415,  math.pi/2),
              ( 0.0,      0.12,   -math.pi/2 ), ( 0.0,     0.1156,   0.0    )],
}

def _dh(a, d, alpha, theta):
    ct, st = math.cos(theta), math.sin(theta)
    ca, sa = math.cos(alpha), math.sin(alpha)
    return np.array([[ct, -st*ca,  st*sa, a*ct],
                     [st,  ct*ca, -ct*sa, a*st],
                     [0.,     sa,     ca,    d],
                     [0.,     0.,     0.,   1.]], dtype=np.float64)

def ur_fk(joint_angles, robot_model="UR5e", up_to_joint=5):
    dh = _UR_DH[robot_model]
    T  = np.eye(4)
    for i in range(up_to_joint):
        a, d, alpha = dh[i]
        T = T @ _dh(a, d, alpha, joint_angles[i])
    return T

def Rmat(rvec):
    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64))
    return R

def pose6d_to_H(vec):
    H = np.eye(4)
    H[:3, :3] = Rmat(vec[3:])
    H[:3,  3] = vec[:3]
    return H

def rvec_tvec_to_H(rvec, tvec):
    H = np.eye(4)
    H[:3, :3] = Rmat(rvec)
    H[:3,  3] = np.asarray(tvec).flatten()
    return H

def H_to_euler_deg(H):
    R  = H[:3, :3]
    sy = math.sqrt(R[0, 0]**2 + R[1, 0]**2)
    if sy > 1e-6:
        x = math.atan2( R[2, 1],  R[2, 2])
        y = math.atan2(-R[2, 0],  sy)
        z = math.atan2( R[1, 0],  R[0, 0])
    else:
        x = math.atan2(-R[1, 2],  R[1, 1])
        y = math.atan2(-R[2, 0],  sy)
        z = 0.0
    return np.degrees([x, y, z])

def reproj_err(obj_pts, img_pts, rvec, tvec, K, dist):
    proj, _ = cv2.projectPoints(obj_pts.astype(np.float32), rvec, tvec, K, dist)
    return float(np.linalg.norm(img_pts - proj.reshape(-1, 2), axis=1).mean())

def order_quad(pts):
    pts = np.array(pts, np.float32)
    s   = pts.sum(1)
    d   = np.diff(pts, axis=1).flatten()
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)],
                     pts[np.argmax(s)], pts[np.argmax(d)]], np.float32)

def _clamp_pts_for_subpix(pts, h, w, win=5):
    out = pts.copy()
    out[:, 0] = np.clip(out[:, 0], win, w - 1 - win)
    out[:, 1] = np.clip(out[:, 1], win, h - 1 - win)
    return out

# 1) ArUco

def build_aruco_detector(dict_name="DICT_4X4_250"):
    key = getattr(cv2.aruco, dict_name, None)
    if key is None:
        raise ValueError(f"Unknown ArUco dict: {dict_name}")
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod    = cv2.aruco.CORNER_REFINE_SUBPIX
    params.cornerRefinementWinSize   = 5
    params.adaptiveThreshWinSizeMin  = 5
    params.adaptiveThreshWinSizeMax  = 23
    params.adaptiveThreshWinSizeStep = 4
    return cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(key), params)

def detect_aruco(img, detector, target_id=None, target_center_2d=None):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners_list, ids, _ = detector.detectMarkers(gray)
    if ids is None:
        return None
        
    best_corners = None
    min_dist = float('inf')
    
    for i, mid in enumerate(ids.flatten()):
        if target_id is None or mid == target_id:
            c = corners_list[i].reshape(4, 2).astype(np.float32)
            if target_center_2d is not None:
                cx, cy = c.mean(axis=0)
                dist = math.hypot(cx - target_center_2d[0], cy - target_center_2d[1])
                if dist < min_dist:
                    min_dist = dist
                    best_corners = c
            else:
                return c
    return best_corners

#  2) HSV top-face
def _quad_from_mask(mask, gray_unblurred, h, w, min_area=1500,
                    max_aspect=5.0, min_solidity=0.70, target_center_2d=None):
    
    detached_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(detached_mask, connectivity=8)
    
    if num_labels <= 1:
        return None
        
    if target_center_2d is not None:
        best_label = -1
        min_dist = float('inf')
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] < min_area:
                continue
            cx, cy = centroids[i]
            dist = math.hypot(cx - target_center_2d[0], cy - target_center_2d[1])
            if dist < min_dist:
                min_dist = dist
                best_label = i
        
        if best_label == -1:
            return None
        target_label = best_label
    else:
        target_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        
    clean_mask = np.where(labels == target_label, 255, 0).astype(np.uint8)

    cnts, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
        
    main_cnt = max(cnts, key=cv2.contourArea)
    solid_mask = np.zeros_like(clean_mask)
    cv2.drawContours(solid_mask, [main_cnt], -1, 255, thickness=cv2.FILLED)
    
    cnts, _ = cv2.findContours(solid_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in sorted(cnts, key=cv2.contourArea, reverse=True)[:3]:
        if cv2.contourArea(cnt) < min_area:
            break

        hull  = cv2.convexHull(cnt)
        peri  = cv2.arcLength(hull, True)
        approx = None
        for eps in np.linspace(0.01, 0.12, 40):
            tmp = cv2.approxPolyDP(hull, eps * peri, True)
            if len(tmp) == 4:
                approx = tmp
                break
        if approx is None:
            continue

        pts = np.float32([p[0] for p in approx])

        hull_area = cv2.contourArea(cv2.convexHull(approx))
        if hull_area < min_area:
            continue
        if cv2.contourArea(cnt) / (hull_area + 1e-6) < min_solidity:
            continue

        side1 = np.linalg.norm(pts[0] - pts[1])
        side2 = np.linalg.norm(pts[1] - pts[2])
        if min(side1, side2) < 20:
            continue
        if max(side1, side2) / (min(side1, side2) + 1e-6) > max_aspect:
            continue

        win  = 3  
        pts  = _clamp_pts_for_subpix(pts, h, w, win)
        crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-4)
        pts  = cv2.cornerSubPix(gray_unblurred, pts, (win, win), (-1, -1), crit)

        return order_quad(pts)

    return None

def detect_top_face_hsv(img, debug_prefix=None, seg_mask=None, target_center_2d=None, hsv_lower=(5, 15, 30), hsv_upper=(40, 255, 255)):
    h, w = img.shape[:2]
    blur      = cv2.bilateralFilter(img, 5, 75, 75)
    hsv       = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    gray_unblurred = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 

    if seg_mask is not None:
        box_mask = (seg_mask > 127).astype(np.uint8) * 255
    else:
        box_mask = cv2.inRange(hsv, np.array(hsv_lower), np.array(hsv_upper))

    box_mask = cv2.morphologyEx(box_mask, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
    box_mask = cv2.morphologyEx(box_mask, cv2.MORPH_OPEN, np.ones(( 3,  3), np.uint8))

    box_pixels = hsv[:, :, 2][box_mask > 0]
    if len(box_pixels) < 200:
        return None, None

    thresh_val, _ = cv2.threshold(box_pixels, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    _, top_mask = cv2.threshold(hsv[:, :, 2], thresh_val, 255, cv2.THRESH_BINARY)
    top_mask = cv2.bitwise_and(top_mask, box_mask)

    otsu_ratio = float(np.count_nonzero(top_mask)) / (np.count_nonzero(box_mask) + 1e-6)

    if otsu_ratio > 0.85:
        top_mask = box_mask
    else:
        top_mask = cv2.morphologyEx(top_mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
        top_mask = cv2.morphologyEx(top_mask, cv2.MORPH_OPEN, np.ones(( 5,  5), np.uint8))

    if debug_prefix:
        cv2.imwrite(f"{debug_prefix}_hsv_box_mask.png", box_mask)
        cv2.imwrite(f"{debug_prefix}_hsv_top_mask.png", top_mask)

    quad = _quad_from_mask(top_mask, gray_unblurred, h, w, target_center_2d=target_center_2d)
    return quad, top_mask

# 3) fallback
def _angle_diff(a, b):
    d = abs(a - b) % 180
    return d if d <= 90 else 180 - d

def _cluster_lines(lines, angle_tol=12, dist_tol=30):
    if lines is None: return []
    pts  = [(float(l[0][0]), float(l[0][1])) for l in lines]
    used = [False] * len(pts)
    clusters = []
    for i, (r0, t0) in enumerate(pts):
        if used[i]: continue
        group = [(r0, t0)]
        used[i] = True
        for j, (r1, t1) in enumerate(pts):
            if used[j]: continue
            if (_angle_diff(math.degrees(t0), math.degrees(t1)) < angle_tol and abs(r0 - r1) < dist_tol):
                group.append((r1, t1))
                used[j] = True
        clusters.append((float(np.mean([g[0] for g in group])), float(np.mean([g[1] for g in group]))))
    return clusters

def _line_intersect(r1, t1, r2, t2):
    ct1, st1 = math.cos(t1), math.sin(t1)
    ct2, st2 = math.cos(t2), math.sin(t2)
    denom = ct1 * st2 - st1 * ct2
    if abs(denom) < 1e-6: return None
    x = ( r1 * st2 - r2 * st1) / denom
    y = (-r1 * ct2 + r2 * ct1) / denom
    return x, y

def detect_box_edges(img, debug_prefix=None, target_center_2d=None):
    gray   = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w   = gray.shape
    clahe  = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_e = clahe.apply(gray)
    
    blur   = cv2.bilateralFilter(gray_e, 5, 75, 75)
    edges  = cv2.Canny(blur, 40, 120)
    edges  = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)

    lines = cv2.HoughLines(edges, 1, math.pi / 180, threshold=60)
    if lines is None or len(lines) < 4: return None

    clusters = _cluster_lines(lines, angle_tol=10, dist_tol=25)
    if len(clusters) < 4: return None

    def angle_group(t):
        deg = math.degrees(t) % 180
        return 0 if deg < 45 or deg >= 135 else 1

    grp0 = [(r, t) for r, t in clusters if angle_group(t) == 0]
    grp1 = [(r, t) for r, t in clusters if angle_group(t) == 1]
    if len(grp0) < 2 or len(grp1) < 2: return None

    def two_furthest(group):
        best = (None, None, 0)
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                d = abs(group[i][0] - group[j][0])
                if d > best[2]: best = (group[i], group[j], d)
        return best[0], best[1]

    l0a, l0b = two_furthest(grp0)
    l1a, l1b = two_furthest(grp1)
    if None in (l0a, l0b, l1a, l1b): return None

    corners = []
    for la in (l0a, l0b):
        for lb in (l1a, l1b):
            pt = _line_intersect(*la, *lb)
            if pt is None: return None
            x, y = pt
            if -50 <= x <= w + 50 and -50 <= y <= h + 50:
                corners.append((x, y))
    if len(corners) != 4: return None

    pts   = order_quad(corners)
    side1 = np.linalg.norm(pts[0] - pts[1])
    side2 = np.linalg.norm(pts[1] - pts[2])
    if side1 * side2 < 2000: return None

    if target_center_2d is not None:
        cx, cy = pts.mean(axis=0)
        if math.hypot(cx - target_center_2d[0], cy - target_center_2d[1]) > 150:
            return None

    win  = 7
    pts  = _clamp_pts_for_subpix(pts, h, w, win)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-4)
    gray_f = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    pts    = cv2.cornerSubPix(gray_f, pts, (win, win), (-1, -1), crit)

    return pts

#  4) Segmentacao de instancias
class SegmentationBackend:
    def __init__(self, backend: str, model: str = "",
                 class_ids=None, conf_thresh: float = 0.35,
                 input_size=(640, 640),
                 sam_variant: str = "mobile_sam",
                 discriminate_top_face: bool = True):

        self.backend   = backend.lower()
        self.class_ids = class_ids if class_ids is not None else []
        self.conf      = conf_thresh
        self.input_sz  = input_size
        self.disc_top  = discriminate_top_face
        self._model    = None

        if self.backend == "sam":
            self._init_sam(model, sam_variant)

        elif self.backend == "ultralytics":
            try:
                from ultralytics import YOLO
            except ImportError:
                raise ImportError("pip install ultralytics")
            self._model = YOLO(model or "yolov8n-seg.pt")
            self._model.fuse()

        elif self.backend == "onnx":
            if not model:
                raise ValueError("onnx backend requires model path")
            self._net = cv2.dnn.readNetFromONNX(model)

        else:
            raise ValueError(f"Unknown backend: {backend!r}. Choose 'sam', 'ultralytics', or 'onnx'.")

    def _init_sam(self, model_path, variant):
        self._sam_variant = variant
        if variant == "mobile_sam":
            _sam_mod = None
            _import_errors = []
            for _mod_name, _reg_key in [("mobile_sam", "vit_t"), ("MobileSAM", "vit_t"), ("mobilesam", "vit_t"), ("segment_anything", "vit_b")]:
                try:
                    import importlib
                    _sam_mod = importlib.import_module(_mod_name)
                    _reg_key_use = _reg_key
                    break
                except ImportError as e:
                    _import_errors.append(f"{_mod_name}: {e}")

            if _sam_mod is None:
                raise ImportError("Could not import MobileSAM under any known module name.\n")

            sam_model_registry = _sam_mod.sam_model_registry
            SamPredictor        = _sam_mod.SamPredictor

            if not os.path.isfile(model_path): raise FileNotFoundError(f"SAM checkpoint not found: {model_path!r}")

            sam = sam_model_registry[_reg_key_use](checkpoint=model_path)
            sam.eval()
            self._sam_predictor = SamPredictor(sam)

        elif variant == "sam2":
            try:
                from sam2.build_sam import build_sam2
                from sam2.sam2_image_predictor import SAM2ImagePredictor
            except ImportError:
                raise ImportError("Could not import sam2.")
            sam2 = build_sam2("sam2_hiera_small.yaml", model_path)
            self._sam_predictor = SAM2ImagePredictor(sam2)
        else:
            raise ValueError(f"Unknown SAM variant: {variant!r}")

    def __call__(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        blank = np.zeros((h, w), np.uint8)

        if self.backend == "sam": return self._run_sam(img, blank)
        elif self.backend == "ultralytics": return self._run_ultralytics(img, blank)
        else: return self._run_onnx(img, blank)

    def _run_sam(self, img, blank):
        h, w = img.shape[:2]

        blur = cv2.bilateralFilter(img, 9, 100, 100)
        hsv  = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
        hsv_mask = cv2.inRange(hsv, np.array([ 5, 30, 40]), np.array([42, 255, 255]))
        M = cv2.moments(hsv_mask)
        if M["m00"] < 100: return blank

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        bg_pts = np.array([[10, 10], [w-10, 10], [10, h-10], [w-10, h-10]], np.float32)
        fg_pts = np.array([[cx, cy]], np.float32)

        point_coords  = np.vstack([fg_pts, bg_pts])
        point_labels  = np.array([1] + [0]*len(bg_pts), np.int32)

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pred    = self._sam_predictor
        pred.set_image(img_rgb)

        masks, scores, _ = pred.predict(point_coords=point_coords, point_labels=point_labels, multimask_output=True)
        order  = np.argsort(scores)[::-1]
        result = blank.copy()
        for idx in order:
            m = (masks[idx] * 255).astype(np.uint8)
            if self.disc_top and not self._is_top_face(m, img): continue
            result = m
            break
        return result

    def _run_ultralytics(self, img, blank):
        results = self._model(img, conf=self.conf, verbose=False)
        h, w    = blank.shape
        best_mask, best_area = None, 0

        for r in results:
            if r.masks is None: continue
            for i, seg in enumerate(r.masks.data):
                cls_id = int(r.boxes.cls[i].item())
                if self.class_ids and cls_id not in self.class_ids: continue
                mask_np  = seg.cpu().numpy()
                mask_u8  = cv2.resize((mask_np * 255).astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
                _, mask_bin = cv2.threshold(mask_u8, 127, 255, cv2.THRESH_BINARY)
                if self.disc_top and not self._is_top_face(mask_bin, img): continue
                area = int(mask_bin.sum())
                if area > best_area: best_area, best_mask = area, mask_bin

        return best_mask if best_mask is not None else blank

    def _run_onnx(self, img, blank):
        ih, iw = img.shape[:2]
        tw, th  = self.input_sz
        blob    = cv2.dnn.blobFromImage(img, 1.0/255.0, (tw, th), (0, 0, 0), swapRB=True, crop=False)
        self._net.setInput(blob)
        out = self._net.forward()
        if out.ndim == 4: out = out[0]

        if self.class_ids: prob = out[self.class_ids].sum(axis=0)
        else: prob = out.max(axis=0)

        prob_u8  = ((prob - prob.min()) / (prob.max() - prob.min() + 1e-6) * 255).astype(np.uint8)
        _, mask  = cv2.threshold(prob_u8, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        mask     = cv2.resize(mask, (iw, ih), interpolation=cv2.INTER_NEAREST)
        if self.disc_top and not self._is_top_face(mask, img): return blank
        return mask

    def _is_top_face(self, mask: np.ndarray, img: np.ndarray) -> bool:
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts: return False

        cnt = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(cnt) < 1000: return False

        _, (bw, bh), _ = cv2.minAreaRect(cnt)
        if bw < 1 or bh < 1: return False
        ratio = max(bw, bh) / (min(bw, bh) + 1e-6)
        if ratio > 4.5: return False

        hsv     = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        v       = hsv[:, :, 2]
        dilated = cv2.dilate(mask, np.ones((15, 15), np.uint8))
        ring    = cv2.bitwise_and(dilated, cv2.bitwise_not(mask))

        interior_px = v[mask  > 0]
        ring_px     = v[ring  > 0]

        if len(interior_px) < 50: return False
        if len(ring_px) < 50: return True

        return float(interior_px.mean()) >= float(ring_px.mean()) * 0.90


#  Solver PnP
def solve_pnp_robust(obj_pts, img_pts, K, dist):
    obj = np.asarray(obj_pts, np.float32)
    img = np.asarray(img_pts, np.float32)
    n   = len(obj)

    if n >= 6:
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            obj, img, K, dist,
            iterationsCount=500, reprojectionError=4.0,
            confidence=0.999, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok or inliers is None or len(inliers) < 4:
            return None, None
        ok2, rvec, tvec = cv2.solvePnP(
            obj[inliers.flatten()], img[inliers.flatten()],
            K, dist, rvec, tvec,
            useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok2:
            return None, None
    else:
        ok, rvec, tvec = cv2.solvePnP(
            obj, img, K, dist,
            flags=cv2.SOLVEPNP_IPPE if n == 4 else cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            return None, None

    return rvec, tvec


#  Cadeia cinematica
# Converte poses estimadas para coordenadas mundo
def box_in_base(H_box_cam, entry, T_cam_mount,
                robot_model="UR5e", mount_joint=5):
    if "joint_angles" in entry:
        T_base_mount = ur_fk(entry["joint_angles"],
                             robot_model=robot_model,
                             up_to_joint=mount_joint)
    else:
        tcp_vec = entry.get("pose_real_tcp")
        if tcp_vec is None:
            return None
        T_base_mount = pose6d_to_H(tcp_vec)
    return T_base_mount @ T_cam_mount @ H_box_cam

class PoseFilter:
    def __init__(self, alpha: float = 0.35):
        assert 0 < alpha <= 1.0
        self.alpha  = float(alpha)
        self._rvec  = None
        self._tvec  = None
        self._no_detect_count = 0
        self._reset_after     = 10

    def update(self, rvec, tvec):
        rv = np.asarray(rvec, np.float64).flatten()
        tv = np.asarray(tvec, np.float64).flatten()

        self._no_detect_count = 0

        if self._rvec is None:
            self._rvec, self._tvec = rv.copy(), tv.copy()
            return self._rvec.copy(), self._tvec.copy()

        if np.dot(rv, self._rvec) < 0:
            rv = -rv

        self._rvec = self.alpha * rv + (1.0 - self.alpha) * self._rvec
        self._tvec = self.alpha * tv + (1.0 - self.alpha) * self._tvec
        return self._rvec.copy(), self._tvec.copy()

    def no_detection(self):
        self._no_detect_count += 1
        if self._no_detect_count >= self._reset_after:
            self.reset()

    def reset(self):
        self._rvec = None
        self._tvec = None
        self._no_detect_count = 0

    @property
    def initialised(self):
        return self._rvec is not None

#  Classe orquestardora da estimacao de pose
class BoxPoseEstimator:
    def __init__(self, K, dist, T_cam_mount,
                 box_W, box_D, box_H=0.125,
                 aruco_dict="DICT_4X4_250", aruco_id=None,
                 aruco_size=0.05, aruco_offset_xy=None,
                 robot_model="UR5e", mount_joint=5,
                 seg_backend: SegmentationBackend = None,
                 hsv_lower=(5, 15, 30), hsv_upper=(40, 255, 255)):

        self.K           = K.astype(np.float64)
        self.dist        = dist.astype(np.float64)
        self.T_cam_mount = T_cam_mount.astype(np.float64)
        self.W           = float(box_W)
        self.D           = float(box_D)
        self.H           = float(box_H)
        self.robot_model = robot_model
        self.mount_joint = mount_joint
        self.aruco_id    = aruco_id
        self.mkr_size    = float(aruco_size)
        self.mkr_offset  = aruco_offset_xy
        self.seg_backend = seg_backend
        self.hsv_lower   = hsv_lower
        self.hsv_upper   = hsv_upper

        self.detector = build_aruco_detector(aruco_dict)

        self.top_obj = np.array([
            [0,      0,      0],
            [self.W, 0,      0],
            [self.W, self.D, 0],
            [0,      self.D, 0],
        ], np.float32)

        if aruco_offset_xy is not None:
            x0, y0 = aruco_offset_xy
            ms = self.mkr_size
            self.mkr_obj = np.array([
                [x0,    y0,    0],
                [x0+ms, y0,    0],
                [x0+ms, y0+ms, 0],
                [x0,    y0+ms, 0],
            ], np.float32)
        else:
            hs = self.mkr_size / 2
            self.mkr_obj = np.array([
                [-hs,  hs, 0],
                [ hs,  hs, 0],
                [ hs, -hs, 0],
                [-hs, -hs, 0],
            ], np.float32)

        self.side_wh_obj = np.array([
            [0,      0, 0],
            [self.W, 0, 0],
            [self.W, 0, self.H],
            [0,      0, self.H],
        ], np.float32)

        self.side_dh_obj = np.array([
            [0, 0,      0],
            [0, self.D, 0],
            [0, self.D, self.H],
            [0, 0,      self.H],
        ], np.float32)

    def estimate(self, img, entry=None,
                 use_aruco=True, max_reproj_px=6.0,
                 debug_prefix=None, force_mode="auto", target_center_2d=None):
        K, dist = self.K, self.dist
        candidates = []
        
        GOOD_ENOUGH_ERR = 2.5 

        # 1. ARUCO
        aruco_pts = None
        if use_aruco and force_mode in ("auto", "aruco", "combined"):
            aruco_pts = detect_aruco(img, self.detector, self.aruco_id, target_center_2d=target_center_2d)
            if aruco_pts is not None:
                rvec, tvec = solve_pnp_robust(self.mkr_obj, aruco_pts, K, dist)
                if rvec is not None:
                    err = reproj_err(self.mkr_obj, aruco_pts, rvec, tvec, K, dist)
                    if err <= max_reproj_px:
                        res = self._build(rvec, tvec, err, "aruco", img, aruco_pts, None, entry, debug_prefix)
                        if force_mode == "aruco" or err <= GOOD_ENOUGH_ERR:
                            return res
                        candidates.append(res)

        # 2. HSV TOP-FACE
        hsv_pts, hsv_mask = None, None
        if force_mode in ("auto", "HSV", "combined"):
            hsv_pts, hsv_mask = detect_top_face_hsv(
                img, debug_prefix=debug_prefix,
                seg_mask=None, target_center_2d=target_center_2d,
                hsv_lower=self.hsv_lower, hsv_upper=self.hsv_upper,
            )
            if hsv_pts is not None:
                rvec, tvec = solve_pnp_robust(self.top_obj, hsv_pts, K, dist)
                if rvec is not None:
                    err = reproj_err(self.top_obj, hsv_pts, rvec, tvec, K, dist)
                    if err <= max_reproj_px:
                        res = self._build(rvec, tvec, err, "hsv_top", img, None, hsv_pts, entry, debug_prefix, seg_mask=hsv_mask)
                        if force_mode == "HSV" or err <= GOOD_ENOUGH_ERR:
                            return res
                        candidates.append(res)

        # 3. COMBINED
        if force_mode in ("auto", "combined") and aruco_pts is not None and hsv_pts is not None and self.mkr_offset is not None:
            obj_c = np.vstack([self.top_obj, self.mkr_obj])
            img_c = np.vstack([hsv_pts,      aruco_pts])
            rvec, tvec = solve_pnp_robust(obj_c, img_c, K, dist)
            if rvec is not None:
                err = reproj_err(obj_c, img_c, rvec, tvec, K, dist)
                if err <= max_reproj_px:
                    res = self._build(rvec, tvec, err, "combined", img, aruco_pts, hsv_pts, entry, debug_prefix, seg_mask=hsv_mask)
                    if force_mode == "combined" or err <= GOOD_ENOUGH_ERR:
                        return res
                    candidates.append(res)

        # 4. ML SEGMENTATION
        seg_mask = None
        if self.seg_backend is not None and force_mode in ("auto", "ML"):
            if force_mode == "ML" or not candidates or min(c["reproj_error_px"] for c in candidates) > GOOD_ENOUGH_ERR:
                try:
                    seg_mask = self.seg_backend(img)
                    if seg_mask is not None:
                        h, w = img.shape[:2]
                        gray_unblurred = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        seg_pts = _quad_from_mask(seg_mask, gray_unblurred, h, w, target_center_2d=target_center_2d)
                        if seg_pts is not None:
                            rvec, tvec = solve_pnp_robust(self.top_obj, seg_pts, K, dist)
                            if rvec is not None:
                                err = reproj_err(self.top_obj, seg_pts, rvec, tvec, K, dist)
                                if err <= max_reproj_px:
                                    res = self._build(rvec, tvec, err, "seg_quad", img, None, seg_pts, entry, debug_prefix, seg_mask=seg_mask)
                                    if force_mode == "ML" or err <= GOOD_ENOUGH_ERR:
                                        return res
                                    candidates.append(res)
                except Exception as e:
                    print(f"  [WARN] Seg backend error: {e}")

        # 5. EDGE FALLBACK
        if force_mode == "auto":
            if not candidates or min(c["reproj_error_px"] for c in candidates) > GOOD_ENOUGH_ERR:
                edge_pts = detect_box_edges(img, debug_prefix=debug_prefix, target_center_2d=target_center_2d)
                if edge_pts is not None:
                    best = (None, None, None, float("inf"), None)
                    for obj_m, label in [(self.top_obj,     "top"),
                                          (self.side_wh_obj, "front"),
                                          (self.side_dh_obj, "left")]:
                        rvec, tvec = solve_pnp_robust(obj_m, edge_pts, K, dist)
                        if rvec is None: continue
                        err = reproj_err(obj_m, edge_pts, rvec, tvec, K, dist)
                        if err < best[3]:
                            best = (rvec, tvec, obj_m, err, label)
                    
                    rvec, tvec, obj_m, err, label = best
                    if rvec is not None and err <= max_reproj_px:
                        res = self._build(rvec, tvec, err, f"edge_{label}", img, None, edge_pts, entry, debug_prefix, obj_model=obj_m)
                        candidates.append(res)

        if not candidates:
            return None

        candidates.sort(key=lambda x: x["reproj_error_px"])
        return candidates[0]

    def _build(self, rvec, tvec, err, mode,
               img, aruco_pts, quad_pts, entry,
               debug_prefix, obj_model=None, seg_mask=None):
        if obj_model is None:
            obj_model = self.top_obj
        H_box_cam  = rvec_tvec_to_H(rvec, tvec)
        H_box_base = box_in_base(H_box_cam, entry or {},
                                 self.T_cam_mount,
                                 self.robot_model, self.mount_joint)
        if debug_prefix:
            self._draw_debug(img, rvec, tvec, aruco_pts, quad_pts,
                             H_box_base, err, mode, debug_prefix, obj_model)
        return {
            "H_box_in_cam":    H_box_cam,
            "H_box_in_base":   H_box_base,
            "rvec":            rvec.flatten().tolist(),
            "tvec":            tvec.flatten().tolist(),
            "reproj_error_px": float(err),
            "mode":            mode,
            "aruco_detected":  aruco_pts is not None,
            "quad_detected":   quad_pts  is not None,
            "quad_pts":        quad_pts,
            "aruco_pts":       aruco_pts,
            "seg_mask":        seg_mask,
        }

    def _draw_debug(self, img, rvec, tvec, aruco_pts, quad_pts,
                    H_base, err, mode, prefix, obj_model):
        draw = img.copy()
        os.makedirs(os.path.dirname(prefix) if os.path.dirname(prefix) else ".", exist_ok=True)
        cv2.drawFrameAxes(draw, self.K, self.dist, rvec, tvec, length=self.W * 0.4)
        for pts, color in [(quad_pts, (0, 0, 255)), (aruco_pts, (255, 80, 0))]:
            if pts is None: continue
            for i, (x, y) in enumerate(pts.astype(int)):
                cv2.circle(draw, (x, y), 7, color, -1)
                cv2.putText(draw, str(i), (x+8, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
            for a, b in [(0,1),(1,2),(2,3),(3,0)]:
                cv2.line(draw, tuple(pts[a].astype(int)), tuple(pts[b].astype(int)), (0, 200, 0), 2)
        euler = H_to_euler_deg(H_base) if H_base is not None else [0, 0, 0]
        t     = H_base[:3,3] if H_base is not None else np.asarray(tvec).flatten()
        for j, txt in enumerate([
            f"Mode: {mode}",
            f"Reproj: {err:.2f} px",
            f"t_base = [{t[0]:.3f},{t[1]:.3f},{t[2]:.3f}] m",
            f"euler  = [{euler[0]:.1f},{euler[1]:.1f},{euler[2]:.1f}] deg",
        ]):
            cv2.putText(draw, txt, (10, 30 + 25*j), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        cv2.imwrite(f"{prefix}_pose.png", draw)