"""
1_intrinsic_calibration.py
==========================
Computes camera intrinsic parameters (K, distortion) from a set of chessboard
images captured during the hand-eye calibration sessions.

Usage
-----
    python 1_intrinsic_calibration.py [--sessions DIR [DIR ...]]
                                      [--pattern W H] [--square S]
                                      [--out calibration.npz]
                                      [--max-reproj 1.0]

Output
------
    calibration.npz          → camMat, distCoeffs, imageSize
    calibration_summary.json → human-readable report with per-image errors
    debug_undistortion.png   → side-by-side undistortion check on one frame
"""

import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np

def build_obj_points(pattern: tuple, square_size: float) -> np.ndarray:
    """3-D chessboard corners in the calibration-target frame (Z=0)."""
    obj = np.zeros((pattern[0] * pattern[1], 3), np.float32)
    obj[:, :2] = np.mgrid[0:pattern[0], 0:pattern[1]].T.reshape(-1, 2)
    return obj * square_size


def detect_chessboard(img_path: str, pattern: tuple, subpix_window: int = 11):
    """
    Returns (corners_refined, gray) if the chessboard is found, else (None, None).
    Tries twice: once on the raw image, once with CLAHE pre-processing for
    low-contrast frames.
    """
    img = cv2.imread(img_path)
    if img is None:
        return None, None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    flags = (cv2.CALIB_CB_ADAPTIVE_THRESH
             | cv2.CALIB_CB_NORMALIZE_IMAGE
             | cv2.CALIB_CB_FAST_CHECK)
    found, corners = cv2.findChessboardCorners(gray, pattern, flags)

    # Second attempt with contrast enhancement
    if not found:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray_eq = clahe.apply(gray)
        found, corners = cv2.findChessboardCorners(gray_eq, pattern, flags)
        if found:
            gray = gray_eq          # use enhanced image for sub-pixel

    if not found:
        return None, None

    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-4)
    corners = cv2.cornerSubPix(gray, corners, (subpix_window, subpix_window), (-1, -1), crit)
    return corners, gray


def reprojection_error(obj_pts, img_pts, rvec, tvec, K, dist):
    """Mean per-corner re-projection error in pixels for one image."""
    proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, dist)
    return float(np.linalg.norm(img_pts - proj, axis=2).mean())

def main():
    try:
        from pipeline_config import (
            CALIB_PATTERN_SIZE, CALIB_SQUARE_SIZE_M,
            CALIB_SESSION_DIRS, INTRINSIC_OUTPUT_NPZ,
            INTRINSIC_OUTPUT_JSON, HANDEYE_MAX_REPROJ_PX,
        )
    except ImportError:
        CALIB_PATTERN_SIZE  = (9, 6)
        CALIB_SQUARE_SIZE_M = 0.025
        CALIB_SESSION_DIRS  = []
        INTRINSIC_OUTPUT_NPZ  = "calibration.npz"
        INTRINSIC_OUTPUT_JSON = "calibration_summary.json"
        HANDEYE_MAX_REPROJ_PX = 1.5

    p = argparse.ArgumentParser(description="Camera intrinsic calibration from chessboard images")
    p.add_argument("--sessions",   nargs="+", default=CALIB_SESSION_DIRS)
    p.add_argument("--pattern",    nargs=2,   type=int, default=list(CALIB_PATTERN_SIZE),
                   metavar=("W", "H"), help="Inner corner count per row and column")
    p.add_argument("--square",     type=float, default=CALIB_SQUARE_SIZE_M,
                   metavar="METRES", help="Physical size of one chessboard square")
    p.add_argument("--out",        default=INTRINSIC_OUTPUT_NPZ)
    p.add_argument("--max-reproj", type=float, default=HANDEYE_MAX_REPROJ_PX,
                   help="Drop images above this reprojection error (pixels) in a second pass")
    args = p.parse_args()

    pattern    = tuple(args.pattern)
    sq         = args.square
    obj_single = build_obj_points(pattern, sq)

    image_paths = []
    for session in args.sessions:
        found = sorted(glob.glob(os.path.join(session, "raw", "*.png"))
                     + glob.glob(os.path.join(session, "raw", "*.jpg")))
        image_paths.extend(found)

    # Also accept loose images in the current directory
    if not image_paths:
        image_paths = sorted(glob.glob("calib_images/*.png")
                           + glob.glob("calib_images/*.jpg"))

    if not image_paths:
        print("[ERROR] No images found. Check --sessions or put images in calib_images/")
        sys.exit(1)

    print(f"Found {len(image_paths)} candidate images")

    all_obj = []
    all_img = []
    paths_used = []
    img_shape  = None

    for path in image_paths:
        corners, gray = detect_chessboard(path, pattern)
        if corners is None:
            print(f"  [SKIP] No chessboard in {path}")
            continue
        if img_shape is None:
            img_shape = gray.shape[::-1]   # (width, height)
        all_obj.append(obj_single.copy())
        all_img.append(corners)
        paths_used.append(path)
        print(f"  [OK]   {path}")

    n_detected = len(all_obj)
    print(f"\n{n_detected}/{len(image_paths)} images usable")

    if n_detected < 6:
        print("[ERROR] Need at least 6 valid images for a reliable calibration.")
        sys.exit(1)

    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        all_obj, all_img, img_shape, None, None,
        flags=cv2.CALIB_RATIONAL_MODEL          # 8-coeff rational model – better for wide-angle
    )
    print(f"\nFirst-pass RMS: {rms:.4f} px  (target < 0.5 px)")

    per_img_err = [
        reprojection_error(all_obj[i], all_img[i], rvecs[i], tvecs[i], K, dist)
        for i in range(n_detected)
    ]

    good_mask = [e <= args.max_reproj for e in per_img_err]
    n_bad     = good_mask.count(False)

    if n_bad > 0:
        print(f"\n{n_bad} image(s) exceed {args.max_reproj:.1f} px – re-calibrating without them …")
        all_obj2  = [o for o, g in zip(all_obj, good_mask) if g]
        all_img2  = [i for i, g in zip(all_img, good_mask) if g]
        paths2    = [p for p, g in zip(paths_used, good_mask) if g]

        if len(all_obj2) < 6:
            print("  [WARN] Not enough images remain after outlier removal – keeping all.")
        else:
            rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
                all_obj2, all_img2, img_shape, None, None,
                flags=cv2.CALIB_RATIONAL_MODEL
            )
            per_img_err = [
                reprojection_error(all_obj2[i], all_img2[i], rvecs[i], tvecs[i], K, dist)
                for i in range(len(all_obj2))
            ]
            all_obj    = all_obj2
            all_img    = all_img2
            paths_used = paths2
            print(f"  Refined RMS: {rms:.4f} px")

    print("\n" + "=" * 60)
    print("CAMERA INTRINSIC CALIBRATION RESULT")
    print("=" * 60)
    print(f"  Images used   : {len(paths_used)}")
    print(f"  Image size    : {img_shape[0]} × {img_shape[1]} px")
    print(f"  RMS error     : {rms:.4f} px")
    print()
    print("Camera matrix K:")
    print(np.round(K, 4))
    print("\nDistortion coefficients:")
    print(np.round(dist, 6))
    print()
    print("Per-image reprojection errors:")
    for path, err in zip(paths_used, per_img_err):
        flag = "⚠" if err > 0.8 else "✓"
        print(f"  {flag}  {os.path.basename(path):40s}  {err:.4f} px")

    np.savez(args.out, camMat=K, distCoeffs=dist, imageSize=img_shape)
    print(f"\nSaved → {args.out}")

    summary = {
        "rms_px":            float(rms),
        "image_size_wh":     list(img_shape),
        "n_images_used":     len(paths_used),
        "camera_matrix":     K.tolist(),
        "dist_coeffs":       dist.tolist(),
        "dist_model":        "rational_8coeff",
        "per_image_errors":  [
            {"path": p, "reproj_px": float(e)}
            for p, e in zip(paths_used, per_img_err)
        ],
    }
    with open(INTRINSIC_OUTPUT_JSON, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved → {INTRINSIC_OUTPUT_JSON}")

    ref_img = cv2.imread(paths_used[0])
    undist  = cv2.undistort(ref_img, K, dist)
    side    = np.hstack([ref_img, undist])
    cv2.putText(side, "Original",   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(side, "Undistorted",
                (ref_img.shape[1] + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    debug_path = "debug_undistortion.png"
    cv2.imwrite(debug_path, side)
    print(f"Saved → {debug_path}")

    fx, fy = K[0, 0], K[1, 1]
    fov_x  = np.degrees(2 * np.arctan(img_shape[0] / (2 * fx)))
    fov_y  = np.degrees(2 * np.arctan(img_shape[1] / (2 * fy)))
    print(f"\nHorizontal FoV ≈ {fov_x:.1f}°   Vertical FoV ≈ {fov_y:.1f}°")
    if rms > 1.0:
        print("\n[WARN] RMS > 1.0 px – calibration quality is marginal.")
        print("       Consider: more images (>20), wider angular spread, better lighting.")
    elif rms < 0.5:
        print("\n[OK]   RMS < 0.5 px – calibration looks good.")


if __name__ == "__main__":
    main()
