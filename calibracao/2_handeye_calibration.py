"""
2_handeye_calibration.py
========================
Offline hand-eye (eye-in-hand) calibration.

Reads pre-collected calibration sessions (each with a metadata.json recording
the TCP pose per image) and computes T_cam→TCP  –  the rigid transform from
the camera optical frame into the robot TCP frame.

Usage
-----
    python 2_handeye_calibration.py [--sessions DIR [DIR ...]]
                                    [--intrinsic calibration.npz]
                                    [--method TSAI]
                                    [--out handeye_calibration.json]

The script runs all five OpenCV hand-eye solvers and prints a quality report
so you can pick the most physically plausible result.

Quality assessment notes
------------------------
Hand-eye calibration is poorly conditioned when:
  • All robot poses lie in a single plane  (no out-of-plane rotation)
  • Angular spread of TCP rotations < ~45 °
  • Fewer than ~15 pose pairs

If the AX=XB residuals are large, collect more data with larger rotation
diversity (tilt the robot wrist, not just pan it in a circle).
"""

import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np


# ── helpers ───────────────────────────────────────────────────────────────────

def build_obj_points(pattern, square_size):
    obj = np.zeros((pattern[0] * pattern[1], 3), np.float32)
    obj[:, :2] = np.mgrid[0:pattern[0], 0:pattern[1]].T.reshape(-1, 2)
    return obj * square_size


def detect_and_solve_pnp(img_path, pattern, obj_pts, K, dist, subpix_win=11):
    """
    Detect chessboard in img_path, run solvePnP.
    Returns (R_target2cam, t_target2cam, reproj_err) or None on failure.
    """
    img = cv2.imread(img_path)
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    flags = (cv2.CALIB_CB_ADAPTIVE_THRESH
             | cv2.CALIB_CB_NORMALIZE_IMAGE
             | cv2.CALIB_CB_FAST_CHECK)
    found, corners = cv2.findChessboardCorners(gray, pattern, flags)
    if not found:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        found, corners = cv2.findChessboardCorners(clahe.apply(gray), pattern, flags)
    if not found:
        return None

    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-4)
    corners = cv2.cornerSubPix(gray, corners, (subpix_win, subpix_win), (-1, -1), crit)

    ok, rvec, tvec = cv2.solvePnP(
        obj_pts, corners, K, dist, flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not ok:
        return None

    # Reprojection error
    proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, dist)
    err = float(np.linalg.norm(corners - proj, axis=2).mean())

    R, _ = cv2.Rodrigues(rvec)
    return R, tvec, err


def pose6d_to_matrix(tcp_vec):
    """Convert UR-style [x, y, z, rx, ry, rz] to 4×4 homogeneous matrix."""
    t = np.array(tcp_vec[:3])
    rv = np.array(tcp_vec[3:])
    R, _ = cv2.Rodrigues(rv)
    H = np.eye(4)
    H[:3, :3] = R
    H[:3, 3]  = t
    return H


def ax_xb_residuals(R_g2b_list, t_g2b_list, R_t2c_list, t_t2c_list, R_x, t_x):
    """
    For each consecutive pair (i, j) compute the AX = XB residuals
    where A = T_j^{-1} T_i (gripper-to-base relative),
          B = T_j T_i^{-1} (target-to-cam relative).

    Returns (mean_rot_deg, mean_trans_m).
    """
    n = len(R_g2b_list)
    rot_errs, t_errs = [], []
    X = np.eye(4)
    X[:3, :3] = R_x
    X[:3, 3]  = t_x.flatten()

    for i in range(n):
        for j in range(i + 1, n):
            # Build A  (gripper-to-base relative motion)
            Ti = np.eye(4); Ti[:3,:3]=R_g2b_list[i]; Ti[:3,3]=t_g2b_list[i].flatten()
            Tj = np.eye(4); Tj[:3,:3]=R_g2b_list[j]; Tj[:3,3]=t_g2b_list[j].flatten()
            A  = np.linalg.inv(Tj) @ Ti

            # Build B  (target-to-cam relative motion)
            Bi = np.eye(4); Bi[:3,:3]=R_t2c_list[i]; Bi[:3,3]=t_t2c_list[i].flatten()
            Bj = np.eye(4); Bj[:3,:3]=R_t2c_list[j]; Bj[:3,3]=t_t2c_list[j].flatten()
            B  = Bj @ np.linalg.inv(Bi)

            LHS = A @ X
            RHS = X @ B
            rot_errs.append(np.degrees(
                np.arccos(np.clip((np.trace(LHS[:3,:3].T @ RHS[:3,:3]) - 1) / 2, -1, 1))
            ))
            t_errs.append(np.linalg.norm(LHS[:3, 3] - RHS[:3, 3]))

    return float(np.mean(rot_errs)), float(np.mean(t_errs))


def rotation_diversity_degrees(R_list):
    """Mean pairwise rotation angle between all tcp rotations (higher = better conditioning)."""
    angles = []
    for i in range(len(R_list)):
        for j in range(i + 1, len(R_list)):
            dR = R_list[i].T @ R_list[j]
            a  = np.degrees(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1)))
            angles.append(a)
    return float(np.mean(angles)) if angles else 0.0


# ── UR forward-kinematics (DH, up to joint N-1) ───────────────────────────────

# Standard UR DH parameters (a, d, alpha) for each joint.
# The table below covers UR3 / UR5 / UR5e / UR10 / UR10e / UR16e.
# Units: metres and radians.
#
# Source: Universal Robots script manual, appendix B.
#
#         a [m]          d [m]         alpha [rad]
_UR_DH = {
    # ---------- UR3 ----------
    "UR3": [
        ( 0.0,        0.1519,   np.pi/2),
        (-0.24365,    0.0,      0.0    ),
        (-0.21325,    0.0,      0.0    ),
        ( 0.0,        0.11235,  np.pi/2),
        ( 0.0,        0.08535, -np.pi/2),
        ( 0.0,        0.0819,   0.0    ),
    ],
    # ---------- UR5 ----------
    "UR5": [
        ( 0.0,        0.089159,  np.pi/2),
        (-0.425,      0.0,       0.0    ),
        (-0.39225,    0.0,       0.0    ),
        ( 0.0,        0.10915,   np.pi/2),
        ( 0.0,        0.09465,  -np.pi/2),
        ( 0.0,        0.0823,    0.0    ),
    ],
    # ---------- UR5e ----------
    "UR5e": [
        ( 0.0,        0.1625,   np.pi/2),
        (-0.425,      0.0,      0.0    ),
        (-0.3922,     0.0,      0.0    ),
        ( 0.0,        0.1333,   np.pi/2),
        ( 0.0,        0.0997,  -np.pi/2),
        ( 0.0,        0.0996,   0.0    ),
    ],
    # ---------- UR10 ----------
    "UR10": [
        ( 0.0,        0.1273,   np.pi/2),
        (-0.612,      0.0,      0.0    ),
        (-0.5723,     0.0,      0.0    ),
        ( 0.0,        0.1639,   np.pi/2),
        ( 0.0,        0.1157,  -np.pi/2),
        ( 0.0,        0.0922,   0.0    ),
    ],
    # ---------- UR10e ----------
    "UR10e": [
        ( 0.0,        0.1807,   np.pi/2),
        (-0.6127,     0.0,      0.0    ),
        (-0.57155,    0.0,      0.0    ),
        ( 0.0,        0.17415,  np.pi/2),
        ( 0.0,        0.12,    -np.pi/2),
        ( 0.0,        0.1156,   0.0    ),
    ],
    # ---------- UR16e ----------
    "UR16e": [
        ( 0.0,        0.1807,   np.pi/2),
        (-0.4784,     0.0,      0.0    ),
        (-0.36,       0.0,      0.0    ),
        ( 0.0,        0.17415,  np.pi/2),
        ( 0.0,        0.12,    -np.pi/2),
        ( 0.0,        0.1156,   0.0    ),
    ],
}


def _dh_matrix(a, d, alpha, theta):
    """Standard DH homogeneous transform for one joint."""
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct,  -st*ca,   st*sa,  a*ct],
        [st,   ct*ca,  -ct*sa,  a*st],
        [0.0,     sa,      ca,     d],
        [0.0,    0.0,     0.0,   1.0],
    ])


def ur_fk_to_joint(joint_angles_rad, robot_model="UR5e", up_to_joint=5):
    """
    Forward kinematics for a UR robot up to (and including) `up_to_joint`
    (1-indexed, so up_to_joint=5 means joints 1-5, i.e. the penultimate joint).

    Parameters
    ----------
    joint_angles_rad : list[float]  — 6 joint angles in radians (j1 … j6)
    robot_model      : str          — one of the keys in _UR_DH
    up_to_joint      : int          — 1..6, default 5 (penultimate)

    Returns
    -------
    T : np.ndarray (4×4)  — pose of the requested frame in the robot base frame
    """
    if robot_model not in _UR_DH:
        raise ValueError(f"Unknown robot model '{robot_model}'. "
                         f"Choose from: {list(_UR_DH.keys())}")
    dh = _UR_DH[robot_model]
    T  = np.eye(4)
    for i in range(up_to_joint):
        a, d, alpha = dh[i]
        T = T @ _dh_matrix(a, d, alpha, joint_angles_rad[i])
    return T


def camera_mount_pose(entry, robot_model="UR5e", mount_joint=5):
    """
    Return the 4×4 pose of the camera-mounting joint frame in the robot base
    frame, given a metadata entry that contains 'joint_angles'.

    If 'joint_angles' is absent (old metadata), falls back to the TCP pose so
    the script stays backward-compatible (with a printed warning).

    Parameters
    ----------
    entry       : dict   — one record from metadata.json
    robot_model : str    — UR model string (must be in _UR_DH)
    mount_joint : int    — joint number the camera is attached to (1-indexed)
                           5 = penultimate joint of a 6-DOF UR robot

    Returns
    -------
    R : np.ndarray (3×3)
    t : np.ndarray (3×1)
    """
    if "joint_angles" not in entry:
        # Backward-compatible fallback — uses TCP pose as before.
        # This is WRONG for a penultimate-joint-mounted camera but keeps
        # old sessions runnable with a clear warning.
        tcp = entry["pose_real_tcp"]
        R, _ = cv2.Rodrigues(np.array(tcp[3:], dtype=float))
        t    = np.array(tcp[:3], dtype=float).reshape(3, 1)
        return R, t

    q  = entry["joint_angles"]          # [j1 … j6] in radians
    T  = ur_fk_to_joint(q, robot_model=robot_model, up_to_joint=mount_joint)
    R  = T[:3, :3]
    t  = T[:3, 3].reshape(3, 1)
    return R, t



def main():
    try:
        from pipeline_config import (
            CALIB_PATTERN_SIZE, CALIB_SQUARE_SIZE_M,
            CALIB_SESSION_DIRS, INTRINSIC_OUTPUT_NPZ,
            HANDEYE_OUTPUT_JSON, HANDEYE_PREFERRED_METHOD,
            HANDEYE_MAX_REPROJ_PX,
        )
    except ImportError:
        CALIB_PATTERN_SIZE         = (9, 6)
        CALIB_SQUARE_SIZE_M        = 0.025
        CALIB_SESSION_DIRS         = []
        INTRINSIC_OUTPUT_NPZ       = "calibration.npz"
        HANDEYE_OUTPUT_JSON        = "handeye_calibration.json"
        HANDEYE_PREFERRED_METHOD   = "TSAI"
        HANDEYE_MAX_REPROJ_PX      = 1.5

    p = argparse.ArgumentParser()
    p.add_argument("--sessions",    nargs="+", default=CALIB_SESSION_DIRS)
    p.add_argument("--intrinsic",   default=INTRINSIC_OUTPUT_NPZ)
    p.add_argument("--method",      default=HANDEYE_PREFERRED_METHOD,
                   choices=["TSAI","PARK","HORAUD","ANDREFF","DANIILIDIS"])
    p.add_argument("--max-reproj",  type=float, default=HANDEYE_MAX_REPROJ_PX)
    p.add_argument("--out",         default=HANDEYE_OUTPUT_JSON)
    p.add_argument("--robot-model", default="UR10",
                   choices=list(_UR_DH.keys()),
                   help="UR robot model for DH forward-kinematics "
                        "(used when camera is NOT on the TCP)")
    p.add_argument("--mount-joint", type=int, default=6,
                   help="Joint number (1-indexed) the camera is physically "
                        "attached to. 6=TCP (standard eye-in-hand). "
                        "5=penultimate joint (default for this setup).")
    args = p.parse_args()

    # ── Load intrinsics ───────────────────────────────────────────────────────
    if not os.path.exists(args.intrinsic):
        print(f"[ERROR] Intrinsic file not found: {args.intrinsic}")
        print("        Run 1_intrinsic_calibration.py first.")
        sys.exit(1)
    calib_data = np.load(args.intrinsic)
    K    = calib_data["camMat"]
    dist = calib_data["distCoeffs"]
    print(f"Loaded intrinsics from {args.intrinsic}")

    using_tcp_pose = (args.mount_joint == 6)
    if using_tcp_pose:
        print("Camera mount : TCP frame  (standard eye-in-hand)")
    else:
        print(f"Camera mount : joint {args.mount_joint} of {args.robot_model}  "
              f"(FK will be computed up to joint {args.mount_joint}, "
              f"joint {args.mount_joint + 1}..6 rotation is NOT part of the camera motion)")

    pattern    = CALIB_PATTERN_SIZE
    sq         = CALIB_SQUARE_SIZE_M
    obj_single = build_obj_points(pattern, sq)

    # ── Collect pose pairs ────────────────────────────────────────────────────
    R_g2b, t_g2b = [], []   # gripper (TCP) to robot base
    R_t2c, t_t2c = [], []   # calibration target to camera
    skipped = []

    for session in args.sessions:
        meta_path = os.path.join("calib_images", session, "metadata.json")
        if not os.path.exists(meta_path):
            print(f"[SKIP] No metadata.json in {session}")
            continue
        with open(meta_path) as f:
            meta = json.load(f)

        imgs_by_id = {
            int(os.path.basename(pth).split("_")[1]): pth
            for pth in glob.glob(os.path.join("calib_images", session, "raw", "*.png"))
                      + glob.glob(os.path.join("calib_images", session, "raw", "*.jpg"))
        }

        for entry in meta:
            img_id = entry["id"]
            if img_id not in imgs_by_id:
                skipped.append(f"{session}/raw/img_{img_id:04d}  [missing file]")
                continue

            result = detect_and_solve_pnp(
                imgs_by_id[img_id], pattern, obj_single, K, dist
            )
            if result is None:
                skipped.append(f"{imgs_by_id[img_id]}  [no chessboard]")
                continue

            R_cam, tvec_cam, reproj_err = result
            if reproj_err > args.max_reproj:
                skipped.append(f"{imgs_by_id[img_id]}  "
                               f"[reproj {reproj_err:.2f}px > {args.max_reproj:.2f}px]")
                continue

            R_t2c.append(R_cam)
            t_t2c.append(tvec_cam)

            # ── Gripper pose: frame the camera is actually attached to ──────
            # For a standard eye-in-hand (camera on TCP) use the TCP pose.
            # For a camera on joint N-1, compute FK up to that joint so that
            # the last-joint rotation is excluded from the "gripper" motion.
            if "joint_angles" not in entry and args.mount_joint != 6:
                print(f"  [WARN] entry id={entry.get('id','?')} has no 'joint_angles' "
                      f"— falling back to TCP pose (incorrect for joint {args.mount_joint} mount).")

            R_rob, t_rob = camera_mount_pose(
                entry,
                robot_model=args.robot_model,
                mount_joint=args.mount_joint,
            )
            R_g2b.append(R_rob)
            t_g2b.append(t_rob)

            print(f"  [OK] {imgs_by_id[img_id]}  reproj={reproj_err:.3f}px")

    n_pairs = len(R_g2b)
    print(f"\nValid pairs : {n_pairs}")
    if skipped:
        print("Skipped     :")
        for s in skipped:
            print(f"  • {s}")

    if n_pairs < 3:
        print("[ERROR] Need at least 3 valid pairs.")
        sys.exit(1)
    if n_pairs < 10:
        print("[WARN] < 10 pairs – results may be noisy. Collect more diverse data.")

    # ── Rotation-diversity report ─────────────────────────────────────────────
    div = rotation_diversity_degrees(R_g2b)
    print(f"\nMean pairwise TCP rotation spread: {div:.1f}° ", end="")
    if div < 30:
        print("⚠  LOW  – risk of degenerate solution. Use more varied wrist orientations.")
    elif div < 60:
        print("(moderate – more diversity would help)")
    else:
        print("(good)")

    # ── Run all five solvers ──────────────────────────────────────────────────
    method_map = {
        "TSAI":       cv2.CALIB_HAND_EYE_TSAI,
        "PARK":       cv2.CALIB_HAND_EYE_PARK,
        "HORAUD":     cv2.CALIB_HAND_EYE_HORAUD,
        "ANDREFF":    cv2.CALIB_HAND_EYE_ANDREFF,
        "DANIILIDIS": cv2.CALIB_HAND_EYE_DANIILIDIS,
    }

    results = {}
    print("\n" + "=" * 72)
    print(f"{'Method':<14}  {'||t|| cm':>10}  {'AX=XB rot °':>12}  {'AX=XB t cm':>12}")
    print("=" * 72)
    for name, flag in method_map.items():
        try:
            R_x, t_x = cv2.calibrateHandEye(R_g2b, t_g2b, R_t2c, t_t2c, method=flag)
        except cv2.error as e:
            print(f"  {name:<12}  FAILED: {e}")
            continue
        T = np.eye(4); T[:3,:3]=R_x; T[:3,3]=t_x.flatten()
        rot_res, t_res = ax_xb_residuals(R_g2b, t_g2b, R_t2c, t_t2c, R_x, t_x)
        t_norm = np.linalg.norm(t_x) * 100
        orth = np.linalg.norm(R_x @ R_x.T - np.eye(3))
        results[name] = {
            "T_cam2tcp": T.tolist(),
            "t_norm_cm": float(t_norm),
            "axb_rot_deg": float(rot_res),
            "axb_t_cm": float(t_res * 100),
            "orthogonality_err": float(orth),
        }
        marker = " ← chosen" if name == args.method else ""
        print(f"  {name:<12}  {t_norm:>10.2f}  {rot_res:>12.3f}  {t_res*100:>12.3f}{marker}")
    print("=" * 72)

    if args.method not in results:
        fallback = min(results, key=lambda k: results[k]["axb_rot_deg"])
        print(f"\n[WARN] {args.method} failed – falling back to {fallback}")
        args.method = fallback

    best = results[args.method]
    T_best = np.array(best["T_cam2tcp"])

    print(f"\nChosen method : {args.method}")
    print("T_cam → TCP  (4×4):")
    print(np.round(T_best, 6))

    # ── Print data-quality recommendation ────────────────────────────────────
    if best["axb_rot_deg"] > 5.0:
        print("\n[WARN] AX=XB rotation residual > 5° – calibration may be inaccurate.")
        print("       Recommendations:")
        print("       1. Collect ≥ 20 poses with varied wrist tilt (not just pan).")
        print("       2. Ensure the chessboard is fixed firmly in the scene.")
        print("       3. Move the robot to at least 3 distinct wrist orientations.")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    output = {
        "chosen_method":  args.method,
        "n_pairs_used":   n_pairs,
        "T_cam2tcp":      T_best.tolist(),
        "camera_matrix":  K.tolist(),
        "dist_coeffs":    dist.tolist(),
        "all_methods":    results,
        "mount_config": {
            "robot_model":  args.robot_model,
            "mount_joint":  args.mount_joint,
            "note": (
                "T_cam2tcp is actually T_cam -> mount_joint_frame "
                "when mount_joint != 6"
            ) if args.mount_joint != 6 else "standard eye-in-hand: T_cam -> TCP",
        },
        "data_quality": {
            "n_pairs":              n_pairs,
            "rotation_diversity_deg": div,
            "chosen_axb_rot_deg":   best["axb_rot_deg"],
            "chosen_axb_t_cm":      best["axb_t_cm"],
        },
    }
    with open(args.out, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved → {args.out}")


if __name__ == "__main__":
    main()
