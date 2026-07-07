"""
paletizacao.py
==============
Estimação de pose em tempo real e sequência de pick-and-place por
visual servoing (versão de linha de comando, com janelas OpenCV).
"""

import argparse
import json
import math
import os
import sys
import time

os.environ.setdefault("QT_LOGGING_RULES", "*.debug=false;qt.qpa.*=false;qt.text.*=false")
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import cv2
import numpy as np

from estimar_pose import (
    BoxPoseEstimator, H_to_euler_deg, rvec_tvec_to_H, ur_fk, _UR_DH,
)

try:
    from detector_pallet import ConfigDetector, processar_frame, desenhar_overlay
    HAS_PALLET_DETECTOR = True
except ImportError:
    HAS_PALLET_DETECTOR = False
    print("[WARN] detector_pallet.py not found. Pallet scanning will be disabled.")

HOME_POSE = [0.685, -0.165, 0.650, math.pi, 0.0, 0.0]
PALLET_POSE = [-0.165, -0.685, 0.650, 2.2, -2.2, 0.0]

class RobotLink:
    def __init__(self, ip):
        from rtde_receive import RTDEReceiveInterface
        from rtde_control import RTDEControlInterface
        self.iface = RTDEReceiveInterface(ip)
        self.ctrl = RTDEControlInterface(ip)

    def get_joint_angles(self):
        return self.iface.getActualQ()

    def get_tcp_pose(self):
        return self.iface.getActualTCPPose()
        
    def move_to_pose(self, target_pose, speed=0.5, accel=0.5):
        success = self.ctrl.moveJ_IK(target_pose, speed, accel)
        if not success:
            raise RuntimeError("RTDE moveJ_IK returned False. Ensure free-drive is OFF and pose is reachable.")

class DummyRobotLink:
    def __init__(self):
        self._q = [0.0, -1.57, 1.57, -1.57, -1.57, 0.0]

    def get_joint_angles(self):
        return self._q

    def get_tcp_pose(self):
        H = ur_fk(self._q, robot_model="UR5e", up_to_joint=6)
        rvec, _ = cv2.Rodrigues(H[:3, :3])
        return [H[0, 3], H[1, 3], H[2, 3], *rvec.flatten().tolist()]
        
    def move_to_pose(self, target_pose, speed=0.5, accel=0.5):
        print(f"\n[DUMMY ROBOT] Simulating moveJ_IK to: {target_pose}")

def draw_pose_axes(img, K, dist, rvec, tvec, length):
    out = img.copy()
    cv2.drawFrameAxes(out, K, dist, np.asarray(rvec, np.float64), np.asarray(tvec, np.float64), length=length)
    return out

def make_data_panel(res, tcp_pose, robot_model, width=640, height=420):
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    panel[:] = (30, 30, 30)
    font = cv2.FONT_HERSHEY_SIMPLEX
    white, yellow, cyan, red = (255, 255, 255), (0, 255, 255), (255, 255, 0), (60, 60, 255)
    y, dy = 30, 26

    def line(txt, color=white, bold=1, indent=0):
        nonlocal y
        cv2.putText(panel, txt, (15 + indent, y), font, 0.6, color, bold)
        y += dy

    line("BOX POSE ESTIMATE", cyan, 2)
    if res is None:
        line("  no detection this frame", red)
    else:
        line(f"  mode          : {res['mode']}")
        line(f"  reproj err    : {res['reproj_error_px']:.2f} px")
        H_cam = res["H_box_in_cam"]
        t_cam = H_cam[:3, 3]
        e_cam = H_to_euler_deg(H_cam)
        line("  -- in camera frame --", yellow)
        line(f"     t  = [{t_cam[0]:+.4f}, {t_cam[1]:+.4f}, {t_cam[2]:+.4f}] m", indent=10)
        line(f"     RPY= [{e_cam[0]:+6.1f}, {e_cam[1]:+6.1f}, {e_cam[2]:+6.1f}] deg", indent=10)
        H_base = res["H_box_in_base"]
        line("  -- in robot base frame --", yellow)
        if H_base is not None:
            t_b = H_base[:3, 3]
            e_b = H_to_euler_deg(H_base)
            line(f"     t  = [{t_b[0]:+.4f}, {t_b[1]:+.4f}, {t_b[2]:+.4f}] m", indent=10)
            line(f"     RPY= [{e_b[0]:+6.1f}, {e_b[1]:+6.1f}, {e_b[2]:+6.1f}] deg", indent=10)
        else:
            line("     (unavailable - no joint angles)", red, indent=10)

    y += 10
    line("ROBOT TCP POSE (base frame)", cyan, 2)
    if tcp_pose is not None:
        t = tcp_pose[:3]
        rvec = np.array(tcp_pose[3:], np.float64)
        H_tcp = rvec_tvec_to_H(rvec, t)
        e = H_to_euler_deg(H_tcp)
        line(f"  t  = [{t[0]:+.4f}, {t[1]:+.4f}, {t[2]:+.4f}] m", indent=10)
        line(f"  RPY= [{e[0]:+6.1f}, {e[1]:+6.1f}, {e[2]:+6.1f}] deg", indent=10)
    else:
        line("  (no robot connection)", red, indent=10)

    line("")
    line(f"robot model: {robot_model}", (180, 180, 180))
    line("press 'q' quit, 's' snap, 'm' mode, 'h' home, SPACE start seq", (150, 150, 150))
    return panel


def main():
    pa = argparse.ArgumentParser(description="Real-time box pose estimation")
    pa.add_argument("--calib", default="handeye_calibration.json")
    pa.add_argument("--camera", default=1)
    pa.add_argument("--robot-ip", default="192.168.0.10")
    pa.add_argument("--no-robot", action="store_true")
    pa.add_argument("--robot-model", default="UR10", choices=list(_UR_DH.keys()))
    pa.add_argument("--mount-joint", type=int, default=6)
    pa.add_argument("--box-w", type=float, default=0.125)
    pa.add_argument("--box-d", type=float, default=0.125)
    pa.add_argument("--box-h", type=float, default=0.100)
    pa.add_argument("--no-aruco", action="store_true")
    pa.add_argument("--max-reproj", type=float, default=8.0)
    pa.add_argument("--every", type=int, default=1)
    
    seg_grp = pa.add_argument_group("Semantic segmentation (optional)")
    seg_grp.add_argument("--seg-backend", default=None)
    seg_grp.add_argument("--seg-model", default="")
    seg_grp.add_argument("--seg-classes", nargs="*", type=int, default=[])
    seg_grp.add_argument("--seg-conf", type=float, default=0.35)
    seg_grp.add_argument("--sam-variant", default="mobile_sam")
    seg_grp.add_argument("--no-face-discriminator", action="store_true")
    
    pal_grp = pa.add_argument_group("Pallet Detector Configuration")
    pal_grp.add_argument("--pallet-grid", default="3x4")
    pal_grp.add_argument("--pallet-modo", default="branco", choices=["branco", "reference", "heuristic", "combined", "yolo"])
    pal_grp.add_argument("--pallet-cantos", default="4boxes", choices=["aruco", "auto", "manual", "4boxes"])
    args = pa.parse_args()

    if HAS_PALLET_DETECTOR:
        pallet_cfg = ConfigDetector()
        try:
            l, c = args.pallet_grid.lower().split('x')
            pallet_cfg.linhas, pallet_cfg.colunas = int(l), int(c)
        except Exception:
            print("[WARN] Invalid --pallet-grid format. Defaulting to 3x4.")

    if not os.path.exists(args.calib):
        print(f"[ERROR] Calibration file not found: {args.calib}")
        sys.exit(1)
    with open(args.calib) as f: cal = json.load(f)

    K           = np.array(cal["camera_matrix"], np.float64)
    dist        = np.array(cal["dist_coeffs"],   np.float64)
    T_cam_mount = np.array(cal["T_cam2tcp"],     np.float64)

    calib_mount = cal.get("mount_config", {}).get("mount_joint", None)
    if calib_mount is not None: args.mount_joint = calib_mount
    calib_model = cal.get("mount_config", {}).get("robot_model", None)
    if calib_model is not None: args.robot_model = calib_model

    seg_backend = None
    if args.seg_backend:
        try:
            from estimar_pose import SegmentationBackend
            seg_backend = SegmentationBackend(
                backend=args.seg_backend, model=args.seg_model, class_ids=args.seg_classes,
                conf_thresh=args.seg_conf, sam_variant=args.sam_variant, discriminate_top_face=not args.no_face_discriminator,
            )
        except Exception as e: print(f"[WARN] Could not init seg backend: {e}")

    estimator = BoxPoseEstimator(
        K=K, dist=dist, T_cam_mount=T_cam_mount, box_W=args.box_w, box_D=args.box_d, box_H=args.box_h,
        robot_model=args.robot_model, mount_joint=args.mount_joint, seg_backend=seg_backend,
    )

    if args.no_robot or args.robot_ip is None:
        robot = DummyRobotLink()
    else:
        try:
            robot = RobotLink(args.robot_ip)
            print(f"[INFO] Connected to robot at {args.robot_ip}")
        except Exception as e:
            print(f"[WARN] Robot connection failed ({e}). Fallback to dummy.")
            robot = DummyRobotLink()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened(): sys.exit(1)

    cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Pose Data", cv2.WINDOW_NORMAL)
    os.makedirs("snapshots", exist_ok=True)
    
    frame_idx, last_res = 0, None
    modes = ["auto", "HSV", "ML", "aruco"]
    current_mode_idx = 0

    print("[INFO] Ready. Press 'SPACE' to start Pick & Place sequence.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.05)
                continue

            try: q = robot.get_joint_angles()
            except: q = None

            try: tcp_pose = robot.get_tcp_pose()
            except: tcp_pose = None

            entry = {"joint_angles": q} if q is not None else {}
            current_mode = modes[current_mode_idx]

            if frame_idx % max(1, args.every) == 0:
                last_res = estimator.estimate(
                    frame, entry=entry, use_aruco=not args.no_aruco,
                    max_reproj_px=args.max_reproj, debug_prefix=None,
                    force_mode=current_mode
                )

            display = frame.copy()
            cv2.putText(display, f"Request Mode: {current_mode}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            if last_res is not None:
                mask = last_res.get("seg_mask")
                if mask is not None:
                    blue_layer = display.copy()
                    blue_layer[mask > 0] = [255, 0, 0]
                    cv2.addWeighted(blue_layer, 0.4, display, 0.6, 0, display)

                quad_pts = last_res.get("quad_pts")
                if quad_pts is not None:
                    for pt in quad_pts.astype(int):
                        cv2.circle(display, tuple(pt), 6, (0, 0, 255), -1)

                rvec, tvec = last_res["rvec"], last_res["tvec"]
                display = draw_pose_axes(display, K, dist, rvec, tvec, length=args.box_w * 0.6)
                cv2.putText(display, f"Success mode: {last_res['mode']}  err: {last_res['reproj_error_px']:.1f}px", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                cv2.putText(display, "no detection", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            panel = make_data_panel(last_res, tcp_pose, args.robot_model)
            cv2.imshow("Camera", display)
            cv2.imshow("Pose Data", panel)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27): break
            elif key == ord('s'):
                ts = time.strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(f"snapshots/{ts}_camera.png", display)
                cv2.imwrite(f"snapshots/{ts}_data.png", panel)
            elif key == ord('m'):
                current_mode_idx = (current_mode_idx + 1) % len(modes)
            elif key == ord('h'):
                try: robot.move_to_pose(HOME_POSE)
                except Exception as e: print(f"[ERROR] Robot movement failed: {e}")
            
            # --- 4-STEP VISUAL SERVOING SEQUENCE + TRACKER ---
            elif key == ord(' '):
                print("\n[INFO] Starting 4-Step Visual Servoing Sequence...")
                
                tracked_center_2d = None
                
                def run_phase(phase_num, duration, target_z):
                    nonlocal tracked_center_2d
                    print(f"\n--- [PHASE {phase_num}] Sampling for {duration}s... ---")
                    
                    for _ in range(10): cap.read()
                        
                    samples = []
                    t_start = time.time()
                    
                    while time.time() - t_start < duration:
                        ok, s_frame = cap.read()
                        if not ok: continue

                        try: q = robot.get_joint_angles()
                        except: q = None
                        sample_entry = {"joint_angles": q} if q is not None else {}

                        res = estimator.estimate(
                            s_frame, entry=sample_entry, use_aruco=not args.no_aruco,
                            max_reproj_px=2.0, debug_prefix=None, force_mode=current_mode,
                            target_center_2d=tracked_center_2d,
                        )

                        if res is not None and res["H_box_in_base"] is not None:
                            samples.append(res)
                            # mantém o rastreio da mesma caixa durante a aproximação
                            pts = res.get("quad_pts")
                            if pts is None: pts = res.get("aruco_pts")
                            if pts is not None: tracked_center_2d = pts.mean(axis=0)

                        display = s_frame.copy()
                        if res is not None:
                            mask = res.get("seg_mask")
                            if mask is not None:
                                blue_layer = display.copy()
                                blue_layer[mask > 0] = [255, 0, 0]
                                cv2.addWeighted(blue_layer, 0.4, display, 0.6, 0, display)
                            quad_pts = res.get("quad_pts")
                            if quad_pts is not None:
                                for pt in quad_pts.astype(int): cv2.circle(display, tuple(pt), 6, (0, 0, 255), -1)
                        
                        cv2.imshow("Camera", display)
                        cv2.waitKey(1)

                    if not samples:
                        print(f"[ERROR] Phase {phase_num} aborted: No valid poses found.")
                        return False

                    samples.sort(key=lambda x: x["reproj_error_px"])
                    top_samples = samples[:5]

                    x_vals, y_vals = [], []
                    for s in top_samples:
                        if s["mode"] == "aruco":
                            box_center_local = np.array([0.0, 0.0, 0.0, 1.0])
                        else:
                            box_center_local = np.array([args.box_w / 2.0, args.box_d / 2.0, 0.0, 1.0])
                            
                        center_base = s["H_box_in_base"] @ box_center_local
                        x_vals.append(center_base[0])
                        y_vals.append(center_base[1])

                    mean_x, mean_y = sum(x_vals) / len(x_vals), sum(y_vals) / len(y_vals)
                    target_pose = [mean_x, mean_y, target_z, math.pi, 0.0, 0.0]

                    print(f"[INFO] Moving to target (centered): x={mean_x:.4f}, y={mean_y:.4f}, z={target_z:.3f}")
                    try:
                        robot.move_to_pose(target_pose)
                        time.sleep(0.5) 
                        return True
                    except Exception as e:
                        print(f"[ERROR] Robot movement failed: {e}")
                        return False

                if run_phase(phase_num=1, duration=5.0, target_z=0.350):
                    if run_phase(phase_num=2, duration=2.5, target_z=0.050):
                        if run_phase(phase_num=3, duration=2.5, target_z=0.050):
                            if run_phase(phase_num=4, duration=2.5, target_z=-0.430):
                                print("\n[INFO] Waiting 5 seconds before moving to PALLET_POSE...")
                                time.sleep(5.0)
                                try:
                                    robot.move_to_pose(PALLET_POSE)
                                    
                                    # --- PHASE 5: PALLET SCANNING ---
                                    if HAS_PALLET_DETECTOR:
                                        print("\n--- [PHASE 5] Scanning Pallet... ---")
                                        time.sleep(1.0)
                                        for _ in range(10): cap.read()
                                        
                                        t_start, final_proxima = time.time(), None
                                        cv2.namedWindow("Pallet View", cv2.WINDOW_NORMAL)
                                        
                                        while time.time() - t_start < 5.0:
                                            ok, p_frame = cap.read()
                                            if not ok: continue
                                                
                                            res = processar_frame(p_frame, pallet_cfg, modo=args.pallet_modo, metodo_cantos=args.pallet_cantos)
                                            if res is not None:
                                                topo, matriz, detalhes, proxima = res
                                                overlay = desenhar_overlay(topo, detalhes, pallet_cfg, proxima)
                                                cv2.imshow("Pallet View", overlay)
                                                final_proxima = proxima
                                            else:
                                                p_display = p_frame.copy()
                                                cv2.putText(p_display, "Searching...", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                                                cv2.imshow("Pallet View", p_display)
                                            cv2.waitKey(1)
                                            
                                        if final_proxima is not None: print(f"\n[SUCCESS] Drop location -> Row: {final_proxima[0]}, Col: {final_proxima[1]}")
                                        else: print("\n[WARN] Pallet completely full or unreadable.")
                                            
                                except Exception as e: print(f"[ERROR] Failed to move to PALLET_POSE: {e}")
                
                print("\n[INFO] Sequence finished. Returning to live view.")
            frame_idx += 1
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()