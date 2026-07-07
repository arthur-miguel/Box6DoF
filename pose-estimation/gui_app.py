"""
gui_app.py
==========
Interface gráfica (CustomTkinter) para estimação de pose em tempo real e
controle da sequência de paletização.
"""

import argparse
import json
import math
import os
import sys
import time
import threading
import queue

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import cv2
import numpy as np
import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from estimar_pose import (
    BoxPoseEstimator, H_to_euler_deg, rvec_tvec_to_H, ur_fk, _UR_DH,
)

try:
    from detector_pallet import ConfigDetector, processar_frame, desenhar_overlay
    HAS_PALLET_DETECTOR = True
except ImportError:
    HAS_PALLET_DETECTOR = False
    print("[WARN] detector_pallet.py not found. Pallet scanning will be disabled.")

# ── GLOBAL VARIABLES ─────────────────────────────────────────────────────────
HOME_POSE = [0.685, -0.165, 0.650, math.pi, 0.0, 0.0]
PALLET_POSE = [-0.165, -0.685, 0.650, 2.2, -2.2, 0.0]

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")


class RobotLink:
    def __init__(self, ip):
        from rtde_receive import RTDEReceiveInterface
        self.ip = ip
        self.iface = RTDEReceiveInterface(self.ip)
        self.ctrl = None

    def get_joint_angles(self): 
        return self.iface.getActualQ()
        
    def get_tcp_pose(self): 
        return self.iface.getActualTCPPose()
        
    def move_to_pose(self, target_pose, speed=0.5, accel=0.5):
        from rtde_control import RTDEControlInterface
        if self.ctrl is None:
            self.ctrl = RTDEControlInterface(self.ip)
        elif not self.ctrl.isConnected():
            try:
                self.ctrl.reconnect()
            except Exception as e:
                self.ctrl.disconnect()
                self.ctrl = RTDEControlInterface(self.ip)

        success = self.ctrl.moveJ_IK(target_pose, speed, accel)
        if not success:
            raise RuntimeError("RTDE moveJ_IK returned False.")
            
    def disconnect(self):
        try:
            if self.ctrl is not None: self.ctrl.disconnect()
        except: pass
        try:
            if self.iface is not None: self.iface.disconnect()
        except: pass

class DummyRobotLink:
    def __init__(self): self._q = [0.0, -1.57, 1.57, -1.57, -1.57, 0.0]
    def get_joint_angles(self): return self._q
    def get_tcp_pose(self):
        H = ur_fk(self._q, robot_model="UR5e", up_to_joint=6)
        rvec, _ = cv2.Rodrigues(H[:3, :3])
        return [H[0, 3], H[1, 3], H[2, 3], *rvec.flatten().tolist()]
    def move_to_pose(self, target_pose, speed=0.5, accel=0.5):
        time.sleep(1.0)
        print(f"[DUMMY ROBOT] Simulating moveJ_IK to: {target_pose}")
    def disconnect(self): pass

def draw_pose_axes(img, K, dist, rvec, tvec, length):
    out = img.copy()
    cv2.drawFrameAxes(out, K, dist, np.asarray(rvec, np.float64), np.asarray(tvec, np.float64), length=length)
    return out

class VideoWorker(threading.Thread):
    def __init__(self, args, estimator, robot, pallet_cfg, gui_queue):
        super().__init__()
        self.daemon = True
        self._run_flag = True
        self.args = args
        self.estimator = estimator
        self.robot = robot
        self.pallet_cfg = pallet_cfg
        self.gui_queue = gui_queue
        
        self.modes = ["auto", "HSV", "ML", "aruco"]
        self.current_mode_idx = 0
        self.command_queue = queue.Queue()
        self.debug_saves_enabled = False
        self.current_ip = args.robot_ip if not args.no_robot else None

    def emit_log(self, msg):
        self.gui_queue.put(("LOG", msg))

    def run(self):
        cap = cv2.VideoCapture(self.args.camera)
        if not cap.isOpened():
            self.emit_log(f"[ERROR] Could not open camera {self.args.camera}")
            return
            
        frame_idx = 0
        last_res = None
        self.emit_log("[INFO] System Ready. Press Space or 'Start Sequence' to begin.")

        try:
            while self._run_flag:
                while not self.command_queue.empty():
                    cmd = self.command_queue.get()
                    if isinstance(cmd, str):
                        if cmd == "MODE":
                            self.current_mode_idx = (self.current_mode_idx + 1) % len(self.modes)
                            self.emit_log(f"[INFO] Detection mode changed to: {self.modes[self.current_mode_idx]}")
                        elif cmd == "HOME":
                            self.emit_log("[INFO] Moving to HOME_POSE...")
                            try:
                                self.robot.move_to_pose(HOME_POSE)
                                self.emit_log("[INFO] Arrived at HOME_POSE.")
                            except Exception as e:
                                self.emit_log(f"[ERROR] Move to HOME failed: {e}")
                        elif cmd == "SEQUENCE":
                            self.execute_sequence(cap)
                    elif isinstance(cmd, tuple) and cmd[0] == "UPDATE_CONFIG":
                        self.update_configurations(cmd[1])
                    elif isinstance(cmd, tuple) and cmd[0] == "UPDATE_HYPERPARAMS":
                        self.update_hyperparams(cmd[1])

                ok, frame = cap.read()
                if not ok:
                    time.sleep(0.05)
                    continue

                try: q = self.robot.get_joint_angles()
                except: q = None
                try: tcp_pose = self.robot.get_tcp_pose()
                except: tcp_pose = None

                entry = {"joint_angles": q} if q is not None else {}
                current_mode = self.modes[self.current_mode_idx]

                if frame_idx % max(1, self.args.every) == 0:
                    last_res = self.estimator.estimate(
                        frame, entry=entry, use_aruco=not self.args.no_aruco,
                        max_reproj_px=self.args.max_reproj, debug_prefix=None,
                        force_mode=current_mode
                    )

                display = self.draw_overlays(frame, last_res, current_mode)
                
                self.gui_queue.put(("IMAGE", display))
                self.gui_queue.put(("DATA", {
                    "res": last_res,
                    "tcp": tcp_pose,
                    "mode": current_mode,
                    "robot_model": self.args.robot_model
                }))
                
                frame_idx += 1

        finally:
            cap.release()
            if hasattr(self.robot, 'disconnect'):
                self.robot.disconnect()

    def update_configurations(self, cfg):
        self.emit_log("[INFO] Applying new configurations...")
        if "debug_mode" in cfg:
            self.debug_saves_enabled = cfg["debug_mode"]
            self.emit_log(f"[INFO] Debug intermediate saves enabled: {self.debug_saves_enabled}")

        if cfg["ip"]:
            if cfg["ip"] != self.current_ip:
                try:
                    if hasattr(self.robot, 'disconnect'):
                        self.robot.disconnect()
                    self.robot = RobotLink(cfg["ip"])
                    self.current_ip = cfg["ip"]
                    self.emit_log(f"[SUCCESS] Connected to robot at {cfg['ip']}")
                except Exception as e:
                    self.emit_log(f"[WARN] Failed to connect to {cfg['ip']} ({e}). Using DummyRobot.")
                    self.robot = DummyRobotLink()
                    self.current_ip = None
                
        backend_name = cfg["backend"]
        if backend_name == "None":
            self.estimator.seg_backend = None
            self.emit_log("[INFO] ML Backend disabled.")
        else:
            try:
                from estimar_pose import SegmentationBackend
                self.estimator.seg_backend = SegmentationBackend(
                    backend=backend_name, model=cfg["weights"], class_ids=self.args.seg_classes,
                    conf_thresh=self.args.seg_conf, sam_variant=self.args.sam_variant, 
                    discriminate_top_face=not self.args.no_face_discriminator
                )
                self.emit_log(f"[SUCCESS] ML Backend '{backend_name}' loaded successfully.")
            except Exception as e:
                self.emit_log(f"[ERROR] Failed to load ML backend: {e}")
                self.estimator.seg_backend = None

    def update_hyperparams(self, cfg):
        """Live injects new hyperparameters into the backend instances."""
        self.estimator.hsv_lower = cfg["hsv_lower"]
        self.estimator.hsv_upper = cfg["hsv_upper"]
        self.args.seg_conf = cfg["yolo_conf"]
        self.args.no_face_discriminator = cfg["no_face_disc"]
        
        if self.estimator.seg_backend is not None:
            self.estimator.seg_backend.conf = cfg["yolo_conf"]
            self.estimator.seg_backend.disc_top = not cfg["no_face_disc"]
            
        self.emit_log(f"[INFO] Hyperparameters updated (HSV limits: {cfg['hsv_lower']} - {cfg['hsv_upper']})")

    def draw_overlays(self, frame, last_res, current_mode):
        display = frame.copy()
        cv2.putText(display, f"Request Mode: {current_mode}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
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
            display = draw_pose_axes(display, self.estimator.K, self.estimator.dist, rvec, tvec, length=self.args.box_w * 0.6)
            cv2.putText(display, f"Success: {last_res['mode']}  err: {last_res['reproj_error_px']:.1f}px", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(display, "no detection", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return display

    def execute_sequence(self, cap):
        self.emit_log("\n[INFO] Starting Visual Servoing Sequence...")
        self.gui_queue.put(("CLEAR_GRAPH", None))
        
        seq_timestamp = time.strftime("%Y%m%d_%H%M%S")
        log_dir = os.path.join("logs", f"sequence_{seq_timestamp}")
        debug_dir = os.path.join(log_dir, "debug")
        os.makedirs(log_dir, exist_ok=True)
        if self.debug_saves_enabled: os.makedirs(debug_dir, exist_ok=True)
            
        self.emit_log(f"[INFO] Logging sequence data to: {log_dir}/")
        
        sequence_data = {
            "timestamp": seq_timestamp,
            "debug_saves_enabled": self.debug_saves_enabled,
            "phases": {}
        }
        
        tracked_center_2d = None
        current_mode = self.modes[self.current_mode_idx]
        seq_start_time = time.time()

        def run_phase(phase_num, duration, target_z):
            nonlocal tracked_center_2d
            self.emit_log(f"--- [PHASE {phase_num}] Sampling for {duration}s... ---")
            for _ in range(10): cap.read()
            
            samples = []
            t_start = time.time()
            last_frame_for_log = None
            
            while time.time() - t_start < duration:
                ok, s_frame = cap.read()
                if not ok: continue

                try: q = self.robot.get_joint_angles()
                except: q = None
                sample_entry = {"joint_angles": q} if q is not None else {}

                res = self.estimator.estimate(
                    s_frame, entry=sample_entry, use_aruco=not self.args.no_aruco,
                    max_reproj_px=2.0, force_mode=current_mode, target_center_2d=tracked_center_2d
                )

                if res is not None and res["H_box_in_base"] is not None:
                    res["raw_frame"] = s_frame.copy() 
                    samples.append(res)
                    pts = res.get("quad_pts") if res.get("quad_pts") is not None else res.get("aruco_pts")
                    if pts is not None: tracked_center_2d = pts.mean(axis=0)
                    self.gui_queue.put(("PLOT", (time.time() - seq_start_time, res["reproj_error_px"])))

                display = self.draw_overlays(s_frame, res, current_mode)
                self.gui_queue.put(("IMAGE", display))
                last_frame_for_log = display.copy()
            
            if not samples:
                self.emit_log(f"[ERROR] Phase {phase_num} aborted: No valid poses found.")
                return False

            samples.sort(key=lambda x: x["reproj_error_px"])
            top_samples = samples[:5]

            x_vals, y_vals = [], []
            for s in top_samples:
                if s["mode"] == "aruco":
                    box_center_local = np.array([0.0, 0.0, 0.0, 1.0])
                else:
                    box_center_local = np.array([self.args.box_w / 2.0, self.args.box_d / 2.0, 0.0, 1.0])
                center_base = s["H_box_in_base"] @ box_center_local
                x_vals.append(center_base[0])
                y_vals.append(center_base[1])

            mean_x, mean_y = sum(x_vals)/len(x_vals), sum(y_vals)/len(y_vals)
            target_pose = [mean_x, mean_y, target_z, math.pi, 0.0, 0.0]

            self.emit_log(f"Moving to: x={mean_x:.3f}, y={mean_y:.3f}, z={target_z:.3f}")
            
            best_res = top_samples[0]
            err_vector = [round(s["reproj_error_px"], 4) for s in top_samples] 
            
            img_filename = f"phase_{phase_num}.jpg"
            if last_frame_for_log is not None:
                cv2.imwrite(os.path.join(log_dir, img_filename), last_frame_for_log)
                
            debug_prefix = None
            if self.debug_saves_enabled:
                debug_prefix = os.path.join(debug_dir, f"phase_{phase_num}_best")
                self.estimator.estimate(
                    best_res["raw_frame"], entry={}, use_aruco=not self.args.no_aruco,
                    max_reproj_px=8.0, force_mode=current_mode, target_center_2d=tracked_center_2d,
                    debug_prefix=debug_prefix
                )

            sequence_data["phases"][f"phase_{phase_num}"] = {
                "mode": best_res["mode"],
                "reproj_errors_px_vector": err_vector,
                "target_pose_base": [round(v, 4) for v in target_pose],
                "image_file": img_filename,
                "debug_prefix": debug_prefix
            }

            try:
                self.robot.move_to_pose(target_pose)
                time.sleep(0.5)
                return True
            except Exception as e:
                self.emit_log(f"[ERROR] Robot movement failed: {e}")
                return False

        if run_phase(1, 5.0, 0.350) and run_phase(2, 2.5, 0.050) and run_phase(3, 2.5, 0.050) and run_phase(4, 2.5, -0.430):
            self.emit_log("[INFO] Waiting 5 seconds before moving to PALLET_POSE...")
            time.sleep(5.0)
            try:
                self.robot.move_to_pose(PALLET_POSE)
                
                if HAS_PALLET_DETECTOR:
                    self.emit_log("--- [PHASE 5] Scanning Pallet... ---")
                    time.sleep(1.0)
                    for _ in range(10): cap.read()
                    
                    t_start, final_proxima = time.time(), None
                    last_overlay = None
                    
                    while time.time() - t_start < 5.0:
                        ok, p_frame = cap.read()
                        if not ok: continue
                        
                        res = processar_frame(p_frame, self.pallet_cfg, modo=self.args.pallet_modo, metodo_cantos=self.args.pallet_cantos)
                        if res is not None:
                            topo, matriz, detalhes, proxima = res
                            overlay = desenhar_overlay(topo, detalhes, self.pallet_cfg, proxima)
                            self.gui_queue.put(("IMAGE", overlay))
                            final_proxima = proxima
                            last_overlay = overlay.copy()
                        else:
                            p_display = p_frame.copy()
                            cv2.putText(p_display, "Searching for pallet...", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                            self.gui_queue.put(("IMAGE", p_display))
                            last_overlay = p_display.copy()
                            
                    if last_overlay is not None:
                        cv2.imwrite(os.path.join(log_dir, "phase_5.jpg"), last_overlay)

                    if final_proxima is not None: 
                        self.emit_log(f"[SUCCESS] Drop location -> Row: {final_proxima[0]}, Col: {final_proxima[1]}")
                        sequence_data["phases"]["phase_5"] = {
                            "drop_location": [final_proxima[0], final_proxima[1]],
                            "image_file": "phase_5.jpg"
                        }
                    else: 
                        self.emit_log("[WARN] Pallet completely full or unreadable.")
                        sequence_data["phases"]["phase_5"] = {
                            "drop_location": None,
                            "error": "Pallet full or unreadable",
                            "image_file": "phase_5.jpg"
                        }
                        
            except Exception as e: 
                self.emit_log(f"[ERROR] Failed to move to PALLET: {e}")
        
        json_path = os.path.join(log_dir, "sequence_log.json")
        with open(json_path, "w") as f:
            json.dump(sequence_data, f, indent=4)
        self.emit_log(f"[INFO] Sequence log saved to: {json_path}")
        self.emit_log("[INFO] Sequence finished. Returning to live tracking.")


class App(ctk.CTk):
    def __init__(self, args, estimator, robot, pallet_cfg):
        super().__init__()
        self.title("Vision & Palletization Control Panel")
        self.geometry("1300x800")
        
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=2)
        self.grid_rowconfigure(0, weight=1)

        self.current_hsv_lower = (5, 15, 30)
        self.current_hsv_upper = (40, 255, 255)
        self.current_yolo_conf = args.seg_conf
        self.current_no_face_disc = args.no_face_discriminator

        # ---------------------------------------------------------
        # LEFT PANEL
        # ---------------------------------------------------------
        self.left_frame = ctk.CTkFrame(self)
        self.left_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        # 1. Configuration Group
        self.config_group = ctk.CTkFrame(self.left_frame)
        self.config_group.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(self.config_group, text="Dynamic Configuration", font=("Arial", 14, "bold")).pack(pady=5)
        
        self.ip_frame = ctk.CTkFrame(self.config_group, fg_color="transparent")
        self.ip_frame.pack(fill="x", padx=5, pady=2)
        ctk.CTkLabel(self.ip_frame, text="Robot IP:", width=100, anchor="w").pack(side="left")
        self.ip_input = ctk.CTkEntry(self.ip_frame)
        self.ip_input.insert(0, args.robot_ip)
        self.ip_input.pack(side="left", fill="x", expand=True)

        self.ml_frame = ctk.CTkFrame(self.config_group, fg_color="transparent")
        self.ml_frame.pack(fill="x", padx=5, pady=2)
        ctk.CTkLabel(self.ml_frame, text="ML Backend:", width=100, anchor="w").pack(side="left")
        self.ml_combo = ctk.CTkComboBox(self.ml_frame, values=["None", "ultralytics", "sam", "onnx"])
        self.ml_combo.set(args.seg_backend if args.seg_backend else "None")
        self.ml_combo.pack(side="left", fill="x", expand=True)

        self.weight_frame = ctk.CTkFrame(self.config_group, fg_color="transparent")
        self.weight_frame.pack(fill="x", padx=5, pady=2)
        ctk.CTkLabel(self.weight_frame, text="Weights:", width=100, anchor="w").pack(side="left")
        self.weight_input = ctk.CTkEntry(self.weight_frame)
        self.weight_input.insert(0, args.seg_model)
        self.weight_input.pack(side="left", fill="x", expand=True, padx=(0, 5))
        self.btn_browse = ctk.CTkButton(self.weight_frame, text="Browse", width=60, command=self.browse_weights)
        self.btn_browse.pack(side="left")

        self.chk_debug_var = tk.BooleanVar(value=False)
        self.chk_debug = ctk.CTkCheckBox(self.config_group, text="Enable Debug Intermediate Image Saves", variable=self.chk_debug_var)
        self.chk_debug.pack(anchor="w", padx=10, pady=10)

        self.conf_btns_frame = ctk.CTkFrame(self.config_group, fg_color="transparent")
        self.conf_btns_frame.pack(fill="x", padx=10, pady=10)
        
        self.btn_apply = ctk.CTkButton(self.conf_btns_frame, text="Apply Configuration", fg_color="#1E88E5", hover_color="#1565C0", command=self.apply_config)
        self.btn_apply.pack(side="left", fill="x", expand=True, padx=(0, 5))
        
        self.btn_hp = ctk.CTkButton(self.conf_btns_frame, text="Advanced Settings", fg_color="#5E35B1", hover_color="#4527A0", command=self.open_hyperparams)
        self.btn_hp.pack(side="right", fill="x", expand=True, padx=(5, 0))

        # 2. Telemetry Data
        telemetry_font = ctk.CTkFont(family="Courier", size=12)
        
        self.lbl_cam_pose = ctk.CTkLabel(self.left_frame, text="CAMERA POSE:\nWaiting...", justify="left", font=telemetry_font, fg_color="#1a1a1a", text_color="#00FF00", corner_radius=5, height=85)
        self.lbl_cam_pose.pack(fill="x", padx=10, pady=5)
        
        self.lbl_base_pose = ctk.CTkLabel(self.left_frame, text="BASE POSE:\nWaiting...", justify="left", font=telemetry_font, fg_color="#1a1a1a", text_color="#00FF00", corner_radius=5, height=85)
        self.lbl_base_pose.pack(fill="x", padx=10, pady=5)
        
        self.lbl_tcp_pose = ctk.CTkLabel(self.left_frame, text="TCP POSE:\nWaiting...", justify="left", font=telemetry_font, fg_color="#1a1a1a", text_color="#00FF00", corner_radius=5, height=85)
        self.lbl_tcp_pose.pack(fill="x", padx=10, pady=5)

        # 3. System Logs
        ctk.CTkLabel(self.left_frame, text="System Logs:", anchor="w").pack(fill="x", padx=10, pady=(10, 0))
        self.log_box = ctk.CTkTextbox(self.left_frame, font=("Courier", 11), fg_color="#121212", text_color="#FFFFFF")
        self.log_box.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        # 4. Action Buttons
        self.btn_frame = ctk.CTkFrame(self.left_frame, fg_color="transparent")
        self.btn_frame.pack(fill="x", padx=10, pady=10)
        self.btn_frame.grid_columnconfigure((0, 1), weight=1)

        self.btn_mode = ctk.CTkButton(self.btn_frame, text="Cycle Mode (M)", command=lambda: self.queue_cmd("MODE"))
        self.btn_mode.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        
        self.btn_home = ctk.CTkButton(self.btn_frame, text="Move Home (H)", command=lambda: self.queue_cmd("HOME"))
        self.btn_home.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        
        self.btn_snap = ctk.CTkButton(self.btn_frame, text="Snapshot (S)", command=self.take_snapshot)
        self.btn_snap.grid(row=1, column=0, padx=5, pady=5, sticky="ew")
        
        self.btn_seq = ctk.CTkButton(self.btn_frame, text="START SEQUENCE (SPACE)", fg_color="#388E3C", hover_color="#2E7D32", text_color="white", command=lambda: self.queue_cmd("SEQUENCE"))
        self.btn_seq.grid(row=1, column=1, padx=5, pady=5, sticky="ew")


        # ---------------------------------------------------------
        # RIGHT PANEL (Video & Graph)
        # ---------------------------------------------------------
        self.right_frame = ctk.CTkFrame(self)
        self.right_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        self.right_frame.grid_rowconfigure(0, weight=3)
        self.right_frame.grid_rowconfigure(1, weight=1)
        self.right_frame.grid_columnconfigure(0, weight=1)

        # Video Label
        self.video_label = tk.Label(self.right_frame, bg="black")
        self.video_label.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        # Matplotlib Graph Configuration 
        self.fig = Figure(figsize=(5, 2), dpi=100)
        self.fig.patch.set_facecolor('#222222')
        self.fig.subplots_adjust(left=0.1, bottom=0.25, right=0.95, top=0.85) 
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor('#222222')
        self.ax.set_title("Reprojection Error During Servoing", color='white', fontsize=10)
        self.ax.set_xlabel("Time (s)", color='white')
        self.ax.set_ylabel("Error (px)", color='white')
        self.ax.tick_params(colors='white')
        self.ax.grid(True, color='#444444')
        self.line, = self.ax.plot([], [], color='green', linewidth=2)
        
        self.graph_canvas = FigureCanvasTkAgg(self.fig, master=self.right_frame)
        self.graph_canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        
        self.error_x = []
        self.error_y = []

        # --- Setup Threading ---
        self.gui_queue = queue.Queue()
        self.worker = VideoWorker(args, estimator, robot, pallet_cfg, self.gui_queue)
        self.worker.start()
        self.last_frame = None

        # --- Key Bindings ---
        self.bind("<Key>", self.handle_keypress)
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Start UI Update Loop
        self.after(15, self.update_gui_from_queue)

    def open_hyperparams(self):
        """Spawns a popup window to configure hyperparameters on the fly."""
        if hasattr(self, "hp_window") and self.hp_window is not None and self.hp_window.winfo_exists():
            self.hp_window.focus()
            return
            
        self.hp_window = ctk.CTkToplevel(self)
        self.hp_window.title("Advanced Hyperparameters")
        self.hp_window.geometry("360x320")
        self.hp_window.attributes("-topmost", True)
        
        def create_hsv_row(parent, label, default_vals):
            frame = ctk.CTkFrame(parent, fg_color="transparent")
            frame.pack(fill="x", padx=10, pady=5)
            ctk.CTkLabel(frame, text=label, width=80, anchor="w").pack(side="left")
            eh = ctk.CTkEntry(frame, width=50)
            eh.insert(0, str(default_vals[0]))
            eh.pack(side="left", padx=5)
            es = ctk.CTkEntry(frame, width=50)
            es.insert(0, str(default_vals[1]))
            es.pack(side="left", padx=5)
            ev = ctk.CTkEntry(frame, width=50)
            ev.insert(0, str(default_vals[2]))
            ev.pack(side="left", padx=5)
            return eh, es, ev

        ctk.CTkLabel(self.hp_window, text="HSV Mask Constraints", font=("Arial", 12, "bold")).pack(pady=(10,0))
        self.eh_l, self.es_l, self.ev_l = create_hsv_row(self.hp_window, "Lower [H,S,V]:", self.current_hsv_lower)
        self.eh_u, self.es_u, self.ev_u = create_hsv_row(self.hp_window, "Upper [H,S,V]:", self.current_hsv_upper)
        
        ctk.CTkLabel(self.hp_window, text="ML Constraints", font=("Arial", 12, "bold")).pack(pady=(15,0))
        
        conf_frame = ctk.CTkFrame(self.hp_window, fg_color="transparent")
        conf_frame.pack(fill="x", padx=10, pady=5)
        ctk.CTkLabel(conf_frame, text="YOLO Confidence:", width=120, anchor="w").pack(side="left")
        self.e_conf = ctk.CTkEntry(conf_frame, width=80)
        self.e_conf.insert(0, str(self.current_yolo_conf))
        self.e_conf.pack(side="left", padx=5)
        
        self.chk_no_face_hp_var = tk.BooleanVar(value=self.current_no_face_disc)
        self.chk_no_face_hp = ctk.CTkCheckBox(self.hp_window, text="Disable Top-Face Discriminator", variable=self.chk_no_face_hp_var)
        self.chk_no_face_hp.pack(anchor="w", padx=10, pady=15)
        
        btn_save = ctk.CTkButton(self.hp_window, text="Save & Inject Live", fg_color="#4CAF50", hover_color="#388E3C", command=self.save_hyperparams)
        btn_save.pack(pady=10)
        
    def save_hyperparams(self):
        """Sends the newly selected hyperparameters directly into the worker thread."""
        try:
            hsv_l = (int(self.eh_l.get()), int(self.es_l.get()), int(self.ev_l.get()))
            hsv_u = (int(self.eh_u.get()), int(self.es_u.get()), int(self.ev_u.get()))
            yolo_c = float(self.e_conf.get())
            no_face = self.chk_no_face_hp_var.get()
            
            self.current_hsv_lower = hsv_l
            self.current_hsv_upper = hsv_u
            self.current_yolo_conf = yolo_c
            self.current_no_face_disc = no_face
            
            cfg = {
                "hsv_lower": hsv_l,
                "hsv_upper": hsv_u,
                "yolo_conf": yolo_c,
                "no_face_disc": no_face
            }
            self.queue_cmd(("UPDATE_HYPERPARAMS", cfg))
            self.hp_window.destroy()
        except ValueError:
            self.append_log("[ERROR] Invalid hyperparameter format. Ensure HSV values are integers and Conf is float.")

    def browse_weights(self):
        file_path = filedialog.askopenfilename(title="Select Weights File", filetypes=(("Model Files", "*.pt *.pth *.onnx"), ("All Files", "*.*")))
        if file_path:
            self.weight_input.delete(0, tk.END)
            self.weight_input.insert(0, file_path)

    def apply_config(self):
        cfg = {
            "ip": self.ip_input.get().strip(),
            "backend": self.ml_combo.get(),
            "weights": self.weight_input.get().strip(),
            "debug_mode": self.chk_debug_var.get()
        }
        self.queue_cmd(("UPDATE_CONFIG", cfg))

    def queue_cmd(self, cmd):
        self.worker.command_queue.put(cmd)

    def handle_keypress(self, event):
        if isinstance(event.widget, (tk.Entry, ctk.CTkEntry)): return
        
        k = event.keysym.lower()
        if k == 'm': self.queue_cmd("MODE")
        elif k == 'h': self.queue_cmd("HOME")
        elif k == 's': self.take_snapshot()
        elif k == 'space': self.queue_cmd("SEQUENCE")

    def take_snapshot(self):
        if self.last_frame is not None:
            ts = time.strftime("%Y%m%d_%H%M%S")
            cv2.imwrite(f"snapshots/{ts}_camera.png", self.last_frame)
            self.append_log(f"[INFO] Snapshot saved -> snapshots/{ts}_camera.png")

    def append_log(self, msg):
        self.log_box.insert("end", msg + "\n")
        self.log_box.see("end")

    def update_gui_from_queue(self):
        while not self.gui_queue.empty():
            try:
                msg_type, data = self.gui_queue.get_nowait()
                
                if msg_type == "LOG":
                    self.append_log(data)
                    
                elif msg_type == "IMAGE":
                    self.last_frame = data.copy()
                    rgb = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(rgb)
                    
                    w, h = self.video_label.winfo_width(), self.video_label.winfo_height()
                    if w > 10 and h > 10:
                        img_w, img_h = pil_img.size
                        scale = min(w / img_w, h / img_h)
                        new_w, new_h = max(1, int(img_w * scale)), max(1, int(img_h * scale))
                        pil_img = pil_img.resize((new_w, new_h), Image.LANCZOS)
                        
                    tk_img = ImageTk.PhotoImage(image=pil_img)
                    self.video_label.configure(image=tk_img)
                    self.video_label.image = tk_img 
                    
                elif msg_type == "DATA":
                    res, tcp = data.get("res"), data.get("tcp")
                    if res:
                        t_c, e_c = res["H_box_in_cam"][:3, 3], H_to_euler_deg(res["H_box_in_cam"])
                        self.lbl_cam_pose.configure(text=f"CAMERA POSE:\nt=[{t_c[0]:+.3f}, {t_c[1]:+.3f}, {t_c[2]:+.3f}]m\nRPY=[{e_c[0]:+5.1f}, {e_c[1]:+5.1f}, {e_c[2]:+5.1f}]°")
                        if res["H_box_in_base"] is not None:
                            t_b, e_b = res["H_box_in_base"][:3, 3], H_to_euler_deg(res["H_box_in_base"])
                            self.lbl_base_pose.configure(text=f"BASE POSE:\nt=[{t_b[0]:+.3f}, {t_b[1]:+.3f}, {t_b[2]:+.3f}]m\nRPY=[{e_b[0]:+5.1f}, {e_b[1]:+5.1f}, {e_b[2]:+5.1f}]°")
                        else:
                            self.lbl_base_pose.configure(text="BASE POSE:\n(No joint angle data)")
                    else:
                        self.lbl_cam_pose.configure(text="CAMERA POSE:\nNo detection")
                        self.lbl_base_pose.configure(text="BASE POSE:\nNo detection")

                    if tcp:
                        H_tcp = rvec_tvec_to_H(np.array(tcp[3:], np.float64), tcp[:3])
                        e_t = H_to_euler_deg(H_tcp)
                        self.lbl_tcp_pose.configure(text=f"TCP POSE:\nt=[{tcp[0]:+.3f}, {tcp[1]:+.3f}, {tcp[2]:+.3f}]m\nRPY=[{e_t[0]:+5.1f}, {e_t[1]:+5.1f}, {e_t[2]:+5.1f}]°")
                    else:
                        self.lbl_tcp_pose.configure(text="TCP POSE:\nDisconnected")

                elif msg_type == "CLEAR_GRAPH":
                    self.error_x.clear()
                    self.error_y.clear()
                    self.line.set_data(self.error_x, self.error_y)
                    self.ax.relim()
                    self.ax.autoscale_view()
                    self.graph_canvas.draw_idle()

                elif msg_type == "PLOT":
                    t, err = data
                    self.error_x.append(t)
                    self.error_y.append(err)
                    self.line.set_data(self.error_x, self.error_y)
                    self.ax.relim()
                    self.ax.autoscale_view()
                    self.graph_canvas.draw_idle()

            except queue.Empty:
                break
                
        self.after(15, self.update_gui_from_queue)

    def on_closing(self):
        self.worker._run_flag = False
        self.worker.join(timeout=2.0)
        self.destroy()

def main():
    pa = argparse.ArgumentParser()
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
    
    seg_grp = pa.add_argument_group("Semantic segmentation")
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

    pallet_cfg = None
    if HAS_PALLET_DETECTOR:
        pallet_cfg = ConfigDetector()
        try:
            l, c = args.pallet_grid.lower().split('x')
            pallet_cfg.linhas, pallet_cfg.colunas = int(l), int(c)
        except Exception:
            pass

    if not os.path.exists(args.calib):
        sys.exit(f"[ERROR] Calibration file not found: {args.calib}")
    with open(args.calib) as f: cal = json.load(f)

    K = np.array(cal["camera_matrix"], np.float64)
    dist = np.array(cal["dist_coeffs"], np.float64)
    T_cam_mount = np.array(cal["T_cam2tcp"], np.float64)

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

    app = App(args, estimator, robot, pallet_cfg)
    app.mainloop()

if __name__ == "__main__":
    main()