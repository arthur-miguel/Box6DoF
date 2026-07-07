"""
0_captura_metadata.py
=====================
Coleta de imagens e metadados (pose TCP e ângulos de junta) para a
calibração hand-eye (eye-in-hand), com diversidade rotacional e
translacional dos waypoints.

Uso:
    python 0_captura_metadata.py
"""

import os
import json
import time
import random
import math
from datetime import datetime

import cv2
import numpy as np

from rtde_receive import RTDEReceiveInterface
from rtde_control import RTDEControlInterface

# CONFIG — ajuste aqui antes de rodar
IP_DO_ROBO    = "192.168.0.10"
CAMERA_INDEX  = 1          # se falhar, o script testa 0-4 automaticamente
USE_UNDISTORT = False

VEL   = 0.8    # rad/s
ACEL  = 0.25   # rad/s²

DELTA_Q4_MAX_GRAUS = 120.0   # limite extra para Pulso 2
Q4_LIMITE_SEGURO   = 150.0   

# Grade esférica
THETAS_GRAUS = [35, 45, 55, 62]

PHIS_POR_THETA = {
    45: 32,
    55: 28,
    62: 16,
    67: 16,
    72: 8,
}

RAIO_MIN_M = 0.3
RAIO_MAX_M = 0.35

# ── Giro de punho ───────────────────────────────────────────
NUMERO_ROLLS   = 1
ROLL_MAX_GRAUS = 40.0

# ── Salvaguardas ────────────────────────────────────────────
MARGEM_COLISAO_M      = 0.12   # Z mínimo acima do tabuleiro
DELTA_Q_MAX_GRAUS     = 160.0  # se qualquer junta mudaria mais que isso → pula
IK_NORM_MAX           = 100.0  # norma máxima aceitável da solução IK (rad)
TIMEOUT_MOVIMENTO_S   = 150.0   # watchdog: máx. segundos para o robô parar
TIMEOUT_ROLL_S        = 8.0    # watchdog para movimentos de punho
PAUSA_ESTABILIZACAO_S = 0.35   # pausa após chegada antes da captura
PAUSA_ROLL_S          = 0.30   # pausa após giro de punho

# ── Saída ───────────────────────────────────────────────────
BASE_DIR         = os.path.dirname(os.path.abspath(__file__))
CONFIG_POSE_FILE = os.path.join(BASE_DIR, "ponto_inicial_handeye.json")
CALIBRATION_FILE = os.path.join(BASE_DIR, "..", "calibration.npz")

def pre_validar_waypoints(rtde_c, P_centro, thetas_graus, phis_por_theta,
                           raio_min, raio_max, margem_z,
                           delta_q_max_rad, ik_norm_max,
                           q_seed, max_tentativas=12):
    """
    Para cada posição esférica desejada, tenta até max_tentativas
    candidatos aleatórios dentro dos mesmos parâmetros angulares até
    encontrar um que passe em TODAS as verificações sem mover o robô.
    Retorna lista de (delta, ik_solution) prontos para execução.
    """
    q_home = [0, -1.5708, 1.5708, -1.5708, -1.5708, 0.0]
    validados = []
    total_desejado = sum(phis_por_theta.get(t, 8) for t in thetas_graus)

    print(f"\n[PRÉ-VALIDAÇÃO] Verificando {total_desejado} posições sem mover o robô...")

    try:
        rtde_c.reuploadScript()
        time.sleep(1.5)
    except Exception as e:
        print(f"[AVISO] reuploadScript falhou: {e}")

    for theta_deg in thetas_graus:
        theta  = np.radians(theta_deg)
        n_phi  = phis_por_theta.get(theta_deg, 8)
        phi_vals = np.linspace(0, 2*np.pi, n_phi, endpoint=False)
        phi_offset = random.uniform(0, 2*np.pi / max(n_phi, 1))

        for phi_base in phi_vals + phi_offset:
            aceito = False

            for tentativa in range(max_tentativas):
                # Varia levemente o azimute e o raio a cada tentativa
                phi     = phi_base + random.uniform(-0.15, 0.15)
                r_horiz = random.uniform(raio_min, raio_max)

                dz = -r_horiz / np.tan(theta)
                dx =  r_horiz * np.cos(phi)
                dy =  r_horiz * np.sin(phi)
                if abs(dz) < margem_z:
                    dz = -margem_z

                drx =  theta * np.sin(phi)
                dry = -theta * np.cos(phi)
                delta = [dx, dy, dz, drx, dry, 0.0]

                try:
                    P_base = rtde_c.poseTrans(P_centro, delta)
                    time.sleep(0.02)
                except RuntimeError:
                    time.sleep(0.1)
                    try:
                        rtde_c.reuploadScript()
                        time.sleep(1.0)
                    except Exception:
                        pass
                    continue
                P_base = rtde_c.poseTrans(P_centro, delta)
                time.sleep(0.2) 
                # 1. Limites de segurança da pose
                if not rtde_c.isPoseWithinSafetyLimits(P_base):
                    continue

                # 2. IK com seed real, fallback para home
                ik = rtde_c.getInverseKinematics(P_base, qnear=q_seed)
                if not ik or len(ik) != 6:
                    ik = rtde_c.getInverseKinematics(P_base, qnear=q_home)

                ok, motivo = ik_valida(ik, q_seed, delta_q_max_rad, ik_norm_max)
                if not ok:
                    continue

                # 3. Verifica cada roll também
                rolls = gerar_rolls(NUMERO_ROLLS, ROLL_MAX_GRAUS)
                rolls_ok = []
                for roll in rolls:
                    P_roll = rtde_c.poseTrans(P_base, [0,0,0,0,0,roll])
                    if not rtde_c.isPoseWithinSafetyLimits(P_roll):
                        continue
                    ik_r = rtde_c.getInverseKinematics(P_roll, qnear=ik)
                    ok_r, _ = ik_valida(ik_r, ik, delta_q_max_rad, ik_norm_max)
                    if ok_r:
                        rolls_ok.append((roll, ik_r))

                if not rolls_ok:
                    continue

                # Passou em tudo
                validados.append({
                    "delta":     delta,
                    "ik_base":   ik,
                    "rolls":     rolls_ok,   # lista de (roll_rad, ik_roll)
                    "theta_deg": theta_deg,
                    "phi":       phi,
                })
                aceito = True
                break

            if not aceito:
                print(f"  [AVISO] theta={theta_deg}°  phi={np.degrees(phi_base):.0f}°"
                      f" — descartado após {max_tentativas} tentativas.")

    random.shuffle(validados)
    print(f"[PRÉ-VALIDAÇÃO] {len(validados)}/{total_desejado} posições aprovadas.\n")
    return validados


def ik_valida(ik_solution, q_ref, delta_q_max_rad, ik_norm_max):
    """
    Retorna (True, motivo) se a solução IK é utilizável, ou
    (False, motivo) indicando o problema encontrado.

    Verificações:
    - len == 6
    - nenhum valor NaN ou Inf
    - não é vetor de zeros (solver falhou silenciosamente)
    - norma total dentro de ik_norm_max
    - nenhuma junta muda mais que delta_q_max_rad em relação a q_ref
    """
    if len(ik_solution) != 6:
        return False, f"len={len(ik_solution)} (esperado 6)"

    arr = np.array(ik_solution, dtype=float)

    if not np.all(np.isfinite(arr)):
        return False, "contém NaN ou Inf"

    if np.all(arr == 0.0):
        return False, "solução toda-zero (IK falhou silenciosamente)"

    norma = np.linalg.norm(arr)
    if norma > ik_norm_max:
        return False, f"norma={norma:.1f} > {ik_norm_max}"

    delta = np.abs(arr - np.array(q_ref, dtype=float))
    idx_max = int(np.argmax(delta))
    if delta[idx_max] > delta_q_max_rad:
        return False, (f"junta J{idx_max+1} mudaria "
                       f"{math.degrees(delta[idx_max]):.1f}° "
                       f"(máx {math.degrees(delta_q_max_rad):.1f}°)")
    
    # Limites absolutos por junta (radianos) com margem de segurança de 10°
    MARGEM_RAD = math.radians(10.0)
    LIMITES_ABSOLUTOS = [
        math.radians(360.0) - MARGEM_RAD,   # J1 base
        math.radians(360.0) - MARGEM_RAD,   # J2 ombro
        math.radians(360.0) - MARGEM_RAD,   # J3 cotovelo
        math.radians(360.0) - MARGEM_RAD,   # J4 pulso 1
        math.radians(360.0) - MARGEM_RAD,   # J5 pulso 2
        math.radians(360.0) - MARGEM_RAD,   # J6 pulso 3
    ]
    for i, (val, lim) in enumerate(zip(arr, LIMITES_ABSOLUTOS)):
        if abs(val) > lim:
            return False, f"J{i+1} em {math.degrees(val):.1f}° ultrapassa limite absoluto"

    # Verificação extra de Pulso 2 (J5) com margem ainda maior
    if abs(arr[4]) > math.radians(Q4_LIMITE_SEGURO):
        return False, f"Pulso2={math.degrees(arr[4]):.1f}° próximo do limite absoluto"

    delta_q4 = abs(arr[4] - np.array(q_ref, dtype=float)[4])
    if delta_q4 > math.radians(DELTA_Q4_MAX_GRAUS):
        return False, f"Pulso2 mudaria {math.degrees(delta_q4):.1f}° (máx {DELTA_Q4_MAX_GRAUS}°)"

    return True, "ok"

def aguardar_robo_parar(rtde_c, rtde_r, cap,
                         cam_mat, dist_coeffs, use_und,
                         timeout_s, texto_ui, frame_overlay_fn):
    """
    Aguarda rtde_c.isSteady() == True com timeout.
    Exibe preview enquanto espera.
    Se timeout estourar, chama stopJ e retorna False.
    Retorna True se parou normalmente.
    """
    t0 = time.time()
    while True:
        if rtde_c.isSteady():
            return True

        elapsed = time.time() - t0
        if elapsed > timeout_s:
            print(f"  [WATCHDOG] Timeout de {timeout_s:.0f}s atingido — "
                  "parando robô.")
            try:
                rtde_c.stopJ(2.0)
            except Exception as e:
                print(f"  [WATCHDOG] stopJ falhou: {e}")
            return False

        if cap is not None and cap.isOpened():
            ret, frame = cap.read()
            if ret and frame_overlay_fn is not None:
                disp = frame_overlay_fn(
                    frame, cam_mat, dist_coeffs, use_und,
                    texto_ui,
                    f"Aguardando... {elapsed:.1f}s / {timeout_s:.0f}s",
                )
                cv2.imshow("Interface", disp)

        if cv2.waitKey(1) & 0xFF == 27:
            raise KeyboardInterrupt

        time.sleep(0.01)

def gerar_waypoints(P_centro, thetas_graus, phis_por_theta,
                    raio_min, raio_max, margem_z):
    waypoints = []
    for theta_deg in thetas_graus:
        theta  = np.radians(theta_deg)
        n_phi  = phis_por_theta.get(theta_deg, 8)
        phi_vals = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
        phi_offset = random.uniform(0, 2 * np.pi / max(n_phi, 1))

        for phi in phi_vals + phi_offset:
            r_horiz = random.uniform(raio_min, raio_max)

            dz = -r_horiz / np.tan(theta)
            dx =  r_horiz * np.cos(phi)
            dy =  r_horiz * np.sin(phi)

            if abs(dz) < margem_z:
                dz = -margem_z

            drx =  theta * np.sin(phi)
            dry = -theta * np.cos(phi)

            waypoints.append([dx, dy, dz, drx, dry, 0.0])

    random.shuffle(waypoints)
    return waypoints


def gerar_rolls(numero_rolls, roll_max_graus):
    if numero_rolls <= 1:
        return [0.0]
    return list(np.linspace(
        -np.radians(roll_max_graus),
         np.radians(roll_max_graus),
        numero_rolls
    ))

def abrir_camera(camera_index):
    """
    Tenta abrir a câmera no índice solicitado.
    Se falhar, varre 0..4 e usa o primeiro que funcionar.
    Retorna (cap, indice_usado) ou (None, -1).
    """
    indices_tentar = [camera_index] + [i for i in range(5) if i != camera_index]
    for idx in indices_tentar:
        cap = cv2.VideoCapture(idx)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        time.sleep(0.4)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                print(f"[CÂMERA] Aberta no índice {idx}.")
                return cap, idx
        cap.release()
    print("[CÂMERA] AVISO: nenhuma câmera encontrada. Continuando sem preview.")
    return None, -1


def frame_com_overlay(frame, cam_mat, dist_coeffs, use_undistort,
                       texto_linha1, texto_linha2="", cor_barra=(0, 0, 0)):
    if frame is None:
        return None
    out = cv2.undistort(frame, cam_mat, dist_coeffs) if use_undistort else frame.copy()
    h, w = out.shape[:2]
    cv2.rectangle(out, (0, 0), (w, 50), cor_barra, -1)
    cv2.putText(out, texto_linha1, (12, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
    if texto_linha2:
        cv2.putText(out, texto_linha2, (12, 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 200, 200), 1, cv2.LINE_AA)
    return out

def capturar_foto_estatica(cap, cam_mat, dist_coeffs, use_undistort,
                            raw_dir, und_dir, contador, label=""):
    if cap is None or not cap.isOpened():
        return None, None

    for _ in range(4):
        cap.read()
    ret, frame = cap.read()
    if not ret:
        return None, None

    und = cv2.undistort(frame, cam_mat, dist_coeffs) if use_undistort else frame.copy()

    fname = f"img_{contador:04d}{('_' + label) if label else ''}.png"
    cv2.imwrite(os.path.join(raw_dir, fname), frame)
    cv2.imwrite(os.path.join(und_dir, fname), und)
    return frame, und

def main():
    run_id     = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(BASE_DIR, "dados_handeye", run_id)
    raw_dir    = os.path.join(output_dir, "raw")
    und_dir    = os.path.join(output_dir, "undistorted")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(und_dir, exist_ok=True)

    metadata = []
    contador = 0

    use_und    = USE_UNDISTORT
    cam_mat    = np.eye(3, dtype=float)
    dist_coeffs = np.zeros(5,  dtype=float)
    if use_und:
        try:
            calib       = np.load(CALIBRATION_FILE)
            cam_mat     = calib["camMat"]
            dist_coeffs = calib["distCoeffs"]
            print("[OK] Calibração intrínseca carregada.")
        except Exception:
            print("[AVISO] Arquivo de calibração não encontrado — raw.")
            use_und = False

    print(f"[ROBÔ] Conectando em {IP_DO_ROBO}...")
    rtde_r = RTDEReceiveInterface(IP_DO_ROBO)
    rtde_c = RTDEControlInterface(IP_DO_ROBO)
    print("[OK] Interfaces RTDE sincronizadas.")

    cap, cam_idx_usado = abrir_camera(CAMERA_INDEX)

    total_posicoes = sum(PHIS_POR_THETA.get(t, 8) for t in THETAS_GRAUS)
    total_fotos    = total_posicoes * NUMERO_ROLLS
    rolls_graus    = [round(math.degrees(r), 1)
                      for r in gerar_rolls(NUMERO_ROLLS, ROLL_MAX_GRAUS)]
    delta_q_max_rad = math.radians(DELTA_Q_MAX_GRAUS)

    print("\n" + "=" * 60)
    print("  PLANO DE COLETA — CALIBRAÇÃO HAND-EYE")
    print("=" * 60)
    print(f"  Elevações (theta)       : {THETAS_GRAUS}°")
    print(f"  Posições na esfera      : {total_posicoes}")
    print(f"  Giros de punho/posição  : {NUMERO_ROLLS}  {rolls_graus}°")
    print(f"  TOTAL fotos previsto    : {total_fotos}")
    print(f"  Raio horizontal         : [{RAIO_MIN_M*100:.0f}, {RAIO_MAX_M*100:.0f}] cm")
    print(f"  Delta-junta máx aceito  : {DELTA_Q_MAX_GRAUS:.0f}°")
    print(f"  Watchdog movimento      : {TIMEOUT_MOVIMENTO_S:.0f}s")
    print(f"  Câmera                  : índice {cam_idx_usado}")
    print("=" * 60)

    # Alias para facilitar chamadas de overlay
    def overlay(frame, *args, **kwargs):
        return frame_com_overlay(frame, cam_mat, dist_coeffs, use_und,
                                  *args, **kwargs)

    try:
        # FASE 1: PONTO INICIAL
        P_centro = None

        if os.path.exists(CONFIG_POSE_FILE):
            print("\n[MEMÓRIA] Ponto inicial salvo detectado.")
            resp = input("Reutilizar? (s/n): ").strip().lower()
            if resp == "s":
                with open(CONFIG_POSE_FILE) as f:
                    P_centro = json.load(f)["P_centro"]
                print("[OK] Posição carregada da memória.")

        if P_centro is None:
            print("\n[FREEDRIVE] Posicione o TCP no CENTRO do tabuleiro.")
            rtde_c.reuploadScript()
            time.sleep(0.2)
            rtde_c.teachMode()
            print("[FREEDRIVE ATIVO] Pressione 'S' para salvar | ESC para abortar.")

            if cap is not None:
                cv2.namedWindow("Interface", cv2.WINDOW_AUTOSIZE)
            while True:
                if cap is not None and cap.isOpened():
                    ret, frame = cap.read()
                    if ret:
                        disp = overlay(
                            frame,
                            "1. CENTRALIZE NO TABULEIRO  2. ENCOSTE LEVEMENTE",
                            "Pressione S para salvar | ESC para abortar",
                        )
                        if disp is not None:
                            h_f, w_f = disp.shape[:2]
                            cv2.line(disp, (w_f//2, 0), (w_f//2, h_f), (0,255,0), 1)
                            cv2.line(disp, (0, h_f//2), (w_f, h_f//2), (0,255,255), 1)
                            cv2.imshow("Interface", disp)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("s"):
                    P_centro = rtde_r.getActualTCPPose()
                    rtde_c.endTeachMode()
                    with open(CONFIG_POSE_FILE, "w") as f:
                        json.dump({
                            "P_centro": P_centro,
                            "salvo_em": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        }, f, indent=4)
                    print(f"[OK] Ponto central salvo.")
                    time.sleep(0.8)
                    break
                elif key == 27:
                    rtde_c.endTeachMode()
                    return

        # FASE 2: GERAR WAYPOINTS
        q_atual = rtde_r.getActualQ()
        waypoints_validos = pre_validar_waypoints(
            rtde_c, P_centro,
            THETAS_GRAUS, PHIS_POR_THETA,
            RAIO_MIN_M, RAIO_MAX_M, MARGEM_COLISAO_M,
            delta_q_max_rad, IK_NORM_MAX,
            q_seed=q_atual, max_tentativas=20
        )

        print(f"\n[GERADO] {len(waypoints_validos)}.")
        print("[INÍCIO] Iniciando varredura esférica...\n")

        q_atual = rtde_r.getActualQ()

        pulados_timeout = 0

        # FASE 3: LOOP PRINCIPAL
        for idx_wp, wp in enumerate(waypoints_validos):

            delta       = wp["delta"]
            ik_solution = wp["ik_base"]
            rolls_rad   = wp["rolls"]   # lista de (roll_rad, ik_roll)
            prefixo     = f"  [WP {idx_wp+1:3d}/{len(waypoints_validos)}]"

            P_base = rtde_c.poseTrans(P_centro, delta)
            time.sleep(0.2) 

            rtde_c.moveJ(ik_solution, VEL, ACEL, asynchronous=True)

            chegou = aguardar_robo_parar(
                rtde_c, rtde_r, cap, cam_mat, dist_coeffs, use_und,
                TIMEOUT_MOVIMENTO_S,
                f"MOVENDO → posição {idx_wp+1}/{len(waypoints_validos)}  "
                f"| capturadas: {contador}  | ESC=parar",
                frame_com_overlay,
            )
            if not chegou:
                print(f"{prefixo} PULADO — timeout de movimento.")
                pulados_timeout += 1
                q_atual = rtde_r.getActualQ()
                continue

            q_atual = rtde_r.getActualQ()
            time.sleep(PAUSA_ESTABILIZACAO_S)

            for roll, ik_roll in rolls_rad:

                rtde_c.moveJ(ik_roll, VEL * 0.6, ACEL * 0.6, asynchronous=True)

                chegou_roll = aguardar_robo_parar(
                    rtde_c, rtde_r, cap, cam_mat, dist_coeffs, use_und,
                    TIMEOUT_ROLL_S,
                    f"Posição {idx_wp+1} | Punho {math.degrees(roll):+.0f}°",
                    frame_com_overlay,
                )
                if not chegou_roll:
                    print(f"    [ROLL {math.degrees(roll):+.0f}°] PULADO — timeout.")
                    pulados_timeout += 1
                    q_atual = rtde_r.getActualQ()
                    continue

                q_atual = rtde_r.getActualQ()
                time.sleep(PAUSA_ROLL_S)

                label = f"wp{idx_wp:03d}_roll{math.degrees(roll):+.0f}"
                raw_f, und_f = capturar_foto_estatica(
                    cap, cam_mat, dist_coeffs, use_und,
                    raw_dir, und_dir, contador, label=label
                )

                if raw_f is None:
                    print(f"    [CAPTURA] FALHA — frame descartado.")
                    continue

                pose_tcp    = rtde_r.getActualTCPPose()
                q_snapshot  = rtde_r.getActualQ()

                metadata.append({
                    "id":            contador,
                    "waypoint_idx":  idx_wp,
                    "theta_graus":   wp["theta_deg"],
                    "phi_graus":     float(math.degrees(wp["phi"])),
                    "roll_graus":    float(math.degrees(roll)),
                    "raio_horiz_m":  float(math.sqrt(delta[0]**2 + delta[1]**2)),
                    "pose_real_tcp": pose_tcp,
                    # ângulos de junta no instante da captura (necessários para
                    # a FK truncada em 2_handeye_calibration.py)
                    "joint_angles":  [float(v) for v in q_snapshot],
                })

                if cap is not None and cap.isOpened():
                    img_ok   = und_f if und_f is not None else raw_f
                    disp_ok  = frame_com_overlay(
                        img_ok, cam_mat, dist_coeffs, False,
                        f"  FOTO {contador+1} SALVA  | wp {idx_wp+1}  "
                        f"roll {math.degrees(roll):+.0f}°",
                        cor_barra=(0, 110, 0),
                    )
                    if disp_ok is not None:
                        cv2.imshow("Interface", disp_ok)
                    cv2.waitKey(1)

                contador += 1

            rtde_c.moveJ(wp["ik_base"], VEL * 0.7, ACEL * 0.7, asynchronous=True)
            aguardar_robo_parar(
                rtde_c, rtde_r, cap, cam_mat, dist_coeffs, use_und,
                TIMEOUT_ROLL_S,
                f"Retornando punho a roll=0...",
                frame_com_overlay,
            )
            q_atual = rtde_r.getActualQ()
        print(f"\n{'='*60}")
        print(f"[CONCLUÍDO] {contador} fotos capturadas.")
        print(f"  Pulados por timeout: {pulados_timeout}")
        print(f"{'='*60}")

    except KeyboardInterrupt:
        print("\n" + "="*60)
        print("[PARADA DE EMERGÊNCIA] Operador interrompeu.")
        print("="*60)
        try:
            rtde_c.stopJ(2.0)
            print("[ROBÔ] Braço imobilizado.")
        except Exception:
            pass

    finally:
        print("\n[FECHAMENTO] Liberando recursos...")
        cv2.destroyAllWindows()

        # Para o robô antes de desconectar
        try:
            if not rtde_c.isSteady():
                rtde_c.stopJ(2.0)
                time.sleep(0.5)
        except Exception:
            pass

        try:
            cap.release()
        except Exception:
            pass

        try:
            rtde_r.disconnect()
        except Exception:
            pass
        try:
            rtde_c.disconnect()
        except Exception:
            pass

        if metadata:
            meta_path = os.path.join(output_dir, "metadata.json")
            with open(meta_path, "w") as f:
                json.dump(metadata, f, indent=4)
            print(f"[DADOS] metadata.json — {len(metadata)} registros.")

            thetas = [e["theta_graus"]  for e in metadata]
            rolls  = [e["roll_graus"]   for e in metadata]
            raios  = [e["raio_horiz_m"] for e in metadata]
            print("\n── Diversidade da sessão ──")
            print(f"   Theta : min={min(thetas):.1f}°  max={max(thetas):.1f}°  "
                  f"std={float(np.std(thetas)):.1f}°")
            print(f"   Roll  : min={min(rolls):.1f}°   max={max(rolls):.1f}°   "
                  f"std={float(np.std(rolls)):.1f}°")
            print(f"   Raio  : min={min(raios)*100:.1f}cm  max={max(raios)*100:.1f}cm  "
                  f"std={float(np.std(raios))*100:.1f}cm")
            print(f"   Saída : {output_dir}")

        print("\nScript encerrado.\n")


if __name__ == "__main__":
    main()