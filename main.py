import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance as dist
from collections import deque
import winsound
import threading
import time
from twilio.rest import Client

TWILIO_SID   = "AC548cd89da9c0c7dc241bdd88fec4e702"
TWILIO_TOKEN = "a5bb6fe2e83b83f473865687be2ed888"
TWILIO_FROM  = "+14155238886"    
ALERT_TO     = "+918178944939"   

EAR_THRESHOLD  = 0.20   
CONSEC_FRAMES  = 15     
NOD_THRESHOLD  = 20     
NOD_FRAMES     = 25   

WHATSAPP_COOLDOWN = 60
BEEP_COOLDOWN     = 3

LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33,  160, 158, 133, 153, 144]

MODEL_POINTS = np.array([
    (0.0,    0.0,    0.0),     
    (0.0,   -330.0, -65.0),    
    (-225.0, 170.0, -135.0), 
    (225.0,  170.0, -135.0),   
    (-150.0,-150.0, -125.0),   
    (150.0, -150.0, -125.0),   
], dtype=np.float64)

POSE_LM_IDS = [1, 152, 263, 33, 287, 57]

GREEN  = (0, 220, 90)
YELLOW = (0, 200, 255)
RED    = (0, 0, 255)
CYAN   = (255, 220, 0)
WHITE  = (255, 255, 255)
GRAY   = (160, 160, 160)
BLACK  = (0, 0, 0)


try:
    twilio_client = Client(TWILIO_SID, TWILIO_TOKEN)
    TWILIO_READY = True
except Exception as e:
    print(f"[TWILIO] Not configured: {e}")
    TWILIO_READY = False


# ─── EAR Formula ──────────────────────────────────────────────────────────────
def eye_aspect_ratio(landmarks, indices, w, h):
    pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in indices]
    A = dist.euclidean(pts[1], pts[5])
    B = dist.euclidean(pts[2], pts[4])
    C = dist.euclidean(pts[0], pts[3])
    return (A + B) / (2.0 * C)



def get_head_pose(landmarks, w, h):
    """
    Returns (pitch, yaw, roll) in degrees.
    Pitch > 0  = head tilting DOWN  (nodding / falling asleep)
    Pitch < 0  = head tilting UP
    """
    image_points = np.array([
        (landmarks[i].x * w, landmarks[i].y * h) for i in POSE_LM_IDS
    ], dtype=np.float64)

    focal_length = w
    cam_matrix   = np.array([
        [focal_length, 0,            w / 2],
        [0,            focal_length, h / 2],
        [0,            0,            1    ]
    ], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1))

    success, rot_vec, trans_vec = cv2.solvePnP(
        MODEL_POINTS, image_points, cam_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not success:
        return 0.0, 0.0, 0.0

    rot_mat, _ = cv2.Rodrigues(rot_vec)
    # Decompose into Euler angles
    sy = np.sqrt(rot_mat[0,0]**2 + rot_mat[1,0]**2)
    singular = sy < 1e-6
    if not singular:
        pitch = np.degrees(np.arctan2( rot_mat[2,1], rot_mat[2,2]))
        yaw   = np.degrees(np.arctan2(-rot_mat[2,0], sy))
        roll  = np.degrees(np.arctan2( rot_mat[1,0], rot_mat[0,0]))
    else:
        pitch = np.degrees(np.arctan2(-rot_mat[1,2], rot_mat[1,1]))
        yaw   = np.degrees(np.arctan2(-rot_mat[2,0], sy))
        roll  = 0.0

    return pitch, yaw, roll


def draw_pose_axes(frame, landmarks, w, h, rot_vec, trans_vec):
    """Draw XYZ axes on the nose tip to visualise head orientation."""
    focal_length = w
    cam_matrix   = np.array([
        [focal_length, 0,            w / 2],
        [0,            focal_length, h / 2],
        [0,            0,            1    ]
    ], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1))

    nose_tip = (int(landmarks[1].x * w), int(landmarks[1].y * h))
    axis_len = 80.0
    axis_pts = np.float32([
        [axis_len, 0, 0], [0, axis_len, 0], [0, 0, axis_len]
    ])
    projected, _ = cv2.projectPoints(axis_pts, rot_vec, trans_vec, cam_matrix, dist_coeffs)
    p = lambda i: tuple(projected[i].ravel().astype(int))
    cv2.line(frame, nose_tip, p(0), (0, 0, 255),   2)   # X red
    cv2.line(frame, nose_tip, p(1), (0, 255, 0),   2)   # Y green
    cv2.line(frame, nose_tip, p(2), (255, 0, 0),   2)   # Z blue



def play_beep():
    def _beep():
        for _ in range(3):
            winsound.Beep(1000, 300)
            time.sleep(0.1)
    threading.Thread(target=_beep, daemon=True).start()



def send_alert(ear_value, alert_num, reason="DROWSINESS"):
    if not TWILIO_READY:
        print("[ALERT] Twilio not configured.")
        return

    def _send():
        try:
            msg = (
                f"ALERT #{alert_num} — {reason}\n"
                f"Driver may be falling asleep!\n"
                f"EAR: {ear_value:.3f}\n"
                f"Time: {time.strftime('%H:%M:%S')}\n"
                f"Please check on the driver immediately."
            )
            twilio_client.messages.create(
                body=msg,
                from_=f"whatsapp:{TWILIO_FROM}",
                to=f"whatsapp:{ALERT_TO}"
            )
            print(f"[WHATSAPP] {reason} alert #{alert_num} sent.")
        except Exception as e:
            print(f"[WHATSAPP ERROR] {e}")

    threading.Thread(target=_send, daemon=True).start()



def draw_eye(frame, landmarks, indices, w, h, color):
    pts = np.array(
        [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in indices],
        dtype=np.int32
    )
    hull = cv2.convexHull(pts)
    cv2.drawContours(frame, [hull], -1, color, 1)
    for (x, y) in pts:
        cv2.circle(frame, (x, y), 2, color, -1)



def draw_ear_bar(frame, ear, threshold, x=10, y=110, bar_w=18, bar_h=140):
    cv2.rectangle(frame, (x, y), (x + bar_w, y + bar_h), (40, 40, 40), -1)
    cv2.rectangle(frame, (x, y), (x + bar_w, y + bar_h), GRAY, 1)
    level = int(min(ear / 0.5, 1.0) * bar_h)
    color = GREEN if ear >= threshold else (YELLOW if ear >= threshold * 0.85 else RED)
    cv2.rectangle(frame, (x, y + bar_h - level), (x + bar_w, y + bar_h), color, -1)
    thresh_y = y + bar_h - int(min(threshold / 0.5, 1.0) * bar_h)
    cv2.line(frame, (x - 4, thresh_y), (x + bar_w + 4, thresh_y), YELLOW, 1)
    cv2.putText(frame, "EAR",     (x - 1, y - 8),         cv2.FONT_HERSHEY_SIMPLEX, 0.38, WHITE, 1)
    cv2.putText(frame, f"{ear:.2f}", (x - 2, y + bar_h + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.36, color, 1)



GRAPH_W, GRAPH_H = 500, 200
GRAPH_LEN = 150   # number of frames to show

def make_ear_graph(ear_history, threshold, nod_history):
    """
    Renders a 500×200 BGR image with:
      - Scrolling EAR line (green/red)
      - Threshold dashed line (yellow)
      - Nod events marked as cyan dots at the top
    """
    canvas = np.zeros((GRAPH_H, GRAPH_W, 3), dtype=np.uint8)
    canvas[:] = (18, 18, 18)

    
    for i in range(1, 5):
        y = int(GRAPH_H * i / 5)
        cv2.line(canvas, (0, y), (GRAPH_W, y), (40, 40, 40), 1)
        val = 0.5 * (5 - i) / 5
        cv2.putText(canvas, f"{val:.2f}", (4, y - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (80, 80, 80), 1)

    
    cv2.putText(canvas, "EAR", (GRAPH_W - 34, 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (120, 120, 120), 1)

    
    thresh_y = int(GRAPH_H - (threshold / 0.5) * GRAPH_H)
    thresh_y = max(2, min(GRAPH_H - 2, thresh_y))
    for x in range(0, GRAPH_W, 10):
        cv2.line(canvas, (x, thresh_y), (min(x + 6, GRAPH_W), thresh_y), (0, 200, 255), 1)
    cv2.putText(canvas, f"thresh {threshold:.2f}", (4, thresh_y - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 200, 255), 1)

    
    pts = list(ear_history)
    n   = len(pts)
    if n >= 2:
        step = GRAPH_W / GRAPH_LEN
        for i in range(1, n):
            v0, v1 = pts[i - 1], pts[i]
            if v0 is None or v1 is None:
                continue
            x0 = int((i - 1) * step)
            x1 = int(i * step)
            y0 = int(GRAPH_H - min(v0 / 0.5, 1.0) * GRAPH_H)
            y1 = int(GRAPH_H - min(v1 / 0.5, 1.0) * GRAPH_H)
            y0 = max(1, min(GRAPH_H - 1, y0))
            y1 = max(1, min(GRAPH_H - 1, y1))
            col = RED if v1 < threshold else GREEN
            cv2.line(canvas, (x0, y0), (x1, y1), col, 2)

    
    nod_pts = list(nod_history)
    for i, nodding in enumerate(nod_pts):
        if nodding:
            x = int(i * GRAPH_W / GRAPH_LEN)
            cv2.circle(canvas, (x, 10), 4, CYAN, -1)

    
    cv2.circle(canvas, (10, GRAPH_H - 12), 5, GREEN, -1)
    cv2.putText(canvas, "EAR", (20, GRAPH_H - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.32, GREEN, 1)
    cv2.circle(canvas, (55, GRAPH_H - 12), 5, CYAN, -1)
    cv2.putText(canvas, "Nod", (65, GRAPH_H - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.32, CYAN, 1)

    
    cv2.putText(canvas, "LIVE EAR GRAPH", (GRAPH_W // 2 - 60, 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 160, 160), 1)

    return canvas



def draw_hud(frame, ear, pitch, consec, nod_consec, threshold, max_frames,
             alert_count, total_frames, drowsy_frames, fps, state):
    h, w = frame.shape[:2]
    drowsy_pct = (drowsy_frames / max(total_frames, 1)) * 100


    banner_col = (RED         if state == "ALERT"   else
                  CYAN        if state == "NODDING" else
                  (30,100,200)if state == "DROWSY"  else
                  (30, 30, 30))
    cv2.rectangle(frame, (0, 0), (w, 40), banner_col, -1)

    status_text = {
        "AWAKE":   "  MONITORING",
        "DROWSY":  "  DROWSINESS DETECTED",
        "ALERT":   "  WAKE UP !!",
        "NODDING": "  HEAD NODDING — WAKE UP",
        "NO FACE": "  NO FACE DETECTED",
    }.get(state, state)
    cv2.putText(frame, status_text, (8, 28), cv2.FONT_HERSHEY_DUPLEX, 0.75, BLACK if state == "NODDING" else WHITE, 2)
    cv2.putText(frame, f"FPS {fps:.0f}", (w - 75, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)

    
    panel_y = h - 80
    cv2.rectangle(frame, (0, panel_y), (w, h), (20, 20, 20), -1)
    cv2.line(frame, (0, panel_y), (w, panel_y), (70, 70, 70), 1)

    ear_col = GREEN if ear >= threshold else (YELLOW if ear >= threshold * 0.85 else RED)
    cv2.putText(frame, f"EAR: {ear:.3f}",              (10, panel_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, ear_col, 2)
    cv2.putText(frame, f"EAR frames: {consec}/{max_frames}", (10, panel_y + 42), cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE, 1)

    pitch_col = CYAN if pitch > NOD_THRESHOLD else WHITE
    cv2.putText(frame, f"Pitch: {pitch:+.1f} deg",     (10, panel_y + 62), cv2.FONT_HERSHEY_SIMPLEX, 0.45, pitch_col, 1)

    cv2.putText(frame, f"Alerts: {alert_count}",        (w//2 - 50, panel_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, RED if alert_count else WHITE, 2)
    cv2.putText(frame, f"Drowsy: {drowsy_pct:.1f}%",    (w//2 - 50, panel_y + 42), cv2.FONT_HERSHEY_SIMPLEX, 0.45, YELLOW, 1)
    cv2.putText(frame, f"Nod frames: {nod_consec}/{NOD_FRAMES}", (w//2 - 50, panel_y + 62), cv2.FONT_HERSHEY_SIMPLEX, 0.45, CYAN, 1)

    cv2.putText(frame, f"Thresh: {threshold:.2f}",      (w - 145, panel_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE, 1)
    cv2.putText(frame, "+/- thresh  [/] frames",        (w - 185, panel_y + 42), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (120,120,120), 1)
    cv2.putText(frame, "T=test alert  Q=quit",          (w - 175, panel_y + 62), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (120,120,120), 1)


def main():
    global EAR_THRESHOLD, CONSEC_FRAMES

    print("[INFO] Starting drowsiness detector...")
    print("[INFO] Controls: Q=quit  +/-=threshold  [/]=frames  T=test alert")

    mp_face_mesh = mp.solutions.face_mesh

    consec          = 0
    nod_consec      = 0
    alert_count     = 0
    total_frames    = 0
    drowsy_frames   = 0
    last_beep_t     = 0
    last_whatsapp_t = 0
    prev_time       = time.time()

    
    ear_history = deque([None] * GRAPH_LEN, maxlen=GRAPH_LEN)
    nod_history = deque([False] * GRAPH_LEN, maxlen=GRAPH_LEN)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        return

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:

        while True:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] No frame from camera.")
                break

            frame = cv2.resize(frame, (720, 540))
            h, w  = frame.shape[:2]
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = face_mesh.process(rgb)
            rgb.flags.writeable = True

            total_frames += 1
            ear   = 0.0
            pitch = 0.0
            state = "AWAKE"
            nodding = False

            if results.multi_face_landmarks:
                lm = results.multi_face_landmarks[0].landmark

                
                left_ear  = eye_aspect_ratio(lm, LEFT_EYE,  w, h)
                right_ear = eye_aspect_ratio(lm, RIGHT_EYE, w, h)
                ear       = (left_ear + right_ear) / 2.0

                eye_color = GREEN if ear >= EAR_THRESHOLD else (YELLOW if ear >= EAR_THRESHOLD * 0.85 else RED)
                draw_eye(frame, lm, LEFT_EYE,  w, h, eye_color)
                draw_eye(frame, lm, RIGHT_EYE, w, h, eye_color)

                
                pitch, yaw, roll = get_head_pose(lm, w, h)

                
                image_points = np.array(
                    [(lm[i].x * w, lm[i].y * h) for i in POSE_LM_IDS], dtype=np.float64
                )
                focal_length = w
                cam_matrix   = np.array([
                    [focal_length, 0, w / 2],
                    [0, focal_length, h / 2],
                    [0, 0, 1]
                ], dtype=np.float64)
                dist_coeffs  = np.zeros((4, 1))
                ok, rot_vec, trans_vec = cv2.solvePnP(
                    MODEL_POINTS, image_points, cam_matrix, dist_coeffs,
                    flags=cv2.SOLVEPNP_ITERATIVE
                )
                if ok:
                    draw_pose_axes(frame, lm, w, h, rot_vec, trans_vec)

                
                if pitch > NOD_THRESHOLD:
                    nod_consec += 1
                    nodding     = True
                else:
                    nod_consec  = 0

                
                now = time.time()

                if ear < EAR_THRESHOLD:
                    consec        += 1
                    drowsy_frames += 1
                    state = "DROWSY"

                    if consec >= CONSEC_FRAMES:
                        state = "ALERT"
                        if now - last_beep_t > BEEP_COOLDOWN:
                            play_beep()
                            last_beep_t = now
                        if now - last_whatsapp_t > WHATSAPP_COOLDOWN:
                            alert_count       += 1
                            last_whatsapp_t    = now
                            send_alert(ear, alert_count, "EAR DROWSINESS")

                        overlay = frame.copy()
                        cv2.rectangle(overlay, (0, 0), (w, h), RED, -1)
                        cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)
                        cv2.putText(frame, "!! WAKE UP !!", (w//2 - 120, h//2),
                                    cv2.FONT_HERSHEY_DUPLEX, 1.2, RED, 3)
                else:
                    consec = 0
                    state  = "AWAKE"

                
                if nod_consec >= NOD_FRAMES and state != "ALERT":
                    state = "NODDING"
                    if now - last_beep_t > BEEP_COOLDOWN:
                        play_beep()
                        last_beep_t = now
                    if now - last_whatsapp_t > WHATSAPP_COOLDOWN:
                        alert_count       += 1
                        last_whatsapp_t    = now
                        send_alert(ear, alert_count, "HEAD NODDING")

                    overlay = frame.copy()
                    cv2.rectangle(overlay, (0, 0), (w, h), (255, 200, 0), -1)
                    cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
                    cv2.putText(frame, "!! HEAD NODDING !!", (w//2 - 150, h//2),
                                cv2.FONT_HERSHEY_DUPLEX, 1.0, CYAN, 3)

            else:
                state     = "NO FACE"
                consec    = 0
                nod_consec = 0

        
            ear_history.append(ear if results.multi_face_landmarks else None)
            nod_history.append(nodding)

    
            now_t     = time.time()
            fps       = 1.0 / max(now_t - prev_time, 1e-9)
            prev_time = now_t

            
            draw_ear_bar(frame, ear, EAR_THRESHOLD)
            draw_hud(frame, ear, pitch, consec, nod_consec, EAR_THRESHOLD,
                     CONSEC_FRAMES, alert_count, total_frames, drowsy_frames, fps, state)

            cv2.imshow("Drowsiness Detector", frame)

            
            graph = make_ear_graph(ear_history, EAR_THRESHOLD, nod_history)
            cv2.imshow("EAR Live Graph", graph)
            cv2.moveWindow("EAR Live Graph", 740, 100)   # position next to main window

            
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            elif key in (ord('+'), ord('=')):
                EAR_THRESHOLD = min(0.40, round(EAR_THRESHOLD + 0.01, 2))
                print(f"[INFO] EAR Threshold -> {EAR_THRESHOLD:.2f}")
            elif key == ord('-'):
                EAR_THRESHOLD = max(0.10, round(EAR_THRESHOLD - 0.01, 2))
                print(f"[INFO] EAR Threshold -> {EAR_THRESHOLD:.2f}")
            elif key == ord(']'):
                CONSEC_FRAMES = min(60, CONSEC_FRAMES + 1)
                print(f"[INFO] Frame count -> {CONSEC_FRAMES}")
            elif key == ord('['):
                CONSEC_FRAMES = max(3, CONSEC_FRAMES - 1)
                print(f"[INFO] Frame count -> {CONSEC_FRAMES}")
            elif key in (ord('t'), ord('T')):
                print("[TEST] Sending test WhatsApp alert...")
                send_alert(0.18, 99, "TEST")

    
    drowsy_pct = (drowsy_frames / max(total_frames, 1)) * 100
    print("\n══ SESSION SUMMARY ══════════════════")
    print(f"  Total frames  : {total_frames}")
    print(f"  Drowsy frames : {drowsy_frames}  ({drowsy_pct:.1f}%)")
    print(f"  Alerts fired  : {alert_count}")
    print("═════════════════════════════════════\n")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()