import threading
import time
import cv2
import numpy as np
import csv
from datetime import datetime
from ultralytics import YOLO

from ucl.common import lib_version
from ucl.highCmd import highCmd
from ucl.unitreeConnection import unitreeConnection, HIGH_WIFI_DEFAULTS
from ucl.enums import MotorModeHigh, GaitType

# ================= CONFIG =================
front_cam_url = "http://192.168.137.4/stream"
target_label = "medicines"

model_path = r"C:\Users\pramod.na\OneDrive - Texas A&M University\Desktop\YoloV8\runs\detect\yolo_ver2_train\weights\best.pt"
#model_path = YOLO("yolov8n.pt")

Kp = 0.0007
min_speed = 0.05
max_speed = 0.111
command_dt = 0.002

# ================= CSV LOGGING =================
log_filename = f"log_{target_label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
log_file = open(log_filename, mode='w', newline='')
csv_writer = csv.writer(log_file)
csv_writer.writerow(["object", "time_sec", "ex", "ey", "vx", "vy"])

# ================= GLOBAL STATE =================
latest_frame = None
frame_lock = threading.Lock()

vx = 0.0
vy = 0.0
vel_lock = threading.Lock()

stop_event = threading.Event()

# ================= LOAD YOLO =================
print("[YOLO] Loading model...")
model = YOLO(model_path)
print("[YOLO] Model loaded.")

# ================= CAMERA THREAD =================
def camera_thread():
    global latest_frame

    cap = cv2.VideoCapture(front_cam_url)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print("Camera failed")
        stop_event.set()
        return

    print("[Camera] Opened successfully.")

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            continue

        with frame_lock:
            latest_frame = frame.copy()

    cap.release()

# ================= PERCEPTION THREAD =================
def perception_thread():
    global vx, vy

    start_time = time.time()

    while not stop_event.is_set():

        with frame_lock:
            if latest_frame is None:
                continue
            frame = latest_frame.copy()

        frame_h, frame_w = frame.shape[:2]
        frame_cx = frame_w // 2
        frame_bottom = frame_h - 1

        tolerance = int(0.05 * min(frame_w, frame_h))

        results = model.predict(frame, conf=0.3, verbose=False)

        new_vx = 0.0
        new_vy = 0.0

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                label = model.names[cls]

                if label != target_label:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                obj_cx = (x1 + x2) // 2
                obj_bottom = y2   # Correct bottom

                error_x = obj_cx - frame_cx
                error_y = obj_bottom - frame_bottom

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

                # Draw reference bottom center (red)
                cv2.circle(frame, (frame_cx, frame_bottom), 6, (0,0,255), -1)

                # Draw object bottom center (blue)
                cv2.circle(frame, (obj_cx, obj_bottom), 6, (255,0,0), -1)

                # Check alignment
                if abs(error_x) < tolerance and abs(error_y) < tolerance:
                    print("[Aligned] Within tolerance. Stopping.")
                    stop_event.set()

                if abs(error_x) > tolerance:
                    new_vy = -Kp * error_x

                if abs(error_y) > tolerance:
                    new_vx = -Kp * error_y

                new_vx = np.clip(new_vx, -max_speed, max_speed)
                new_vy = np.clip(new_vy, -max_speed, max_speed)

                if abs(error_y) > tolerance and 0 < abs(new_vx) < min_speed:
                    new_vx = min_speed * np.sign(new_vx)

                if abs(error_x) > tolerance and 0 < abs(new_vy) < min_speed:
                    new_vy = min_speed * np.sign(new_vy)

                # ================= LOG DATA =================
                elapsed = time.time() - start_time
                csv_writer.writerow([
                    target_label,
                    round(elapsed, 4),
                    round(error_x, 2),
                    round(error_y, 2),
                    round(new_vx, 4),
                    round(new_vy, 4)
                ])
                log_file.flush()

                break

        with vel_lock:
            vx = new_vx
            vy = new_vy

        cv2.imshow("Front Camera", frame)
        if cv2.waitKey(1) == ord('q'):
            stop_event.set()

# ================= CONTROL THREAD =================
def control_thread():
    global vx, vy

    print(f"[Robot] SDK lib version: {lib_version()}")

    conn = unitreeConnection(HIGH_WIFI_DEFAULTS)
    conn.startRecv()
    hcmd = highCmd()

    conn.send(hcmd.buildCmd(debug=False))
    time.sleep(0.5)

    # Stand up AFTER camera running
    print("[Robot] Standing up...")
    for _ in range(500):
        hcmd.mode = MotorModeHigh.STAND_UP
        conn.send(hcmd.buildCmd(debug=False))
        time.sleep(command_dt)

    while not stop_event.is_set():
        with vel_lock:
            send_vx = vx
            send_vy = vy

        hcmd.mode = MotorModeHigh.VEL_WALK
        hcmd.gaitType = GaitType.TROT
        hcmd.velocity = [send_vx, send_vy]
        hcmd.yawSpeed = 0.0
        hcmd.footRaiseHeight = 0.05

        conn.send(hcmd.buildCmd(debug=False))
        time.sleep(command_dt)

    # Safe stop
    for _ in range(200):
        hcmd.velocity = [0.0, 0.0]
        conn.send(hcmd.buildCmd(debug=False))
        time.sleep(command_dt)

    conn.stopRecv()
    print("[Robot] Stopped.")

# ================= MAIN =================
if __name__ == "__main__":
    try:
        t_cam = threading.Thread(target=camera_thread)
        t_perc = threading.Thread(target=perception_thread)
        t_ctrl = threading.Thread(target=control_thread)

        t_cam.start()
        t_perc.start()
        t_ctrl.start()

        t_cam.join()
        t_perc.join()
        t_ctrl.join()

    except KeyboardInterrupt:
        stop_event.set()

    finally:
        log_file.close()
        cv2.destroyAllWindows()
        print(f"CSV saved as: {log_filename}")
        print("Program ended.")