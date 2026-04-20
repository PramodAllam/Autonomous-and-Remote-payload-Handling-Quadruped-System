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
down_cam_url = "http://192.168.137.171/stream"
target_label = "first_aid_box"

model_path = r"C:\Users\pramod.na\OneDrive - Texas A&M University\Desktop\YoloV8\runs\detect\yolo_ver2_train\weights\best.pt"

Kp = 0.0007
min_speed = 0.111
max_speed = 0.111
threshold_px = 30

command_dt = 0.002        # 500 Hz control
perception_dt = 0.1      # 25 Hz perception
print_rate = 5            # print 5 times per second

max_miss = 5              # allow 5 consecutive missed detections

# ================= CSV LOGGING =================
log_filename = f"down_align_{target_label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
log_file = open(log_filename, mode='w', newline='')
csv_writer = csv.writer(log_file)
csv_writer.writerow(["object", "cam", "time_sec", "ex", "ey", "vx", "vy"])

# ================= GLOBAL STATE =================
latest_frame = None
frame_lock = threading.Lock()

vx = 0.0
vy = 0.0
vel_lock = threading.Lock()

stop_event = threading.Event()

print("[YOLO] Loading model...")
model = YOLO(model_path)
print("[YOLO] Model loaded.")

# ================= CAMERA THREAD =================
def camera_thread():
    global latest_frame

    cap = cv2.VideoCapture(down_cam_url)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print("Down camera failed.")
        stop_event.set()
        return

    print("Down camera started.")

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            continue

        with frame_lock:
            latest_frame = frame.copy()

    cap.release()
    print("Down camera stopped.")

# ================= CONTROL THREAD =================
def control_thread():
    global vx, vy

    print(f"[Robot] SDK lib version: {lib_version()}")

    conn = unitreeConnection(HIGH_WIFI_DEFAULTS)
    conn.startRecv()
    hcmd = highCmd()

    conn.send(hcmd.buildCmd(debug=False))
    time.sleep(0.5)

    print("[Robot] Standing up...")
    for _ in range(1000):  # longer stand-up for stability
        hcmd.mode = MotorModeHigh.STAND_UP
        conn.send(hcmd.buildCmd(debug=False))
        time.sleep(command_dt)

    # ================= Gait Activation Kick =================
    '''print("[Robot] Activating gait with forward kick...")
    for _ in range(1):  # ~0.4 seconds at 0.111 m/s
        hcmd.mode = MotorModeHigh.VEL_WALK
        hcmd.gaitType = GaitType.TROT
        hcmd.velocity = [0.111, 0.0]  # small forward push
        hcmd.velocity = [-0.111, 0.0]
        hcmd.yawSpeed = 0.0
        hcmd.footRaiseHeight = 0.01
        conn.send(hcmd.buildCmd(debug=False))
        time.sleep(0.002)


    print("[Robot] Gait active.")'''

    # ================= Normal Control Loop =================
    while not stop_event.is_set():

        with vel_lock:
            send_vx = vx
            send_vy = vy

        hcmd.mode = MotorModeHigh.VEL_WALK
        hcmd.gaitType = GaitType.TROT
        hcmd.velocity = [send_vx, send_vy]
        hcmd.yawSpeed = 0.0
        hcmd.footRaiseHeight = 0.01

        conn.send(hcmd.buildCmd(debug=False))
        time.sleep(command_dt)

    # ================= Safe Stop =================
    for _ in range(300):
        hcmd.velocity = [0.0, 0.0]
        conn.send(hcmd.buildCmd(debug=False))
        time.sleep(command_dt)

    conn.stopRecv()
    print("[Robot] Stopped.")

# ================= PERCEPTION THREAD =================
def perception_thread():
    global vx, vy

    start_time = time.time()
    frame_counter = 0
    miss_count = 0

    while not stop_event.is_set():

        with frame_lock:
            if latest_frame is None:
                time.sleep(perception_dt)
                continue
            frame = latest_frame.copy()

        frame_h, frame_w = frame.shape[:2]
        frame_cx = frame_w // 2
        frame_cy = frame_h // 2

        results = model.predict(frame, conf=0.3, verbose=False)

        detected = False
        new_vx = 0.0
        new_vy = 0.0
        ex = 0
        ey = 0

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                label = model.names[cls]
                if label != target_label:
                    continue

                detected = True
                miss_count = 0  # reset miss counter

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                obj_cx = (x1 + x2) // 2
                obj_cy = (y1 + y2) // 2

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Red dot = frame center
                cv2.circle(frame, (frame_cx, frame_cy), 6, (0, 0, 255), -1)

                # Blue dot = object center
                cv2.circle(frame, (obj_cx, obj_cy), 6, (255, 0, 0), -1)

                ex = obj_cx - frame_cx
                ey = obj_cy - frame_cy

                new_vx = Kp * ey
                new_vy = Kp * ex

                new_vx = np.clip(new_vx, -max_speed, max_speed)
                new_vy = np.clip(new_vy, -max_speed, max_speed)

                if abs(ey) > threshold_px and 0 < abs(new_vx) < min_speed:
                    new_vx = min_speed * np.sign(new_vx)

                if abs(ex) > threshold_px and 0 < abs(new_vy) < min_speed:
                    new_vy = min_speed * np.sign(new_vy)

                break

        # If not detected, increment miss counter
        if not detected:
            miss_count += 1

        # Only stop robot after consecutive misses
        if miss_count > max_miss:
            new_vx = 0.0
            new_vy = 0.0

        with vel_lock:
            vx = new_vx
            vy = new_vy

        elapsed = time.time() - start_time

        csv_writer.writerow([
            target_label,
            "down",
            round(elapsed, 3),
            ex,
            ey,
            round(vx, 3),
            round(vy, 3)
        ])
        log_file.flush()

        # Controlled printing rate
        frame_counter += 1
        if frame_counter % int(1 / (perception_dt * print_rate)) == 0:
            print(f"[Perception] ex={ex:.1f}, ey={ey:.1f}, vx={vx:.3f}, vy={vy:.3f}, miss={miss_count}")

        # Stop when aligned
        if detected and abs(ex) < threshold_px and abs(ey) < threshold_px:
            print("[Aligned] Within threshold. Ending program.")
            stop_event.set()

        cv2.imshow("Down Camera Alignment", frame)
        if cv2.waitKey(1) == ord('q'):
            stop_event.set()

        time.sleep(perception_dt)

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
        print("CSV saved as:", log_filename)
        print("Program ended.")