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
down_cam_url  = "http://192.168.137.31/stream"
target_label = "energy_bars"

model_path = r"C:\Users\pramod.na\OneDrive - Texas A&M University\Desktop\YoloV8\runs\detect\yolo_ver2_train\weights\best.pt"

Kp = 0.0012
min_speed = 0.05
max_speed = 0.111
command_dt = 0.002

# ================= CSV =================
log_filename = f"log_{target_label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
log_file = open(log_filename, mode='w', newline='')
csv_writer = csv.writer(log_file)
csv_writer.writerow(["object", "camera", "time_sec", "ex", "ey", "vx", "vy"])

# ================= GLOBAL =================
latest_frame = None
frame_lock = threading.Lock()

vx = 0.0
vy = 0.0
vel_lock = threading.Lock()

stop_event = threading.Event()
phase = "front"
camera_running = False

print("[YOLO] Loading model...")
model = YOLO(model_path)
print("[YOLO] Model loaded.")

# ================= CAMERA THREAD =================
def camera_thread(cam_url):
    global latest_frame, camera_running

    cap = cv2.VideoCapture(cam_url)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print("Camera failed")
        stop_event.set()
        return

    camera_running = True
    print("Camera started:", cam_url)

    while camera_running and not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            continue
        with frame_lock:
            latest_frame = frame.copy()

    cap.release()
    print("Camera stopped:", cam_url)

# ================= CONTROL THREAD =================
def control_thread():
    global phase

    print(f"[Robot] SDK lib version: {lib_version()}")

    conn = unitreeConnection(HIGH_WIFI_DEFAULTS)
    conn.startRecv()
    hcmd = highCmd()

    conn.send(hcmd.buildCmd(debug=False))
    time.sleep(0.5)

    # Stand up
    print("[Robot] Standing up...")
    for _ in range(500):
        hcmd.mode = MotorModeHigh.STAND_UP
        conn.send(hcmd.buildCmd(debug=False))
        time.sleep(command_dt)

    while not stop_event.is_set():

        if phase == "forward_move":
            print("[Robot] Moving forward 3 seconds...")
            start = time.time()
            while time.time() - start < 7.5:
                hcmd.mode = MotorModeHigh.VEL_WALK
                hcmd.gaitType = GaitType.TROT
                hcmd.velocity = [0.111, 0.0]
                hcmd.yawSpeed = 0.0
                conn.send(hcmd.buildCmd(debug=False))
                time.sleep(command_dt)

            hcmd.velocity = [0.0, 0.0]
            conn.send(hcmd.buildCmd(debug=False))
            phase = "down"
            break

        with vel_lock:
            send_vx = vx
            send_vy = vy

        print(f"[Control] Sending vx={send_vx:.3f}, vy={send_vy:.3f}")

        hcmd.mode = MotorModeHigh.VEL_WALK
        hcmd.gaitType = GaitType.TROT
        hcmd.velocity = [send_vx, send_vy]
        hcmd.yawSpeed = 0.0
        hcmd.footRaiseHeight = 0.05

        conn.send(hcmd.buildCmd(debug=False))
        time.sleep(command_dt)

    conn.stopRecv()

# ================= PERCEPTION =================
def perception_thread():
    global vx, vy, phase

    start_time = time.time()

    while not stop_event.is_set():

        with frame_lock:
            if latest_frame is None:
                continue
            frame = latest_frame.copy()

        results = model.predict(frame, conf=0.3, verbose=False)

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                label = model.names[cls]

                if label != target_label:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx = (x1 + x2)//2
                cy = (y1 + y2)//2

                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)

                elapsed = time.time() - start_time

                if phase == "front":

                    frame_cx = frame.shape[1]//2
                    frame_bottom = frame.shape[0] - 1
                    obj_bottom = y2

                    # Draw red dot at frame bottom center
                    cv2.circle(frame, (frame_cx, frame_bottom), 6, (0,0,255), -1)

                    # Draw blue dot at object bottom center
                    cv2.circle(frame, (cx, obj_bottom), 6, (255,0,0), -1)

                    ex = cx - frame_cx
                    ey = obj_bottom - frame_bottom

                    new_vx = -Kp * ey
                    new_vy = -Kp * ex

                    new_vx = np.clip(new_vx, -max_speed, max_speed)
                    new_vy = np.clip(new_vy, -max_speed, max_speed)

                    if abs(ey) > 20 and 0 < abs(new_vx) < min_speed:
                        new_vx = min_speed * np.sign(new_vx)

                    if abs(ex) > 20 and 0 < abs(new_vy) < min_speed:
                        new_vy = min_speed * np.sign(new_vy)

                    with vel_lock:
                        vx = new_vx
                        vy = new_vy

                    print(f"[Front] ex={ex:.1f} ey={ey:.1f} vx={vx:.3f} vy={vy:.3f}")

                    csv_writer.writerow([target_label,"front",round(elapsed,3),ex,ey,vx,vy])
                    log_file.flush()

                    if abs(ex) < 20 and abs(ey) < 20:
                        print("[Front] Alignment reached")
                        phase = "forward_move"

                elif phase == "down":

                    frame_cx = frame.shape[1]//2
                    frame_cy = frame.shape[0]//2

                    # Draw red dot at frame center
                    cv2.circle(frame, (frame_cx, frame_cy), 6, (0,0,255), -1)

                    # Draw blue dot at object center
                    cv2.circle(frame, (cx, cy), 6, (255,0,0), -1)

                    ex = cx - frame_cx
                    ey = cy - frame_cy

                    print(f"[Down] ex={ex:.1f} ey={ey:.1f}")

                    csv_writer.writerow([target_label,"down",round(elapsed,3),ex,ey,0,0])
                    log_file.flush()

                break

        cv2.imshow("Camera", frame)

        if cv2.waitKey(1) == ord('q'):
            stop_event.set()

# ================= MAIN =================
if __name__ == "__main__":
    try:
        cam_thread = threading.Thread(target=camera_thread, args=(front_cam_url,))
        cam_thread.start()

        perc_thread = threading.Thread(target=perception_thread)
        ctrl_thread = threading.Thread(target=control_thread)

        perc_thread.start()
        ctrl_thread.start()

        ctrl_thread.join()

        # Stop front camera
        camera_running = False
        cam_thread.join()

        # Start down camera
        if phase == "down":
            print("[System] Switching to down camera")
            camera_running = True
            cam_thread = threading.Thread(target=camera_thread, args=(down_cam_url,))
            cam_thread.start()

            perc_thread.join()

    except KeyboardInterrupt:
        stop_event.set()

    finally:
        log_file.close()
        cv2.destroyAllWindows()
        print("CSV saved:", log_filename)
        print("Program ended.")