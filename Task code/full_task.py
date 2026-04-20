import threading
import time
import cv2
import numpy as np
import requests
from ultralytics import YOLO

from ucl.common import lib_version
from ucl.highCmd import highCmd
from ucl.unitreeConnection import unitreeConnection, HIGH_WIFI_DEFAULTS
from ucl.enums import MotorModeHigh, GaitType

# ================= CONFIG =================
FRONT_CAM_URL = "http://"ENTER URL HERE"/stream"
DOWN_CAM_URL  = "http:/"ENTER URL HERE"/stream"

OBJECT_MODEL_PATH = "best.pt"
HUMAN_MODEL_PATH  = "yolov8n.pt"

ESP32_IP = "http://"ENTER URL HERE""

OBJECT_LABEL = " " #enter target label here
HUMAN_LABEL  = "person"

REFERENCE_IMAGE_PATH = "empty_gripper.jpg"
SIMILARITY_THRESHOLD = 20  # tune based on testing
empty_reference_gray = None

Kp = 0.0008
max_speed = 0.111
min_speed = 0.111
threshold_px = 10
forward_time = 7.5
command_dt = 0.002

last_human_ex = 0
human_miss_count = 0
max_human_miss = 5

drift_retry_count = 0
max_drift_retries = 1
down_detect_start_time = None
down_wait_time = 5.0

grasp_retry_count = 0
max_grasp_retries = 2
# ================= GLOBAL STATE =================
latest_frame = None
frame_lock = threading.Lock()

vx = 0.0
vy = 0.0
yaw_rate = 0.0
vel_lock = threading.Lock()

phase = "FRONT_ALIGN"
current_camera = "front"
forward_start_time = None
human_start_time=None

stop_event = threading.Event()

# ================= LOAD MODELS =================
object_model = YOLO(OBJECT_MODEL_PATH)
human_model  = YOLO(HUMAN_MODEL_PATH)

reference_img = cv2.imread(REFERENCE_IMAGE_PATH)
reference_img = cv2.resize(reference_img, (320, 240))
empty_reference_gray = cv2.cvtColor(reference_img, cv2.COLOR_BGR2GRAY)

# ================= ROBOT CONNECTION =================
print("[Robot] SDK:", lib_version())
conn = unitreeConnection(HIGH_WIFI_DEFAULTS)
conn.startRecv()
hcmd = highCmd()
conn.send(hcmd.buildCmd(debug=False))
time.sleep(0.5)

# ================= COMPUTE SIMILARITY =================
def compute_similarity(reference_gray, current_frame):
    current_resized = cv2.resize(current_frame, (320, 240))
    current_gray = cv2.cvtColor(current_resized, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(reference_gray, current_gray)
    score = np.mean(diff)
    return score

# ================= CONTROL THREAD =================
def control_thread():
    global vx, vy, yaw_rate

    for _ in range(500):
        hcmd.mode = MotorModeHigh.STAND_UP
        conn.send(hcmd.buildCmd(debug=False))
        time.sleep(command_dt)

    while not stop_event.is_set():
        with vel_lock:
            send_vx = vx
            send_vy = vy
            send_yaw = yaw_rate

        hcmd.mode = MotorModeHigh.VEL_WALK
        hcmd.gaitType = GaitType.TROT
        hcmd.velocity = [send_vx, send_vy]
        hcmd.yawSpeed = send_yaw
        hcmd.footRaiseHeight = 0.05

        conn.send(hcmd.buildCmd(debug=False))
        time.sleep(command_dt)

    conn.stopRecv()

# ================= CAMERA THREAD =================
def camera_thread():
    global latest_frame, current_camera
    cap = None
    active_cam = None

    while not stop_event.is_set():

        if current_camera != active_cam:
            if cap is not None:
                cap.release()
                cap = None
                time.sleep(0.3)
                with frame_lock:
                    latest_frame = None
                time.sleep(0.3)

            url = FRONT_CAM_URL if current_camera == "front" else DOWN_CAM_URL
            cap = cv2.VideoCapture(url)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            active_cam = current_camera
            print("[Camera] Opened", active_cam)

        ret, frame = cap.read()
        if ret:
            with frame_lock:
                latest_frame = frame.copy()

    if cap:
        cap.release()

# ================= PERCEPTION THREAD =================
def perception_thread():
    global vx, vy, yaw_rate, phase, current_camera
    global forward_start_time
    global drift_retry_count
    global down_detect_start_time
    global grasp_retry_count

    prev_time=time.perf_counter()
    fps=0
    while not stop_event.is_set():

        with frame_lock:
            if latest_frame is None:
                time.sleep(0.01)
                continue
            frame = latest_frame.copy()

        H, W = frame.shape[:2]
        detected = False
        verify_object_present = False   
        ex = ey = 0

        # Select model
        if phase in ["FRONT_ALIGN", "FORWARD_MOVE", "DOWN_ALIGN", "VERIFY"]:
            model = object_model
            label_needed = OBJECT_LABEL
        else:
            model = human_model
            label_needed = HUMAN_LABEL

        results = model.predict(frame, conf=0.2, verbose=False)

        for r in results:
            for box in r.boxes:
                class_name = model.names[int(box.cls[0])]
                if class_name == OBJECT_LABEL:
                    detected = True
                    verify_object_present = True

                # Human detection
                if class_name == HUMAN_LABEL:
                    detected = True

                if class_name != label_needed:
                    continue

                x1,y1,x2,y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)

                # ---------------- FRONT ALIGN ----------------
                if phase == "FRONT_ALIGN":

                    obj_cx = (x1+x2)//2
                    obj_bottom = y2
                    frame_cx = W//2
                    frame_bottom = H-1

                    ex = obj_cx - frame_cx
                    ey = obj_bottom - frame_bottom

                    vx_cmd = 0
                    vy_cmd = 0

                    if abs(ex) > threshold_px:
                        vy_cmd = -Kp * ex
                    if abs(ey) > threshold_px:
                        vx_cmd = -Kp * ey

                    if abs(ey) > threshold_px and 0 < abs(vx_cmd) < min_speed:
                        vx_cmd = min_speed * np.sign(vx_cmd)
                    if abs(ex) > threshold_px and 0 < abs(vy_cmd) < min_speed:
                        vy_cmd = min_speed * np.sign(vy_cmd)

                    vx_cmd = np.clip(vx_cmd, -max_speed, max_speed)
                    vy_cmd = np.clip(vy_cmd, -max_speed, max_speed)

                    cv2.circle(frame, (frame_cx, frame_bottom), 6, (0,0,255), -1)
                    cv2.circle(frame, (obj_cx, obj_bottom), 6, (255,0,0), -1)

                    with vel_lock:
                        vx = vx_cmd
                        vy = vy_cmd

                    if abs(ex) < threshold_px and abs(ey) < threshold_px:
                        print("[Front] Aligned")
                        with vel_lock:
                            vx = 0
                            vy = 0
                        requests.get(f"{ESP32_IP}/open1")
                        phase = "FORWARD_MOVE"
                        forward_start_time = time.time()
                        current_camera = "down"

                # ---------------- DOWN ALIGN ----------------
                elif phase == "DOWN_ALIGN":

                    obj_cx = (x1+x2)//2
                    obj_cy = (y1+y2)//2
                    frame_cx = W//2
                    frame_cy = H//2

                    ex = obj_cx - frame_cx
                    ey = obj_cy - frame_cy

                    vx_cmd = 0
                    vy_cmd = 0

                    if abs(ex) > threshold_px:
                        vy_cmd = Kp * ex
                    if abs(ey) > threshold_px:
                        vx_cmd = Kp * ey

                    if abs(ey) > threshold_px and 0 < abs(vx_cmd) < min_speed:
                        vx_cmd = min_speed * np.sign(vx_cmd)
                    if abs(ex) > threshold_px and 0 < abs(vy_cmd) < min_speed:
                        vy_cmd = min_speed * np.sign(vy_cmd)

                    vx_cmd = np.clip(vx_cmd, -max_speed, max_speed)
                    vy_cmd = np.clip(vy_cmd, -max_speed, max_speed)

                    cv2.circle(frame, (frame_cx, frame_cy), 6, (0,0,255), -1)
                    cv2.circle(frame, (obj_cx, obj_cy), 6, (255,0,0), -1)

                    with vel_lock:
                        vx = vx_cmd
                        vy = vy_cmd

                    if abs(ex) < threshold_px+10 and abs(ey) < threshold_px+10:
                        with vel_lock:
                            vx = 0
                            vy = 0
                        phase = "GRASP"

                # ---------------- ALIGN HUMAN ----------------
                elif phase == "ALIGN_HUMAN":

                    if detected:

                        human_miss_count = 0

                        obj_cx = (x1+x2)//2
                        frame_cx = W//2
                        ex = obj_cx - frame_cx

                        last_human_ex = ex

                    else:
                        human_miss_count += 1
                        ex = last_human_ex   # use last known error

                    # keep rotating using last error
                    yaw_rate = -0.002 * ex

                    # stop only if centered AND detected
                    if detected and abs(ex) < threshold_px:
                        yaw_rate = 0
                        phase = "CENTER_HUMAN"

                    # if completely lost for long time → rotate search
                    if human_miss_count > max_human_miss:
                        yaw_rate = 0.5   # slow scanning rotation

                # ---------------- APPROACH HUMAN ----------------
                elif phase == "CENTER_HUMAN":

                    obj_cx = (x1+x2)//2
                    obj_bottom = y2
                    frame_cx = W//2
                    frame_bottom = H-1

                    ex = obj_cx - frame_cx
                    ey = obj_bottom - frame_bottom

                    vx_cmd = -Kp * ey
                    vy_cmd = -Kp * ex

                    if abs(ey) > threshold_px and 0 < abs(vx_cmd) < min_speed:
                        vx_cmd = min_speed * np.sign(vx_cmd)
                    if abs(ex) > threshold_px and 0 < abs(vy_cmd) < min_speed:
                        vy_cmd = min_speed * np.sign(vy_cmd)

                    vx_cmd = np.clip(vx_cmd, -max_speed, max_speed)
                    vy_cmd = np.clip(vy_cmd, -max_speed, max_speed)

                    with vel_lock:
                        vx = vx_cmd
                        vy = vy_cmd

                    if abs(ex) < threshold_px and abs(ey) < threshold_px:
                        with vel_lock:
                            vx = 0
                            vy = 0
                        phase = "APPROACH_HUMAN"
                        human_start_time = time.time()
                break

        if not detected:
            with vel_lock:
                vx = 0
                vy = 0

        print(f"[{phase}] ex={ex:.1f} ey={ey:.1f} vx={vx:.3f} vy={vy:.3f} yaw={yaw_rate:.3f}")

        # ---------------- STATE MACHINE ----------------

        if phase == "FORWARD_MOVE":
            with vel_lock:
                vx = 0.111
                vy = 0
            if time.time() - forward_start_time > forward_time:
                with vel_lock:
                    vx = 0
                current_camera = "down"
                down_detect_start_time = time.time()
                phase = "CHECK_DOWN"

        elif phase == "CHECK_DOWN":
            
            if detected:
                print("[Down] Object found")
                phase = "DOWN_ALIGN"

            else:
                # wait 3 seconds before declaring drift
                if time.time() - down_detect_start_time > down_wait_time:

                    drift_retry_count += 1
                    print(f"[Drift Recovery] Attempt {drift_retry_count}")

                    if drift_retry_count > max_drift_retries:
                        print("Object drifted away. Mission aborted.")
                        # Close gripper
                        requests.get(f"{ESP32_IP}/close1")
                        stop_event.set()
                        return
                    requests.get(f"{ESP32_IP}/close1")
                    # Walk backward same distance
                    back_start = time.time()
                    while time.time() - back_start < forward_time+5:
                        hcmd.mode = MotorModeHigh.VEL_WALK
                        hcmd.gaitType = GaitType.TROT
                        hcmd.velocity = [-0.111, 0]
                        hcmd.footRaiseHeight = 0.05
                        conn.send(hcmd.buildCmd(debug=False))
                        time.sleep(command_dt)

                    # Switch back to front cam
                    current_camera = "front"
                    time.sleep(0.5)

                    phase = "FRONT_ALIGN"

        elif phase == "GRASP":

            #current_camera = None
            #time.sleep(0.5)
            #cv2.destroyWindow("Mission")

            for _ in range(int(1.2/command_dt)):
                hcmd.mode = MotorModeHigh.FORCE_STAND
                hcmd.bodyHeight = -0.11
                conn.send(hcmd.buildCmd(debug=False))
                time.sleep(command_dt)

            requests.get(f"{ESP32_IP}/close1")
            time.sleep(2)

            for _ in range(int(1.0/command_dt)):
                hcmd.mode = MotorModeHigh.FORCE_STAND
                hcmd.bodyHeight = 0
                conn.send(hcmd.buildCmd(debug=False))
                time.sleep(command_dt)
            #current_camera="down"
            #time.sleep(10)
            phase = "VERIFY_GRASP"

        elif phase == "VERIFY_GRASP":

            current_camera = "down"
            time.sleep(1.0)  # allow fresh frame

            with frame_lock:
                if latest_frame is None:
                    continue
                current_frame = latest_frame.copy()

            similarity_score = compute_similarity(empty_reference_gray, current_frame)

            print("Similarity score:", similarity_score)

            if similarity_score < SIMILARITY_THRESHOLD:
                print("[Verify] Gripper empty")

                grasp_retry_count += 1

                if grasp_retry_count > max_grasp_retries:
                    print("Grasp failed multiple times. Aborting.")
                    stop_event.set()
                    return

                requests.get(f"{ESP32_IP}/open1")
                time.sleep(1)
                phase = "DOWN_ALIGN"

            else:
                print("[Verify] Object secured")

                grasp_retry_count = 0
                current_camera = None
                time.sleep(1)
                current_camera="front"
                phase = "SEARCH_HUMAN"
                
        elif phase=="APPROACH_HUMAN":
            with vel_lock:
                vx = 0.111
                vy = 0
            if time.time() - human_start_time > forward_time-3:
                with vel_lock:
                    vx = 0
                    phase = "DROP"

        elif phase == "SEARCH_HUMAN":
            yaw_rate = 0.5
            if detected:
                yaw_rate = 0
                phase = "ALIGN_HUMAN"

        elif phase == "DROP":

            '''for _ in range(int(1.2/command_dt)):
                hcmd.mode = MotorModeHigh.FORCE_STAND
                hcmd.bodyHeight = -0.11
                conn.send(hcmd.buildCmd(debug=False))
                time.sleep(command_dt)'''

            requests.get(f"{ESP32_IP}/open1")
            requests.get(f"{ESP32_IP}/open2")
            time.sleep(2)
            requests.get(f"{ESP32_IP}/close1")
            requests.get(f"{ESP32_IP}/close2")

            '''for _ in range(int(1.0/command_dt)):
                hcmd.mode = MotorModeHigh.FORCE_STAND
                hcmd.bodyHeight = 0
                conn.send(hcmd.buildCmd(debug=False))
                time.sleep(command_dt)'''

            stop_event.set()

        #FPS calculation
        curr_time=time.perf_counter()
        fps=1/(curr_time-prev_time)
        prev_time=curr_time
        cv2.putText(frame,f"FPS:{fps:.2f}",(20,40),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
        cv2.imshow("Mission", frame)
        if cv2.waitKey(1) == ord('q'):
            stop_event.set()

# ================= MAIN =================
if __name__ == "__main__":
    t1 = threading.Thread(target=control_thread)
    t2 = threading.Thread(target=camera_thread)
    t3 = threading.Thread(target=perception_thread)

    t1.start()
    t2.start()
    t3.start()

    t1.join()
    t2.join()
    t3.join()

    cv2.destroyAllWindows()
    print("[Mission] Complete")
