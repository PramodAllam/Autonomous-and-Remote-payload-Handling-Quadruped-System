import requests
import time

ESP32_IP = 'http://192.168.137.25'  # Confirmed working

def trigger_gripper():
    try:
        print("[Gripper] Sending open request...")
        r = requests.get(f'{ESP32_IP}/open', timeout=7)
        print(f"[Gripper] Response: {r.status_code} - {r.text}")

        time.sleep(2)

        print("[Gripper] Sending close request...")
        r = requests.get(f'{ESP32_IP}/close', timeout=7)
        print(f"[Gripper] Response: {r.status_code} - {r.text}")

    except requests.exceptions.Timeout:
        print("[Gripper] Request timed out.")
    except requests.exceptions.ConnectionError:
        print("[Gripper] Could not connect to ESP32.")
    except Exception as e:
        print(f"[Gripper] Unexpected error: {e}")

# Call the function
trigger_gripper()
