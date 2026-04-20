from ucl.common import lib_version
from ucl.highCmd import highCmd
from ucl.unitreeConnection import unitreeConnection, HIGH_WIFI_DEFAULTS
from ucl.enums import MotorModeHigh
import time

print(f"[Robot] SDK lib version: {lib_version()}")

# Establish connection
conn = unitreeConnection(HIGH_WIFI_DEFAULTS)
conn.startRecv()
time.sleep(0.5)

# Initialize command
hcmd = highCmd()

# -------------------------
# Abrupt squat down
# -------------------------
for _ in range(250):  # 250 * 0.002s = 0.5 sec
    hcmd.mode = MotorModeHigh.FORCE_STAND
    hcmd.bodyHeight = -0.11 # Instantly command squat height
    conn.send(hcmd.buildCmd(debug=False))
    time.sleep(0.002)

# -------------------------
# Hold squat position
# -------------------------
for _ in range(1000):  # Hold for another 0.5 sec
    hcmd.mode = MotorModeHigh.FORCE_STAND
    hcmd.bodyHeight = -0.11
    conn.send(hcmd.buildCmd(debug=False))
    time.sleep(0.002)

# -------------------------
# Stand back up
# -------------------------
for _ in range(250):  # Stand back instantly
    hcmd.mode = MotorModeHigh.FORCE_STAND
    hcmd.bodyHeight = 0.0
    conn.send(hcmd.buildCmd(debug=False))
    time.sleep(0.002)

conn.stopRecv()
print("[Done] Abrupt squat sequence completed.")
