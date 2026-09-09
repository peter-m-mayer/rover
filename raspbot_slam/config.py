"""
Central configuration for RASPBOT-V2 Visual SLAM.

All tunable parameters in one place. Adjust these based on calibration
results and your specific environment.
"""

# =============================================================================
# Feature Detection (ORB)
# =============================================================================
ORB_N_FEATURES = 1000
ORB_SCALE_FACTOR = 1.2
ORB_N_LEVELS = 8
ORB_EDGE_THRESHOLD = 31
ORB_PATCH_SIZE = 31

# Matching
LOWE_RATIO_THRESHOLD = 0.75        # ratio test for frame-to-frame
LOWE_RATIO_LOOP_CLOSURE = 0.65     # stricter for loop closure
MIN_INLIER_MATCHES = 8             # minimum for Essential matrix
RANSAC_PROB = 0.999
RANSAC_THRESHOLD = 1.0             # pixels

# =============================================================================
# Visual Odometry
# =============================================================================
KEYFRAME_TRANSLATION_M = 0.15      # new keyframe every 15 cm
KEYFRAME_ROTATION_DEG = 15.0       # or every 15 degrees
KEYFRAME_OVERLAP_THRESHOLD = 0.40  # or when feature overlap drops below 40%
KEYFRAME_MAX_INTERVAL = 30         # or every 30 frames (time-based fallback)
KEYFRAME_BUFFER_SIZE = 20          # active keyframe window in memory

# =============================================================================
# Synthetic Stereo
# =============================================================================
STRAFE_SPEED = 80                  # motor speed for lateral strafe (of 255 max)
STRAFE_DISTANCE_MM = 50            # calibrated baseline distance
STRAFE_DURATION_MS = 300           # calibrated strafe time at STRAFE_SPEED
SETTLE_TIME_MS = 100               # vibration damping after strafe
STEREO_INTERVAL_KEYFRAMES = 3      # perform stereo every N keyframes
MIN_DISPARITY_PX = 2               # reject triangulations with disparity below this

# =============================================================================
# EKF-SLAM
# =============================================================================
MAX_ACTIVE_LANDMARKS = 30          # cap on landmarks in EKF state vector
PROCESS_NOISE_POSITION = 0.05      # fraction of displacement magnitude
PROCESS_NOISE_HEADING = 0.02       # fraction of rotation magnitude
PROCESS_NOISE_SCALE = 0.0001       # slow scale drift per prediction step
OBSERVATION_NOISE_BEARING = 0.03   # radians (~1.7 degrees)
OBSERVATION_NOISE_RANGE = 0.10     # meters (10 cm)
ULTRASONIC_NOISE_MM = 20           # measurement noise on ultrasonic
LANDMARK_PROMOTION_OBS = 3         # observations before candidate -> active
LANDMARK_FREEZE_UNSEEN = 50        # keyframes before active -> frozen
LANDMARK_REACTIVATION_INFLATION = 10.0  # covariance scale on reactivation
CHI_SQUARED_GATE = 9.21           # 99% gate for 2-DOF observation (bearing, range)

# =============================================================================
# Occupancy Grid
# =============================================================================
GRID_RESOLUTION_M = 0.05           # 5 cm per cell
GRID_SIZE_CELLS = 400              # 400x400 = 20m x 20m coverage
GRID_ORIGIN_OFFSET = 200           # robot starts at cell (200, 200) = center
LOG_ODDS_FREE = -0.4
LOG_ODDS_OCCUPIED = 0.85
LOG_ODDS_PRIOR = 0.0               # unknown
LOG_ODDS_MIN = -5.0                # clamp to avoid certainty lock
LOG_ODDS_MAX = 5.0
OBSTACLE_HEIGHT_MIN_M = 0.05       # below = floor noise, ignore
OBSTACLE_HEIGHT_MAX_M = 2.0        # above = ceiling, ignore

# =============================================================================
# Navigation & Motion
# =============================================================================
WAYPOINT_TOLERANCE_M = 0.10        # 10 cm arrival threshold
OBSTACLE_STOP_MM = 200             # ultrasonic emergency stop distance
SCAN_INTERVAL_M = 0.50             # scanning stop every 50 cm of travel
NAV_SPEED = 60                     # conservative motor speed (of 255 max)
HEADING_PID_P = 0.8
HEADING_PID_I = 0.0
HEADING_PID_D = 0.01
CROSSTRACK_PID_P = 0.2
CROSSTRACK_PID_I = 0.0
CROSSTRACK_PID_D = 0.002

# =============================================================================
# Cat Chaser (catchaser package)
# =============================================================================
# Heading PID: input is normalized horizontal error of the cat centroid,
# err = (cx - W/2) / (W/2) in [-1, 1] (+ = cat to the right of frame center).
# Output is a mecanum turn command in motor units (+ = rotate clockwise/right).
# KP calibrated on hardware (2026-09-06 floor run): at ~6 Hz real loop rate,
# KP=80 overshot — turn=+44 flipped err +0.38 -> -0.27 in one frame
# (~0.015 err per turn-unit per frame). Target ~60% correction per frame.
CHASE_HEADING_KP = 40.0            # motor units per unit normalized error
CHASE_HEADING_KI = 0.0             # off by default (avoids windup while searching)
CHASE_HEADING_KD = 6.0             # damps overshoot on fast centroid swings
CHASE_FORWARD_SPEED = NAV_SPEED    # base approach speed when cat is centered
CHASE_MAX_SPEED = 160              # cap on any single wheel magnitude (of 255);
                                   # raised from 120 to give prey lunges headroom
CHASE_TURN_MAX = 90                # clamp on the turn command
CHASE_STOP_MM = OBSTACLE_STOP_MM   # ultrasonic stop distance (200 mm) — don't maul the cat
CHASE_SEARCH_SPIN_SPEED = 50       # in-place spin speed while hunting for a lost cat
# Pulsed search ("spin-and-stare"): at ~6 Hz loop the robot sweeps a large
# fraction of the ~65 deg FOV per frame and motion blur kills detection while
# rotating, so alternate short spin bursts with stationary look frames.
CHASE_SEARCH_SPIN_FRAMES = 2       # frames of spinning per search cycle
CHASE_SEARCH_STARE_FRAMES = 3      # stationary detection frames per cycle
CHASE_LOST_GRACE_FRAMES = 12       # hold (camera fixed where the cat was) this many lost
                                   # frames before giving up to search — was 3, too twitchy:
                                   # a few blurry/occluded frames dropped it straight to spin
CHASE_CENTER_DEADBAND = 0.06       # |err| below this counts as centered (no turn)
CHASE_TURN_ONLY_ERROR = 0.5        # |err| at/above this: turn in place, no forward drive
# 0.50 calibrated from floor-run snapshots (2026-09-06): every real-cat
# detection scored >=0.53; false positives (legs, blurred objects) hit
# 0.41-0.45. Dim iPad-screen decoys can drop below 0.5 — max screen
# brightness fixes that.
CHASE_MIN_CONFIDENCE = 0.50        # ignore detections below this confidence

# Fast shutter + auto-brightness (ON by default; --no-fast-shutter for native
# auto exposure). Keeps a SHORT exposure to kill motion blur and adapts GAIN to
# reach a target brightness — only lengthening exposure in genuinely dim light.
# v4l2 exposure units, measured range 10-626 (native auto ~156, often blown out).
CHASE_CAM_FAST_SHUTTER = True      # default-apply adaptive short-exposure on chase/rcbot
CHASE_CAM_FAST_EXPOSURE = 78       # the short exposure floor (~half native auto)
CHASE_CAM_EXP_MAX = 220            # auto-brightness may lengthen up to here when dim
CHASE_CAM_TARGET_BRIGHTNESS = 125  # mean frame brightness the auto-gain aims for
CHASE_CAM_GAIN_MAX = 8             # sensor gain ceiling (of 1-8)
CHASE_LOOP_HZ = 20                 # loop pacing cap; detector (~50 ms) is the
                                   # real limiter, so this yields ~9 Hz on Pi 5
CHASE_APPROACH_TAPER_MM = 400      # start slowing this far beyond stop range
CHASE_APPROACH_MIN_FACTOR = 0.35   # floor of the close-range speed taper

# --- Pan-servo tracking (camera tracks the cat; the body follows the pan) ----
# The camera pan servo re-centers the cat in-frame far faster (and with far
# less motion blur) than spinning the whole chassis. The body then rotates to
# follow the pan back toward center. Off by default until the servo direction
# is confirmed on hardware (flip CHASE_PAN_SIGN if it tracks the wrong way).
CHASE_PAN_ENABLED = False
CHASE_PAN_GAIN = 22.0              # degrees of pan correction per unit err/frame
CHASE_PAN_SIGN = 1                 # +1 = sim (pan>center looks left); -1 if hw differs
CHASE_PAN_TURN_ONLY_DEG = 55.0     # if pan is past this, stop forward, let body catch up

# Coarse/fine hand-off: the camera does ALL fine tracking within a wide body
# dead-zone; the body only makes a slow, coarse rotation when the pan swings
# out near the FOV edge (cat about to leave frame), then hands back to the
# camera. Hysteresis (engage > release) stops the body from chattering.
CHASE_PAN_BODY_ENGAGE_DEG = 42.0   # body starts coarse-rotating past this pan offset
CHASE_PAN_BODY_RELEASE_DEG = 14.0  # ...and keeps going until back within this
# Halved twice (2026-09-07): coarse-align/search rotation was blurring frames
# enough that the detector lost the cat mid-turn. Slower body rotate = less
# motion blur (and less motor-stall current on carpet). Now ~1/4 the original.
CHASE_PAN_BODY_ROTATE = 11         # slow, steady body-rotate speed while coarse-aligning
CHASE_PAN_SEARCH_ROTATE = 10       # slow CONTINUOUS search rotate (pan mode; no stop-and-go)

# --- Prey / play mode (behavioral: dart, freeze, flee — what cats hunt) ------
# Relentless smooth pursuit reads as boring or threatening. Prey darts and
# FREEZES (the pause triggers the pounce), zig-zags (mecanum strafe), and
# FLEES when the cat charges — the single most engaging move for a cat.
CHASE_PREY_DART_FRAMES = 4         # frames of darting per cycle
CHASE_PREY_FREEZE_FRAMES = 5       # frames frozen per cycle (pounce bait)
CHASE_PREY_DART_SPEED = 105        # forward lunge speed (exaggerated +50% — cats love it)
CHASE_PREY_STRAFE = 55             # lateral zig-zag magnitude (alternates per cycle)
CHASE_PREY_FLEE_MM = 300           # cat within this -> flee (back away)
CHASE_PREY_FLEE_SPEED = 135        # retreat lunge speed (exaggerated +50%)

# =============================================================================
# Camera
# =============================================================================
CAMERA_DEVICE = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# =============================================================================
# Servos
# =============================================================================
SERVO_PAN_ID = 1
SERVO_TILT_ID = 2
SERVO_PAN_CENTER = 73             # measured mechanical forward for this unit
                                  # (nominal 90; the pan mount sits ~17 deg off,
                                  # confirmed dead-ahead by eye after re-seating
                                  # a loose screw)
SERVO_PAN_MIN = 0
SERVO_PAN_MAX = 180
SERVO_TILT_REST = 74               # the dead-quiet sweet spot (see below)
# Operational tilt band = the truly quiet zone found by a SUSTAINED-hold test
# (2026-09-09). A short-burst map called 68-104 "flat", but holding each angle
# 2.5s revealed a slow LIMIT CYCLE: tilt 72-76 is dead-still (jitter 0.26)
# while 80-100 hunts (jitter 2-4, the oscillation Peter caught). So the real
# quiet band is narrow, ~72-78; tilt is a FIXED level framing (~6 deg of quiet
# travel). Active elevation tracking waits on the new camera/servo. Lesson:
# verify servo stability with SUSTAINED holds, not short bursts. Re-map after
# any remount / new hardware.
SERVO_TILT_MIN = 72
SERVO_TILT_MAX = 78
SERVO_SETTLE_MS = 200              # wait after servo move
PAN_SWEEP_ANGLES = [30, 60, 90, 120, 150]

# =============================================================================
# Motors
# =============================================================================
MOTOR_L1_ID = 0                    # front-left
MOTOR_L2_ID = 1                    # rear-left
MOTOR_R1_ID = 2                    # front-right
MOTOR_R2_ID = 3                    # rear-right
MOTOR_SPEED_MAX = 255

# =============================================================================
# LEDs (status indication)
# =============================================================================
LED_COLOR_MAPPING = "mapping"       # blue while mapping
LED_COLOR_LOCALIZING = "green"      # green when localized
LED_COLOR_LOST = "red"              # red when tracking lost
LED_COLOR_SCANNING = "yellow"       # yellow during scan stop

# =============================================================================
# IMU (ICM-20948, optional upgrade)
# =============================================================================
IMU_ENABLED = True                     # auto-detect; set False to force disable
IMU_I2C_ADDRESS = 0x69                 # default ICM-20948 address (no conflict with 0x2B)
IMU_GYRO_RATE_HZ = 100                # gyro sampling rate
IMU_ACCEL_RATE_HZ = 50                # accelerometer sampling rate
IMU_MAG_RATE_HZ = 10                  # magnetometer sampling rate
IMU_GYRO_NOISE_RAD = 0.001            # gyro noise std (rad/s)
IMU_ACCEL_NOISE_M = 0.05              # accelerometer noise std (m/s^2)
IMU_MAG_NOISE_RAD = 0.05              # magnetometer heading noise (rad)
IMU_COMPLEMENTARY_ALPHA = 0.98        # gyro weight in complementary filter
IMU_MAG_DECLINATION_DEG = 0.0         # local magnetic declination

# =============================================================================
# File Paths
# =============================================================================
CALIBRATION_DIR = "calibration"
CAMERA_CALIBRATION_FILE = "calibration/camera_calibration.json"
STRAFE_CALIBRATION_FILE = "calibration/strafe_calibration.json"
DEFAULT_MAP_DIR = "maps"
