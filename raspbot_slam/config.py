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
CHASE_HEADING_KP = 80.0            # motor units per unit normalized error
CHASE_HEADING_KI = 0.0             # off by default (avoids windup while searching)
CHASE_HEADING_KD = 6.0             # damps overshoot on fast centroid swings
CHASE_FORWARD_SPEED = NAV_SPEED    # base approach speed when cat is centered
CHASE_MAX_SPEED = 120              # cap on any single wheel magnitude (of 255)
CHASE_TURN_MAX = 90                # clamp on the turn command
CHASE_STOP_MM = OBSTACLE_STOP_MM   # ultrasonic stop distance (200 mm) — don't maul the cat
CHASE_SEARCH_SPIN_SPEED = 50       # in-place spin speed while hunting for a lost cat
# Pulsed search ("spin-and-stare"): at ~6 Hz loop the robot sweeps a large
# fraction of the ~65 deg FOV per frame and motion blur kills detection while
# rotating, so alternate short spin bursts with stationary look frames.
CHASE_SEARCH_SPIN_FRAMES = 2       # frames of spinning per search cycle
CHASE_SEARCH_STARE_FRAMES = 3      # stationary detection frames per cycle
CHASE_LOST_GRACE_FRAMES = 3        # hold position this many lost frames before search-spin
CHASE_CENTER_DEADBAND = 0.06       # |err| below this counts as centered (no turn)
CHASE_TURN_ONLY_ERROR = 0.5        # |err| at/above this: turn in place, no forward drive
CHASE_MIN_CONFIDENCE = 0.35        # ignore detections below this confidence
CHASE_LOOP_HZ = 10                 # target control loop rate

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
SERVO_PAN_CENTER = 90              # degrees
SERVO_PAN_MIN = 0
SERVO_PAN_MAX = 180
SERVO_TILT_REST = 25               # degrees (slightly downward)
SERVO_TILT_MIN = 0
SERVO_TILT_MAX = 110               # hardware limit on servo 2
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
