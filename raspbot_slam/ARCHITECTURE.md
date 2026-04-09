# RASPBOT-V2 Visual SLAM -- Architecture Document

**Platform:** Yahboom RASPBOT-V2 on Raspberry Pi 5
**Purpose:** Map indoor ground floor using monocular vision, then localize and navigate
**Product page:** https://www.yahboom.net/study/RASPBOT-V2

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Hardware Platform](#2-hardware-platform)
3. [Architecture Diagram](#3-architecture-diagram)
4. [Module Specifications](#4-module-specifications)
5. [Core Algorithms](#5-core-algorithms)
6. [Map Representation](#6-map-representation)
7. [Operating Modes](#7-operating-modes)
8. [Computational Budget](#8-computational-budget)
9. [Calibration Procedures](#9-calibration-procedures)
10. [Package Structure](#10-package-structure)
11. [Build Order](#11-build-order)
12. [Risks and Mitigations](#12-risks-and-mitigations)
13. [Verification Plan](#13-verification-plan)

---

## 1. System Overview

This system implements monocular visual SLAM (Simultaneous Localization and Mapping) on a
mecanum-wheeled rover with a single camera. The key innovation is **synthetic stereo** --
exploiting the rover's ability to strafe laterally without rotating to create known-baseline
stereo pairs, resolving the scale ambiguity inherent in monocular vision.

The system has two operating modes:

- **Mapping mode:** The rover autonomously explores using frontier-based exploration,
  building a sparse 3D landmark map and a 2D occupancy grid of the floor plan.
- **Localization mode:** The rover matches visual features against the stored map to
  determine its position, then navigates to commanded goals. The map is incrementally
  updated to handle environmental changes (moved furniture, people, etc.).

### Design Philosophy

- **VO is ground truth.** With no encoders or IMU, motor commands are "suggestions" --
  visual odometry measures what actually happened.
- **Stop-and-scan for depth.** Real-time monocular VO provides direction; periodic
  synthetic stereo stops provide absolute scale and depth.
- **Bounded complexity.** Active landmark count is capped at 30 to keep the EKF update
  under 1 ms. Inactive landmarks are frozen, not deleted.
- **Offline refinement.** Bundle adjustment and SIFT-based loop closure run post-mapping,
  not in real-time. The Pi 5 handles ORB at 10+ FPS; heavy lifting happens after.

---

## 2. Hardware Platform

### Sensors and Actuators

| Component | Specification | Interface | Notes |
|-----------|--------------|-----------|-------|
| Camera | USB, 640x480 @ 15-30 FPS | `cv2.VideoCapture(0)` | Mounted on pan/tilt |
| Pan servo (ID 1) | 0-180 degrees | I2C reg 0x02 | Horizontal aim |
| Tilt servo (ID 2) | 0-110 degrees | I2C reg 0x02 | Vertical aim, rest=25 deg |
| Motor L1 (front-left) | -255..+255 | I2C reg 0x01, ID 0 | Mecanum wheel |
| Motor L2 (rear-left) | -255..+255 | I2C reg 0x01, ID 1 | Mecanum wheel |
| Motor R1 (front-right) | -255..+255 | I2C reg 0x01, ID 2 | Mecanum wheel |
| Motor R2 (rear-right) | -255..+255 | I2C reg 0x01, ID 3 | Mecanum wheel |
| Ultrasonic | 16-bit, mm resolution | I2C reg 0x1A/0x1B | Forward-facing, fixed |
| Line tracking | 4 binary IR sensors | I2C reg 0x0A | Floor-facing |
| RGB LEDs | 14x WS2812B | I2C reg 0x03-04, 0x08-09 | Status indication |
| Buzzer | On/off | I2C reg 0x06 | Audio feedback |

### What We Don't Have

- No wheel encoders (open-loop motor control only)
- No IMU or gyroscope
- No lidar or structured light
- No stereo camera (single camera only)
- No GPS (indoor)

### Pi 5 Compute Resources

| Resource | Value |
|----------|-------|
| CPU | Quad-core Cortex-A76 @ 2.4 GHz |
| RAM | 4 or 8 GB LPDDR4X |
| GPU | VideoCore VII (not used for SLAM) |
| Storage | microSD or NVMe via HAT |
| OpenCV | 4.x with ORB, SIFT (patent-free), DNN module |
| Python | 3.11+ |

---

## 3. Architecture Diagram

```
                        ┌─────────────┐
                        │   Camera    │ 640x480 grayscale
                        │  (USB)      │
                        └──────┬──────┘
                               │ frame
                               ▼
                    ┌──────────────────────┐
                    │  FeatureExtractor    │ ORB: 1000 features
                    │  (detect + compute)  │ ~20 ms/frame
                    └─────┬──────────┬────┘
                          │          │
              keypoints + │          │ keypoints +
              descriptors │          │ descriptors
                          │          │
                          ▼          ▼
              ┌───────────────┐  ┌──────────────────┐
              │ VisualOdometry│  │ SyntheticStereo   │
              │               │  │                    │
              │ Essential mat │  │ Strafe 5cm right   │
              │ → R, t_dir    │  │ → triangulate      │
              │               │  │ → depth per point  │
              │ Keyframe mgmt │  │                    │
              └───────┬───────┘  └────────┬───────────┘
                      │ VO delta           │ 3D landmark observations
                      │ (dx, dy, dθ)       │ (x, y, z per feature)
                      │                    │
                      ▼                    ▼
              ┌────────────────────────────────────┐
              │         EKF-SLAM                   │
              │                                    │◀─── Ultrasonic
              │  State: [x, y, θ, scale,           │     (range, mm)
              │          lm1_xyz, ..., lm30_xyz]   │
              │                                    │
              │  Predict: VO delta × scale         │
              │  Update:  landmark observations    │
              │           + ultrasonic range        │
              └──────────────┬─────────────────────┘
                             │ fused pose + landmark positions
                             ▼
              ┌────────────────────────────────────┐
              │         MapManager                 │
              │                                    │
              │  ├─ Sparse landmark DB             │
              │  │  (ORB desc + 3D pos + cov)      │
              │  │                                 │
              │  └─ 2D Occupancy Grid              │
              │     (5 cm cells, log-odds)         │
              └──────────────┬─────────────────────┘
                             │
                ┌────────────┴────────────┐
                │                         │
                ▼                         ▼
    ┌─────────────────┐      ┌─────────────────────┐
    │    Explorer      │      │     Navigator        │
    │  (mapping mode)  │      │  (localization mode)  │
    │                  │      │                       │
    │  Frontier-based  │      │  A* path planning     │
    │  exploration     │      │  + waypoint following  │
    └────────┬────────┘      └──────────┬────────────┘
             │                          │
             └──────────┬───────────────┘
                        │ target waypoint
                        ▼
              ┌──────────────────────┐
              │  MotionController    │
              │                      │
              │  Heading PID         │
              │  Cross-track PID     │
              │  Mecanum kinematics  │
              └──────────┬───────────┘
                         │ motor commands
                         ▼
              ┌──────────────────────┐
              │  Raspbot_Lib (I2C)   │
              │  → Motors, Servos    │
              └──────────────────────┘
```

### Data Flow Summary

1. Camera captures 640x480 grayscale frames at ~15 FPS
2. ORB extracts 1000 keypoints + binary descriptors per frame
3. Visual Odometry matches features frame-to-frame, decomposes Essential matrix into
   rotation R and translation direction t_hat (unit vector, scale unknown)
4. Periodically (~every 50 cm), the robot stops and performs Synthetic Stereo:
   strafe right 5 cm with known baseline, triangulate matched features for depth
5. EKF-SLAM fuses VO deltas (scaled by estimated scale factor) with depth-resolved
   landmark observations and ultrasonic range to maintain a coherent pose estimate
6. MapManager stores 3D landmarks (for relocalization) and updates a 2D occupancy grid
   (for path planning) from ultrasonic ray-casts and projected depth points
7. Explorer (mapping) or Navigator (localization) issues waypoints
8. MotionController drives toward waypoints using PID on heading error, translating
   desired motion into mecanum wheel commands

---

## 4. Module Specifications

### 4.1 `feature_extractor.py`

Wraps OpenCV ORB with project-specific configuration and matching.

```python
class FeatureExtractor:
    """ORB feature detection, description, and matching."""

    def __init__(self, n_features=1000, scale_factor=1.2, n_levels=8):
        """Initialize ORB detector."""

    def detect_and_compute(self, frame: np.ndarray) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
        """Extract ORB keypoints and descriptors from a grayscale frame.
        Returns (keypoints, descriptors) where descriptors is Nx32 uint8."""

    def match(self, desc1: np.ndarray, desc2: np.ndarray,
              ratio_threshold: float = 0.75) -> List[cv2.DMatch]:
        """BFMatcher with Lowe's ratio test. Returns good matches."""

    def match_with_geometric_check(self, kp1, desc1, kp2, desc2, K: np.ndarray
        ) -> Tuple[List[cv2.DMatch], np.ndarray]:
        """Match + RANSAC Essential matrix. Returns (inlier_matches, mask)."""
```

### 4.2 `visual_odometry.py`

Frame-to-frame relative pose estimation from monocular camera.

```python
class Keyframe:
    """Stores image, features, pose, and timestamp for a keyframe."""
    frame: np.ndarray          # grayscale image (640x480)
    keypoints: List[cv2.KeyPoint]
    descriptors: np.ndarray    # Nx32
    pose: np.ndarray           # 4x4 transformation matrix (world frame)
    timestamp: float
    id: int

class VisualOdometry:
    """Monocular visual odometry using ORB features and Essential matrix."""

    def __init__(self, camera: Camera, feature_extractor: FeatureExtractor):
        """Initialize with calibrated camera."""

    def process_frame(self, frame: np.ndarray) -> Optional[Tuple[float, float, float]]:
        """Process a new frame. Returns (dx, dy, dtheta) relative motion estimate,
        or None if tracking is lost (< MIN_INLIER_MATCHES)."""

    def is_keyframe_needed(self) -> bool:
        """Check if current frame should become a keyframe based on
        translation, rotation, and feature overlap thresholds."""

    def create_keyframe(self) -> Keyframe:
        """Promote current frame to keyframe, store in buffer."""

    def get_keyframe_buffer(self) -> List[Keyframe]:
        """Return active keyframe window (last 20)."""

    @property
    def is_tracking(self) -> bool:
        """Whether VO is currently tracking (sufficient feature matches)."""
```

**Algorithm:**
1. Extract ORB from current frame
2. Match against previous frame using BFMatcher + ratio test
3. If >= 8 inliers: `cv2.findEssentialMat(method=RANSAC, prob=0.999, threshold=1.0)`
4. `cv2.recoverPose` -> R, t_direction (unit vector)
5. Scale from last synthetic stereo calibration: `t = t_direction * scale`
6. Compose into cumulative pose: `T_world = T_world @ delta_T`
7. Check keyframe criteria; promote if needed

### 4.3 `synthetic_stereo.py`

Depth estimation via deliberate lateral strafe with known baseline.

```python
class SyntheticStereo:
    """Produces depth estimates by strafing the mecanum-wheeled rover laterally."""

    def __init__(self, camera: Camera, feature_extractor: FeatureExtractor,
                 actuators: Actuators, config: dict):
        """Initialize with calibrated strafe distance."""

    def capture_stereo_pair(self) -> Tuple[np.ndarray, np.ndarray, float]:
        """Stop robot, capture left frame, strafe right by D, capture right frame.
        Returns (left_frame, right_frame, baseline_meters).
        Robot returns to original position after capture."""

    def triangulate(self, left_frame: np.ndarray, right_frame: np.ndarray,
                    baseline: float) -> List[Tuple[np.ndarray, np.ndarray, float]]:
        """Match features between stereo pair and triangulate.
        Returns list of (keypoint_2d, position_3d, confidence) tuples.
        depth = focal_length_px * baseline / disparity"""

    def estimate_scale(self, depth_observations: List, ultrasonic_range_mm: float
        ) -> float:
        """Cross-validate triangulated depths against ultrasonic reading.
        Returns updated scale factor."""
```

**Synthetic Stereo Protocol:**
```
1. Stop all motors, wait 50 ms for settling
2. Capture LEFT frame, extract ORB features
3. Command: move_right(speed=80) for calibrated duration T_strafe
4. Wait 100 ms for vibration damping
5. Capture RIGHT frame, extract ORB features
6. Match LEFT↔RIGHT (BFMatcher + ratio test + RANSAC)
7. For each match with disparity d > 2 pixels:
       Z = (focal_length_px * baseline_m) / d
8. Command: move_left(speed=80) for T_strafe to return
9. Wait 100 ms
```

**Depth range at baseline D=5cm, f≈500px:**

| Disparity (px) | Depth (m) | Use case |
|----------------|-----------|----------|
| 50 | 0.50 | Nearby furniture |
| 25 | 1.00 | Typical room feature |
| 10 | 2.50 | Across room |
| 5 | 5.00 | Down hallway |
| 2 | 12.50 | Maximum reliable range |

### 4.4 `state_estimator.py`

EKF-SLAM fusing visual odometry, synthetic stereo depth, and ultrasonic range.

```python
class EKFSLAM:
    """Extended Kalman Filter for simultaneous localization and mapping."""

    def __init__(self, initial_pose: Tuple[float, float, float], config: dict):
        """Initialize with (x, y, theta) starting pose.
        State vector: [x, y, theta, scale, lm1_x, lm1_y, lm1_z, ...]
        Max 30 active landmarks → 94-dimensional state."""

    def predict(self, vo_delta: Tuple[float, float, float]):
        """Prediction step using VO-estimated motion (dx, dy, dtheta).
        Applies scale factor. Adds process noise proportional to displacement."""

    def update_landmark(self, landmark_id: int,
                        bearing: float, range_m: float, descriptor: np.ndarray):
        """Update step for a single landmark observation.
        If landmark_id is new, consider promotion (requires 3 observations)."""

    def update_ultrasonic(self, range_mm: float, occupancy_grid: np.ndarray):
        """Update using ultrasonic range reading.
        Constrains position along heading direction using ray-cast against grid."""

    def add_landmark(self, position_3d: np.ndarray, descriptor: np.ndarray) -> int:
        """Add a new landmark to the active set. Returns landmark ID.
        If at capacity (30), freeze the least-recently-observed landmark."""

    def freeze_landmark(self, landmark_id: int):
        """Remove landmark from active state (save mean, discard from covariance).
        Reactivated with inflated covariance when robot returns to that area."""

    def reactivate_landmark(self, landmark_id: int):
        """Restore a frozen landmark to the active state with inflated covariance."""

    def get_pose(self) -> Tuple[float, float, float]:
        """Return current (x, y, theta) estimate."""

    def get_pose_covariance(self) -> np.ndarray:
        """Return 3x3 pose covariance submatrix."""

    def get_scale_factor(self) -> float:
        """Return current estimated scale factor."""
```

**State Vector:**
```
x = [x_robot,       # meters, world frame
     y_robot,       # meters, world frame
     theta_robot,   # radians, world frame
     scale_factor,  # multiplicative VO scale correction
     lm1_x, lm1_y, lm1_z,  # landmark 1 position (world frame)
     lm2_x, lm2_y, lm2_z,  # landmark 2 position
     ...
     lm30_x, lm30_y, lm30_z]  # landmark 30 position

dim(x) = 4 + 3*N_active,  max = 4 + 3*30 = 94
```

**Covariance matrix P:** 94x94 at maximum. Update cost O(94^2) ≈ 0.1 ms per observation.

**Process noise (prediction):**
- Position: 5% of displacement magnitude (open-loop motors are unreliable)
- Heading: 2% of rotation magnitude
- Scale: 0.0001 per step (slow drift)
- Landmarks: 0 (static world assumption)

**Landmark lifecycle:**
1. Feature observed once → **candidate** (stored, not in EKF)
2. Observed 3+ times → **promoted** to active EKF landmark
3. In active set, observed regularly → **active** (in covariance matrix)
4. Not observed for 50 keyframes → **frozen** (mean saved, removed from P)
5. Robot returns to area → **reactivated** with 10x inflated covariance

### 4.5 `map_manager.py`

Dual map: sparse 3D landmarks for localization + 2D occupancy grid for navigation.

```python
class Landmark:
    id: int
    position_3d: np.ndarray    # [x, y, z] world frame
    covariance: np.ndarray     # 3x3
    descriptor: np.ndarray     # ORB, 32 bytes
    observation_count: int
    last_seen_keyframe: int
    status: str                # 'candidate', 'active', 'frozen'

class MapManager:
    """Maintains landmark database and occupancy grid."""

    def __init__(self, config: dict):
        """Initialize empty map with configured grid resolution."""

    # --- Landmark operations ---
    def add_candidate(self, position_3d, descriptor) -> int: ...
    def promote_candidate(self, candidate_id) -> int: ...
    def get_visible_landmarks(self, pose, fov_deg=90, max_range=5.0) -> List[Landmark]: ...
    def match_observation(self, descriptor, position_3d) -> Optional[Landmark]: ...

    # --- Occupancy grid ---
    def update_ultrasonic(self, pose, range_mm): ...
    def update_depth_points(self, pose, points_3d: np.ndarray): ...
    def mark_traversed(self, pose): ...
    def get_frontiers(self) -> List[np.ndarray]: ...
    def is_free(self, x, y) -> bool: ...

    # --- Persistence ---
    def save(self, map_dir: str): ...
    def load(self, map_dir: str): ...
```

**Occupancy Grid:**
- Resolution: 5 cm per cell
- Size: 400x400 cells (20m x 20m coverage)
- Values: int8, log-odds representation
  - -1 = unknown
  - 0..100 = log-odds P(occupied)
- Update parameters: `l_free = -0.4`, `l_occupied = 0.85`
- Memory: ~160 KB

**Occupancy update sources:**
1. **Ultrasonic ray-cast:** Cells along ray = free, cell at range = occupied
2. **Stereo depth points:** Project 3D points to ground plane.
   Height < 30 cm → obstacle, 30-200 cm → wall, below feature → free
3. **Robot path:** Cells traversed by robot marked as free

### 4.6 `explorer.py`

Frontier-based autonomous exploration for mapping mode.

```python
class Explorer:
    """Frontier-based exploration strategy."""

    def __init__(self, map_manager: MapManager, config: dict): ...

    def select_target(self, robot_pose) -> Optional[np.ndarray]:
        """Find and score frontier segments, return best target (x, y).
        Score = size_weight * length + dist_weight / distance + info_weight * unknown_behind
        Returns None when mapping is complete (no significant frontiers)."""

    def plan_path(self, start, goal) -> List[np.ndarray]:
        """A* on occupancy grid through free cells. Returns waypoint list."""

    def is_mapping_complete(self) -> bool:
        """True when no frontier segments > 3 cells remain,
        or mapped area hasn't grown for 3 cycles."""
```

**Scanning Stop Protocol (every 50 cm of travel):**
1. Stop motors, wait 50 ms
2. Pan servo sweep: 30, 60, 90, 120, 150 degrees (5 positions)
3. At each: capture frame, extract features, match against landmark DB
4. At center (90 deg): perform synthetic stereo capture
5. Read ultrasonic
6. Update EKF and occupancy grid
7. Total time: ~3-5 seconds per stop

### 4.7 `navigator.py`

Goal-directed navigation using a stored map.

```python
class Navigator:
    """Path planning and goal navigation on a known map."""

    def __init__(self, map_manager: MapManager, state_estimator: EKFSLAM, config: dict): ...

    def relocalize(self, camera: Camera, actuators: Actuators) -> bool:
        """Panoramic sweep + PnP against landmark DB.
        Returns True if localized (>20 PnP inliers)."""

    def navigate_to(self, goal_xy: Tuple[float, float]) -> bool:
        """Plan path with A*, follow waypoints with PID.
        Handles dynamic obstacle detection and replanning."""

    def is_localized(self) -> bool:
        """True if pose covariance trace is below threshold."""
```

**Relocalization procedure:**
1. Pan sweep 0-180 degrees in 30-degree steps (7 frames)
2. Extract ORB from all frames
3. Match against all keyframe descriptors in map
4. Best keyframe match → `cv2.solvePnPRansac` with 3D landmark positions
5. If inliers > 20: accept pose, initialize EKF
6. If inliers < 20: rotate 90 degrees, retry

**Change adaptation:**
- Landmarks consistently missed (3+ times) → flagged "possibly moved"
- After 10 misses → covariance inflated 10x (downweighted)
- New persistent features (5+ consecutive frames, same location) → tentative landmarks
- Occupancy grid updated in real-time (new obstacles from ultrasonic)

### 4.8 `motion_controller.py`

Translates waypoints into mecanum wheel commands.

```python
class MotionController:
    """PID-controlled waypoint following with mecanum kinematics."""

    def __init__(self, actuators: Actuators, config: dict): ...

    def drive_to_waypoint(self, current_pose, target_xy) -> bool:
        """One control step toward waypoint. Returns True when reached.
        Uses heading PID + cross-track PID + mecanum set_deflection."""

    def emergency_stop(self): ...
    def is_obstacle_ahead(self, ultrasonic_range_mm) -> bool: ...
```

### 4.9 Hardware Abstraction: `camera.py`, `sensors.py`, `actuators.py`

```python
class Camera:
    def __init__(self, device=0, width=640, height=480): ...
    def capture(self) -> np.ndarray:              # grayscale
    def capture_color(self) -> np.ndarray:         # BGR
    def undistort_points(self, pts) -> np.ndarray:
    @property
    def K(self) -> np.ndarray:                     # 3x3 intrinsic matrix
    @property
    def dist_coeffs(self) -> np.ndarray:           # distortion coefficients

class Sensors:
    def __init__(self, bot: Raspbot): ...
    def read_ultrasonic_mm(self) -> int:
    def read_line_tracker(self) -> Tuple[bool, bool, bool, bool]:

class Actuators:
    def __init__(self, bot: Raspbot): ...
    def set_servo_pan(self, angle_deg: float): ...
    def set_servo_tilt(self, angle_deg: float): ...
    def move_forward(self, speed: int): ...
    def move_right(self, speed: int): ...         # strafe
    def move_left(self, speed: int): ...          # strafe
    def rotate_left(self, speed: int): ...
    def rotate_right(self, speed: int): ...
    def stop(self): ...
    def set_led_color(self, r, g, b): ...         # status indication
```

### 4.10 `config.py`

Single source of truth for all tunable parameters.

```python
# === Feature Detection ===
ORB_N_FEATURES = 1000
ORB_SCALE_FACTOR = 1.2
ORB_N_LEVELS = 8
LOWE_RATIO_THRESHOLD = 0.75
LOWE_RATIO_LOOP_CLOSURE = 0.65
MIN_INLIER_MATCHES = 8

# === Visual Odometry ===
KEYFRAME_TRANSLATION_M = 0.15       # new keyframe every 15 cm
KEYFRAME_ROTATION_DEG = 15.0        # or every 15 degrees
KEYFRAME_OVERLAP_THRESHOLD = 0.40   # or when overlap drops below 40%
KEYFRAME_MAX_INTERVAL = 30          # or every 30 frames
KEYFRAME_BUFFER_SIZE = 20           # active window

# === Synthetic Stereo ===
STRAFE_SPEED = 80                   # motor speed for strafe
STRAFE_DISTANCE_MM = 50             # calibrated baseline
STRAFE_DURATION_MS = 300            # calibrated time
SETTLE_TIME_MS = 100                # vibration damping
STEREO_INTERVAL_KEYFRAMES = 3       # stereo every 3rd keyframe
MIN_DISPARITY_PX = 2                # reject triangulations below this

# === EKF-SLAM ===
MAX_ACTIVE_LANDMARKS = 30
PROCESS_NOISE_POSITION = 0.05       # fraction of displacement
PROCESS_NOISE_HEADING = 0.02        # fraction of rotation
PROCESS_NOISE_SCALE = 0.0001        # per step
LANDMARK_PROMOTION_OBS = 3          # observations before promotion
LANDMARK_FREEZE_UNSEEN = 50         # keyframes before freezing
LANDMARK_REACTIVATION_INFLATION = 10.0

# === Occupancy Grid ===
GRID_RESOLUTION_M = 0.05            # 5 cm per cell
GRID_SIZE_CELLS = 400               # 20m x 20m
LOG_ODDS_FREE = -0.4
LOG_ODDS_OCCUPIED = 0.85
OBSTACLE_HEIGHT_MIN_M = 0.05        # below = floor noise
OBSTACLE_HEIGHT_MAX_M = 2.0         # above = ceiling, ignore

# === Navigation ===
WAYPOINT_TOLERANCE_M = 0.10         # 10 cm arrival threshold
OBSTACLE_STOP_MM = 200              # ultrasonic stop distance
SCAN_INTERVAL_M = 0.50              # scanning stop every 50 cm
HEADING_PID = (0.8, 0.0, 0.01)     # P, I, D for heading control
CROSSTRACK_PID = (0.2, 0.0, 0.002) # P, I, D for lateral correction
NAV_SPEED = 60                      # conservative speed (of 255 max)

# === Camera ===
CAMERA_DEVICE = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# === Servo ===
SERVO_PAN_CENTER = 90
SERVO_TILT_REST = 25
PAN_SWEEP_ANGLES = [30, 60, 90, 120, 150]
```

---

## 5. Core Algorithms

### 5.1 ORB Feature Extraction and Matching

ORB (Oriented FAST and Rotated BRIEF) is a binary descriptor, ~10x faster than SIFT
on CPU. At 640x480 with 1000 features, extraction takes ~20 ms on Pi 5.

**Matching pipeline:**
1. `cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)`
2. `knnMatch(k=2)` → candidate pairs
3. Lowe's ratio test: keep if `d_best / d_second < 0.75` (~60% false match rejection)
4. Geometric verification: `cv2.findEssentialMat(method=RANSAC, prob=0.999, threshold=1.0)`
5. Chirality check via `cv2.recoverPose` (rejects behind-camera points)

**For loop closure (offline):** Re-extract SIFT from keyframes. Stricter ratio test (0.65).
Bag-of-visual-words for candidate selection before exhaustive matching.

### 5.2 Visual Odometry (Essential Matrix Decomposition)

Given matched points in calibrated coordinates (undistorted, normalized by K^-1):

1. `E, mask = cv2.findEssentialMat(pts1, pts2, K, method=RANSAC)`
2. `n_inliers, R, t, mask = cv2.recoverPose(E, pts1, pts2, K)`
3. R is the rotation matrix; t is the translation DIRECTION (unit vector)
4. Scale is unknown from monocular vision alone → applied from synthetic stereo

**Keyframe management:**
- New keyframe when: translation > 15 cm OR rotation > 15 deg OR overlap < 40% OR 30 frames elapsed
- Active buffer: 20 most recent keyframes
- All keyframes saved to disk for offline processing

### 5.3 Synthetic Stereo Depth Estimation

The mecanum strafe creates a horizontal baseline between two camera positions.
With known baseline D (meters) and focal length f (pixels):

```
depth_z = f * D / disparity

where disparity = left_point_x - right_point_x (pixels)
```

This is identical to standard stereo vision, but the "stereo pair" is captured
sequentially with a known physical displacement.

**Key insight:** The mecanum wheels can strafe purely laterally (no rotation),
making the stereo geometry clean -- the epipolar lines are exactly horizontal.
Any rotation that occurs (detected by VO) can be compensated in the matching.

**Reliability:** At D=5cm baseline:
- Features at 1m have ~25px disparity → depth error ~4%
- Features at 5m have ~5px disparity → depth error ~20%
- Features at 12m have ~2px disparity → depth error ~50%
- Below 2px disparity, depth is unreliable and discarded

### 5.4 EKF-SLAM

**Prediction (every VO frame):**
```
x' = x + scale * (dx * cos(θ) - dy * sin(θ))
y' = y + scale * (dx * sin(θ) + dy * cos(θ))
θ' = θ + dθ

F = Jacobian of motion model w.r.t. state
P' = F @ P @ F.T + Q
```

**Update (each observed landmark):**
```
z_expected = h(x, landmark_pos)    # predicted bearing + range
z_actual = measured bearing + range from triangulation

innovation = z_actual - z_expected
H = Jacobian of h w.r.t. state
S = H @ P @ H.T + R              # innovation covariance
K = P @ H.T @ S^-1               # Kalman gain
x = x + K @ innovation
P = (I - K @ H) @ P
```

**Ultrasonic update:**
```
z = ultrasonic_range_mm / 1000    # meters
h(x) = ray_cast(x, y, θ, occupancy_grid)  # predicted range along heading
innovation = z - h(x)
H = numerical Jacobian (finite difference)
# Standard EKF update with 1D observation
```

### 5.5 Frontier-Based Exploration

**Frontier definition:** A cell on the occupancy grid that is FREE and has at least
one UNKNOWN neighbor.

**Algorithm:**
1. Find all frontier cells via convolution or boundary tracing
2. Cluster into connected segments
3. Score each segment:
   ```
   score = w_size * len(segment)
         + w_dist * (1.0 / distance_to_robot)
         + w_info * count_unknown_behind_segment
   ```
4. Select highest-scoring segment centroid as next target
5. A* path on occupancy grid (only through free cells, 8-connected)
6. Smooth path (remove redundant waypoints on straight segments)

**Completion criteria:**
- No frontier segments > 3 cells remain
- OR total mapped free area unchanged for 3 consecutive exploration cycles
- OR user manually signals done

### 5.6 A* Path Planning

Standard A* on the 2D occupancy grid:
- 8-connected neighbors (diagonal moves allowed)
- Cost: 1.0 for cardinal, 1.414 for diagonal
- Inflated obstacles: cells within 15 cm of occupied cells have extra cost
- Heuristic: Euclidean distance to goal
- Replanning triggered on new obstacle detection

---

## 6. Map Representation

### Sparse Landmark Database

Each landmark stores:
- 3D position (world frame): 24 bytes (3 x float64)
- 3x3 covariance: 72 bytes
- ORB descriptor: 32 bytes
- Metadata (id, obs_count, last_seen, status): ~20 bytes
- **Total: ~150 bytes per landmark**

Typical house mapping: 5,000-10,000 landmarks → **0.75 - 1.5 MB**

Storage format: Python pickle (`.pkl`) for simplicity. SQLite for indexed spatial queries
if landmark count exceeds 50K.

### 2D Occupancy Grid

- 400x400 cells at 5 cm resolution = 20m x 20m coverage
- int8 log-odds: -128 to +127
- **Memory: 160 KB**
- Storage: numpy `.npy` file

### Keyframe Archive

- JPEG compressed images (~30 KB each at 640x480)
- ORB descriptors: ~32 KB per keyframe (1000 features x 32 bytes)
- 500 keyframes for a full house → **~30 MB**

### Map Directory Structure

```
maps/ground_floor/
    metadata.json           # timestamp, camera calibration, grid params, stats
    landmarks.pkl           # list of Landmark objects
    occupancy_grid.npy      # 400x400 int8
    trajectory.npy          # Nx3 array of (x, y, theta) robot poses
    keyframes/
        kf_0001.jpg         # compressed grayscale image
        kf_0001_desc.npy    # 1000x32 ORB descriptors
        kf_0001_kp.npy      # 1000x2 keypoint coordinates
        ...
```

---

## 7. Operating Modes

### 7.1 Mapping Mode (`run_mapping.py`)

```
INIT:
    calibrate camera (if not cached)
    calibrate strafe distance (if not cached)
    initialize EKF at origin (0, 0, 0)
    initialize empty map

LOOP:
    frame = camera.capture()
    vo_delta = visual_odometry.process_frame(frame)

    if vo_delta is not None:
        ekf.predict(vo_delta)
        match visible landmarks, update EKF

    if distance_since_last_scan > 0.50m:
        perform_scanning_stop()
            - pan sweep (5 positions)
            - synthetic stereo at center
            - ultrasonic reading
            - update landmarks + occupancy grid

    if ultrasonic < 200mm:
        emergency_stop()
        update occupancy grid

    target = explorer.select_target(ekf.get_pose())
    if target is None:
        print("Mapping complete!")
        break

    motion_controller.drive_to_waypoint(ekf.get_pose(), target)

POST:
    save map to disk
    run offline bundle adjustment (optional)
    run loop closure detection (optional)
    visualize with visualize_map.py
```

### 7.2 Localization Mode (`run_navigation.py`)

```
INIT:
    load map from disk
    relocalize (panoramic sweep + PnP)
    initialize EKF at relocalized pose

LOOP:
    frame = camera.capture()
    vo_delta = visual_odometry.process_frame(frame)

    if vo_delta is not None:
        ekf.predict(vo_delta)
        match against map landmarks, update EKF

    if tracking lost for 5 frames:
        stop, trigger relocalization

    if goal is set:
        path = navigator.plan_path(ekf.get_pose(), goal)
        motion_controller.drive_to_waypoint(ekf.get_pose(), path[0])

    # Adapt to changes
    flag missing landmarks (>3 unseen)
    add new persistent features as tentative landmarks
    update occupancy grid from ultrasonic
```

---

## 8. Computational Budget

### Per-Frame Budget (target: 10 FPS = 100 ms/frame)

| Component | Time (ms) | Memory | Notes |
|-----------|-----------|--------|-------|
| Frame capture + grayscale | 3 | 300 KB | cv2.VideoCapture + cvtColor |
| Undistort keypoints | 1 | negligible | cv2.undistortPoints |
| ORB detect + compute | 20 | 32 KB desc | 1000 features at 640x480 |
| Feature matching | 10 | — | BFMatcher + Lowe's ratio |
| Essential matrix + pose | 7 | — | RANSAC, ~200 inliers |
| EKF prediction | 0.5 | 70 KB cov | 94x94 matrix multiply |
| EKF update (30 landmarks) | 2 | — | Kalman gain computation |
| Occupancy grid update | 1 | 160 KB grid | Ray-cast, numpy |
| Motion controller | 0.5 | — | PID + I2C write |
| Display / debug overlay | 5 | — | Optional, disable for perf |
| **Total per frame** | **~45-50** | **~600 KB** | **50-55 ms headroom** |

### Synthetic Stereo Stop (~1 second)

| Step | Time (ms) |
|------|-----------|
| Stop motors + settle | 50 |
| Capture left + ORB | 25 |
| Strafe 5 cm | 300 |
| Settle | 100 |
| Capture right + ORB | 25 |
| Match + triangulate | 30 |
| Strafe back | 300 |
| Settle | 100 |
| **Total** | **~930** |

### Pan Sweep Stop (~2 seconds)

| Step | Time (ms) |
|------|-----------|
| 5 positions x (servo 200 + settle 100 + capture+ORB 25) | 1625 |
| Match 5 frames against landmark DB | 200 |
| **Total** | **~1825** |

### Full Scanning Stop (stereo + sweep): ~3-5 seconds

### Offline Processing (post-mapping, on Pi 5)

| Task | Time | Notes |
|------|------|-------|
| SIFT re-extraction (500 keyframes) | ~90 sec | 180 ms/frame |
| Loop closure (all-pairs search) | ~300 sec | BoVW acceleration |
| Bundle adjustment | 60-600 sec | scipy or g2o, depends on graph |
| Occupancy grid refinement | ~10 sec | Re-project with corrected poses |
| **Total** | **8-17 min** | Can transfer to laptop if too slow |

### Total RAM Usage

| Component | Size |
|-----------|------|
| Current frame + previous frame | 600 KB |
| Keyframe buffer (20 frames) | 6 MB |
| ORB descriptors (20 keyframes) | 640 KB |
| EKF covariance (94x94 float64) | 70 KB |
| Landmark database (10K landmarks) | 1.5 MB |
| Occupancy grid (400x400) | 160 KB |
| OpenCV overhead | ~100 MB |
| Python + libraries | ~200 MB |
| **Total** | **~310 MB** | Fits easily in 4 GB Pi 5 |

---

## 9. Calibration Procedures

### 9.1 Camera Intrinsic Calibration (`calibration/camera_calibrate.py`)

**One-time procedure:**
1. Print a checkerboard pattern (9x6 inner corners recommended)
2. Capture 15-20 images from various angles and distances
3. `cv2.findChessboardCorners` + `cv2.calibrateCamera`
4. Store K (3x3 intrinsic matrix) and dist_coeffs to `camera_calibration.json`

**Output:**
```json
{
    "camera_matrix": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
    "dist_coeffs": [k1, k2, p1, p2, k3],
    "image_size": [640, 480],
    "reprojection_error": 0.3
}
```

### 9.2 Strafe Distance Calibration (`calibration/strafe_calibrate.py`)

**Procedure:**
1. Place rover at measured distance from wall (verify with tape measure)
2. Read ultrasonic for baseline distance
3. Strafe right at speed S for duration T
4. Read ultrasonic again (should be same if parallel to wall)
5. Use camera VO to estimate displacement
6. Use triangulation against wall features for ground truth
7. Repeat at multiple speeds: 40, 60, 80, 100
8. Build lookup table: `(speed, duration_ms) → distance_mm`

**Output:** `calibration/strafe_calibration.json`
```json
{
    "floor_surface": "hardwood",
    "calibrations": [
        {"speed": 80, "duration_ms": 300, "distance_mm": 50, "std_mm": 3},
        {"speed": 80, "duration_ms": 600, "distance_mm": 95, "std_mm": 5},
        ...
    ]
}
```

---

## 10. Package Structure

```
rover/raspbot_slam/
    __init__.py
    config.py                      # All tunable parameters

    # --- Core SLAM pipeline ---
    feature_extractor.py           # ORB detection, matching, outlier rejection
    visual_odometry.py             # Frame-to-frame VO, keyframe management
    synthetic_stereo.py            # Mecanum strafe depth estimation
    state_estimator.py             # EKF-SLAM with bounded landmarks
    map_manager.py                 # Landmark DB + occupancy grid

    # --- Navigation ---
    explorer.py                    # Frontier-based exploration (mapping mode)
    navigator.py                   # A* + waypoint following (localization mode)
    motion_controller.py           # PID heading + mecanum commands

    # --- Hardware abstraction ---
    camera.py                      # Capture, calibration, undistortion
    sensors.py                     # Ultrasonic, line tracker
    actuators.py                   # Servos, motors, LEDs

    # --- Existing drivers (from Yahboom, unmodified) ---
    drivers/
        __init__.py
        Raspbot_Lib.py             # I2C hardware driver
        McLumk_Wheel_Sports.py     # Mecanum kinematics
        PID.py                     # PID controllers

    # --- Offline post-processing ---
    offline/
        __init__.py
        bundle_adjustment.py       # Full BA with scipy.optimize
        loop_closure.py            # SIFT re-extraction + BoVW matching
        map_optimizer.py           # Trajectory smoothing + grid refinement

    # --- Calibration tools ---
    calibration/
        __init__.py
        camera_calibrate.py        # Checkerboard → intrinsics
        strafe_calibrate.py        # Motor speed → distance LUT
        camera_calibration.json    # Stored camera intrinsics
        strafe_calibration.json    # Stored strafe LUT

    # --- Entry points ---
    run_mapping.py                 # Main: autonomous mapping
    run_navigation.py              # Main: load map + navigate to goals
    run_calibration.py             # Interactive calibration wizard
    visualize_map.py               # matplotlib map + trajectory viewer

    # --- Map storage ---
    maps/
        <map_name>/
            metadata.json
            landmarks.pkl
            occupancy_grid.npy
            trajectory.npy
            keyframes/
                kf_NNNN.jpg
                kf_NNNN_desc.npy
                kf_NNNN_kp.npy

    ARCHITECTURE.md                # This document
```

---

## 11. Build Order

| Phase | Duration | Modules | Milestone |
|-------|----------|---------|-----------|
| **1. Calibration + Camera** | 1-2 days | `camera.py`, `camera_calibrate.py`, `strafe_calibrate.py`, `sensors.py`, `actuators.py` | Camera captures undistorted frames. Strafe LUT built. |
| **2. Feature Extraction + VO** | 2-3 days | `feature_extractor.py`, `visual_odometry.py` | Robot drives forward 1m, VO reports ~1m. Rotate 90 deg, VO reports ~90 deg. |
| **3. Synthetic Stereo** | 2-3 days | `synthetic_stereo.py` | Strafe, triangulate wall at 1m. Depth agrees with ultrasonic within 10%. |
| **4. State Estimation** | 2-3 days | `state_estimator.py` | Drive 1m square, return to start. EKF pose within 10 cm of origin. |
| **5. Mapping** | 3-4 days | `map_manager.py`, `explorer.py`, `motion_controller.py`, `run_mapping.py` | Map a single room. `visualize_map.py` shows recognizable floor plan. |
| **6. Offline Optimization** | 2-3 days | `offline/bundle_adjustment.py`, `offline/loop_closure.py`, `offline/map_optimizer.py` | Reprocessed map is more consistent. Loop closures detected. |
| **7. Localization + Navigation** | 2-3 days | `navigator.py`, `run_navigation.py` | Place robot on map, command goal. Robot navigates successfully. |

**Total: 15-21 days for working prototype**

---

## 12. Risks and Mitigations

| Risk | Severity | Mitigation |
|------|----------|-----------|
| **Strafe distance varies by floor surface** | High | Calibrate on actual floor. Ultrasonic cross-validation. Accept wider depth uncertainty (inflate covariance). |
| **Textureless walls produce few ORB features** | High | Add textured targets (posters, stickers) during mapping. Tilt camera to include floor texture. Pan sweep gives more viewpoints. Increase ORB features to 2000 if needed. |
| **Scale drift in long corridors** | Medium | Trigger synthetic stereo more frequently (every keyframe). Use ultrasonic wall-following for range constraint. |
| **Open-loop motors → unpredictable motion** | Medium | Keep speeds low (60-80 of 255). VO is ground truth; motor commands are suggestions. PID corrects using VO feedback. |
| **EKF divergence** | Medium | Chi-squared gating on innovation. If innovation > 3σ, reject observation. Monitor covariance trace; if growing unbounded, trigger relocalization. |
| **Computational overload on Pi 5** | Low | Budget is conservative (45 ms of 100 ms). Can drop to 320x240 or 500 features. Disable debug overlay. |
| **Camera-ultrasonic disagreement** | Low | Trust ultrasonic for short range (<1m). Trust camera for bearing. Weight by measurement confidence. |

---

## 13. Verification Plan

### Phase 1-2: Visual Odometry Accuracy
- Drive robot forward 1m (measured with tape). VO should report 0.9-1.1m.
- Rotate in place 90 degrees. VO should report 80-100 degrees.
- Drive a 2m straight line. Check cumulative drift < 10%.

### Phase 3: Synthetic Stereo Depth
- Place object at 1.0m (measured). Triangulated depth should be 0.9-1.1m.
- Repeat at 0.5m, 2.0m, 5.0m. Check error < 20% at each distance.
- Compare all depths against simultaneous ultrasonic reading.

### Phase 4: EKF Consistency
- Drive 1m square (4 turns). Return-to-origin error < 10 cm.
- Drive 3m straight, do synthetic stereo every 1m. Scale drift < 5%.
- Check that covariance trace decreases after each landmark update.

### Phase 5: Mapping Quality
- Map a single room (~4m x 4m). Occupancy grid should show 4 walls.
- Landmark density should be 50-200 per room.
- `visualize_map.py` produces a recognizable floor plan.

### Phase 6: Offline Improvement
- Loop closure detected when robot returns to starting position.
- Bundle-adjusted trajectory has lower total reprojection error.
- Occupancy grid walls become sharper after optimization.

### Phase 7: Navigation Success
- Place robot at known position. Relocalization succeeds in < 30 seconds.
- Command goal 3m away. Robot arrives within 20 cm of target.
- Place new obstacle on path. Robot detects and replans around it.
- Move a piece of furniture. Robot adapts map on next visit.

---

## 14. PyBullet Digital Twin Simulator

### Overview

The simulator provides drop-in replacements for all hardware interfaces:

| Real Hardware | Simulator Replacement | Notes |
|--------------|----------------------|-------|
| `Camera` | `SimCamera` | Renders 640×480 from rover viewpoint via PyBullet |
| `Sensors` | `SimSensors` | Ultrasonic via ray-cast with Gaussian noise |
| `Actuators` | `SimActuators` | Mecanum kinematics via position integration |
| I2C / Raspbot_Lib | `SimWorld` | PyBullet physics world with rooms and furniture |

The SLAM pipeline code is **identical** between hardware and simulation.

### Simulator Architecture

```
SimWorld (PyBullet)
├── Floor plan (walls, floor, furniture)
├── Kinematic rover body
├── Procedural textures (for ORB features)
└── Collision detection
    │
    ├── SimCamera
    │   └── p.getCameraImage() → grayscale / color / depth
    │
    ├── SimSensors
    │   └── p.rayTest() → ultrasonic distance (+ noise)
    │
    └── SimActuators
        └── Mecanum kinematics → p.resetBasePositionAndOrientation()
```

### Kinematic Motion Model

The rover is modeled as a kinematic body (mass=0) -- position is updated
directly each physics step rather than through forces. This avoids friction
and gravity fighting the motion, which is the standard approach for wheeled
robots on flat floors in PyBullet.

```python
# Per step (1/240 second):
vx = (l1 + l2 + r1 + r2) / 4.0            # forward velocity
vy = (-l1 + l2 + r1 - r2) / 4.0           # lateral velocity
omega = (-l1 - l2 + r1 + r2) / (4 * 0.17) # angular velocity

new_x = x + (vx * cos(yaw) - vy * sin(yaw)) * dt
new_y = y + (vx * sin(yaw) + vy * cos(yaw)) * dt
new_yaw = yaw + omega * dt
```

Motor speed 80 (of 255) → 0.157 m/s → validated at 0.314m in 2 seconds.

### Floor Plans

Each floor plan includes walls, textured surfaces, and optional furniture:

- **simple_room:** 4m × 4m single room
- **L_shaped:** 6m × 4m L-shaped with table and chairs
- **corridor:** 6m × 1.5m narrow corridor
- **two_rooms:** Two 3m × 3m rooms connected by 1m doorway

### Procedural Textures

Walls and floors get procedural textures (rectangles, circles, panel lines,
noise) so that ORB finds 500+ features per frame. Without textures,
flat-colored walls yield < 50 features -- insufficient for SLAM.

### Validated Performance

| Metric | Result |
|--------|--------|
| Forward displacement (2s, speed=80) | 0.314m (expected 0.31m) |
| Strafe heading drift | 0.0 degrees |
| ORB features per frame | 500+ |
| Cross-frame ORB matches | 148+ |
| Ultrasonic accuracy | ±20mm noise on ground truth |
| Depth image range | 0.05m to 20m |

### Running the Simulator

```bash
# Headless (fast, for automated testing)
python -m raspbot_slam.simulator.run_sim --floor-plan L_shaped

# With GUI (visual, for demos)
python -m raspbot_slam.simulator.run_sim --floor-plan two_rooms --gui

# With custom map name and step limit
python -m raspbot_slam.simulator.run_sim --floor-plan corridor --map-name sim_test --max-steps 3000
```

---

## 15. Test Suite

155 offline tests covering all modules. No hardware required.

```bash
python -m pytest tests/ -v                    # full suite
python -m pytest tests/test_simulator.py -v   # simulator only
```

| Test File | Count | Coverage |
|-----------|-------|----------|
| `test_config.py` | 10 | Parameter ranges, consistency |
| `test_camera.py` | 7 | Calibration I/O, undistortion math |
| `test_sensors_actuators.py` | 17 | Mock hardware interface |
| `test_feature_extractor.py` | 14 | ORB detection, matching, RANSAC |
| `test_state_estimator.py` | 16 | EKF predict/update, landmark lifecycle, gating |
| `test_map_manager.py` | 20 | Occupancy grid, landmarks, frontiers, persistence |
| `test_explorer.py` | 11 | A*, frontier clustering, path simplification |
| `test_motion_controller.py` | 12 | PID controller, waypoint following |
| `test_synthetic_stereo.py` | 8 | Triangulation formula, scale cross-validation |
| `test_simulator.py` | 34 | All simulator components (world, motion, camera, sensors) |
| `test_integration.py` | 11 | End-to-end VO, stereo, EKF, navigation in sim |
| `test_imu.py` (EKF) | 6 | IMU prediction, mag heading, VO-as-observation |
| `test_imu.py` (mock) | 6 | Mock mode, calibration, data injection |
| **Total** | **172+** | |

### Test Design

- **Synthetic images** generated with OpenCV (checkerboards, random shapes, noise)
- **Mock hardware** via `bot=None` on all hardware abstraction classes
- **Pre-built maps** via pytest fixtures (room grids, landmark databases)
- **No disk/network dependencies** except temp files for persistence tests

---

## 16. ICM-20948 9-DOF IMU (Optional Upgrade)

### Overview

The [Adafruit ICM-20948](https://www.adafruit.com/product/4554) adds 9 axes of
inertial measurement to the rover. This is an **optional upgrade** -- the SLAM system
works without it (proven in simulation), but gains significantly when it is present.

### Hardware

| Sensor | Spec | Rate | SLAM Use |
|--------|------|------|----------|
| 3-axis Gyroscope | ±250 to ±2000 dps | 100 Hz | Primary heading source (replaces VO) |
| 3-axis Accelerometer | ±2g to ±16g | 50 Hz | Dead reckoning between VO frames |
| 3-axis Magnetometer | ±4900 uT | 10 Hz | Absolute compass heading (no drift) |

**I2C address:** 0x69 (no conflict with rover at 0x2B)
**Wiring:** Pi 3.3V → VIN, Pi GND → GND, Pi SCL → SCL, Pi SDA → SDA
**Library:** `pip install adafruit-circuitpython-icm20x`

### Architecture Change

```
Without IMU:                          With IMU:
VO (10 Hz) ──predict──▶ EKF          IMU (100 Hz) ──predict──▶ EKF
                                      VO (10 Hz) ──update──▶ EKF (observation)
                                      Mag (10 Hz) ──update──▶ EKF (heading)
```

With IMU, the gyro becomes the **primary prediction source** at 100 Hz. This is a
fundamental architectural upgrade:

- VO changes from prediction to observation (less trusted, corrective)
- Heading comes from gyro (very accurate short-term) + magnetometer (no drift long-term)
- Accelerometer fills motion gaps between VO frames

### Module: `imu.py`

```python
class IMU:
    def __init__(self, mock_mode=False):     # auto-detect hardware
    def calibrate_gyro(n_samples=200):       # hold still 2 seconds
    def calibrate_magnetometer(duration=15): # rotate slowly for 15 seconds
    def update(self):                        # read sensors, run complementary filter
    def start_background(rate_hz=100):       # threaded reader

    # Properties (thread-safe)
    heading: float      # fused heading (gyro + mag complementary filter)
    gyro_z: float       # yaw rate (rad/s, bias-corrected)
    accel_xy: (float, float)  # horizontal accel (gravity-compensated)
    mag_heading: float  # raw magnetometer heading

    # EKF helpers
    def get_prediction_delta(dt) -> (dx, dy, dtheta)
    def get_heading_observation() -> (heading, noise)
```

### EKF Methods (added to `state_estimator.py`)

| Method | Rate | Purpose |
|--------|------|---------|
| `predict_imu(gyro_z, accel_x, accel_y, dt)` | 100 Hz | High-rate heading + position prediction |
| `update_vo(vo_dx, vo_dy, vo_dtheta)` | 10 Hz | VO as corrective observation |
| `update_heading(mag_heading, noise)` | ~1 Hz | Absolute heading from magnetometer |

### Noise Parameters

| Source | Noise | Compare to VO |
|--------|-------|---------------|
| Gyro heading | 0.001 rad/s | 100x better than VO heading |
| Accelerometer | 0.05 m/s² | Noisy, but fills gaps |
| Magnetometer | 0.05 rad | Absolute (no drift) |
| VO heading | ~0.1 rad | Relative, drifts |

### Complementary Filter

Fuses gyro (fast, accurate short-term, drifts) with magnetometer (slow, noisy,
no drift) using weighted blending:

```
heading = 0.98 * (heading + gyro_z * dt) + 0.02 * mag_heading
```

The 98/2 split means the gyro dominates for fast motion tracking while the
magnetometer slowly corrects drift over seconds.

### Calibration

**Gyro bias** (2 seconds, robot stationary):
- Averages 200 readings to find the zero-rate offset
- Subtracted from all subsequent readings

**Magnetometer hard/soft iron** (15 seconds, rotate robot):
- Collects min/max per axis during rotation
- Hard-iron offset = center of min/max range
- Soft-iron scale = normalize each axis range

### Simulation

`SimIMU` generates realistic sensor data from PyBullet ground truth:
- Gyro: angular velocity from GT + configurable bias + Gaussian noise
- Accel: gravity + body acceleration from velocity differentiation + noise
- Mag: Earth's field rotated by GT heading + noise
