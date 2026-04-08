# RASPBOT-V2 Visual SLAM

Indoor floor mapping and autonomous navigation for the [Yahboom RASPBOT-V2](https://www.yahboom.net/study/RASPBOT-V2) rover on Raspberry Pi 5.

## What This Does

The rover autonomously explores your house, builds a map of the floor plan using its single camera, and then navigates to commanded locations using that map. No lidar, no encoders, no IMU -- just a camera, an ultrasonic sensor, and mecanum wheels.

### Key Idea: Synthetic Stereo

Since the rover has mecanum wheels that can strafe laterally without rotating, it creates **synthetic stereo vision** by capturing two images with a known baseline:

```
1. Capture LEFT image
2. Strafe RIGHT 5 cm (known distance)
3. Capture RIGHT image
4. Triangulate: depth = focal_length × baseline / disparity
```

This resolves the scale ambiguity that normally plagues monocular SLAM.

## Architecture

```
Camera (640x480) → ORB Features → Visual Odometry (Essential matrix)
                                        ↓
Synthetic Stereo (mecanum strafe) → EKF-SLAM ← Ultrasonic
                                        ↓
                                   Map Manager
                                   ├─ 3D Landmarks (ORB descriptors)
                                   └─ 2D Occupancy Grid (5 cm cells)
                                        ↓
                              Explorer / Navigator
                                        ↓
                              PID + Mecanum Motors
```

## Three Ways to Run

### On the physical rover (Pi 5)

```bash
# Map your house
python -m raspbot_slam.run_mapping --map-name ground_floor

# Navigate to a location
python -m raspbot_slam.run_navigation --map-name ground_floor --goal 3.0 2.5
```

### In the PyBullet simulator (any machine)

```bash
# Map a simulated L-shaped room (headless, fast)
python -m raspbot_slam.simulator.run_sim --floor-plan L_shaped

# Map with 3D visualization
python -m raspbot_slam.simulator.run_sim --floor-plan two_rooms --gui

# Available floor plans: simple_room, L_shaped, corridor, two_rooms
```

### Visualize a saved map

```bash
python -m raspbot_slam.visualize_map maps/ground_floor
python -m raspbot_slam.visualize_map maps/ground_floor output.png  # save to file
```

## Project Structure

```
rover/
├── README.md                         # This file
├── ROVER_OVERVIEW.md                 # Yahboom vendor codebase inventory
├── .gitignore
│
├── raspbot_slam/                     # Main package (7,829 lines across 42 files)
│   ├── ARCHITECTURE.md               # Full design document (1,120 lines)
│   ├── config.py                     # All tunable parameters in one place
│   │
│   ├── camera.py                     # Capture, calibration, undistortion
│   ├── sensors.py                    # Ultrasonic, line tracker
│   ├── actuators.py                  # Servos, mecanum motors, LEDs
│   │
│   ├── feature_extractor.py          # ORB detect/compute/match + RANSAC
│   ├── visual_odometry.py            # Frame-to-frame VO, keyframe management
│   ├── synthetic_stereo.py           # Mecanum strafe → triangulated depth
│   ├── state_estimator.py            # EKF-SLAM with bounded active landmarks
│   ├── map_manager.py                # Landmark DB + occupancy grid
│   │
│   ├── explorer.py                   # Frontier-based exploration + A*
│   ├── navigator.py                  # Relocalization + goal navigation
│   ├── motion_controller.py          # PID waypoint following
│   │
│   ├── simulator/                    # PyBullet digital twin
│   │   ├── sim_world.py              # Physics world, floor plans, kinematic rover
│   │   ├── sim_camera.py             # Rendered camera with pan/tilt + depth
│   │   ├── sim_sensors.py            # Ultrasonic via ray-cast
│   │   ├── sim_actuators.py          # Motor commands → kinematic motion
│   │   └── run_sim.py                # Entry point for simulated SLAM
│   │
│   ├── calibration/                  # One-time setup tools
│   │   ├── camera_calibrate.py       # Checkerboard → intrinsics
│   │   └── strafe_calibrate.py       # Motor speed → distance LUT
│   │
│   ├── offline/                      # Post-mapping optimization
│   │   ├── bundle_adjustment.py      # scipy least-squares BA
│   │   ├── loop_closure.py           # SIFT re-extraction + matching
│   │   └── map_optimizer.py          # Outlier removal, trajectory smoothing
│   │
│   ├── run_mapping.py                # Entry point: autonomous mapping
│   ├── run_navigation.py             # Entry point: goal navigation
│   ├── visualize_map.py              # matplotlib map viewer
│   │
│   ├── drivers/                      # Yahboom hardware drivers (unmodified)
│   └── maps/                         # Stored map data
│       └── <name>/
│           ├── metadata.json
│           ├── landmarks.pkl
│           ├── occupancy_grid.npy
│           ├── trajectory.npy
│           └── keyframes/
│
├── tests/                            # Offline test suite (155 tests)
│   ├── conftest.py                   # Shared fixtures, synthetic images
│   ├── test_config.py                # Parameter validation (10)
│   ├── test_camera.py                # Calibration, undistortion (7)
│   ├── test_sensors_actuators.py     # Mock hardware (17)
│   ├── test_feature_extractor.py     # ORB detect/match/RANSAC (14)
│   ├── test_state_estimator.py       # EKF predict/update/landmarks (16)
│   ├── test_map_manager.py           # Occupancy grid, landmarks, I/O (20)
│   ├── test_explorer.py              # A*, frontiers, path planning (11)
│   ├── test_motion_controller.py     # PID, waypoint following (12)
│   ├── test_synthetic_stereo.py      # Triangulation math, scale (8)
│   └── test_simulator.py             # World/motion/camera/sensors (34)
│
├── RaspbotV2-Code/                   # Yahboom vendor code (not tracked in git)
└── Raspbot_V2-Manual/                # Yahboom hardware manual (not tracked)
```

## Hardware

| Component | Spec | Used For |
|-----------|------|----------|
| Camera | USB, 640x480 | Visual odometry, feature extraction |
| Pan servo | 0-180° | Extend FOV during scan stops |
| Tilt servo | 0-110° | Camera angle adjustment |
| 4 mecanum wheels | ±255 speed, no encoders | Omnidirectional motion, lateral strafe |
| Ultrasonic | mm resolution, forward-facing | Obstacle detection, scale validation |
| 14 WS2812B LEDs | RGB addressable | Status indication |
| Buzzer | on/off | Audio feedback |

## Simulator (Digital Twin)

The PyBullet simulator provides a drop-in replacement for real hardware. The same SLAM code runs identically -- only the hardware backend is swapped:

```python
# Real hardware
from raspbot_slam.camera import Camera
from raspbot_slam.sensors import Sensors
from raspbot_slam.actuators import Actuators

# Simulated (same interface)
from raspbot_slam.simulator import SimCamera as Camera
from raspbot_slam.simulator import SimSensors as Sensors
from raspbot_slam.simulator import SimActuators as Actuators
```

**4 floor plans** with textured walls, furniture, and correct physics:

| Floor Plan | Description | Dimensions |
|------------|-------------|------------|
| `simple_room` | Single room | 4m × 4m |
| `L_shaped` | L-shaped room with furniture | 6m × 4m |
| `corridor` | Narrow corridor | 6m × 1.5m |
| `two_rooms` | Two rooms connected by doorway | 6m × 3m |

**Simulator features:**
- Kinematic mecanum rover (validated: 0.314m in 2s at speed=80)
- Textured walls producing 500+ ORB features per frame
- Ultrasonic via ray-cast with configurable Gaussian noise
- Depth camera for ground-truth comparison
- Pan/tilt servo simulation
- Ground-truth pose for error measurement

## Computational Budget (Pi 5)

### Real-time (per frame, 10 FPS target)

| Step | Time | Notes |
|------|------|-------|
| Frame capture + grayscale | 3 ms | |
| ORB detect + compute (1000 features) | 20 ms | 640×480 |
| Feature matching + ratio test | 10 ms | BFMatcher |
| Essential matrix + pose | 7 ms | RANSAC |
| EKF predict + update (30 landmarks) | 3 ms | 94×94 covariance |
| Occupancy grid + control | 2 ms | |
| **Total** | **~45 ms** | **55 ms headroom** |

### Periodic stops

| Operation | Duration |
|-----------|----------|
| Synthetic stereo (strafe + capture + triangulate) | ~1 sec |
| Pan sweep (5 positions) | ~2 sec |
| Full scanning stop | ~3-5 sec |

### Offline (post-mapping)

| Operation | Duration |
|-----------|----------|
| SIFT re-extraction (500 keyframes) | ~90 sec |
| Loop closure detection | ~5 min |
| Bundle adjustment | 1-10 min |

## Testing

All tests run offline without hardware:

```bash
# Run full suite (155 tests, ~3 seconds)
python -m pytest tests/ -v

# Run just SLAM module tests (121 tests)
python -m pytest tests/ -v --ignore=tests/test_simulator.py

# Run just simulator tests (34 tests)
python -m pytest tests/test_simulator.py -v

# Run a single test file
python -m pytest tests/test_state_estimator.py -v
```

**Test coverage:**

| Area | Tests | What's verified |
|------|-------|-----------------|
| Config | 10 | Parameter ranges and consistency |
| Camera | 7 | Calibration I/O, undistortion math |
| Hardware mock | 17 | Sensors + actuators with `bot=None` |
| ORB features | 14 | Detection, matching, RANSAC, utilities |
| EKF-SLAM | 16 | Predict, update, gating, landmark lifecycle |
| Map | 20 | Occupancy grid, landmarks, persistence |
| Explorer | 11 | A*, frontiers, path simplification |
| Motion control | 12 | PID, waypoint following, angle normalization |
| Stereo depth | 8 | Triangulation formula, scale cross-validation |
| Simulator | 34 | World/motion/camera/sensors/actuator interface |
| **Total** | **155** | |

## Getting Started

### Prerequisites

```bash
pip install opencv-python-headless numpy scipy matplotlib pybullet
```

On the Pi 5, also install the I2C driver:
```bash
pip install smbus2
cd "RaspbotV2-Code/Python driver library/py_install"
sudo python3 setup.py install
```

### Quick start with the simulator (no hardware needed)

```bash
# Clone the repo
git clone https://gitlab.com/peter-m-mayer/rover.git
cd rover

# Run tests
pip install opencv-python-headless numpy pybullet pytest
python -m pytest tests/ -v

# Run a simulated mapping session
python -m raspbot_slam.simulator.run_sim --floor-plan L_shaped
```

### On the physical rover

#### 1. Calibrate the camera

Print a 9×6 checkerboard and run:
```bash
python -m raspbot_slam.calibration.camera_calibrate
```

#### 2. Calibrate strafe distance

```bash
python -m raspbot_slam.calibration.strafe_calibrate
```

#### 3. Map a room

```bash
python -m raspbot_slam.run_mapping --map-name my_house
```

Press `Ctrl+C` to stop and save at any time.

#### 4. Visualize the map

```bash
python -m raspbot_slam.visualize_map maps/my_house
```

#### 5. Navigate to a goal

```bash
python -m raspbot_slam.run_navigation --map-name my_house --goal 3.0 2.5
```

#### 6. Offline optimization (optional)

```bash
python -m raspbot_slam.offline.loop_closure maps/my_house
python -m raspbot_slam.offline.bundle_adjustment maps/my_house
python -m raspbot_slam.offline.map_optimizer maps/my_house
```

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **ORB over SIFT** for real-time | 10× faster on CPU. SIFT used offline only for loop closure. |
| **EKF over particle filter** | Deterministic cost with bounded landmarks. 30 active → 94-dim state → 0.1 ms update. |
| **VO is ground truth** | No encoders means motor commands are suggestions. VO measures what actually happened. |
| **Stop-and-scan for depth** | Continuous VO gives direction; periodic synthetic stereo gives absolute scale. |
| **Kinematic simulation** | Direct position integration (not force-based) for reliable, predictable digital twin. |
| **Dual map representation** | Sparse landmarks for localization, occupancy grid for navigation. |

## Documentation

| Document | Description |
|----------|-------------|
| [ARCHITECTURE.md](raspbot_slam/ARCHITECTURE.md) | Full design: algorithms, state vectors, module APIs, calibration, verification plan (1,120 lines) |
| [ROVER_OVERVIEW.md](ROVER_OVERVIEW.md) | Inventory of the Yahboom vendor codebase: drivers, demos, hardware registers |

## Repository

- **GitLab:** https://gitlab.com/peter-m-mayer/rover
- **Platform:** [Yahboom RASPBOT-V2](https://www.yahboom.net/study/RASPBOT-V2)
- **Stats:** 42 Python files, 7,829 lines of code, 155 tests
