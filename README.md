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

## Two Operating Modes

**Mapping** -- Autonomous exploration using frontier-based strategy. Stops every 50 cm for a pan sweep + stereo depth capture. Builds the map from scratch.

```bash
python -m raspbot_slam.run_mapping --map-name ground_floor
```

**Navigation** -- Loads a stored map, relocalizes via panoramic feature matching + PnP, then navigates to a goal using A* path planning.

```bash
python -m raspbot_slam.run_navigation --map-name ground_floor --goal 3.0 2.5
```

## Package Structure

```
raspbot_slam/
├── config.py                 # All tunable parameters
│
├── camera.py                 # Capture, calibration, undistortion
├── sensors.py                # Ultrasonic, line tracker
├── actuators.py              # Servos, mecanum motors, LEDs
│
├── feature_extractor.py      # ORB detect/compute/match, RANSAC
├── visual_odometry.py        # Frame-to-frame VO, keyframe management
├── synthetic_stereo.py       # Strafe-based depth estimation
├── state_estimator.py        # EKF-SLAM (bounded 30 active landmarks)
├── map_manager.py            # Landmark DB + occupancy grid
│
├── explorer.py               # Frontier-based exploration + A*
├── navigator.py              # Relocalization + goal navigation
├── motion_controller.py      # PID waypoint following
│
├── calibration/
│   ├── camera_calibrate.py   # Checkerboard intrinsic calibration
│   └── strafe_calibrate.py   # Motor speed → distance LUT
│
├── offline/
│   ├── bundle_adjustment.py  # Joint pose + landmark optimization
│   ├── loop_closure.py       # SIFT-based distant keyframe matching
│   └── map_optimizer.py      # Outlier removal, trajectory smoothing
│
├── run_mapping.py            # Entry point: autonomous mapping
├── run_navigation.py         # Entry point: map-based navigation
├── visualize_map.py          # matplotlib map viewer
│
├── maps/                     # Stored map data
│   └── <name>/
│       ├── metadata.json
│       ├── landmarks.pkl
│       ├── occupancy_grid.npy
│       ├── trajectory.npy
│       └── keyframes/
│
├── drivers/                  # Yahboom drivers (unmodified)
└── ARCHITECTURE.md           # Full design document
```

## Hardware

| Component | Spec | Used For |
|-----------|------|----------|
| Camera | USB, 640x480 | Visual odometry, feature extraction |
| Pan servo | 0-180° | Extend FOV during scan stops |
| Tilt servo | 0-110° | Camera angle adjustment |
| 4 mecanum wheels | ±255 speed, no encoders | Omnidirectional motion, lateral strafe |
| Ultrasonic | mm resolution, forward | Obstacle detection, scale validation |
| 14 WS2812B LEDs | RGB | Status indication |

## Computational Budget (Pi 5)

| Per-frame (10 FPS target) | Time |
|---------------------------|------|
| ORB extraction (1000 features) | 20 ms |
| Feature matching + Essential | 17 ms |
| EKF update (30 landmarks) | 3 ms |
| Occupancy + control | 3 ms |
| **Total** | **~43 ms** |

Stereo stop: ~1 sec. Pan sweep: ~2 sec. Offline bundle adjustment: 8-17 min.

## Getting Started

### 1. Calibrate the Camera

Print a 9x6 checkerboard and run:
```bash
python -m raspbot_slam.calibration.camera_calibrate
```

### 2. Calibrate Strafe Distance

```bash
python -m raspbot_slam.calibration.strafe_calibrate
```

### 3. Map a Room

```bash
python -m raspbot_slam.run_mapping --map-name my_house
```

### 4. Visualize the Map

```bash
python -m raspbot_slam.visualize_map maps/my_house
```

### 5. Navigate

```bash
python -m raspbot_slam.run_navigation --map-name my_house --goal 3.0 2.5
```

### 6. Offline Optimization (Optional)

```bash
python -m raspbot_slam.offline.loop_closure maps/my_house
python -m raspbot_slam.offline.bundle_adjustment maps/my_house
python -m raspbot_slam.offline.map_optimizer maps/my_house
```

## Dependencies

**On the Pi 5:**
```
opencv-python >= 4.5
numpy
scipy (for offline bundle adjustment)
matplotlib (for visualization, optional)
smbus (for I2C hardware control)
```

**For development (any machine):**
All modules support mock hardware -- pass `bot=None` to hardware abstraction classes.

## Key Design Decisions

- **ORB over SIFT** for real-time: 10x faster, adequate for frame-to-frame tracking. SIFT used offline only for loop closure.
- **EKF over particle filter**: Deterministic compute cost with bounded landmarks. 30 active landmarks → 94-dim state → 0.1 ms update.
- **VO is ground truth**: With no encoders, motor commands are suggestions. Visual odometry measures what actually happened.
- **Stop-and-scan**: Real-time VO provides direction; periodic stops provide absolute depth via synthetic stereo.

## Documentation

See [`ARCHITECTURE.md`](raspbot_slam/ARCHITECTURE.md) for the full design document including algorithm details, state vector definitions, calibration procedures, and verification plan.

See [`ROVER_OVERVIEW.md`](ROVER_OVERVIEW.md) for an inventory of the Yahboom vendor codebase.
