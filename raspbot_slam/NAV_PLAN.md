# Autonomous navigation v2: map-based, contact-aware, self-checking

## Context

Today's tag tour reached 1 of 5 stops. What we learned from real runs (2026-10-11):

| Observation | Evidence | Consequence for design |
|---|---|---|
| Purely reactive driving (look ahead, detour ±50°) bounces between furniture | tour 195634: 3 legs "blocked 5 times" | needs a **persistent map + path planner** |
| Standpoints 0.8 m in front of tags were inside furniture | legs to tags 3, 1, 0 | choose **viewpoints from the map** (free, reachable, line of sight) |
| Low obstacles (desk ring base) invisible to the sonar; overhangs invisible to the camera floor model | 48 s push under the round desk | fuse both, plus a **contact sensor** |
| Pushing furniture is detectable: floor odometry ~0 while driving; accel spikes 25-35° "tilt" in bumps | stall detector worked; imu.csv | bump/stall → mark obstacle, never push > 1 s |
| Gyro heading excellent normally (360° closes to 1°, 0.4° vs tags) but once lost ~170° during repeated collisions | tags 1 vs gyro, 80-102 s | heading must be **checked against tag fixes**, with a reset when they disagree consistently |
| Tag fixes beyond ~1.2 m / few frames can be off 0.5 m | leg 1 fix z 0.155, n 2 | tighter gates; fixes are **corrections with uncertainty**, not truth |
| Camera mount is fragile (pan slipped 38°, re-seat moved it 19° and pitch 4°) | align_check, line run | **startup self-check** every session |
| Clean floor odometry 2-3%; with bumps ~20%/m | eval_tags | dead reckoning OK between fixes if collisions are avoided |
| Carpet: no strafing, hops < 0.2 s don't move, turn-in-place fine | earlier runs | motion primitives = turn-in-place + heading-held hop |
| Compass useless indoors (offset varies 160° across the room) | compass vs tag fixes | don't use it |

Tools (fixed): Pi 5, USB camera 68° HFOV at 10 Hz, pan/tilt servos with slack, one forward sonar (narrow beam, 0.05-3 m), LSM9DS1 gyro/accel, 4 mecanum wheels without encoders, 6 surveyed ArUco tags, offline replay on the PC.

Goal: reach every tag viewpoint without pushes; dead-reckoning error under 15 cm at each stop (no correction); then explore and map the room, scored against the survey.

## Design

One loop at each stop: **sense → update map and pose → plan → one hop → check progress**.

### 1. Startup self-check — `raspbot_slam/selfcheck.py` (new)
- gyro bias; `align_check.measure` (reuse `align_check.py`) → refuse to run if > 5° off
- if a tag is within 1.2 m: compare tag PnP pitch and height with `config.CAMERA_PITCH_DEG`/`CAMERA_HEIGHT_M`; warn if pitch > 1.5° off
- writes `selfcheck.json` into the run folder

### 2. Contact sensing — `raspbot_slam/contact.py` (new)
- `ContactMonitor` thread on `rig.imu`: a bump is a horizontal accel spike above a threshold fitted from the recorded runs (tour 195634 has labelled pushes) while motors are commanded
- stall = floor odometry < 3 cm over a hop that should cover 14 cm (existing logic from `tag_tour.go_to`, moved here)
- either → stop at once, mark the cells ahead as lethal, back off 0.15 m

### 3. Room-frame map — `raspbot_slam/room_map.py` (new; one class used live and offline)
- log-odds grid, 4 cm cells, room frame; **seeded from the survey** (`surveys/office.json` walls and alcove as lethal)
- inputs:
  - camera free-space rays (`free_space.FreeSpaceScanner.scan`)
  - sonar as a 15° cone, not one ray
  - contact cells (lethal, decay slowly)
  - traversed footprint (free)
- reuse `offline/build_map.Grid.ray` logic; replace the copy in `offline/eval_tags.py` with this class
- `costmap()` = obstacles inflated by robot radius 0.13 m plus a 0.07 m soft margin; unknown = passable but costly

### 4. Pose estimate — `raspbot_slam/localizer.py` (new; wraps `online_odometry.OnlineOdometry`)
- prediction: gyro heading and floor odometry, using a heading history for the 75 ms camera latency (pending item)
- 3-state EKF (x, y, θ), growing covariance per metre and per turn
- tag update through `TagLocalizer` with tighter gates: range ≤ 1.5 m, dz ≤ 0.03, n ≥ 3 frames, Mahalanobis gate
- **heading watchdog**: if three consecutive gated fixes disagree with the gyro by > 15° in the same direction, re-anchor the heading and log `heading_reset` (today's 170° event)
- position "no correction" mode is kept for scoring

### 5. Planner — `raspbot_slam/planner.py` (new)
- A* on the costmap (reuse `explorer.Explorer._astar` and `_simplify_path`)
- `tag_viewpoint(tag)`: candidate cells 0.5–1.4 m in front of the tag, within ±35° of its normal. Choose the lowest path cost among cells that are free and have a clear line of sight on the map. Replaces `tag_tour.standpoint`.
- `frontier_goal()`: reuse `explorer.select_target` on the room map

### 6. Executor — `raspbot_slam/nav_executor.py` (new; replaces `tag_tour.go_to`)
- each stop:
  - pan sweep at −40°, 0° and +40° for camera free space (approach the centre from below)
  - sonar read
  - map update
  - replan
- turn to the first waypoint with `heading.turn_to`, then one `heading.drive_straight` hop of 0.2–0.5 s, never further than the free distance minus 0.25 m
- contact or stall → map, back off, replan; give up a goal after 3 contacts or 60 hops

### 7. Missions
- `tag_tour.py` v2: the route uses `tag_viewpoint`; scores as now (`tourstop` events)
- `explore.py`: frontier exploration until no frontier remains or 10 minutes; saves `room_map.npz` and PNG

## Milestones (each: implement → offline replay test → one robot run → score)

| # | Deliverable | Pass criterion |
|---|---|---|
| N0 | selfcheck, tighter tag gates | selfcheck catches a deliberately offset pan (+10) |
| N1 | ContactMonitor | on tour 195634 replay, flags every push (frames 50-61 s, 77-83 s); no false bumps on line runs |
| N2 | RoomMap, survey-seeded; eval_tags uses it | replayed tour map: furniture blobs where the photos show the dresser and desk |
| N3 | EKF and heading watchdog | replay of 195634 detects the 170° heading break; live: 360° spin test unchanged |
| N4 | planner and executor | unit tests on synthetic grids; live: go to a point 3 m away around the dresser with no contact |
| N5 | tag tour v2 | ≥ 5/6 tags reached; median DR error < 15 cm uncorrected; zero pushes > 1 s |
| N6 | explore | map walls within 10 cm of survey; coverage ≥ 80% of reachable floor |

## Files
- new: `selfcheck.py`, `contact.py`, `room_map.py`, `localizer.py`, `planner.py`, `nav_executor.py`, `explore.py`; tests in `tests/test_room_map.py`, `test_planner.py`, `test_contact.py`, `test_localizer.py`
- changed:
  - `tag_tour.py`: uses planner and executor
  - `online_odometry.py`: heading history for latency
  - `tag_localizer.py`: gates
  - `offline/eval_tags.py`: uses RoomMap
  - `config.py`: new tunables
  - `docs/COMMANDS.md` and `AUTONOMY_PLAN.md` updated
- reused: `heading.py`, `floor_odometry.py`, `free_space.py`, `recorder.py`, `explorer.py` (A*, frontiers), `align_check.py`

## Verification
- `python -m pytest tests/ -q` on PC (all existing tests pass)
- offline replays of `datasets/slam_runs/20261011_195634_tagtour` and the line runs for N1–N3
- robot runs: selfcheck → `tag_tour` v2 → `offline/eval_tags --map` → `explore`; compare score table with today's (1/5 stops, 0.42 m median)
- commit per milestone on `slam-autonomy-m0-m2`, push, `git pull` on the Pi
