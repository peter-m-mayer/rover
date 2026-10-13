# Autonomous Mapping + Cat Play — Plan (2026-10-11)

Goal: put the rover on the ground floor of the house, have it explore on its
own, and produce (1) a 2-D obstacle map at robot height and (2) a sparse 3-D
point cloud. When it sees a cat, it switches to play for up to 5 minutes,
then goes back to mapping.

This plan builds on [SLAM_REVIEW.md](SLAM_REVIEW.md) and [SLAM_PLAN.md](SLAM_PLAN.md).
It differs from them in two ways:

- **Hardware first.** The IMU is now fitted, so the review's biggest gaps
  (no metric scale, no real heading) can be closed with real sensors rather
  than in simulation. Record-and-replay logging comes first, and every
  algorithm is developed offline against real recorded runs.
- **A simpler core.** The core becomes 2-D pose-graph SLAM plus an occupancy
  grid, instead of EKF landmark SLAM. The EKF needs the per-landmark camera
  measurements that never got wired in (review #3). A pose graph only needs
  odometry between keyframes and occasional loop closures, which this
  hardware can produce reliably. We keep the occupancy grid, frontier
  explorer, A*, and the mock and simulator test pattern.

## Navigation v2 (2026-10-12)

The evening tag tour (reactive driving) reached 1 of 5 stops: fixed standpoints were inside furniture and
look-ahead-and-detour bounced between obstacles. Plan and lessons: **[NAV_PLAN.md](NAV_PLAN.md)**. Built:
`room_map.py` (survey-seeded log-odds map, sonar cone, contacts, costmap), `planner.py` (Dijkstra cost field,
clearance-checked shortcuts, tag viewpoints), `localizer.py` (Kalman updates from gated tag fixes, heading
watchdog, position recovery), `nav_executor.py` (sense -> map -> plan -> one hop -> stall check), tag_tour v2.
Offline replay of the evening tour: the watchdog re-anchors the 174 deg heading break at 81 s. Findings:
- accelerometer spikes do NOT separate furniture contact from mecanum vibration (clean runs peak as high):
  contact = floor odometry not moving during a commanded hop;
- per-tag heading offsets: tags 1/4/5 agree with each other; tag 2 (tonight) and tag 0 (afternoon, n=2) sat
  ~180 deg away. Unresolved: gyro slip vs survey. eval_tags now prints the per-tag check every run.
- not yet run live (room dark, robot charging).

Update, later on 2026-10-12:
- **The 180 deg mystery was the survey.** Its frame was a mirror image of the room (left-handed). With
  y flipped (`surveys/office.json` `frame_fix`), every tag agrees with the gyro. Replay of the day's runs needs
  no heading resets, and the median error at fixes is 4 cm.
- **Live tours.** The best tour (20261012_090828) reached 2 of 5 tags, with errors of 0.30 m and 0.14 m.
- **Executor fixes since then:**
  - sonar blocks are marked on the map;
  - a relaxed escape plan when boxed in;
  - a carpet start-lag hop model (0.28 m/s × (t − 0.12 s), minimum 0.35 s);
  - stalled-turn detection.

## Mission "learn the room" (2026-10-12 evening)

Goal: each run improves the next: **run → automatic score → merge into a persistent map → refit the models →
next run starts from that knowledge**. It succeeds when a tour after an explore run beats a tour before it.
- `room_map.py`: persistent obstacles. A cell hit by the camera at least 3 times, from two places, in ≥ 25% of
  its observations is solid until the robot drives over it, so furniture no longer erodes into thin edges.
  `save()` keeps the counts, and contacts are saved as solid. `load_prior()` starts a run from a saved map.
- `offline/score_run.py`: reached, stop error, contacts, blocks, stalled turns, give-ups by reason, distance,
  coverage and wall offset, written to `score.json` and `datasets/slam_runs/scores.csv`. Tours and explore runs
  print it at the end.
- `offline/merge_runs.py`: replays runs (tag-corrected) into one prior map, `maps/office_prior.npz`.
- `offline/calibrate.py`: refits the hop speed and start lag, tag-fix noise and dead-reckoning drift. Values go
  to `calibration/nav_calibration.json`, which `config.py` reads, and only within sanity bounds.
- `explore.py`: drives to the reachable place that would show the most unseen floor per metre of path, then
  looks all round. It detours to a tag when the position sigma passes 0.15 m.
- Safety:
  - SIGINT and SIGTERM stop a mission cleanly. Jobs started with `nohup ... &` ignore SIGINT, which is why
    `pkill -INT` failed.
  - `touch /tmp/rover_stop` also stops a mission.
  - Four troubles in one spot with an unchanging sonar range count as **wedged**: stop and ask for a hand.
- Runs: baseline tour → explore → merge + calibrate → tour with `--prior` (same route `1,0,4,3,2`).

## Status (2026-10-11, end of first autonomous session)

| Item | Result | Where |
|---|---|---|
| Recorder (frames 10 Hz, IMU 100 Hz, sonar/IR 10 Hz, commands, power) | done, replay reader | `recorder.py` |
| IMU axes / compass | identity axis map; compass 1.01x gyro after flipping mag X **and** Y | `lsm9ds1.py`, `calibration/imu_calibration.json` |
| Camera intrinsics | checkerboard, self-driven incl. circle moves for oblique views: f 471/469 px, k1 -0.42, **0.17 px RMS** over 28 views (classic corner detector mis-assigned far views -> SB detector). Interim tag result agreed (473 px) | `calibration/auto_camera_cal.py` |
| Camera height / pitch | 0.107 m (tag PnP); pitch +0.45 deg at tilt 30, fitted on sonar runs; **4.0 deg after the 2026-10-11 evening re-seat** (pan centre 74 -> 93) | `config.CAMERA_HEIGHT_M/PITCH_DEG` |
| Tag localiser | ~5 cm / <1 deg within ~1.5 m of a tag; far single tags rejected | `tag_localizer.py` |
| Motor dropouts | vendor lib swallowed I2C errors; a lost "stop" turned 10 deg into 43 deg. Now retried, raises if the bus stays down | `actuators.py` |
| Heading hold | carpet veer was 17-36 deg/m; with gyro hold < 5 deg | `heading.py` |
| Gyro heading | <1 deg error per full turn (vs camera and tags) | |
| Floor odometry (gyro rotation + floor translation) | **M1 met**: distance within 4% of the sonar on two runs (+0.2/+0.7%, -1/-3.9%) | `floor_odometry.py`, `offline/eval_odometry.py` |
| Strafing on carpet | barely moves (10 cm per 9 bursts) and turns 23 deg: plan as forward + turn only | |
| Free-space scan | texture-based (colour off until exposure is locked); close walls match sonar to ~2 cm | `free_space.py` |
| First maps | local occupancy maps from a line run and a 6-leg wander | `offline/build_map.py`, `wander.py` |

**Tag self-test (2026-10-11, afternoon):**
- First legs: dead reckoning vs tag fixes **3 cm / 2.8 deg and 9 cm / 0.8 deg**.
- Then the camera head jammed under the round desk (sonar can't see its low ring
  base; no stuck detection yet) and the **pan horn slipped ~38 deg** on its spline.
  Tags and gyro still agreed on heading, but floor odometry measured motion in a
  rotated frame, so the second tour's dead reckoning drifted out of the room.
- Found and fixed along the way: tag fixes now drop far tags, reject poses outside
  the room and require multi-tag agreement (heading vs gyro: median 0.4 deg over
  370 fixes); camera frames lag the gyro by **75 ms** (`CAMERA_LATENCY_S`);
  stuck detection; floor model learned from the start; fixes taken with the pan
  centred, turning the body instead of trusting the pan servo.
- New: `align_check.py` (camera vs body from a straight drive),
  `offline/eval_tags.py` (dead reckoning vs tags, map over the survey).
- **Hardware to do:** re-seat and tighten the pan horn; then `align_check` and re-run
  `tag_tour`.

**Next:** lock camera exposure / white balance for mapping (then re-enable colour);
checkerboard calibration with the self-driving routine; tag fixes during the wander
(multi-tag, wider pan sweep) so maps land in survey coordinates; pose graph.

## What the robot can sense (and can't)

| Sensor | Gives | Limits |
|---|---|---|
| USB camera 640×480, fx≈425 (~74° HFOV), on pan/tilt | Floor motion, free-space edges, features, cats | Monocular, so scale must come from the floor plane. Blur on fast turns. |
| LSM9DS1 gyro | Heading rate, ~0.1 dps noise | Bias drifts, so re-estimate it at every stop |
| LSM9DS1 magnetometer | Absolute heading | Disturbed indoors by steel and appliances. Hint only. |
| Ultrasonic (fixed, forward) | Range 2 cm–3 m, ~15° cone | One beam. Misses soft or angled things. |
| 4 downward IR line sensors | Reflectance under the chassis | **Candidate cliff/stair detector** (needs a test) |
| Motor commands | Rough speed prior | No encoders. Mecanum wheels slip on carpet. |

The honest expectation: room-scale 2-D maps accurate to about 10–15 cm, and a
whole ground floor that stays consistent **if** there are loop closures
(markers, or revisiting places). Hard cases are glass doors, black furniture,
thin chair legs, dark rooms, and blank walls.

## Architecture

```
camera ─┬─ floor odometry (ground-plane homography, bottom of image) ─┐
        ├─ free-space scan (floor/obstacle boundary per column)       │
        ├─ ArUco markers + keyframe features (loop closure)           ├─► pose graph ─► occupancy grid ─► explorer / A*
        └─ cat detector ───────────────────────► mission state machine│      (scipy)        (log-odds)
gyro ── heading (bias updated at every stop) ─────────────────────────┘
ultrasonic ─ obstacle range + safety stop      IR ─ cliff stop      IMU ─ stuck detection
```

- **Floor odometry.** The camera height and tilt are fixed, and the
  floor is a plane, so floor features tracked between frames give **metric**
  translation directly. This was the review's §4B option 1. Carpet texture
  suits it. The gyro supplies the rotation, so the floor tracker only
  solves for translation, which is a much better-conditioned problem.
- **Free-space "pseudo-laser".** The floor directly in front of the robot
  serves as a colour and texture model. In each image column, the first row
  where the image stops looking like floor is an obstacle. With known camera
  height that row maps to a distance. Panning the camera or spinning in place
  turns this into a 360° scan. This is the classic Ulrich–Nourbakhsh method,
  and it gives the occupancy grid far more than one ultrasonic beam can.
- **Pose graph.** Keyframes come every ~30 cm or 20°. Edges come from
  odometry, plus loop closures from markers or matched keyframe features.
  `scipy.optimize.least_squares` re-solves the graph, and the grid is
  re-rendered from the corrected poses.

## Milestones

Each milestone ends with a measured pass/fail against ground truth that you
supply. The loop is the same each time:

1. You run a hardware session.
2. I pull the logs and replay them on the PC.
3. I fix the code and score the result.
4. Next session.

### M0 — Instrument and calibrate (today)

| Who | Task |
|---|---|
| me | Finish the IMU setup: `imu_check --axes --write`, then `--mag`. Needs about 1 m of clear floor. |
| me | **Logger.** Build `raspbot_slam/recorder.py`: timestamped frames, IMU at 100 Hz, ultrasonic, IR, motor commands, servo angles, battery. Store as one folder per run. Write a matching replay reader. |
| you | Print a checkerboard (I'll generate the page). Hold it at ~15 poses in front of the camera. Gives camera intrinsics (review H1a). |
| me | Measure the camera's height and tilt above the floor by viewing a checkerboard laid flat on the floor. |
| you | Print 4–6 **ArUco markers** (I'll generate them). Tape them low on walls or furniture (10–25 cm up), at positions you measure from a room corner. Tape two floor marks: a start pose and a 1 m square. |
| you | A rough sketch of the office with dimensions and furniture footprints. A phone photo of a pencil drawing is fine. This is the ground-truth map. |

*Pass:* camera reprojection error < 0.5 px. IMU axes and compass written to config.

### M1 — Honest odometry (gyro + floor)

I'll drive slow scripted moves while logging:
- 1 m forward and back
- 0.5 m strafes
- a 360° spin
- the 1 m square, twice

Then offline: gyro heading with the bias taken at stops, plus ground-plane
translation.

*Pass:* distance within 5%, a 360° spin within 3°, and the square closes
within 10 cm and 5°. These are the review's H3/H4 hardware tests, made tighter
now that the IMU is fitted.

### M2 — 2-D map of one room

Free-space scans plus ultrasonic feed the occupancy grid from odometry poses.
At each stop the robot spins once or sweeps the pan. ArUco sightings are added
as pose-graph constraints.

*Pass:* walls and furniture edges within 15 cm of your sketch, no phantom
obstacles in open floor, and doorways show as open.

### M3 — Autonomous exploration of one room, then the ground floor

Frontier exploration (`explorer.py`) plus A* on the live grid. Safety layer:
- ultrasonic stop
- IR cliff stop (tested at a real step first)
- stuck detection (motors driving but the gyro and floor show no motion)
- a low-battery return

You draw no-go zones on the map (stairs, the litter box, the cables behind
the desk). It ends with a return to start.

*Pass:*
- the office is explored to completion without help
- the robot is back at start within 20 cm
- zero collisions it didn't see coming

Then the ground floor, with markers in each room for the first runs and fewer
once keyframe loop closure works.

### M4 — 3-D (sparse point cloud)

- **Online (Pi).** ORB features on keyframes, triangulated using the metric
  poses from M1–M3. This gives a sparse 3-D cloud in metres.
- **Offline (PC).** Use SIFT, which is in mainline OpenCV now that its patent
  has expired, plus full bundle adjustment over the logged keyframes. This
  gives a denser and more accurate cloud. `offline/bundle_adjustment.py`
  already exists as a starting point. SURF is non-free and not in the
  OpenCV build we use, and SIFT is the better choice anyway.
- **Optional.** A small monocular depth network (e.g. Depth Anything V2-Small
  as ONNX; ~1 s/frame on the Pi, faster on the PC), with its scale fixed by
  the known floor plane. Gives dense depth and obstacle heights (table edges,
  overhangs the robot could get stuck under).

I'll compare SIFT and ORB on your rooms by match count and how much
bundle adjustment reduces the error. The results go in an interactive 3-D
viewer page.

### M5 — Cat behaviour

A mission state machine on top of exploration:

```
EXPLORE ──cat seen (N frames)──► PLAY (existing chase --prey --pan)
   ▲                                │ 5 min wall clock, OR cat lost > 30 s, OR battery low
   │                                ▼
   └──── RESUME ◄── RELOCALIZE (gyro kept heading; markers/keyframes fix position)
```

- During play, the chase controller respects the map's no-go zones and cliff
  stops. The cat does not get chased down the stairs.
- After play there's a **cooldown**: default 15 min before it will play again,
  adjustable. Otherwise it would never finish mapping in a house with cats.
- Every cat sighting is pinned on the map with a time, giving a "where the
  cats hang out" layer for free.
- Play is the worst case for odometry (fast spins and lunges). The gyro keeps
  the heading, and the position is re-anchored by relocalisation before
  exploration resumes.

## Iteration loop and tooling

- Every run is logged, so every idea is tested by **replay on the PC** first.
  A physical run is spent only to collect new data or to confirm a fix.
- Scores per run are saved in `datasets/slam_runs/<run>/score.json`:
  - odometry error vs marks
  - map error vs sketch
  - explored area
  - collisions
  
  This shows whether a change actually helped.
- The simulator stays useful for logic (explorer, state machine, safety), but
  not for proving accuracy. SLAM_PLAN Phase 1 (remove the ground-truth
  anchors) is still worth doing for that.

## Today, concretely (carpeted office)

1. Charge, then the IMU calibration (~5 min of me driving).
2. You: print the checkerboard and markers, tape the marks, sketch the room.
   Meanwhile I write the logger and the calibration scripts.
3. Camera intrinsics, plus height and tilt (~10 min, with you holding the
   board).
4. I drive the M1 script and a slow lap of the office with spins at a few
   stops. No autonomy yet, just recording.
5. Offline: odometry score, then the first occupancy grid drawn over your
   sketch.

A realistic outcome today is M0 done, M1 measured, and a first M2 map.
Autonomy (M3) comes in a following session, after the map is trustworthy.
