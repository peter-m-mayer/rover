# raspbot_slam — Critical Review and Test Plans

Reviewed at commit `4e63bd2` (github.com/peter-m-mayer/rover, main), 2026-09-06.
Method: ran the full test suite and the PyBullet simulator in a clean environment, then read the estimation chain line by line (feature_extractor → visual_odometry → synthetic_stereo → state_estimator → run_mapping → navigator) and wrote targeted experiments against simulator ground truth for anything I suspected. Every finding below says whether it is **verified** (reproduced by running code), **read** (established from the source, not executed), or **suspected** (needs a test).

## 1. Verdict

The architecture is sound and the EKF is well implemented, but the system as it stands **cannot produce a metric map on hardware**, and the simulator results that suggest otherwise are anchored to ground truth. The root cause is not any one bug but a missing link: nothing in the hardware path ever converts visual odometry into metres, and the EKF never receives a real camera measurement. Fixing that is a redesign of the estimation core (a few hundred lines, not a rewrite), and it is a well-understood one. Everything downstream — occupancy grid, frontier exploration, A*, motion control, relocalization by PnP — is in reasonable shape and worth keeping.

Confidence in that verdict: high (~90%). The four showstoppers are each independently verified or read directly from the code.

Relevance to the cat chaser: unchanged — SLAM is not on the chase's critical path. But if you want autonomous room navigation later, the fix path in §4 is where the effort goes.

## 2. Empirical baseline

**Test suite.** 176 of the "172+" tests pass in seconds. `tests/test_imu.py` hangs indefinitely at its 5th test (`test_prediction_delta_math`) — a self-deadlock in `imu.py`, finding #7. The full-suite pass claim is therefore not reproducible.

**Simulator run** (`run_sim --floor-plan simple_room --max-steps 400`, headless, ~0.9 m of travel):

| Frame | GT (x, y) | EKF (x, y) | Error |
|---|---|---|---|
| 100 | (0.24, −0.18) | (0.06, +0.04) | 0.29 m |
| 200 | (0.52, −0.33) | (0.35, +0.14) | 0.50 m |
| 400 | (0.89, −0.36) | (0.49, +0.16) | 0.65 m |

Final heading error: **132°**. This is with the ground-truth anchoring described in finding #4 still in place; without it the divergence would be faster. The commit message claiming "sub-3 cm accuracy across all floor plans" (`d0f3b47`) does not reproduce here.

**VO sign/magnitude experiment** (commanded motions in sim, `vo.scale = 1`, raw VO summed over the move):

| Motion | Ground truth | VO raw (dx, dy, yaw) | Note |
|---|---|---|---|
| forward 0.235 m | fwd +0.235 | +3.06, −0.18, −0.3° | sign right |
| backward 0.235 m | fwd −0.235 | −5.91, +0.36, −0.2° | same distance, 2× the magnitude |
| strafe left 0.235 m | left +0.235 | +2.96, +1.86, **−22.2°** | phantom yaw, dx > dy |
| strafe right 0.235 m | left −0.235 | −2.29, **+3.47**, **+25.3°** | dy sign wrong, phantom yaw |
| rotate CCW | +79.3° | −3.03, +0.23, **+82.7°** | rotation good; phantom translation |
| rotate CW | −79.3° | −4.36, −0.11, **−82.9°** | rotation good; phantom translation |

**Baseline experiment** (same 0.235 m strafe, varying per-frame step):

| step | frames | frames gated | VO dy | phantom yaw |
|---|---|---|---|---|
| 8 | 60 | 56 | 1.86 | −22.2° |
| 24 | 20 | 0 | 16.13 | +2.3° |
| 48 | 10 | 0 | 8.95 | +1.3° |

Two things fall out: (a) with enough per-frame baseline the lateral estimate is clean and correctly signed — the ambiguity is a small-baseline artifact; (b) dy ≈ 0.8–0.9 *per frame* regardless of distance — the magnitude counts frames, not metres.

## 3. Findings, ranked by severity

### S0 — showstoppers (hardware mapping cannot work until these are addressed)

**1. VO translation magnitude is a frame count, not a distance.** *Verified.* `cv2.recoverPose` returns a unit translation direction per frame; `process_frame` scales it by a constant and sums (visual_odometry.py:154–176). The sum therefore grows by ~1 per frame that passes the parallax gate, independent of speed. No constant scale factor can map that to metres (the same 0.235 m gave 3.06 forward and 5.91 backward). The whole "scale calibration" design rests on a quantity that carries no distance information.

**2. Scale calibration does not exist on the hardware path.** *Read.* In `run_mapping.py`, `vo.scale` is never assigned, so VO runs at its default `0.001` forever. `cross_validate_with_ultrasonic` adjusts the *stereo baseline* correction (a different variable) and its return value is discarded (run_mapping.py:228). `estimate_vo_scale` is never called — and if it were, it returns the median scene depth, which is not a scale factor (synthetic_stereo.py:250–265; its own docstring admits the scale is "applied externally by the caller", and no caller does). Consequences: the map is in nonsense units, and the scan-stop trigger (`distance_since_scan ≥ 0.5 m`) fires after ~500 gated frames regardless of actual motion.

**3. The EKF never receives a camera measurement.** *Read.* In the pan sweep (run_mapping.py:181–185) the "observation" bearing/range is computed from the *map's stored landmark position* relative to the EKF pose — i.e., from the prior, not from the image. The keypoint pixel, the intrinsics, and the pan angle are never used. The innovation is ~0 by construction, so the update adds no information while still shrinking covariance (overconfidence). Combined with #1–2, the hardware pose is pure VO dead-reckoning in undefined units.

**4. Simulator validation is anchored to ground truth in four places.** *Verified (read + run).* `run_sim.py` resets the EKF pose to GT after every rotation scan (line 168), places landmarks using the GT pose (209–212), builds the occupancy grid from GT pose (222), and calibrates VO scale from GT displacement (234). The sim therefore measures a system periodically teleported to the truth, not SLAM; the accuracy claims in the docs derive from this. Between anchors it drifts 0.65 m / 132° in under a metre (§2).

### S1 — serious

**5. Frame-to-frame VO with tiny baselines.** *Verified.* At the sim's step size, the 1.5 px parallax gate rejects 70–93% of frames, and the survivors are noise-dominated: pure strafes produce ±22–25° phantom yaw and a spurious forward component larger than the real lateral one. The synthetic-stereo strafe *is* this motion. The fix is standard — match against the last keyframe rather than the previous frame, and only emit an estimate when median parallax exceeds a threshold (the §2 baseline table shows this works: phantom yaw → 1–2°).

**6. Pure rotation injects phantom translation.** *Verified.* An 80° in-place turn yields ~3–4 raw units of translation (comparable to a forward move). The essential matrix is degenerate under pure rotation; `recoverPose` emits an arbitrary unit `t`. Every rotation scan corrupts position. Needs rotation-only detection (homography-vs-essential model selection, or zero translation when the commanded motion is a spin).

**7. IMU self-deadlock.** *Verified.* `IMU.get_prediction_delta` holds the non-reentrant `threading.Lock` (imu.py:408) and then reads the `accel_xy` property, which takes the same lock (imu.py:319). Hangs forever. On hardware with the IMU attached, the first prediction step would freeze the loop. Fix: `RLock`, or read `_accel_raw` directly inside the held lock.

**8. `update_vo` treats a relative rotation as an absolute heading observation.** *Read.* state_estimator.py:212–229 applies `K · vo_dtheta` as a correction to the absolute heading. A fraction of an increment is neither a prediction nor a valid observation; `theta_vo` is computed and unused. The comment admits it's a placeholder. IMU path only, so lower blast radius.

**9. Relocalization mixes frame conventions.** *Suspected (needs a map to test).* Landmarks are stored with z = up (`wz = −Y_cam`, run_mapping.py:204), but `relocalize` returns `(cam_pos[0], cam_pos[2])` as the planar pose (navigator.py:129) — that's (x, up) — and extracts yaw with the camera-axis formula. ~75% likely to return a wrong pose even with a good PnP inlier set. Worth stating: PnP against triangulated landmarks is exactly the metric-pose mechanism the mapping loop is missing (§4B); this code is close to reusable once the frames are reconciled.

**10. IMU is not integrated into the hardware loop.** *Read.* No reference to the IMU in `run_mapping.py`; the integration exists only in tests and the sim path.

### S2 — moderate

**11. Synthetic-stereo baseline is open-loop.** A timed strafe on encoder-less mecanum wheels; the 5 cm baseline will differ between carpet and hardwood, and depth error scales with it. The ultrasonic cross-check helps but compares a 50 px image window against a ~15° sonar cone — loosely aligned at best. `confidence = disparity/20` is a heuristic, not an uncertainty.

**12. Uncalibrated intrinsics default to fx = fy = 500** (camera.py). VO and PnP are systematically wrong without running `camera_calibrate.py` first.

**13. Tracking loss has no recovery during mapping** — stop 0.5 s and retry the next frame; `vo.reset()` after scans discards the accumulated pose. No relocalization against keyframes while mapping.

**14. Pan-sweep matching is O(N·M) Hamming** over all landmarks per descriptor, and `visible` (the FOV-filtered set) is computed but unused (run_mapping.py:176). Fine for small maps; will not scale.

**15. Tests are written to pass, not to discriminate.** `test_rotation_detected` asserts only `|dθ| > 0.1` (sign unchecked); `test_ekf_bounded_error_forward` accepts up to 5 m error; stereo depth checks accept a ratio between 0.2 and 5. None of the S0 findings could be caught by the current suite.

### What is good and should be kept

The EKF-SLAM core: full cross-covariance propagation, correct motion and measurement Jacobians, chi-squared gating, Joseph-form updates, bounded active-landmark set with freeze/reactivate. Rotation estimation from the essential matrix (within ~4% of truth). The occupancy grid, frontier clustering and A*. The PyBullet twin as infrastructure (once its GT leaks are removed). The mock-hardware pattern that makes everything unit-testable. The module boundaries — the fix in §4 slots in without touching most of the code.

## 4. Recommended fix path (ranked by leverage)

**A. Keyframe VO.** Match the current frame to the last keyframe, not the previous frame; emit a motion estimate only when median parallax exceeds ~5–10 px; create a new keyframe when overlap drops. Removes findings #5 and most of #6's impact. ~100 lines in `visual_odometry.py`. Confidence it resolves the phantom-yaw problem: high (the §2 baseline table is the evidence).

**B. Get metric scale from somewhere real.** Three options, in rough order of preference for *this* platform:
- *Ground-plane homography (recommended to evaluate first).* The camera height and tilt are fixed on the chassis, and the floor is a plane. Features on the floor give **metric** translation directly (no scale ambiguity) via a homography with known camera height. This is the classic trick for floor robots and sidesteps monocular scale entirely. Depends on floor texture — carpet is ideal, plain tile poor. Estimated effort: ~150 lines plus a one-time height/tilt calibration. Confidence it works on a textured floor: ~70%.
- *PnP against stereo-triangulated landmarks.* Synthetic stereo already yields metric 3D points; feeding the *actual* 2D keypoint observations into the EKF (or solving PnP as in `navigator.relocalize`) recovers metric pose. This is what the current code was reaching for and never wired. Fixes #2–3 properly; needs #9's frame reconciliation.
- *Motor-command distance prior.* Generalize `strafe_calibrate`'s speed→distance LUT to forward motion and use it as the translation magnitude, with VO supplying direction and rotation. Crude (surface-dependent, no encoders) but cheap and would make the current pipeline produce plausible maps quickly.

**C. Make the simulator honest.** Remove the four GT anchors from `run_sim.py` (keep GT only for *scoring*), and report ATE/RPE per run. Until this is done, no sim result is evidence.

**D. IMU.** Fix the deadlock (#7, one line), integrate the gyro as the heading prediction in `run_mapping` (rotation is what VO already does well, but the gyro survives blur and scan rotations), and either fix or delete `update_vo`.

**E. Relocalization.** Reconcile frames (#9), then test in sim against a GT-built map.

**F. Tests that can fail.** Sign-checked VO tests against sim GT (the §2 experiment, made permanent), ATE thresholds that would have caught #1–4.

## 5. Virtual test plan

Tiered so each tier is a gate for the next. All run headless against the PyBullet twin; GT is used **only for scoring**.

**Tier 0 — regression suite must be green and terminate** (fixes #7). Add a per-test timeout (`pytest-timeout`) so a hang is a failure, not a stall.

**Tier 1 — VO unit-motion sign and magnitude tests** (the §2 experiment as permanent tests): forward/backward/strafe-left/strafe-right/rotate-CCW/rotate-CW at three per-frame baselines. Pass: correct sign on the dominant axis; non-dominant axes < 25% of dominant; phantom yaw < 3° for pure translations; rotation within 10% of GT. Today: fails on strafe sign, phantom yaw, and phantom translation.

**Tier 2 — metric scale.** Whatever mechanism §4B adopts: drive 1 m forward in sim, report estimated distance. Pass: within 10%. Repeat for 0.25 m and 2 m (linearity). Today: not measurable — no metric output exists.

**Tier 3 — dead-reckoning drift without anchors.** Square loop 1×1 m and a 2 m out-and-back, no GT resets. Metrics: absolute trajectory error (ATE) and relative pose error per metre (RPE). Pass (initial target): ATE < 10 cm at loop closure, heading < 10°. Today: 0.65 m / 132° over 0.9 m.

**Tier 4 — degenerate and adversarial cases.** (a) Pure 360° rotation: position must stay within 5 cm. (b) Low-texture room: reduce wall texture in `sim_world` until ORB yields < 100 features; require graceful tracking-loss reporting, not silent garbage. (c) Motion blur: convolve sim frames with a directional kernel proportional to angular rate; measure tracking-loss rate vs blur. (d) Dynamic object: a moving box in view (a cat) — VO must reject its features via RANSAC; measure ATE with and without.

**Tier 5 — synthetic stereo robustness.** Perturb the executed strafe by ±20% relative to the assumed baseline (models carpet vs hardwood); report depth error and whether ultrasonic cross-validation converges. Pass: depth error < 15% after 3 stops.

**Tier 6 — relocalization.** Build a map with GT (allowed here — it's the map, not the estimator), teleport the robot to 20 random poses, call `relocalize`, score pose error. Pass: < 15 cm / 10° on ≥ 80% of trials. This directly tests finding #9.

**Tier 7 — full mapping + navigation, no anchors.** Explore `two_rooms`, then navigate to 5 goals. Metrics: occupancy-grid IoU vs the true floor plan, goal-arrival rate, collisions. Pass targets to be set after Tiers 1–6 are green.

**Tier 8 — timing.** ORB + match + VO + EKF per frame on x86, scaled to Pi 5 by the measured factor from the YOLO benchmark (x86 21 ms → Pi 5 54 ms, ≈ 2.6×). Budget on Pi 5: ≤ 100 ms/frame for 10 Hz.

## 6. Hardware test plan

Staged; every stage has a pass criterion, and no stage starts until the previous one passes. The enabler for all of it is **record-and-replay**: log raw camera frames, motor commands, ultrasonic and (if fitted) IMU with timestamps on every run, so VO/EKF can be re-run offline against the same data after each fix. Without this, every hypothesis costs a physical run. This logger is the first thing to build (~50 lines; the harness already captures frames).

**H0 — Safety and setup.** Wheels off the ground for every first run of new motion code. IR-remote kill switch verified (the chase loop already has this). Battery voltage logged — motor speed→distance is voltage-dependent, which silently changes the strafe baseline.

**H1 — Calibration.** (a) Camera intrinsics with the 9×6 checkerboard (`camera_calibrate.py`); reprojection error < 0.5 px. (b) Camera height and tilt measured (needed for §4B ground-plane option). (c) Strafe LUT on **each floor surface you care about** (`strafe_calibrate.py`), because the baseline differs by surface; log the surface with the LUT.

**H2 — Static noise floor.** Robot parked, 60 s of VO. Pass: reported motion < 1 cm and < 1° total. Anything else is a bug (or a vibrating fan).

**H3 — Single-DOF motions with a tape measure.** Forward 1.00 m, backward 1.00 m, strafe left/right 0.50 m, rotate 360° (mark on floor). Pass: distance within 10%, rotation within 5°, non-commanded axes < 10% of commanded. Repeat on two floor surfaces. This is the hardware twin of Tier 1–2 and the first place metric scale is tested for real.

**H4 — Closed loops.** 1×1 m square (tape corners on the floor), then a 2 m out-and-back. Pass: return-to-start error < 15 cm and < 10°. Compare against the sim Tier 3 numbers to calibrate how much the sim under- or overstates reality.

**H5 — Synthetic stereo vs a ruler.** Park facing a wall at 1.0 m, 2.0 m, 3.0 m (tape measure). Run `capture_and_triangulate`; report median depth of centre features and the ultrasonic reading. Pass: within 15% at all three ranges. Also run it on the least textured wall in the house and record the observation count — that number bounds what synthetic stereo can do in your home.

**H6 — Single-room map.** Map one room; visualize the occupancy grid over a hand-measured floor plan. Pass: walls within 20 cm, no phantom walls, doorways open.

**H7 — Relocalization.** Power-cycle, place the robot at 5 marked poses, call `relocalize`. Pass: within 20 cm / 15° on 4 of 5.

**H8 — Navigate-to-goal.** Three goals in the mapped room, one through a doorway. Pass: arrival within `WAYPOINT_TOLERANCE_M` on 2 of 3, no collisions.

**Instrumentation for every stage:** ATE/RPE where GT exists (tape marks), per-frame feature counts, tracking-loss events, per-stage timing. Keep the recorded runs; they become the regression corpus for offline replay.

## 7. Effort and sequencing

If the goal is autonomous room navigation: §4A + §4B (ground-plane option) + §4C + record-and-replay is roughly a week of focused work, and would produce the first honest measurement of what this platform can do. Fix #7 (one line) regardless — a hang on hardware is worse than a wrong answer.

If the goal is the cat chaser: none of this is required. The reusable pieces are the occupancy grid fed by ultrasonic (for local obstacle memory, no VO needed), the PyBullet twin, and the motion controller — all of which already work.
