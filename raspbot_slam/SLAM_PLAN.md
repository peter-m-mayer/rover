# SLAM Improvement Plan

Response to [SLAM_REVIEW.md](SLAM_REVIEW.md) (external review at `4e63bd2`,
2026-09-06). This is the longer-term track — **not** on the cat chaser's
critical path. Finding numbers (#N) and section refs (§N) point into the review.

## Accepted verdict

The review is right, and its experiments are reproducible: hardware mapping
cannot work today because (1) VO translation is a frame count, not metres
(#1), (2) no metric scale ever enters the hardware path (#2), (3) the EKF
never sees a real camera measurement (#3), and (4) the simulator results were
anchored to ground truth (#4). The EKF core, grid/exploration/A*, the PyBullet
twin, and the module boundaries are worth keeping — the fix is a rebuild of
the estimation core's inputs, not a rewrite.

## Status ledger

| Item | Status |
|------|--------|
| #7 IMU self-deadlock | ✅ **Fixed** (2026-09-06): `get_prediction_delta` read `accel_xy` while holding the same non-reentrant lock; now reads `_accel_raw` directly. `tests/test_imu.py` terminates (hung forever before). |
| SimIMU gyro blind to rotation | 🆕 **New finding** exposed by the deadlock fix: `TestSimIMU::test_sim_imu_detects_rotation` and `..heading_tracks_rotation` fail — always did, but sat *after* the hanging test so they never ran. Likely cause: the kinematic rover moves via `resetBasePositionAndOrientation`, so PyBullet reports ~0 angular velocity and `SimIMU` reads ~0 gyro. Fix in Phase 1 (compute sim IMU rates from pose deltas, not PyBullet velocities). |
| Everything else | 📋 Planned below |

## Phases

Ordering follows the review's leverage ranking (§4) with one change: honest
measurement comes first, because until the sim stops cheating we can't tell
whether any fix works.

### Phase 1 — Honest measurement (review §4C + Tier 0/1)
*Goal: a sim harness that can say "this fix helped" truthfully.*

1. Remove the four GT anchors from `run_sim.py` (#4): EKF pose resets,
   GT-placed landmarks, GT-built occupancy grid, GT-derived VO scale. Keep GT
   **only** for scoring.
2. Add ATE / RPE scoring to `run_sim` output (per-run trajectory error vs GT).
3. Make the §2 VO experiments permanent tests (Tier 1): six unit motions
   (fwd/back/strafe-L/R/rotate-CCW/CW) at three baselines; assert dominant-axis
   sign, cross-axis < 25%, phantom yaw < 3°, rotation within 10%.
   These tests **fail today** — mark `xfail(strict=False)` until Phase 2 lands,
   then flip to hard asserts.
4. `pytest-timeout` on the suite so a hang is a failure (Tier 0).
5. Fix `SimIMU` rate computation (pose-delta based) so the two newly exposed
   IMU tests pass; they're the Tier-0 gate for Phase 5.
6. Discriminating-test cleanup (#15): tighten the sign-unchecked and
   anything-goes tolerances flagged in the review.

*Exit gate: unanchored sim run produces honest ATE/RPE numbers (expected: bad).*

### Phase 2 — Keyframe VO (review §4A, fixes #5, most of #6)
*~100 lines in `visual_odometry.py`.*

- Match current frame against the **last keyframe**, not the previous frame.
- Emit motion only when median parallax exceeds ~5–10 px (the review's
  baseline table shows this kills the ±22–25° phantom yaw).
- Rotation-only degeneracy (#6): homography-vs-essential model selection, or
  zero the translation when commanded motion is a pure spin.
- Flip the Tier-1 xfails to hard asserts.

*Exit gate: Tier 1 green.*

### Phase 3 — Metric scale (review §4B, fixes #1/#2)
Evaluate in the review's preference order:

1. **Ground-plane homography** (evaluate first, ~70% confidence): fixed camera
   height/tilt + floor features → metric translation directly. Needs the
   one-time height/tilt calibration (H1b). Carpet at home = favorable.
2. **PnP against stereo-triangulated landmarks**: what the current code was
   reaching for; requires frame reconciliation (#9) and feeds Phase 4.
3. **Motor-command distance prior** (cheap stopgap): generalized strafe LUT
   as translation magnitude, VO for direction/rotation.

Decision point: run Tier 2 (1 m / 0.25 m / 2 m drives, ±10%) against option 1
in sim; fall back to 2/3 if floor texture defeats it.

*Exit gate: Tier 2 within 10%; Tier 3 (1×1 m square, no anchors) ATE < 10 cm,
heading < 10°.*

### Phase 4 — Real EKF measurements (fixes #3, #8, #9)

- Feed actual keypoint pixels + intrinsics + pan angle into the EKF update —
  innovation computed against the image, not the prior (#3).
- Fix or delete `update_vo`'s relative-as-absolute heading hack (#8).
- Reconcile the landmark/camera frame conventions in `relocalize` (#9) —
  z-up vs camera-frame mix; then Tier 6 (20 random relocalizations, ≥80%
  within 15 cm / 10°).

### Phase 5 — IMU integration (review §4D, #10)

- Gyro as the heading prediction source in `run_mapping` (rotation survives
  blur and scan spins; that's where VO is weakest on hardware).
- Requires the SimIMU fix from Phase 1 for sim validation.

### Phase 6 — Hardware ladder (review §6)

- **First build record-and-replay** (~50 lines): log frames, motor commands,
  ultrasonic, IMU with timestamps every run; re-run VO/EKF offline against
  recorded data. One physical run per hypothesis is unaffordable.
- Then H0→H8 as written in the review (safety → calibration → static noise →
  tape-measure single-DOF → closed loops → stereo vs ruler → single-room map →
  relocalize → navigate). Battery voltage in every log (speed→distance is
  voltage-dependent).
- H1 calibration items double as Phase 3 prerequisites (intrinsics; camera
  height/tilt; per-surface strafe LUT).

### Deferred / opportunistic

- Tier 4 adversarial sims (low texture, blur kernels, moving-cat RANSAC
  rejection), Tier 5 baseline perturbation, Tier 8 timing budget.
- O(N·M) pan-sweep matching (#14) — only matters once maps grow.
- Synthetic-stereo confidence model (#11) — revisit after Phase 3 picks the
  scale mechanism (ground-plane may demote stereo to a cross-check).

## Effort estimate

Per the review (§7): Phases 1–3 ≈ one focused week and produce the first
honest measurement of the platform. Phases 4–5 are a second chunk of similar
size. Phase 6 is spread across hardware sessions.

## Relation to the cat chaser

None of this blocks the chase. Shared components already proven by the chase
work: motion controller, ultrasonic, camera path, PyBullet twin, and the
record-and-replay logger (Phase 6) will reuse the chase's dataset-harvest
pattern (`catchaser/dataset.py`). If the chaser later wants "remember where
the cat's favorite spots are", the occupancy grid + ultrasonic (no VO) is the
review-endorsed shortcut.
