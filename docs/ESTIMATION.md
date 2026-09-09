# Cat & Robot State Estimation — design proposal (2026-09-09)

Peter's observation: the tracking is a patchwork of reactive image-frame loops
with no unified estimate of *where the cat is* or *where the robot is pointing*.
This proposes a principled design. Not built yet — this is the plan to react to.

## What's wrong with today's design

- **Image-frame only.** The cat is known as a pixel centroid in the *current*
  frame. There's no world-frame ("absolute") notion of the cat's direction, so
  the instant detection drops (blur, occlusion) we have nothing to coast on →
  the twitchy LOST→search behavior we keep band-aiding (grace frames, hold-pan).
- **Open-loop heading.** No encoders; body yaw is a guess. We built an
  optical-flow yaw estimator (`catchaser/flow.py`) but it isn't fused into
  tracking. No IMU fitted.
- **Two loops that barely talk.** Pan tracks the centroid; the body follows the
  pan; forward approaches — but nothing shares a common state. Lead/prediction
  is impossible without a velocity estimate.
- **No elevation.** Tilt isn't tracked (and is currently mechanically inert).

## Proposed state

Track the cat in **absolute angular coordinates**, decoupled from the camera:

- `theta_r` — robot heading (relative to start), integrated from a **fused
  yaw-rate**: wheel-command model ⊕ optical flow ⊕ IMU gyro (when added).
- `pan`, `tilt` — camera angles (commanded; known but with mechanical slop).
- **Cat absolute bearing** `beta = theta_r + (pan - pan0) + img_bearing_x(cx)`
  and **elevation** `eps = (tilt - tilt0) + img_bearing_y(cy)`, where
  `img_bearing_x = atan(( cx - W/2 ) / fx)` (same for y with fy).
- **Cat range** `r` — fused from bbox height (monocular) ⊕ ultrasonic (when the
  cat is ~ahead).
- Track `(beta, beta_dot, eps, eps_dot, r)` with a small constant-velocity
  filter (alpha-beta or a 1-D Kalman per axis). This buys **prediction (lead
  pursuit)** and **coasting through dropouts** for free.

## Sensor → estimator map

| Sensor | Updates |
|--------|---------|
| Detector centroid (cx, cy, bbox) | `beta`, `eps` measurement; `r` from bbox |
| Optical flow (`flow.py`) | `theta_r` (yaw-rate prediction) |
| IMU gyro (if fitted) | `theta_r` (fast, drift-corrected by flow) |
| Ultrasonic | `r` when cat centered & ahead |
| Servo commands | `pan`, `tilt` control inputs |

## Control (two-rate, both off the SAME estimate)

- **Fast (~15 Hz):** drive **pan → center `beta`**, **tilt → center `eps`**
  (once tilt works). Servos are quick and low-blur.
- **Slow:** body yaw to re-center the pan (coarse hand-off, already built), and
  forward to approach (gate on `r`, ultrasonic stop).
- **Lead pursuit:** aim at `beta + beta_dot * dt` instead of the raw centroid.
- **Dropout handling:** *coast* the estimate (keep predicting `beta`) for a
  bounded time; only fall back to SEARCH when the estimate's uncertainty grows
  past a threshold. This *replaces* the grace-frame / hold-pan hacks with one
  principled rule.

## Phasing

1. **Absolute-angle cat estimator** (`beta, beta_dot`) fusing detection +
   flow-derived yaw, with coast-through-dropout. Biggest bang; no new hardware.
   Subsumes the current lost/grace/hold-pan logic.
2. **Fused yaw** for `theta_r`: optical flow ⊕ wheel-command model (⊕ IMU gyro
   when added). Makes rotation trustworthy on carpet.
3. **Elevation tracking** (tilt) — **blocked** until the tilt mount actually
   moves the camera (hardware).
4. **Range fusion** (bbox ⊕ ultrasonic) + lead pursuit in the approach.

## Dependencies / caveats

- **Tilt: FIXED — was bent metal, servo is fine** (2026-09-09). The tilt mount
  had bent metal jamming the servo (early sweeps showed no/tiny motion + stall
  risk). After Peter straightened it, the servo sweeps freely and repeatably
  (20+ full 0→110 cycles unloaded; observed travel ~80° for a 0-110 command —
  normal servo under-travel, plenty for framing). **Still TODO once the camera
  is reattached:** (1) verify the tilt *sign* (an early test read up↔down
  inverted) with a gentle 65/35 test, (2) set the true `SERVO_TILT_REST` to
  frame a floor-cat. Then Phase 3 (elevation tracking) is unblocked.
- **No IMU fitted** (I2C 0x69 empty). Recommended: Adafruit ICM-20948 — the
  driver (`raspbot_slam/imu.py`) already exists. Improves Phase 2; not required.
- **Slop:** commanded pan/tilt ≠ true angle. Flow-correction of `theta_r` helps
  the yaw axis; a similar trick for pan would need a known visual reference.
- Ties into the SLAM effort's honest-odometry problem — see
  `raspbot_slam/SLAM_PLAN.md`; the yaw-fusion work is shared.
