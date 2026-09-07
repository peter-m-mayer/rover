# Cat Chaser 3000 — working notes for Claude

Context for future sessions working on the `catchaser/` package. Records what's
been verified, what's still blocked, and the hard-won environment facts.

Status legend: ✅ verified · ⚠️ assumed/unverified · ⛔ blocked

---

## 1. Hardware bring-up on the Pi — ✅ ALL GREEN (2026-09-06)

Ran on the real robot (`ssh pi@192.168.0.32`, pw `yahboom`). See root
`CLAUDE.md` for connection facts.

- **Board:** Raspberry Pi 5 Model B Rev 1.1, **aarch64**, Python 3.11.2,
  hostname `yahboom`. → **ONNX Runtime is the path** (the Pi-3/32-bit/TFLite
  worry from the handoff is moot; this is a 64-bit Pi 5).
- **`hello --no-motion`:** all PASS — driver connect (HARDWARE), led+buzzer,
  camera (640×480 → hello_frame.jpg), ultrasonic (170 mm). ✅
- **`hello` (motion):** PASS — in-place spin 1.0 s @ 60 ran; forward drive was
  **auto-skipped** because the ultrasonic saw 170 mm < the 400 mm clearance
  gate (correct safety behavior). To exercise forward drive, give it ≥0.5 m of
  clear space and re-run.
- **Benchmark (`catchaser.benchmark`), Pi 5, ONNX, 320 px:**

  | Mode | Inference (median / p90) | End-to-end | Rate |
  |------|--------------------------|-----------|------|
  | synthetic frame | 54.1 / 60.1 ms | 60.3 ms | **16.6 Hz** |
  | camera (capture+detect) | 57.2 / 61.1 ms | 66.2 ms | **15.1 Hz** |
  | forced `--threads 4` | 56.1 / 59.5 ms | — | 15.9 Hz |

  → **≥1 Hz target crushed** — ~15 Hz with real camera capture (+~9 ms/frame).
  Auto-threading is already optimal (forcing 4 threads doesn't help).
  `CHASE_LOOP_HZ=10` is comfortably within budget.

### Gotchas fixed during bring-up (important for future sessions)

1. **`cv2` was corrupted** — `import cv2` failed with a nonsense
   `'y: cannot open shared object file'`; `ldd` on `cv2.abi3.so` showed garbage
   DT_NEEDED entries (damaged ELF). Root cause: a broken binary **and** both
   `opencv-python` + `opencv-contrib-python` installed. Fix:
   `pip3 uninstall -y opencv-python opencv-contrib-python` then
   `pip3 install --user --no-cache-dir --break-system-packages "numpy<2" "opencv-python-headless<5"`.
   → **numpy must stay <2.0** (onnxruntime 1.18.1 requires it) and **opencv
   pinned <5** (opencv 5 drags in numpy 2). Landed on numpy 1.26.4 +
   opencv-python-headless 4.11.0.86 + onnxruntime 1.18.1. Headless is correct
   for a display-less robot.
2. **Vendor driver method names differ from `ROVER_OVERVIEW.md`.** The real
   `Raspbot` class has **no `Ctrl_WS2812B` / `Ctrl_Buzzer`**. Actual API:
   - LEDs: `Ctrl_WQ2812_ALL(state, color_index)` and
     `Ctrl_WQ2812_Alone(number, state, color_index)` — **color is a palette
     index, not RGB.** Verified indices: red=0, green=1, blue=2, yellow=3.
   - Brightness: `Ctrl_WQ2812_brightness_ALL(R, G, B)`.
   - Buzzer: `Ctrl_BEEP_Switch(state)`.
   - Motors: `Ctrl_Muto(motor_id, motor_speed)` — **handles negative speed**
     (dir bit), so signed speeds work for reverse/rotate. Also `Ctrl_Car(id,
     dir, speed)`.
   - Servo `Ctrl_Servo(id, angle)`, ultrasonic enable `Ctrl_Ulatist_Switch(state)`.
   `raspbot_slam/actuators.py` was updated to use these real names (LED via
   index palette, buzzer via BEEP). Motors/servo/ultrasonic already matched.
3. **`i2cdetect` is not installed** (`i2c-tools` missing). The sanity check in
   root `CLAUDE.md` needs `sudo apt install i2c-tools` first. Not required for
   normal operation — the vendor driver talks I2C via `smbus` directly, and it
   connects fine (`0x2b`).
4. Vendor code lives on the Pi at `~/project_demo/raspbot/` and
   `~/py_install/` (not in git). Repo cloned to `~/rover` on the Pi.
5. **IR remote register idles at 0xFF** (255), not 0 — verified on hardware.
   `sensors.read_ir_remote()` must treat both 0 and 255 as "no key", or the
   chase loop's IR kill switch fires instantly at startup (it did). Also the
   receiver must be powered on first: `Ctrl_IR_Switch(1)` (exposed as
   `Sensors.enable_ir()`, called by `ChaseController.run()`).

### Floor runs with live cat (2026-09-06 evening)

Two 120 s untethered-WiFi floor runs, real cat in room + iPad decoy.
**Run 2 ended in HOLD at the actual cat** (snapshot-confirmed point-blank
approach sequence) — first successful chase. What the data taught us:

- **Continuous search spin outruns the detector** (~6 Hz loop, ~65° FOV,
  heavy rotation blur): run 1 had 15+ single-frame edge acquisitions.
  → fixed with pulsed spin-and-stare (2 spin / 3 stare frames).
- **KP=80 (sim-tuned, 10 Hz) overshoots at the real ~6 Hz**: telemetry showed
  err sign-flips in one frame (turn=+44 flipped err +0.38→-0.27, ~0.015
  err/turn-unit/frame). → KP=40, and the PID D-term now needs two consecutive
  tracked frames (fresh-acquisition D-kick was a bug).
- **Detection snapshots (`--save-dir`) are the debugging superpower** — the
  14 annotated frames settled recognition-vs-control instantly. Findings:
  every real-cat hit ≥0.53; false positives (legs 0.45, motion-blurred trash
  can 0.41, jacket 0.63) → `CHASE_MIN_CONFIDENCE` raised 0.35→0.50. Camera is
  fixed-focus: sharp ~0.4 m→∞, mush closer than ~0.3 m. iPad decoy works at
  max screen brightness only (dim screen → 0.37-0.39, below threshold now).
- Other deployed tuning: close-range approach taper (CHASE_APPROACH_*),
  camera CAP_PROP_BUFFERSIZE=1 (stale-frame latency), CHASE_LOOP_HZ 20.
- **Fine-tuning the ONNX on the specific cat is NOT needed** — recognition is
  solid; every failure was control-side. Revisit only if a *second* pet needs
  discriminating.

### Training-data pipeline (2026-09-06)

End-to-end path from chase runs to a fine-tuning dataset (only needed if a
second pet ever requires discrimination — recognition itself is solid):

1. **Harvest** during any chase: `python3 -m catchaser.chase --save-dir datasets/<name>`
   → `images/` (clean frames), `labels/` (YOLO weak labels from the detector,
   class 0; empty = negative), `preview/` (annotated, for humans). Detections
   ≤1/s, negatives ≤1/10 s, 300-frame cap. `catchaser/dataset.py`.
2. **Triage** from the couch: `python3 -m catchaser.review datasets/<name>`
   → phone at `http://<pi-ip>:5000`; g=Good / n=No cat / f=Fix (Fix copies
   into `review_fix/` for a real box editor); lossless undo/resume.
3. **Stats** anytime: `python3 -m catchaser.dataset_stats` — per-session +
   combined detector precision (of frames where the detector *claimed* a cat)
   and pos/neg composition. Reads each `review_status.json`.
4. **Train** elsewhere (ultralytics on a PC), export ONNX 320 px opset 12,
   drop into `catchaser/models/`.

**Confusion matrix (2 reviewed sessions, 244 frames, 2026-09-07):**
TP 198 · FP 0 · FN 23 · TN 23 → **precision 100%, recall 89.6%, F1 94.5%**.
Run `python3 -m catchaser.dataset_stats` for the live matrix. The detector
never false-alarms (0 FP, thanks to the 0.50 threshold) but misses ~10% — all
hard cases (motion blur, extreme close-ups, dark cat under furniture,
legs-only crops). Recognition is not the bottleneck; don't fine-tune yet.

⚠️ **Two findings from the matrix work (act before any fine-tune):**
1. **23 missed cats are mislabeled as negatives.** The reviewer pressed "good"
   on no-box frames meaning "yes, cat" — but `review.py`'s "good" *keeps the
   empty label*, so those frames are stored as negatives (a cat labeled as
   background = training poison). Verified by eye: every "good"-on-empty frame
   contains the cat. They must be re-boxed (→ positives) before training;
   they're the highest-value hard examples.
2. **Review-tool trap — FIXED (2026-09-07).** `review.py` now shows
   context-aware buttons: boxed frames get Good/No-cat/Fix; no-box frames get
   No-cat/**Missed** (🐾, key `m`), which routes to `review_fix/` as a
   to-be-boxed positive instead of a poison negative. To reopen the 23 already
   mislabeled frames for re-triage:
   `python3 -m catchaser.review datasets/<name> --fix-traps` (clears their
   "good" decision so they return to the queue), then review and press Missed.
   `dataset_stats` counts the explicit "missed" decision as FN (and still
   treats legacy good/fix-on-empty as FN for pre-fix data). Earlier figures
   ("93%", then "99.5%") predate identifying the FN quadrant.

**Chase loop rate (2026-09-07):** `run()` used to `sleep(period)`
unconditionally after perception (~50-70 ms), throttling the loop to ~6 Hz —
half the detector rate — and adding tracking latency. Fixed to sleep only the
period *remainder*, so the detector (~15 Hz) is the limiter. Higher loop rate
= tighter centroid tracking. See [[chase-controller]] tuning.

### Pan-servo tracking + prey mode (2026-09-07) — built, sim-validated

- **Pan tracking** (`--pan`, `CHASE_PAN_*`): the camera servo tracks the
  centroid; the body turn just follows the pan back toward center. Sim result:
  mean |err| 0.067 → **0.026** (2.5× tighter centering). ⚠️ **Hardware check
  before trusting it:** confirm servo direction — if the camera pans the wrong
  way, flip `CHASE_PAN_SIGN` to -1. (Sim convention: pan>90 = look left.)
  Default `CHASE_PAN_ENABLED=False` until that check passes.
- **Prey mode** (`--prey`, `CHASE_PREY_*`): motion policy = DART (mecanum
  zig-zag, alternating per cycle) → FREEZE (fully still, the pounce bait) →
  and FLEE when the cat closes within `CHASE_PREY_FLEE_MM` (300 mm). Heading
  tracking (pan or PID) still centers the cat throughout. LEDs: DART green,
  FREEZE white, FLEE purple. Composable with `--pan`.
- **Loop rate**: also fixed the self-throttling sleep (~6→~15 Hz) — see above.
- Run: `python3 -m catchaser.chase --pan --prey` (hardware) or
  `python3 -m catchaser.chase_sim --scenario approach --pan --prey --verbose`.

**Pan servo verified on hardware (2026-09-07):** direction sign is the sim
convention (pan > center = camera looks LEFT) → `CHASE_PAN_SIGN = +1` (default,
confirmed by eye). Tilt works (higher = looks up). The pan mount sat ~13° off
true-forward (mechanical slop, fixed by re-seating a screw), so
`SERVO_PAN_CENTER` is now **77** (measured), not the nominal 90 — this is the
body-follow target, so getting it right removes a steady-state heading bias.
`python3 -m catchaser.pan_check` re-measures the sign anytime.
**`--pan` is validated — safe to use.**

**Phone aliases** (in the Pi's `~/.bashrc`; reconnect or `source ~/.bashrc`):
`play` (prey + pan + harvest to a dated dataset), `playbody` (prey, no pan),
`chaseonly` (plain chase + pan), `review` (triage newest dataset),
`stats`, `bench`, `hellobot`, `catpull` (git pull).

Still open, in rough value order:
1. ~~Hardware servo-sign check for pan~~ — DONE (sign +1, center 77).
2. Exposure control: shorter camera exposure (v4l2) to cut rotation blur.
3. Centroid velocity prediction (lead pursuit) once tracking is stable.
4. Pan *sweep* during SEARCH (park body, sweep camera) — bigger FOV, no blur.

### Wheels-off hardware test results (2026-09-06)

`python3 -m catchaser.chase --forward-speed 0 --search-speed 0` with a phone
cat-photo, rover propped up:
- Sign convention verified: photo right of center → `+err`/`+turn`
  (clockwise); left → negative. Proportional, symmetric, deadband quiet when
  centered.
- Grace-hold (3 frames) → search-spin transitions clean; spin runs at
  `turn=+50` and physically rotates the wheels.
- Ultrasonic HOLD engages at <200 mm (phone/hand in front) — forward locked
  to 0, steering stays live.
- **Bonus: the detector locked onto the actual house cat at ~1–3.5 m** for
  ~35 straight frames (stable `err≈-0.05`, in-deadband). First real-cat
  contact: PASS.
- Detector in-loop: ~43 ms/frame; effective loop ~6 Hz (0.1 s period + capture
  + inference).
- ⚠️ Still unverified on hardware: IR-remote kill (post-fix), forward-drive
  TRACKING, and SEARCHING→TRACKING reacquire — the phone photo was never
  detected in the full-loop run (screen glare/angle defeats the detector;
  a printed photo works better).

---

## 2. ✅ RESOLVED: the network saga (historical — kept as a cautionary tale)

**Resolution:** `192.168.1.11` was a **dead relic address** — a *different*
device on the direct cable answered its ARP but dropped all IP traffic, which
burned hours. The Pi was actually on the normal LAN at **`192.168.0.32`** (DHCP
from the TP-Link router) the whole time. Connect via `ssh pi@192.168.0.32`
(pw `yahboom`). The static `192.168.1.10` added to the Windows Ethernet adapter
is now pointless and can be removed:
`netsh interface ipv4 delete address name="Ethernet" 192.168.1.10`.

Lesson: when ARP resolves but ICMP/TCP time out (not *refused*) and it survives
a reboot, suspect **wrong host / relic address**, not just a firewall.

Original diagnosis (2026-09-05), before we knew the address was stale:

- This host (WSL2, **mirrored** networking) and the Windows host are both on
  Wi-Fi `192.168.0.38/24`. The Pi reports `192.168.1.11` on its OLED — a
  **different subnet**, reached via a **direct ethernet cable** to the Windows
  host (adapter "Ethernet", 1 Gbps link up).
- That ethernet adapter had only an APIPA address (`169.254.x`) — no DHCP on a
  point-to-point link — so it couldn't talk to `192.168.1.x`.
- **Fix applied:** added a static secondary IP to the Windows Ethernet adapter:
  `netsh interface ipv4 add address name="Ethernet" 192.168.1.10 255.255.255.0`
  (elevated). WSL mirrors this, so both can now route to the Pi's subnet.
  Remove later with `netsh interface ipv4 delete address name="Ethernet" 192.168.1.10`.
- After the fix: **ARP resolves** the Pi (`88-A2-9E-3F-96-7F`) → L2 works, routing
  is correct (egress source `192.168.1.10`). But **ICMP and TCP/22 both time out**
  — even from native Windows, and with the Windows firewall fully OFF. Persisted
  across a Pi power-cycle.
- Conclusion: **the block is on the Pi.** A *timeout* (not "connection refused"/RST)
  means packets are being dropped, which points to a Pi-side firewall or the SSH
  service not serving on that interface. Not a Windows/WSL problem.

**Next step (needs a keyboard+monitor on the Pi):**
```bash
ip -4 addr show            # does eth0 really hold 192.168.1.11? which iface has the 88:A2:9E MAC?
sudo systemctl status ssh  # RPi OS ships sshd DISABLED by default
sudo ss -tlnp | grep :22   # listening? on which address?
sudo ufw status verbose    # likely culprit (dropped, not refused)
sudo iptables -S
```
Most likely fixes: `sudo systemctl enable --now ssh` and/or `sudo ufw allow 22`
(or `sudo ufw disable` on this trusted direct link).

> ⚠️ The ARP'd MAC OUI `88:A2:9E` is **not** a standard Raspberry Pi onboard NIC
> prefix (Pi 5 = `2C:CF:67`, Pi 4 = `DC:A6:32`/`E4:5F:01`, Pi ≤3 = `B8:27:EB`).
> Worth confirming `192.168.1.11` really is the Pi's ethernet (vs a USB NIC or
> another device on the cable).

---

## 3. Dev-machine environment (WSL) — ✅

- Default `python3` (3.10) lacked `cv2` and `pybullet`; the whole test suite
  auto-skips without OpenCV (`tests/conftest.py`).
- Installed (user site) to run the sim locally:
  `pip install --user opencv-python-headless pybullet`
  (this bumped `numpy` 1.21 → 2.2; `onnxruntime` 1.23 already present).
- On the Pi, follow the repo README instead: `opencv-python-headless numpy`
  (+ `onnxruntime` **only if aarch64**) and `sudo python3 setup.py install` for
  the vendor `Raspbot_Lib`.

---

## 3b. Command reference + RC mode + coarse/fine tracking (2026-09-07)

- **[../docs/COMMANDS.md](../docs/COMMANDS.md)** is the living command/alias
  reference — update it whenever commands/flags/aliases change.
- **`rcbot`** (`catchaser/rc.py`): Flask web drive controller on port 5001 —
  live MJPEG camera + keyboard/touch driving (numpad/arrows), pan/tilt, home,
  set-home (recalibrate), speed, dead-man stop. Mecanum vector combines held
  keys (diagonals). `RCController` is pure/tested; endpoints tested; camera
  stream + motion are manual. Home persists to `~/.catchaser_rc_home.json`.
- **Coarse/fine pan tracking**: pan mode now keeps the body still within a wide
  dead-zone (`CHASE_PAN_BODY_ENGAGE_DEG`) and only slow-rotates the body to
  re-center the pan near the FOV edge (hysteresis). Search in pan mode is a slow
  continuous rotate (no stop-and-go). This is what makes the pan actually get
  used. Prey lunges exaggerated +50% (dart 105, flee 135; `CHASE_MAX_SPEED` 160).

## 4. Chase controller — ✅ built (`catchaser/chase.py`)

Pipeline: **detector centroid → normalized heading error → heading PID →
mecanum drive**.

- `ChaseController.compute(detections, distance_mm, dt)` is **pure** (given PID
  state) → unit-testable with no hardware. Returns a `DriveCommand`
  (forward, turn, state).
- Heading error `err = (cx - W/2)/(W/2)` ∈ [-1, 1]; `+err` = cat right of center
  → `+turn` = clockwise/right (sign convention matches `Actuators.drive` and the
  sim's mecanum kinematics — verified in tests).
- **Forward** speed is full when centered and tapers to 0 as the cat drifts;
  pure turn-in-place past `CHASE_TURN_ONLY_ERROR`.
- **States → LED:** TRACKING(green), LOST(yellow), SEARCHING(red), HOLD(cyan),
  IDLE(off).
- **Search-spin:** on cat loss, hold for `CHASE_LOST_GRACE_FRAMES` (debounce a
  dropped frame), then spin in place *toward the side the cat was last seen*.
- **Ultrasonic stop:** within `CHASE_STOP_MM` (200 mm) → HOLD (stop advancing,
  keep steering to stay pointed at the cat). Never rams the cat.
- **Speed cap:** `_cap()` scales so no wheel exceeds `CHASE_MAX_SPEED`.
- **Kill switches:** `Ctrl+C`, a `kill_check` callback, and (hardware) any
  **IR-remote key** all stop the loop; motors are stopped in a `finally` block
  no matter how it exits.
- Hardware entry point: `run_hardware_chase(camera, detector, sensors, actuators)`.

**Tunables live in `raspbot_slam/config.py`** (the `Cat Chaser` section):
`CHASE_HEADING_KP/KI/KD = 80/0/6`, `CHASE_FORWARD_SPEED = 60` (=NAV_SPEED),
`CHASE_MAX_SPEED = 120`, `CHASE_TURN_MAX = 90`, `CHASE_STOP_MM = 200`
(=OBSTACLE_STOP_MM), `CHASE_SEARCH_SPIN_SPEED = 50`, `CHASE_LOST_GRACE_FRAMES = 3`,
`CHASE_CENTER_DEADBAND = 0.06`, `CHASE_TURN_ONLY_ERROR = 0.5`,
`CHASE_MIN_CONFIDENCE = 0.35`, `CHASE_LOOP_HZ = 10`.

> Added a `drive(forward, turn, strafe=0)` mecanum-mixing method to **both**
> `raspbot_slam/actuators.py` and `simulator/sim_actuators.py`. Verified it
> matches the existing `move_forward` / `rotate_right` / `move_right` primitives
> and the sim's inverse kinematics.

**Gains are sim-tuned only.** Expect to re-tune `KP`/`KD` and speeds on hardware
(no encoders — "motor commands are suggestions"), and `CHASE_LOOP_HZ` must not
exceed the measured detector Hz from §1.

---

## 5. Simulator validation — ✅ (`catchaser/chase_sim.py`)

The real YOLO detector can't see a PyBullet primitive as a "cat", so the sim
validates the **control loop** (the new/risky part) with a synthetic detector
that projects a virtual cat's true bearing to a pixel centroid matching
`SimCamera`'s FOV. The cat is a real physical box, so the simulated ultrasonic
detects it and the 200 mm stop is exercised for real. It drives the *actual*
`ChaseController.run()` loop.

```bash
python -m catchaser.chase_sim --scenario approach   # center + approach + stop
python -m catchaser.chase_sim --scenario search     # cat out of frame -> spin -> acquire
python -m catchaser.chase_sim --gui                 # watch it
```

Verified results (headless):
- **approach:** acquires, centers, drives in, HOLDs at ~238 mm — no collision.
- **search:** grace-holds, spins to the last-seen side, acquires, HOLDs at
  ~231 mm — no collision.

Tests: `tests/test_catchaser_chase.py` — 27 tests (pure-logic + 2 PyBullet
end-to-end). Run: `python -m pytest tests/test_catchaser_chase.py -q`.

---

## 6. Reused from `raspbot_slam` (don't reinvent)

`actuators.Actuators` (mecanum + `drive()`, servos, LEDs, buzzer; `bot=None`
mock), `camera.Camera` (`capture_color()`), `sensors.Sensors`
(`read_ultrasonic_mm`, `enable_ultrasonic()` first!, `read_ir_remote`),
`config.py` (all tunables), and the `simulator/` twin (drop-in
`SimCamera/SimSensors/SimActuators`). Hardware degrades to mock when the vendor
`Raspbot_Lib` isn't installed — see `catchaser/hw.py`.
