# Cat Chaser 3000 — Command Reference

Peter's quick reference. **Kept up to date as features land** (if you add a
command/alias and this file doesn't mention it, that's a bug — ping Claude).

Last updated: 2026-09-07.

---

## Connect

```bash
ssh pi@192.168.0.41       # WiFi — STATIC, always this (password: yahboom)
ssh pi@192.168.0.40       # Ethernet tether (when plugged into the router)
```
The dev machine's SSH key is installed, so no password prompt from there.
The Pi's OLED shows its current WiFi IP (should read `192.168.0.41`).

---

## Aliases (type these after `ssh`, from anywhere on the Pi)

Reload after changes with `source ~/.bashrc` (or just reconnect).

| Alias | What it does |
|-------|--------------|
| **`play`** | **Prey mode + pan tracking**, harvests a dated dataset. The main event. |
| `playbody` | Prey mode, no pan (fallback if the pan servo misbehaves). |
| `chaseonly` | Plain steady chase + pan tracking (no prey darts). |
| **`rcbot`** | **Manual web drive controller** → open `http://192.168.0.41:5001`. |
| `review` | Triage the newest dataset → open `http://192.168.0.41:5000`. |
| `stats` | Detector precision/recall + confusion matrix. |
| `bench` | Detector speed (Hz) on the Pi. |
| `hellobot` | Full hardware smoke test (LEDs, camera, ultrasonic, motion). |
| `catpull` | `git pull` the latest code. |

Stop any chase/play run: **Ctrl+C**, any **IR-remote key**, or pick the bot up.

---

## Autonomous chase — `python3 -m catchaser.chase [flags]`

```bash
python3 -m catchaser.chase --prey --pan               # = `play` (minus the dataset)
python3 -m catchaser.chase --prey --pan --save-dir datasets/session1
python3 -m catchaser.chase                            # plain body-only chase
python3 -m catchaser.chase --forward-speed 0 --search-speed 0   # steer-only, wheels-off test
```

| Flag | Meaning |
|------|---------|
| `--prey` | Dart / freeze / flee behavior (engaging). Off = steady pursuit. |
| `--pan` | Camera pan tracking; body coarse-follows only near the FOV edge. |
| `--fast-shutter` | Short camera exposure + gain to kill motion blur (in `play`). |
| `--forward-speed N` | Base approach speed (0 = steer/spin only, no advance). |
| `--search-speed N` | Spin speed when the cat is lost (0 = hold still). |
| `--stop-mm N` | Ultrasonic hold distance (default 200). |
| `--min-confidence F` | Detection threshold (default 0.50). |
| `--max-runtime S` | Auto-stop after S seconds (default 120). |
| `--max-frames N` | Auto-stop after N frames. |
| `--save-dir DIR` | Harvest a training dataset (frames + weak labels + previews). |
| `--quiet` | Log only state changes, not every frame. |

LED status: **green** tracking · **white** frozen (prey) · **purple** fleeing ·
**red** searching · **cyan** holding (at stop distance).

---

## RC drive — `rcbot` (`python3 -m catchaser.rc`)

Open **`http://192.168.0.41:5001`** in a browser. Live camera + drive controls.
Hold keys to move; release to stop. Combine keys for diagonals (mecanum).

| Key(s) | Action | Key(s) | Action |
|--------|--------|--------|--------|
| `8` / `↑` | forward | `4` | rotate left (CCW) |
| `2` / `↓` | back | `6` | rotate right (CW) |
| `a` / `←` | slide left | `s` / `→` | slide right |
| `space` / `5` | **STOP** | `+` / `-` | faster / slower |
| `j` | pan left | `k` | pan right |
| `i` | tilt up | `m` | tilt down |
| `h` | camera home | `0` | set current pose as home (recalibrate) |

Status bar shows a **pwr** indicator (Pi undervoltage flag): **OK** (green) =
rail fine · **⚠ dipped** (amber) = under-voltage happened earlier (likely a
motor-stall brownout) · **⚠ LOW NOW** (red) = rail sagging right now → back off
speed / charge. Numpad 4/6 rotate with NumLock on OR off.

Safety: releasing keys stops the wheels; a server dead-man stops them if the
browser goes quiet (~0.6 s); closing the tab stops the robot.

---

## Data pipeline (harvest → triage → stats)

```bash
./harvest.sh [seconds]                 # chase + collect a dataset (default 180s)
./review.sh                            # triage newest dataset → http://192.168.0.41:5000
python3 -m catchaser.review datasets/<name>            # triage a specific dataset
python3 -m catchaser.review datasets/<name> --fix-traps  # reopen old mislabeled "misses"
python3 -m catchaser.dataset_stats                     # precision/recall + confusion matrix
```

Review taps: **g** good · **n** no-cat · **f** fix box · **m** missed (cat present,
no box) · **u** undo. On no-box frames only No-cat / Missed are offered.

---

## Checks & calibration

```bash
python3 -m catchaser.hello                 # full smoke test (add --no-motion to skip wheels)
python3 -m catchaser.benchmark             # detector Hz (add --camera for capture+detect)
python3 -m catchaser.pan_check             # re-measure the pan servo direction sign
i2cdetect -y 1                             # should show 0x2b (MCU) — needs i2c-tools

# Rotation feedback (optical flow — measures ACTUAL yaw, no encoders):
python3 -m catchaser.flowcheck --deg 90    # closed-loop: turn until flow says 90 deg (CCW +)
python3 -m catchaser.flowcheck --deg -90   # 90 deg clockwise
python3 -m catchaser.flowcheck --open 1.5  # open-loop spin 1.5s, report measured yaw (shows the error)

# Wheels-off motor/mecanum diagnostic (watch the wheels):
python3 -m catchaser.wheelcheck --describe # print the wheel pattern for each move
python3 -m catchaser.wheelcheck --each     # spin motors 0,1,2,3 one at a time (verify IDs)
python3 -m catchaser.wheelcheck --move strafe_right   # or rotate_right, forward, ...
python3 -m catchaser.wheelcheck --all      # every primitive in sequence
```

**Rotate vs. strafe** (they're different!): *rotate right* spins in place
(left wheels forward, right wheels back = `+ + - -`); *strafe right* slides
sideways (`+ - - +`, a diagonal the mecanum rollers turn into lateral motion).
If strafe feels like a rotation/shuffle, the mecanum wheels are likely mounted
wrong — from above the roller axes must form an **X**. Use `wheelcheck` to see.

---

## Dev machine (not the Pi)

```bash
python3 -m pytest tests/ -q                                   # full suite
python3 -m pytest tests/test_catchaser_chase.py -q            # chase controller
python3 -m catchaser.chase_sim --scenario approach --pan --prey --verbose   # PyBullet
```

---

## Maintenance / undo

```bash
# Static IPs (Pi-side): undo either with method auto
sudo nmcli connection modify LomasV_5G ipv4.method auto            # WiFi back to DHCP
sudo nmcli connection modify "Wired connection 1" ipv4.method auto # ethernet back to DHCP
```
