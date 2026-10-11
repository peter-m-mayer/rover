# Rover — project notes for Claude

## The physical robot (Yahboom RASPBOT-V2)

- **WiFi (untethered): `192.168.0.41` — STATIC, fixed.** Pi 5 built-in WiFi on
  `LomasV_5G`, pinned Pi-side (`nmcli` manual 192.168.0.41/24, gw/dns
  192.168.0.1, metric 600, autoconnect). No more lease drift — the OLED and
  `ssh pi@192.168.0.41` agree every boot. (Earlier it roamed: `.27/.35/.113`.)
  Undo: `sudo nmcli connection modify LomasV_5G ipv4.method auto`. Not a router
  reservation (no TP-Link admin needed); wlan0 MAC `88:a2:9e:3f:96:80` if you
  ever do reserve it there instead.
- **Ethernet tether (static): `192.168.0.40`** — bound to the `Wired
  connection 1` profile (eth0 MAC `88:a2:9e:3f:96:7f`), `manual`
  192.168.0.40/24, gw/dns 192.168.0.1, route-metric 700 (so WiFi stays the
  default route when both are up), autoconnect on. Plug the Pi into the router
  (or a LAN switch) and it's always `ssh pi@192.168.0.40`. Undo:
  `sudo nmcli connection modify "Wired connection 1" ipv4.method auto`.
  Picked .40 because .40–.49 were all free (below the DHCP pool); if it ever
  collides, reserve it on the router or pick another sub-pool address.
- **Phone-driven operation:** helper scripts live on the Pi —
  `./harvest.sh [secs]` (10 s countdown, chase + dataset harvest, prints
  summary) and `./review.sh` (triage server on the newest dataset; open
  `http://<pi-ip>:5000`). SSH from any phone SSH app: `pi`/`yahboom`.
- **SSH:** `ssh pi@192.168.0.35` (or `.32` wired) — user `pi`, password
  **`yahboom`**; the dev machine's key is installed, so no password needed.
  - ⚠️ The old **`192.168.1.11` is dead — do NOT use it.** Post-mortem: that
    was the Pi's *WiFi* address from a leftover NetworkManager profile named
    `Raspbot` (hidden SSID, isolated network — ARP answered but all IP traffic
    dropped, which burned hours). That profile is now **autoconnect-disabled**;
    the Pi joins `LomasV_5G` on boot instead. The OLED may show a stale address
    right after boot — trust `nmcli`/router DHCP list, not the OLED.
- **Board:** Raspberry Pi 5, **64-bit OS (aarch64)** → **ONNX Runtime** is the
  inference path for the cat detector (onnxruntime wheels are aarch64-only; this
  board qualifies).
- **Spare rescue SD card** exists in the drawer: RPi OS 2024-07-04, SSH
  pre-enabled, I2C already configured. Use it if the primary card gets wedged.
- **Boot card replaced 2026-10-11.** The old root was a microSD in a cheap USB
  reader that flipped bits (corrupt wheels, inconsistent reads). It's archived
  as "old, flaky", so don't reuse it. The Pi now boots a fresh Bookworm
  (2026-09-15 arm64) from a SanDisk Ultra 64 GB in its **built-in slot**.
  Rebuilt from `catChaser/pi_backup_2026-10-10/`:
  - static IPs as above
  - pip pins (numpy 1.26.4, opencv-headless 4.11.0.86, onnxruntime 1.18.1, scipy 1.14.0)
  - Raspbot_Lib
  - OLED and app autostart; `reservo.py` now rests the servos at pan 74 / tilt 34
  - `.bashrc` aliases and the phone scripts
  - datasets

  The Yahboom tutorials, Jupyter and the ROS 2 Docker image were not restored.
  The camera is the **USB** camera (`/dev/video0`), not CSI, so `rpicam-*` reports "no cameras".

## Hardware I2C sanity check

If the vendor driver ever fails to connect, verify the MCU is on the bus:

```bash
i2cdetect -y 1      # should show 0x2b (the RASPBOT-V2 MCU)
```

No `0x2b` → hardware/I2C problem, not a software bug. The OLED is `0x3c`. The
LSM9DS1 IMU is on its own bus, **i2c-3** (GPIO22/23, `dtoverlay=i2c3-pi5,pins_22_23`):
`i2cdetect -y 3` should show `0x1e` (magnetometer) and `0x6b` (accel/gyro).

## Gotchas found 2026-10-11

- **The vendor `Raspbot_Lib` swallows I2C write errors** (prints and returns). A
  dropped motor "stop" left the robot turning. `Actuators` now writes the MCU
  registers directly with retries and raises `I2CWriteError`. Don't call
  `bot.Ctrl_Muto`/`Ctrl_Servo` directly in new code.
- **`pkill -f <pattern>` over SSH kills its own shell** when the pattern appears in
  the command line. Use a bracket pattern: `pkill -f "[a]uto_camera_cal"`.
- **OpenCV 5 on the PC vs 4.11 on the Pi:** `findChessboardCorners` returns (N, 2)
  vs (N, 1, 2), and `calcBackProject` returned zeros on 5. Reshape explicitly, and
  use numpy for histogram lookups.
- **`CALIB_CB_FAST_CHECK` rejected every view** of the printed board on this camera.
  Don't use it.

## Vendor driver (not in git)

Installed on the Pi from the Yahboom package:
`cd "RaspbotV2-Code/Python driver library/py_install" && sudo python3 setup.py install`
Provides `Raspbot_Lib.Raspbot`; `catchaser/hw.py` discovers it and falls back to
mock mode off-robot.

## Layout

- `raspbot_slam/` — visual-SLAM + navigation package (46 files); `config.py`
  holds all tunables. The `simulator/` subpackage is a PyBullet drop-in for the
  hardware classes.
- `catchaser/` — Cat Chaser 3000 (detector + chase controller). See
  **`catchaser/CLAUDE.md`** for cat-chaser specifics, tuning, and the sim harness.

**[docs/COMMANDS.md](docs/COMMANDS.md) is Peter's living command/alias reference —
keep it current whenever a command, flag, or alias changes.**

## Tests

`python -m pytest tests/ -v` (needs `opencv-python-headless numpy pybullet`).
Cat-chaser controller tests: `python -m pytest tests/test_catchaser_chase.py -q`.
