# Rover — project notes for Claude

## The physical robot (Yahboom RASPBOT-V2)

- **WiFi (untethered): `192.168.0.35`** — Pi 5 built-in WiFi on `LomasV_5G`,
  DHCP from the TP-Link router. This is the address for floor runs.
- **Ethernet (when plugged): `192.168.0.32`** — same router.
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

## Hardware I2C sanity check

If the vendor driver ever fails to connect, verify the MCU is on the bus:

```bash
i2cdetect -y 1      # should show 0x2b (the RASPBOT-V2 MCU)
```

No `0x2b` → hardware/I2C problem, not a software bug. (Optional IMU, if fitted,
appears at `0x69`.)

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

## Tests

`python -m pytest tests/ -v` (needs `opencv-python-headless numpy pybullet`).
Cat-chaser controller tests: `python -m pytest tests/test_catchaser_chase.py -q`.
