# Rover — project notes for Claude

## The physical robot (Yahboom RASPBOT-V2)

- **WiFi (untethered):** Pi 5 built-in WiFi on `LomasV_5G`, DHCP from the
  TP-Link router — but **the lease moves between boots** (seen: `.35`, then
  `.113` on 2026-09-06). Find it by ping-sweeping port 22 on `192.168.0.x`,
  or fix it permanently with a **DHCP reservation** in the TP-Link admin page
  (recommended; MAC is the wlan0 one).
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
