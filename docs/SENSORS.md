# Sensor & Actuator Inventory — Cat Chaser 3000

What the robot has, what we currently use, and where fusing signals could buy
new information. Based on the RASPBOT-V2 hardware + a live I2C scan of this unit
(2026-09-07): bus 1 has **`0x2b` (MCU)** and **`0x3c` (OLED)** — **no IMU at
`0x69`** (not fitted).

## Inventory

| Sensor / actuator | Bus / where | Used by us? | Notes |
|---|---|---|---|
| USB camera 640×480 | Pi USB | ✅ heavily | detector + optical-flow yaw. The one rich sensor. Brownout-prone. |
| Ultrasonic (forward, mm) | MCU `0x2b` regs 0x1A/1B | ✅ | stop@200 mm, prey-flee trigger, rcbot readout. Single forward cone. |
| IR remote receiver | MCU reg 0x0C | ✅ | kill switch (any key). Idles at 0xFF. |
| 4-ch line tracker (down IR) | MCU reg 0x0A | ❌ **unused** | `sensors.read_line_tracker()`. Points at the floor. |
| 4 mecanum motors | MCU | ✅ | open-loop, **no encoders** — commanded ≠ actual. |
| 2 servos (pan/tilt) | MCU | ✅ | pan tracking, rcbot. Pan center 73°, tilt rest 25°. |
| 14 WS2812B LEDs | MCU | ✅ | status colors. |
| Buzzer | MCU reg 0x06 | ⚠️ hello only | not used in chase/play. |
| OLED display (SSD1306) | I2C `0x3c` | ❌ (boot script owns it) | shows the IP; **writable** — could show status. |
| ICM-20948 IMU (9-DOF) | I2C `0x69` | ❌ **not fitted** | code exists (`raspbot_slam/imu.py`); would need the add-on board. |
| Battery voltage | — | ❌ **no telemetry** | driver exposes no read; brownouts suggest it matters (see below). |
| USB mic / speaker | Pi USB | ❌ | vendor voice demos only; presence unconfirmed. |

## Fusion opportunities (ranked)

1. **Camera optical flow ⊕ motor command → closed-loop rotation.** *(building
   now — `catchaser/flow.py`)* No encoders means rotation is a guess, and on
   carpet the wheels scrub so the guess is badly wrong. Measuring real yaw from
   how the scene slides fixes "I can't rotate on my floors." Biggest immediate
   win. (Would fuse with an IMU gyro if one were fitted.)
2. **Line tracker → cliff/edge safety.** *(unused today; free win)* The 4
   down-facing IR sensors read floor-vs-void. A sudden "no floor" on the front
   pair = a table/stair edge → emergency stop before driving off. Pure safety,
   no new hardware. Highest value-per-effort after #1.
3. **Ultrasonic ⊕ detector bbox → distance.** Cross-check the cat's range: the
   bbox height gives a rough monocular distance, the ultrasonic gives a metric
   one when the cat is dead ahead. Fusing sharpens the flee/approach timing and
   flags "big bbox but far ultrasonic = false positive / not the cat".
4. **Battery voltage → power-aware behavior.** *(needs a readable register —
   investigate)* If we can read pack voltage, cap speed when low and warn
   before the brownout that kills the camera mid-chase. Directly targets
   today's blackout. TODO: probe MCU registers / vendor demos for a battery read.
5. **Buzzer → cat engagement.** A chirp during the prey FREEZE (the pounce
   bait) or on FLEE could pull the cat in. Behavioral, cheap.
6. **OLED (`0x3c`) → glanceable status.** Show state / cat confidence / battery
   without a laptop. Careful: a boot script already writes the IP to it.
7. **IMU (if added) → gyro yaw + pickup/bump + compass.** Would make rotation
   robust even when the camera is blinded, and detect being picked up. Requires
   the ICM-20948 add-on (I2C 0x69, no bus conflict). See SLAM_PLAN.md Phase 5.

## Known gaps / cautions

- **Power/brownout:** high motor-stall current (mecanum on carpet) sags the 5V
  rail and can reset the USB camera → black stream. Low battery makes weak
  rotation *and* dropouts. No battery telemetry yet to warn us. Charge first.
- **No encoders / no IMU:** all odometry is open-loop or vision-derived.
- **Single ultrasonic cone** (forward only): no side/rear obstacle sensing.
