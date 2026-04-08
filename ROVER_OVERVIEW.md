# Yahboom RASPBOT-V2 Rover -- Codebase Overview

**Product page & docs:** https://www.yahboom.net/study/RASPBOT-V2

---

## Directory Structure

```
rover/
├── Raspbot-V2.STEP                          CAD model (69MB)
├── Download APP.pdf                         Mobile app docs
├── Raspbot_V2-Manual/                       Hardware manual (29 subdirs)
└── RaspbotV2-Code/
    ├── Python driver library/
    │   └── py_install/
    │       ├── Raspbot_Lib/Raspbot_Lib.py   Core driver (481 lines)
    │       └── setup.py                     v0.0.2
    │
    ├── Library Files/
    │   ├── KMachineLearning-master/         ML utilities
    │   ├── MNIST_data dataset/              Digit recognition data
    │   ├── Opencv classifier/              Haar cascades, etc.
    │   └── Chassis driver library/          py_install.zip
    │
    ├── Appendix/
    │   ├── speech_py_install_v0.0.3/        Speech lib (TTS/STT)
    │   ├── yolov3/                          YOLOv3-tiny weights + config
    │   └── speech_music/                    192 audio files
    │
    └── Source Code/.../project_demo/
        ├── 03.Basic_car_course/             10 notebooks + 6 RGB demos
        ├── 04.Car_motion_control/           5 notebooks (mecanum)
        ├── 05.Comprehensive_gameplay/       6 notebooks (line follow, avoidance)
        ├── 06.Open_source_cv_fundamentals/  35 notebooks (OpenCV + ML)
        ├── 09.AI_Big_Model/                 30+ scripts (LLM agent system)
        └── 10.Basic_voice_control/          13 scripts (speech-driven)
```

---

## Hardware Summary

| Component         | Detail                                        |
|--------------------|-----------------------------------------------|
| **MCU interface**  | I2C bus 1, address `0x2B`                     |
| **Motors**         | 4 DC (mecanum wheels), speed -255..+255       |
| **Servos**         | 2 (0-180 deg; servo 2 limited to 110 deg)    |
| **RGB LEDs**       | 14 WS2812B addressable, 7 preset colors       |
| **Ultrasonic**     | Distance sensor (registers 0x1A/0x1B)         |
| **Line tracking**  | 4 IR sensors (register 0x0A)                  |
| **IR remote**      | Receiver (register 0x0C)                      |
| **Buzzer**         | On/off (register 0x06)                        |
| **Camera**         | Raspberry Pi camera (for CV / AI)             |
| **Microphone**     | USB mic for speech input                      |
| **Speaker**        | Audio output for TTS                          |

### I2C Register Map

| Register | Function                   |
|----------|----------------------------|
| 0x01     | Motor control (4 motors)   |
| 0x02     | Servo control (2 servos)   |
| 0x03-04  | RGB LED global color       |
| 0x06     | Buzzer on/off              |
| 0x08-09  | RGB LED brightness         |
| 0x0A     | Line tracking sensor state |
| 0x0C     | IR remote key code         |
| 0x1A-1B  | Ultrasonic distance (L/H)  |

---

## Core Driver Library

**File:** `RaspbotV2-Code/Python driver library/py_install/Raspbot_Lib/Raspbot_Lib.py` (481 lines)

### Key Classes and Methods

**`Raspbot` class** -- main hardware interface:

| Method                        | Purpose                                       |
|-------------------------------|-----------------------------------------------|
| `Ctrl_Muto(id, speed)`       | Set motor `id` (0-3) to `speed` (-255..255)   |
| `Ctrl_Servo(id, angle)`      | Set servo `id` (1-2) to `angle` (0-180)       |
| `Ctrl_Buzzer(on_off)`        | Buzzer control (1=on, 0=off)                  |
| `Ctrl_WS2812B(num, R, G, B)` | Set LED `num` color (0=all, 1-14=individual)  |
| `Ctrl_WS2812B_Brig(brig)`    | Set LED brightness (0-255)                    |
| `Ctrl_ReadUltrasonicData()`  | Read ultrasonic distance (cm)                 |
| `Ctrl_ReadLineTrackerData()` | Read 4 line-tracking sensor bits               |
| `Ctrl_ReadIR_Data()`         | Read IR remote key code                       |

**`LightShow` class** -- LED animation effects:

| Effect              | Description                    |
|---------------------|--------------------------------|
| `river_light()`     | Flowing/running light          |
| `breathing_light()` | Fade in/out                    |
| `gradient_light()`  | Color gradient sweep           |
| `random_running()`  | Random color per LED           |
| `starlight()`       | Twinkling star effect          |

### Installation

```bash
cd "RaspbotV2-Code/Python driver library/py_install"
sudo python3 setup.py install
```

---

## Motion Control

**File:** `McLumk_Wheel_Sports.py` (in project 09/10 directories)

Mecanum wheel kinematics for omnidirectional movement:

| Function                     | Motion              |
|------------------------------|----------------------|
| `move_forward(speed)`        | Forward              |
| `move_backward(speed)`       | Backward             |
| `move_left(speed)`           | Strafe left          |
| `move_right(speed)`          | Strafe right         |
| `move_upper_left(speed)`     | Diagonal upper-left  |
| `move_lower_left(speed)`     | Diagonal lower-left  |
| `move_upper_right(speed)`    | Diagonal upper-right |
| `move_lower_right(speed)`    | Diagonal lower-right |
| `rotate_left(speed)`         | Rotate CCW           |
| `rotate_right(speed)`        | Rotate CW            |
| `set_deflection(speed, ang)` | Vector-based motion  |
| `stop()`                     | Stop all motors      |

Motor mapping: L1=front-left(0), L2=rear-left(1), R1=front-right(2), R2=rear-right(3).

---

## AI / LLM Agent System (Project 09)

Multi-layer architecture for natural-language rover control:

```
User Input (Speech / Text)
    |
Decision Agent  -- Qwen VL parses intent into action plan
    |
Execution APIs  -- Specialized modules per task
    |
Motion / Sensor Primitives
    |
I2C Hardware
```

### Supported LLM Backends

| Provider       | Config key         | Models                    |
|----------------|--------------------|---------------------------|
| Alibaba Qwen   | `TONYI_key`        | qwen-vl-max, qwen-turbo  |
| Spark/XingHuo  | `XINGHOU_KEY`      | spark variants            |
| OpenRouter     | `OPENROUTER_KEY`   | various (foreign access)  |
| Dify           | `DIFY_SWITCH`      | custom workflows          |

### Agent Modules

| Module                      | Purpose                              |
|-----------------------------|--------------------------------------|
| `Car_decision_agent.py`     | NL instruction -> JSON action plan   |
| `Car_execute_api.py`        | Execute movement commands            |
| `Car_image_api.py`          | Camera capture + image analysis      |
| `Car_audio.py`              | Recording, VAD, playback             |
| `Car_tts.py` / `*_tongyi_*` | Text-to-speech (multiple backends)  |
| `Car_speak_iat.py`          | Speech-to-text                       |
| `Track_color_Follow_api.py` | HSV color tracking with PID          |
| `Track_Face_Follow_api.py`  | Face detection + following           |
| `Track_Food_api.py`         | Food/object detection + tracking     |
| `Car_avoid_api.py`          | Ultrasonic obstacle avoidance        |
| `Car_music_api.py`          | Music playback                       |

---

## Computer Vision (Project 06)

35 Jupyter notebooks covering:

| Topic Area                | Notebooks | Key techniques                        |
|---------------------------|-----------|---------------------------------------|
| A. Introduction           | 6         | OpenCV basics, TensorFlow intro       |
| B. Geometric Transforms   | 8         | Scale, crop, rotate, affine, warp     |
| C. Image Processing       | 7         | Drawing, text, color spaces           |
| D. Image Enhancement      | 8         | Blur, edge detect, erode/dilate       |
| E. Machine Learning       | 6         | KNN + SVM digit recognition (MNIST)   |

### Object Detection

- **YOLOv3-tiny** (pre-trained COCO, 80 classes) in `Appendix/yolov3/`
- **Haar cascades** for face detection in `Library Files/Opencv classifier/`
- **HSV color tracking** with PID feedback for real-time following

---

## Voice Control (Project 10)

13 scripts across 5 categories:

| Demo                       | What it does                              |
|----------------------------|-------------------------------------------|
| Speech control sport       | Voice commands for movement               |
| Speech control RGB         | Voice-driven LED colors                   |
| Speech color identify      | Say a color, rover finds it               |
| Speech car line patrol     | Voice-activated line following            |
| Speech track color/face    | Voice-activated object/face tracking      |

### Serial Protocol (Voice Module)

- Baud: 115200
- Packet: `[0xAA, 0x55, CMD_ID, 0x00, 0xFB]`
- Port: `/dev/ttyUSB0` or `/dev/myspeech`

---

## PID Control

**File:** `PID.py` (in project 09)

Two implementations:

| Class             | Type          | Use case                        |
|-------------------|---------------|----------------------------------|
| `IncrementalPID`  | Incremental   | Smooth delta-based correction   |
| `PositionalPID`   | Positional    | Absolute error-based correction |

Used for color/face tracking to steer the rover toward a target centroid.

---

## Dependencies

### Python packages (from imports across codebase)

**Hardware:**
`smbus`, `serial`, `RPi.GPIO` (implied)

**Audio:**
`pyaudio`, `librosa`, `soundfile`, `scipy`

**Vision:**
`opencv-python` (cv2), `numpy`, `Pillow`

**ML/AI:**
`scikit-learn`, `openai`, `requests`

**System:**
`threading`, `subprocess`, `os`, `sys`, `time`, `base64`

---

## Stats

| Metric                 | Count  |
|------------------------|--------|
| Python files           | 138+   |
| Jupyter notebooks      | 70     |
| Core driver lines      | 481    |
| Demo categories        | 6      |
| AI agent modules       | 30+    |
| Audio files (speech)   | 192    |
| YOLO object classes    | 80     |
| Addressable LEDs       | 14     |
