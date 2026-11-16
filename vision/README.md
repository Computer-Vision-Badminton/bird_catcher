# Bird Catcher vision pipeline (Raspberry Pi 5 + AI HAT+ 2)

This branch holds only what the shuttle-tracking pipeline needs to run on the Pi, plus 20 recorded test shots to replay.
It predicts where the shuttle will land, frame by frame, from the ZED stereo camera.

**Current status (2026-11-15, Pi 5 + Hailo-10H):** 50.5 ms per stereo frame (about 20 fps), landing predicted within
25 cm on average (23 cm median) on the 20 test shots. The first landing estimate is ready 7–10 frames (~0.12–0.17 s) after the hit.

**Not in this branch yet:** a live camera loop and automatic hit detection (each test shot starts at a hand-labelled hit
frame). For motor testing, `vision/shot_landing.py` replays a recorded shot and returns (or sends to the ESP32) its
predicted landing point; see "Landing point for the motors" below.

## What runs where

| Step | Runs on | File |
| --- | --- | --- |
| Gray conversion of both eyes | Pi CPU | `pipeline.py` |
| Motion view + YOLO11n bird detector (640 × 640 crop per eye) | **Hailo-10H** | `models/bird_yolo11n_v1_gray_stack.hef`, `detectors.py` |
| Duplicate-box removal | Pi CPU (HailoRT) | inside the `.hef` |
| Crop aiming, rising-chain tracking, left/right matching | Pi CPU | `crop_tracking.py`, `bird_tracker.py` |
| Stereo triangulation to court coordinates | Pi CPU | `court_3d.py`, `calibrate_net.py` |
| Flight fit (gravity + drag, RK4) and landing point | Pi CPU, C++ | `flight_model.py`, `flight_cpp/flight_core.cpp` |

## Files

```
models/bird_yolo11n_v1_gray_stack.hef   compiled detector for the Hailo-10H (HailoRT 5.1.1 tested)
models/bird_yolo11n_v1_gray_stack.onnx  the same detector for a regular CPU (onnxruntime), for testing without a Hailo
pi_test_package/clips/shot_*.avi        20 recorded test shots (original ZED frames, 2560 x 720, 60 fps)
pi_test_package/manifest.json           per shot: hit frame, frame times, measured landing; tuned drag + correction
pi_test_package/net_calibration.json    camera position and lens calibration for these recordings
vision/pipeline.py                      ShotPipeline: one stereo frame in, updated landing estimate out
vision/experiment/pi_realtime_test.py   replays the 20 shots through ShotPipeline and times every stage
vision/deploy/setup_pi.sh               one-time Pi setup (Python environment + C++ flight core)
vision/shot_landing.py                  test helper: replay a shot, get (or send) its landing point
vision/esp32_link.py                    Pi -> ESP32 message format and UART sending
vision/deploy/hailo_check.py            quick check that the .hef loads and finds the bird
vision/flight_cpp/                      C++ flight simulation and its build script
```

## Setup on the Pi (once)

Needs Raspberry Pi OS Trixie (64-bit), the AI HAT+ 2, and internet access.

```bash
sudo apt update
sudo apt install dkms hailo-h10-all g++
sudo reboot

hailortcli fw-control identify          # should report Device Architecture: HAILO10H

git clone https://github.com/Computer-Vision-Badminton/bird_catcher
cd bird_catcher
git checkout vision-pipeline
bash vision/deploy/setup_pi.sh          # should end with "flight fit engine: c++" and "hailo_platform 5.1.1"
```

Use the official 27 W USB-C power supply: the Pi has shut down under load on a weaker supply.

## Run it

Check the detector on one known frame (about 10 seconds):

```bash
.venv/bin/python vision/deploy/hailo_check.py models/bird_yolo11n_v1_gray_stack.hef --input gray-stack
```

Expected: `hailo finds: [(≈320, ≈266, ≈0.88)]` and `hailo on an empty crop: []`.

Replay all 20 test shots through the full pipeline (a few minutes):

```bash
.venv/bin/python vision/experiment/pi_realtime_test.py --detector hailo --model models/bird_yolo11n_v1_gray_stack.hef --input gray-stack
```

`--shots 37 74` runs only some shots. The run prints time per stage, the per-shot landing prediction and miss, and saves
`pi_test_package/results_*.json`.

## On a regular computer (no Hailo)

The `.hef` only runs on a Hailo chip. On a Mac, Linux or Windows computer use the same network exported as ONNX,
`models/bird_yolo11n_v1_gray_stack.onnx`, which runs on the CPU through `onnxruntime`. It gives the same accuracy
(23.9 / 19.6 cm mean / median on the 20 test shots on a MacBook Air) but CPU speed (about 140 ms per frame there), so
use it to check behaviour and predictions, not timing.

Mac or Linux:

```bash
git clone https://github.com/Computer-Vision-Badminton/bird_catcher
cd bird_catcher
git checkout vision-pipeline
python3 -m venv .venv
.venv/bin/pip install numpy scipy opencv-python onnxruntime pybind11
bash vision/flight_cpp/build.sh
.venv/bin/python vision/experiment/pi_realtime_test.py --detector onnx --model models/bird_yolo11n_v1_gray_stack.onnx --input gray-stack
```

`build.sh` needs a C++ compiler (`xcode-select --install` on a Mac, `sudo apt install g++` on Linux). It is optional:
without it the flight fit runs in Python, with the same results but about 20× slower per fit.

Windows (PowerShell): the same steps with `python -m venv .venv`, `.venv\Scripts\pip install ...` and
`.venv\Scripts\python vision\experiment\pi_realtime_test.py ...`; skip `build.sh` (the Python flight fit is used).

## Using the landing prediction

`ShotPipeline.push(frame_index, stereo_frame, time_s)` takes one 2560 × 720 BGR stereo frame and its capture time in
seconds. It returns `None` until the shot window starts, then a dict per frame:

| Key | Meaning |
| --- | --- |
| `landing_cm` | latest landing estimate `[x, y]` in cm, or `None` until there are enough 3D points (≈7 frames after the hit) |
| `landing_from_frame` | the frame that estimate was computed on |
| `stages_ms` | time spent in each stage for this frame |

`ShotPipeline.finish()` returns the final landing from all points of the shot. The estimate is refreshed every frame
and is usually within ~25 cm by 15 frames (0.25 s) after the hit.

Court coordinates (centimetres, origin on the floor directly under the net on the middle line):

- **x**: from the middle line toward the sideline; positive toward the post on the right of the camera image
- **y**: from the net toward the back line; positive on the camera's (landing) side
- **z**: height above the floor (not part of the landing output)

These are court coordinates, not robot coordinates: converting a landing point into a motor target needs the robot's
own position on the court.

See `vision/experiment/pi_realtime_test.py` (`run_shot`) for a complete example of feeding frames and reading estimates.

## Landing point for the motors (test helper)

`vision/shot_landing.py` replays one recorded shot through the full pipeline and returns where it predicts the shuttle
lands. It picks the Hailo model on the Pi and the ONNX model anywhere else. This is a stand-in for the live camera: the
robot gets real predictions from real recorded flights.

From Python (run from the `vision/` folder, or add it to `sys.path`):

```python
from shot_landing import ROBOT_HOME_CM, landing_for_shot

result = landing_for_shot(37)
result.landing_cm            # (-271.7, 614.2)  final landing, court cm (x, y)
result.offset_from_home_cm   # (-121.7, 284.2)  landing minus ROBOT_HOME_CM
result.measured_cm           # (-260.0, 612.0)  where it really landed (tape measure)
result.miss_cm               # 11.9
result.estimates             # every in-flight estimate: frame, seconds_after_hit, x_cm, y_cm
```

`landing_for_shot(37, on_estimate=callback)` calls `callback(estimate)` as soon as each new in-flight estimate is ready,
the way a live system would feed the robot (first one about 0.12–0.17 s after the hit).

From the command line:

```bash
.venv/bin/python vision/shot_landing.py                  # list the 20 test shots
.venv/bin/python vision/shot_landing.py 37 74            # print their landings
.venv/bin/python vision/shot_landing.py --all --send-uart --pause 5
```

`--send-uart` sends each shot's final landing to the ESP32 (`/dev/serial0`, 921600 baud; another port can follow the
flag). `--stream` also sends every in-flight estimate. `--pause 5` waits 5 s between shots so the robot can drive back.

**Robot home** is `ROBOT_HOME_CM = (-150, 330)`: 150 cm from the middle line on the left-post side, 330 cm from the net
(from `docs/tasks/measurement_note.md`). Change it in `shot_landing.py` if the robot starts elsewhere.

### All 20 test shots

Predicted final landing for every test shot, against where it really landed (tape measure). Court cm, x from the
middle line, y from the net. "Pi" is the Hailo model (INT8) on the Pi 5, 2026-11-15; "computer" is the ONNX model
(full precision) that `shot_landing.py` uses without a Hailo. The two differ by a few cm because the Hailo model runs in 8-bit.

| Shot | Measured x, y | Predicted on Pi x, y | Pi miss | Predicted on computer x, y | Computer miss | First estimate after hit |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | -171.0, 445.0 | -181.3, 476.0 | 33 cm | -181.8, 470.6 | 28 cm | 0.17 s |
| 6 | -56.0, 590.0 | -57.2, 626.3 | 36 cm | -57.1, 625.9 | 36 cm | 0.12 s |
| 12 | -131.0, 599.0 | -140.6, 621.2 | 24 cm | -139.3, 616.5 | 19 cm | 0.12 s |
| 18 | -126.0, 658.0 | -124.8, 634.9 | 23 cm | -124.7, 631.4 | 27 cm | 0.12 s |
| 24 | -58.0, 655.0 | -71.4, 664.2 | 16 cm | -71.0, 662.3 | 15 cm | 0.18 s |
| 25 | -216.0, 566.0 | -209.5, 627.5 | 62 cm | -210.0, 617.5 | 52 cm | 0.12 s |
| 31 | -67.0, 587.0 | -78.6, 606.3 | 23 cm | -75.0, 594.8 | 11 cm | 0.12 s |
| 37 | -260.0, 612.0 | -271.2, 614.4 | 11 cm | -271.7, 614.2 | 12 cm | 0.13 s |
| 43 | -110.0, 570.0 | -124.3, 541.3 | 32 cm | -124.2, 536.0 | 37 cm | 0.12 s |
| 49 | -202.0, 650.0 | -195.2, 644.6 | 9 cm | -194.5, 644.8 | 9 cm | 0.13 s |
| 50 | -168.0, 621.0 | -162.5, 632.2 | 12 cm | -162.3, 627.5 | 9 cm | 0.13 s |
| 56 | -194.0, 520.0 | -201.9, 519.9 | 8 cm | -201.6, 523.8 | 8 cm | 0.12 s |
| 62 | -21.0, 578.0 | -23.7, 566.4 | 12 cm | -22.9, 558.3 | 20 cm | 0.12 s |
| 68 | -59.0, 509.0 | -74.6, 461.9 | 50 cm | -74.7, 467.5 | 44 cm | 0.12 s |
| 74 | -269.0, 602.0 | -272.3, 621.2 | 19 cm | -273.7, 619.4 | 18 cm | 0.12 s |
| 75 | -270.0, 523.0 | -276.1, 566.2 | 44 cm | -277.5, 564.5 | 42 cm | 0.12 s |
| 81 | -232.0, 598.0 | -221.0, 587.3 | 15 cm | -220.7, 589.3 | 14 cm | 0.12 s |
| 87 | -170.0, 548.0 | -177.2, 535.8 | 14 cm | -177.5, 537.9 | 13 cm | 0.13 s |
| 93 | -264.0, 562.0 | -255.2, 531.9 | 31 cm | -256.9, 526.5 | 36 cm | 0.17 s |
| 99 | -227.0, 564.0 | -231.9, 534.6 | 30 cm | -231.5, 536.9 | 28 cm | 0.12 s |
| **all 20** | | | **25.2 / 23 cm** mean / median | | **23.9 / 19.5 cm** mean / median | |

Shots 25, 68 and 75 miss by 40–60 cm on both machines, so their error comes from tracking and the flight model, not the
Hailo. `python vision/shot_landing.py --all` reproduces the computer columns.

### Wiring and enabling the Pi UART

| Pi 5 | ESP32-S3 |
| --- | --- |
| GPIO14 TX (header pin 8) | GPIO18 (UART1 RX) |
| GPIO15 RX (header pin 10) | GPIO17 (UART1 TX) |
| GND (header pin 6) | GND |

Enable it once on the Pi: `sudo raspi-config` → Interface Options → Serial Port → login shell over serial: **No**,
serial port hardware: **Yes**, then reboot. The port is `/dev/serial0`.

Sending needs `pyserial`. If the Pi was set up before `shot_landing.py` existed, run `bash vision/deploy/setup_pi.sh`
once more to install it. On a regular computer, printing landings needs nothing extra; to send through a USB-serial
adapter, run `.venv/bin/pip install pyserial` and pass the adapter's port, e.g. `--send-uart /dev/ttyUSB0`
(`COM3` on Windows).

### Message format (Pi → ESP32)

One frame per landing estimate, as planned in `docs/tasks/hardware/communication.md`:

| Bytes | Field | Value |
| --- | --- | --- |
| 2 | sync | `0xAA 0x55` |
| 1 | type | `0x01` = landing target |
| 1 | len | payload length, `5` |
| 1 | seq | 0–255, +1 per frame, wraps; a gap means a lost frame |
| 2 | x | int16 little-endian, **millimetres** from the middle line (positive toward the right post in the camera image) |
| 2 | y | int16 little-endian, **millimetres** from the net (positive on the robot's side) |
| 1 | flags | bit 0 = 1: final landing of the shot; 0: in-flight estimate |
| 2 | crc | CRC-16/CCITT-FALSE (poly 0x1021, init 0xFFFF, no reflection) over type, len, seq and payload; little-endian |

Example, final landing x = −271.9 cm, y = 613.8 cm, seq 1: `aa 55 01 05 01 61 f5 fa 17 01 39 74`.

CRC on the ESP32 (check value: the bytes of `"123456789"` give `0x29B1`):

```cpp
uint16_t crc16Ccitt(const uint8_t *data, size_t length) {
  uint16_t crc = 0xFFFF;
  for (size_t i = 0; i < length; i++) {
    crc ^= static_cast<uint16_t>(data[i]) << 8;
    for (int bit = 0; bit < 8; bit++) {
      crc = (crc & 0x8000) ? static_cast<uint16_t>((crc << 1) ^ 0x1021) : static_cast<uint16_t>(crc << 1);
    }
  }
  return crc;
}
```

The Python side of the format is `vision/esp32_link.py` (`frame`, `decode_frames`, `decode_landing`), handy for checking
what the ESP32 receives.
