# Bird Catcher vision: guide for the robot side

The vision side runs on the Raspberry Pi 5 with the AI HAT+ 2 (Hailo-10H). It watches the ZED stereo camera, finds the
shuttle, and tells the ESP32-S3 where the shuttle will land: a stream of landing estimates while the shuttle flies,
then one final landing per shot, sent over UART.

This guide is for the robot and firmware side. It covers what you get, how to test it on the Pi, how to drive the
ESP32 from recorded shots, and how to receive the landings on the ESP32. The vision internals are summed up at the end
in "How it works".

Commands are run from the repository root (`~/bird_catcher` on the Pi). Commands marked **Pi only** need the Hailo.

## 1. What the vision side gives the robot

For each shot:

1. **In-flight estimates.** A new landing estimate (x, y) on almost every camera frame while the shuttle flies. The
   first one is coarse; they settle as the flight goes on.
2. **A final landing.** One message with the final flag set, once tracking for that shot ends.

Positions are court coordinates (section 5): centimetres inside the vision code, **millimetres** in the UART messages.
The vision side does not know where the robot is; turning a landing into a motor target is the robot's job.

### Current numbers

All measured on the 20 test shots of the `clears_nov10` session (recorded video, tape-measured landings).

| What | Number | Where it comes from |
| --- | --- | --- |
| Hits found without labels (live mode) | 20 / 20, 19 within ±1 frame, 0 false triggers | Pi, `live_replay_test.py`, 2026-11-17 |
| Final landing miss (live mode) | 15.0 cm mean, 14.2 cm median | same run |
| First landing estimate | about 6 frames (100 ms) after the hit | same run |
| Final landing | about 40 frames (675 ms) after the hit | same run |
| Time per frame on the Pi | searching 17.6 ms, following 11.7 ms, tracking 11.7 ms | same run |
| Final landing miss, shots started at the labelled hit | 16.6 cm mean, 13.6 cm median; 6.2 ms of CPU work per frame | Pi, `pi_realtime_test.py` setup D + overlap + Kalman, 2026-11-16 |

The camera gives a frame every 16.7 ms (60 fps). Searching (no shuttle seen yet) is just over that; following and
tracking fit. The full run logs are in the "Bird Catcher: Real-Time Performance on the Pi" doc.

### Not done yet

- **The live ZED capture loop.** Everything above runs on recorded video. The robot can already be driven by real
  predictions from real recorded flights (section 3), but not from the camera live.
- Clears only: smashes and rallies are not tested.

## 2. Set up and test on the Pi

### One-time setup (Pi only)

Needs Raspberry Pi OS Trixie (64-bit), the AI HAT+ 2, internet access, and the official 27 W USB-C power supply (the
Pi has shut down under load on a weaker one).

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

`setup_pi.sh` makes the Python environment (`.venv`), installs `pyserial`, and builds the C++ flight core. Run it again
after a pull that changes `vision/deploy/requirements-pi.txt`. To rebuild only the C++ core:

```bash
bash vision/flight/core_source/build.sh
.venv/bin/python -c "from vision.flight import flight_model; print(flight_model.FLIGHT_ENGINE)"   # must print c++
```

From a Mac with the full repository, `vision/deploy/push_to_pi.sh` copies the code, the models and the test packages
to `catcher@catcher.local:~/bird_catcher` instead of a `git pull`.

### Check the detector (Pi only, about 10 s)

```bash
.venv/bin/python vision/deploy/hailo_check.py models/bird_yolo11n_v1_gray_stack.hef --input gray-stack
```

Expected: `hailo finds: [(≈320, ≈266, ≈0.88)]` and `hailo on an empty crop: []`.

### Replay the 20 test shots (Pi only, a few minutes)

Each shot starts at its hand-labelled hit frame. This is the quickest end-to-end check.

```bash
.venv/bin/python vision/experiment/pi_realtime_test.py --detector hailo --model models/bird_yolo11n_v1_gray_stack.hef --input gray-stack
```

It prints the time per stage, then each shot's landing and miss, and saves `test/pi_test_package/results_*.json`.
Expected (2026-11-15): about 23 cm mean miss. `--shots 37 74` runs only some shots.

The faster setup used for the numbers in section 1 adds the 320 model, overlap and the Kalman landing:

```bash
.venv/bin/python vision/experiment/pi_realtime_test.py --detector hailo --model models/bird_yolo11n_v1_gray_stack.hef --input gray-stack --follow-model models/bird_yolo11n_v1_gray_stack_320.hef --search-with-follow-model once-seen --overlap --landing-method kalman
```

Expected (2026-11-16): 16.6 / 13.6 cm mean / median miss.

### Live replay test (Pi only for the Hailo; about 1 minute)

This is the closest thing to the real robot: one continuous video of the 20 test shots with the gaps between them
(3,520 frames). Nothing is labelled; the live loop finds each hit itself, tracks the shot and gives a landing.

```bash
.venv/bin/python vision/experiment/live_replay_test.py run --detector hailo
.venv/bin/python vision/experiment/live_replay_test.py score --detector hailo
```

`run` prints a line at each hit and each final landing, then
`3520 frames in … s, 20 shots, 20 hits, 0 replay fallbacks []`. `score` prints the hit and landing summary. Expected
(2026-11-17): `hits: 20/20 found`, `false triggers 0`, landing miss mean 15.0, median 14.2 cm, and the per-state
times of section 1.

The test package `test/pi_live_test_package/` (about 600 MB) is not in git. If it is missing on the Pi, make it on the
Mac that holds the recordings (`data/`), then copy it over:

```bash
.venv/bin/python vision/experiment/make_live_test_package.py make
.venv/bin/python vision/experiment/make_live_test_package.py verify
vision/deploy/push_to_pi.sh
```

### Without a Hailo (Mac, Linux, Windows)

The `.hef` only runs on a Hailo chip. On a computer, the same networks run on the CPU as ONNX files. Results match the
Pi within a few cm (the Hailo runs in 8 bit), but timing does not: use it to check behaviour, not speed.

```bash
python3 -m venv .venv
.venv/bin/pip install numpy scipy opencv-python onnxruntime pybind11 pyserial
bash vision/flight/core_source/build.sh
.venv/bin/python vision/experiment/pi_realtime_test.py --detector onnx --model models/bird_yolo11n_v1_gray_stack.onnx --input gray-stack
.venv/bin/python vision/experiment/live_replay_test.py run
.venv/bin/python vision/experiment/live_replay_test.py score
```

`build.sh` needs a C++ compiler (`xcode-select --install` on a Mac, `sudo apt install g++` on Linux). Without it the
flight fit runs in Python: same results, about 20 times slower. On Windows use `.venv\Scripts\python` and skip
`build.sh`. On a MacBook Air M2 the live replay gives 20/20 hits and 17.8 / 15.1 cm (2026-11-17).

## 3. Drive the ESP32 from the Pi with test data

Two ways to send real predictions from recorded flights to the ESP32, both through `vision/robot/esp32_link.py`. Both
default to `/dev/serial0` at 921600 baud; give another port after `--send-uart` (for example a USB-serial adapter,
`/dev/ttyUSB0`, or `COM3` on Windows).

### One shot at a time: `shot_landing.py`

Replays recorded test shots (starting at the labelled hit) and sends their landings. Good for motor tests: you choose
the shot and the pause.

```bash
.venv/bin/python vision/robot/shot_landing.py                  # list the 20 test shots
.venv/bin/python vision/robot/shot_landing.py 37 74            # print their landings, send nothing
.venv/bin/python vision/robot/shot_landing.py --all --send-uart --pause 5
.venv/bin/python vision/robot/shot_landing.py 37 --send-uart --stream --landing-method kalman
```

| Option | What it does |
| --- | --- |
| `--send-uart [PORT]` | send each shot's final landing |
| `--stream` | also send every in-flight estimate as soon as it is ready |
| `--pause 5` | wait 5 s between shots, so the robot can drive back |
| `--landing-method kalman` | the Kalman landing used by live mode (default: `fit`) |
| `--baud` | default 921600 |

It uses the Hailo on the Pi and the ONNX model anywhere else. Checked on the Mac (2026-11-17) through a
pseudo-terminal: shot 37 with `--stream --landing-method kalman` sent 37 frames, seq 0 to 36, the last one final at
x −273.0 cm, y 609.8 cm (measured landing −260, 612).

From Python:

```python
from vision.robot.shot_landing import ROBOT_HOME_CM, landing_for_shot

result = landing_for_shot(37)
result.landing_cm            # final landing, court cm (x, y)
result.offset_from_home_cm   # landing minus ROBOT_HOME_CM
result.measured_cm           # (-260.0, 612.0), tape measure
result.miss_cm
result.estimates             # every in-flight estimate: frame, seconds_after_hit, x_cm, y_cm
```

`landing_for_shot(37, on_estimate=callback)` calls `callback(estimate)` as each in-flight estimate is ready.

### The whole live loop: `live_replay_test.py run --send-uart`

Runs the live loop on the 20-shot video (section 2) and sends landings the moment the loop produces them: the robot
sees exactly what live mode decides, hits found without labels included.

```bash
.venv/bin/python vision/experiment/live_replay_test.py run --detector hailo --send-uart              # Pi only: final landing per shot
.venv/bin/python vision/experiment/live_replay_test.py run --detector hailo --send-uart --stream     # Pi only: every estimate too
```

At the end it prints `sent N landing frames to PORT at BAUD baud`. Without `--send-uart` nothing is opened and the
loop runs exactly as before. The replay does not wait for camera time: on the Pi it handles the 58.7 s of video in
about 46 s, so messages arrive a little faster than they would live.

**What the ESP32 receives.** Checked on the Mac (ONNX, 2026-11-17) with a pseudo-terminal as the port and
`decode_frames` / `decode_landing` reading the other end (`--baud 115200`, because a Mac pseudo-terminal refuses
921600; a real UART does not care):

- with `--stream`: 736 frames (8,832 bytes) for the 20 shots, seq 0 to 223 with no gaps (wrapping at 255), 20 of them
  final; every frame equals the run's saved results;
- without `--stream`: 20 frames, one final landing per shot, seq 0 to 19.

Shot 37 with `--stream` is 37 frames: in-flight estimates from about 6 frames after the hit, then the final one. It is
the 8th shot of the video, so seq has already wrapped past 255 and starts at 2 here.

| Bytes | seq | x mm | y mm | final |
| --- | --- | --- | --- | --- |
| `aa 55 01 05 02 bc f4 e7 13 00 22 07` | 2 | −2884 | 5095 | no |
| `aa 55 01 05 03 49 f5 c2 15 00 a7 ef` | 3 | −2743 | 5570 | no |
| `aa 55 01 05 04 99 f4 f2 16 00 46 f6` | 4 | −2919 | 5874 | no |
| … 32 more estimates … | | | | |
| `aa 55 01 05 25 60 f5 b6 17 00 2c 95` | 37 | −2720 | 6070 | no |
| `aa 55 01 05 26 60 f5 b6 17 01 ed 4b` | 38 | −2720 | 6070 | **yes** |

The early estimates move by tens of cm; by the final one this shot is 13 cm from the measured landing (−260, 612 cm).
Plan the motors for that: start toward the early estimate, keep correcting, stop on the final.

### All 20 test shots (`shot_landing.py`, labelled hits, `fit` landing)

Predicted final landing against the measured one. Court cm. "Pi" is the Hailo model on the Pi 5 (2026-11-15);
"computer" is the ONNX model that `shot_landing.py --all` uses without a Hailo. The live mode with the Kalman landing
(section 1) is better: 15.0 / 14.2 cm.

| Shot | Measured x, y | Predicted on Pi x, y | Pi miss | Predicted on computer x, y | Computer miss | First estimate after hit |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | -171.0, 445.0 | -181.6, 471.6 | 29 cm | -181.8, 470.1 | 27 cm | 0.18 s |
| 6 | -56.0, 590.0 | -54.9, 611.9 | 22 cm | -57.0, 616.8 | 27 cm | 0.12 s |
| 12 | -131.0, 599.0 | -140.3, 614.9 | 18 cm | -140.1, 623.1 | 26 cm | 0.12 s |
| 18 | -126.0, 658.0 | -124.7, 635.6 | 22 cm | -124.7, 632.1 | 26 cm | 0.12 s |
| 24 | -58.0, 655.0 | -72.5, 666.0 | 18 cm | -72.3, 664.7 | 17 cm | 0.18 s |
| 25 | -216.0, 566.0 | -209.7, 623.9 | 58 cm | -210.1, 619.9 | 54 cm | 0.12 s |
| 31 | -67.0, 587.0 | -78.6, 608.1 | 24 cm | -76.7, 601.9 | 18 cm | 0.12 s |
| 37 | -260.0, 612.0 | -272.3, 611.8 | 12 cm | -271.4, 615.2 | 12 cm | 0.13 s |
| 43 | -110.0, 570.0 | -124.2, 543.4 | 30 cm | -124.2, 537.6 | 35 cm | 0.12 s |
| 49 | -202.0, 650.0 | -195.9, 637.6 | 14 cm | -194.8, 641.3 | 11 cm | 0.13 s |
| 50 | -168.0, 621.0 | -162.5, 629.7 | 10 cm | -162.4, 630.9 | 11 cm | 0.13 s |
| 56 | -194.0, 520.0 | -201.8, 522.4 | 8 cm | -201.6, 524.4 | 9 cm | 0.12 s |
| 62 | -21.0, 578.0 | -26.0, 578.2 | 5 cm | -22.8, 561.6 | 16 cm | 0.12 s |
| 68 | -59.0, 509.0 | -74.1, 468.4 | 43 cm | -74.3, 472.6 | 39 cm | 0.12 s |
| 74 | -269.0, 602.0 | -270.9, 625.0 | 23 cm | -273.3, 620.3 | 19 cm | 0.12 s |
| 75 | -270.0, 523.0 | -276.1, 564.4 | 42 cm | -277.0, 564.8 | 42 cm | 0.12 s |
| 81 | -232.0, 598.0 | -220.4, 590.2 | 14 cm | -219.8, 596.8 | 12 cm | 0.12 s |
| 87 | -170.0, 548.0 | -177.2, 535.7 | 14 cm | -176.8, 545.6 | 7 cm | 0.13 s |
| 93 | -264.0, 562.0 | -254.9, 532.5 | 31 cm | -256.8, 526.8 | 36 cm | 0.17 s |
| 99 | -227.0, 564.0 | -230.4, 542.4 | 22 cm | -231.2, 540.2 | 24 cm | 0.12 s |
| **all 20** | | | **23.0 / 21.9 cm** mean / median | | **23.5 / 21.5 cm** mean / median | |

Shots 25, 68 and 75 miss by 40–60 cm on both machines: their error comes from tracking and the flight model, not the
Hailo.

## 4. Receive the landings on the ESP32

### Wiring

| Pi 5 | ESP32-S3 |
| --- | --- |
| GPIO14 TX (header pin 8) | GPIO18 (UART1 RX) |
| GPIO15 RX (header pin 10) | GPIO17 (UART1 TX) |
| GND (header pin 6) | GND |

921600 baud, 8N1. Enable the Pi UART once: `sudo raspi-config` → Interface Options → Serial Port → login shell over
serial: **No**, serial port hardware: **Yes**, then reboot. The port is `/dev/serial0`.

### Message format (Pi → ESP32)

One 12-byte frame per landing estimate, as planned in `docs/tasks/hardware/communication.md`:

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

seq starts at 0 each time a Pi program opens the port, so a restart on the Pi shows up once as a jump in seq. At 60
frames per second the link carries 720 bytes/s, under 1 % of 921600 baud.

The Python side of the format is `vision/robot/esp32_link.py` (`frame`, `decode_frames`, `decode_landing`).

### Receiving example

[`firmware/experiments/landing_receiver/`](../firmware/experiments/landing_receiver/) holds a complete receiver for the
Arduino framework, in two files:

- `landing_frame_parser.h`: plain C++, no Arduino. `crc16Ccitt` (the same routine as before), and
  `LandingFrameParser::push(byte, landing)`, a byte-at-a-time state machine. It waits for `0xAA 0x55` (and resyncs on
  the next one after any bad frame), reads type, len and seq, rejects a len over 32, reads the payload and the CRC,
  checks the CRC, decodes x / y mm and the final flag, and counts frames, lost frames (seq gaps), CRC errors, bad
  lengths and other message types in `counters`. It returns `true` when a landing is ready.
- `landing_receiver.cpp`: `setup()` opens UART1 on GPIO18 RX / GPIO17 TX at 921600 with a 1 KB receive buffer;
  `loop()` feeds every received byte to the parser and calls
  `onLanding(int16_t xMm, int16_t yMm, bool final)`, which prints the landing on the USB serial port at 115200.
  Replace its body with your motor code. Once a second, if frames arrived, it prints the counters.

The core of it:

```cpp
HardwareSerial piLink(1);
LandingFrameParser landingParser;

void setup() {
  Serial.begin(115200);
  piLink.setRxBufferSize(PI_LINK_RX_BUFFER_BYTES);
  piLink.begin(PI_LINK_BAUD, SERIAL_8N1, PI_LINK_RX_PIN, PI_LINK_TX_PIN);
}

void loop() {
  LandingTarget landing;
  while (piLink.available() > 0) {
    if (landingParser.push(static_cast<uint8_t>(piLink.read()), landing)) {
      onLanding(landing.xMm, landing.yMm, landing.final);
    }
  }
}
```

To build it as its own PlatformIO environment, the same way as `motor_test`, add to `firmware/platformio.ini`:

```ini
[env:landing_receiver]
extends = env:esp32s3
build_src_filter = -<*> +<../experiments/landing_receiver/>
```

then `pio run -e landing_receiver -t upload` and `pio device monitor`.

**What was checked (2026-11-17, Mac):** PlatformIO is not installed there, so the example is **not yet compiled for
the ESP32-S3**, and the environment above has not been tried. What was run, with Apple clang and
`-Wall -Wextra -Werror`:

- the parser decodes the README example `aa 55 01 05 01 61 f5 fa 17 01 39 74` to x −2719 mm, y 6138 mm, final, seq 1;
  `crc16Ccitt("123456789")` is `0x29B1`;
- a noisy stream (stray bytes, a repeated `0xAA`, a frame with one flipped bit, a frame with seq 4, a frame with
  len 255) gives 2 landings, 1 CRC error, 1 bad length and 2 lost frames (seq 2 and 3);
- the 8,832 bytes the live loop sent with `--stream` (section 3) give 736 landings, 20 final, 0 lost, 0 CRC errors;
- `landing_receiver.cpp` builds and runs against a small stand-in for `Arduino.h` (`HardwareSerial`, `millis`), which
  checks the C++ but not the real ESP32 core.

## 5. Coordinates for the motors

Court coordinates, origin on the floor under the net on the middle line:

- **x**: from the middle line toward the sidelines; positive toward the post on the right of the camera image.
- **y**: from the net toward the back line; positive on the camera's (landing) side, where the robot is.
- Height is not sent.

Units: cm in the Python code, **mm** (int16) in the UART frames. int16 holds ±32.7 m, far more than the court.

**Robot home** is (−150, 330) cm: 150 cm from the middle line on the left-post side, 330 cm from the net
(`docs/tasks/measurement_note.md`; `ROBOT_HOME_CM` in `vision/robot/shot_landing.py`).

The vision side only gives the landing point on the court. Turning it into a motor target needs the robot's own
position and heading on the court, which the ESP32 knows (from home and odometry / the IMU). For a robot that starts
at home facing along the court axes, the target offset is landing − home: for shot 37's final landing
(−2720, 6070) mm that is (−2720 − (−1500), 6070 − 3300) = (−1220, 2770) mm from home. `shot_landing.py` prints this
offset as "from home".

## 6. Troubleshooting

| Problem | Fix |
| --- | --- |
| `catcher.local` does not resolve (ssh or `push_to_pi.sh` fails) | Use the Pi's IP address (on the Pi: `hostname -I`); `push_to_pi.sh` takes the target as its first argument, `vision/deploy/push_to_pi.sh catcher@<ip>`. On the Pi, `sudo systemctl restart avahi-daemon` usually brings the name back. |
| A Hailo run hangs or times out in `create_infer_model` | With two `.hef` files on one Hailo device, every model must be created before the first inference. The scripts here do that; custom code must too. |
| `score` fails reading a JSON file after an interrupted run | Results are written to `*.writing` and then renamed, so a stopped run leaves at most a `*.writing` file. Delete it and run `run` again. Old caches from before that fix: re-run with `--redo`. |
| A pasted command fails with an odd "unrecognized arguments" or a wrong file name | Copying long commands out of chat or PDFs can drop or add spaces at line breaks (`--inputgray-stack`). Paste one line at a time and check it; each command here is a single line. |
| `--send-uart` fails with `Inappropriate ioctl for device` | The port does not accept 921600 (a Mac pseudo-terminal, for example); pass `--baud 115200`. |
| The ESP32 counts CRC errors or lost frames | Check the shared GND and the TX/RX crossing; keep the UART wires away from motor wires. Motor noise is the reason for the CRC and seq. |
| `flight fit engine: python` instead of `c++` | `sudo apt install g++`, then `bash vision/flight/core_source/build.sh`. |
| The Pi shuts down under load | Use the official 27 W power supply. |

## How it works

Per stereo frame (2560 × 720, 60 fps): convert both eyes to gray; aim a crop at the shuttle (follow the last
detections, or search moving blobs when it is lost); run YOLO11n on the Hailo on one crop per eye (640 × 640 while
searching, 320 × 320 once seen), with a motion view built from frames t−1, t, t+1 inside the network, so detections lag
one frame; link detections into a rising chain and match left and right; triangulate to court cm; update the landing
estimate (a Kalman filter with gravity and drag in live mode, or a least-squares flight fit). In live mode a hit rule
on the rising chain starts each shot, so no labels are needed.

| Area | Files |
| --- | --- |
| Frames from the camera / clips | `vision/camera/frames.py`, `vision/camera/record.py` |
| Calibration, pixels ↔ court cm | `vision/calibration/court_3d.py`, `vision/calibration/calibrate_net.py` |
| Detection (YOLO on the Hailo or ONNX, crop aiming) | `vision/detection/detectors.py`, `vision/detection/crop_tracking.py`, `vision/detection/motion_view.py` |
| Tracking and the hit rule | `vision/tracking/bird_tracker.py`, `vision/tracking/hit_detection.py` |
| Flight physics, C++ core | `vision/flight/flight_model.py`, `vision/flight/core_source/` |
| Landing prediction (fit, Kalman) | `vision/landing_prediction/` |
| One shot, frame by frame | `vision/pipeline/shot_pipeline.py` (`ShotPipeline.push`) |
| Live loop: searching, following, tracking | `vision/pipeline/live_mode.py` (`LiveShotLoop.push`) |
| Talking to the ESP32, replaying shots | `vision/robot/esp32_link.py`, `vision/robot/shot_landing.py` |
| Pi setup, push, Hailo check | `vision/deploy/setup_pi.sh`, `vision/deploy/push_to_pi.sh`, `vision/deploy/hailo_check.py` |
| Test harnesses | `vision/experiment/pi_realtime_test.py`, `vision/experiment/live_replay_test.py` |
| Models | `models/bird_yolo11n_v1_gray_stack.hef` and `_320.hef` (Pi), `.onnx` (computer) |
| Test data | `test/pi_test_package/` (20 shot clips, in git), `test/pi_live_test_package/` (20-shot video, not in git) |
