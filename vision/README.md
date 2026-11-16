# Bird Catcher vision pipeline (Raspberry Pi 5 + AI HAT+ 2)

This branch holds only what the shuttle-tracking pipeline needs to run on the Pi, plus 20 recorded test shots to replay.
It predicts where the shuttle will land, frame by frame, from the ZED stereo camera.

**Current status (2026-11-15, Pi 5 + Hailo-10H):** 50.5 ms per stereo frame (about 20 fps), landing predicted within
25 cm on average (23 cm median) on the 20 test shots. The first landing estimate is ready 7–10 frames (~0.12–0.17 s) after the hit.

**Not in this branch yet:** a live camera loop, automatic hit detection (each test shot starts at a hand-labelled hit
frame), and anything sent to the ESP32. The Pi–ESP32 link is specified in `docs/tasks/hardware/communication.md`.

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
pi_test_package/clips/shot_*.avi        20 recorded test shots (original ZED frames, 2560 x 720, 60 fps)
pi_test_package/manifest.json           per shot: hit frame, frame times, measured landing; tuned drag + correction
pi_test_package/net_calibration.json    camera position and lens calibration for these recordings
vision/pipeline.py                      ShotPipeline: one stereo frame in, updated landing estimate out
vision/experiment/pi_realtime_test.py   replays the 20 shots through ShotPipeline and times every stage
vision/deploy/setup_pi.sh               one-time Pi setup (Python environment + C++ flight core)
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
