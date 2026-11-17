# Goal: tidy `vision/` into subfolders

The user asked (2026-11-17) to split `vision/` into subfolders by area before the next commit and push. The layout is
the manager's choice. Behaviour must not change at all: the same landings, hits and timings paths as before.

## Layout (manager, 2026-11-17)

`vision/` becomes a Python package (`__init__.py` in every folder) with absolute imports such as
`from vision.tracking.bird_tracker import longest_rising_chain`.

| Folder | What | Files (now → new) |
| --- | --- | --- |
| `vision/camera/` | frames from the ZED: reading clips and frames, recording | `frames.py`, `record.py` |
| `vision/calibration/` | camera and court calibration, pixels ↔ court cm | `calibrate_net.py`, `calibrate_net.html`, `court_3d.py` |
| `vision/detection/` | finding the shuttle in a frame: YOLO runners, motion view, crop aiming | `detectors.py`, `motion_view.py`, `crop_tracking.py` |
| `vision/tracking/` | linking detections over time: rising chain, stereo matching, moving blobs, the hit rule and waiting stage, the offline tracker | `bird_tracker.py`, `hit_detection.py`, `yolo_tracker.py` |
| `vision/flight/` | shuttle physics: the flight model and the compiled core | `flight_model.py`, `flight_cpp/` → `core_source/` (moved and renamed; `flight_core/` would shadow the module, see Status), the built `flight_core*.so` |
| `vision/landing_prediction/` | turning a track into a landing: least-squares fit worker, Kalman filter model and settings | `landing_worker.py`, `kalman_landing_model.py`, `kalman_landing_settings.json` |
| `vision/pipeline/` | the per-frame pipeline and the live loop | `pipeline.py` → `shot_pipeline.py`, `live_mode.py` |
| `vision/robot/` | talking to the robot, and replaying shots for it | `esp32_link.py`, `shot_landing.py` |
| `vision/deploy/`, `vision/experiment/`, `vision/helper/` | unchanged roles; imports updated | — |
| `test/` (repo root, user 2026-11-17) | all test packages, each folder intact | `pi_test_package/` (git mv), `pi_hit_test_package/`, `pi_live_test_package/` (plain mv); in git only `test/pi_test_package/` without `results_*.json` (`.gitignore`: `test/*`, `!test/pi_test_package/`, `test/pi_test_package/results_*.json`) |
| `notebooks/` | deleted by the manager (`git rm -r`, staged); row and mentions removed from `CLAUDE.md` | — |

`vision/README.md` stays at the top of `vision/`.

## Rules for the move
- `git mv` tracked files (keeps history); plain moves for untracked ones. No code changes beyond imports and paths.
- Every entry script keeps working **from the repo root with the same command** as today
  (`.venv/bin/python vision/experiment/pi_realtime_test.py …`, `vision/deploy/push_to_pi.sh`, …): scripts that today
  put `vision/` on `sys.path` put the repo root there instead. `vision/robot/shot_landing.py` (was
  `vision/shot_landing.py`) and `vision/camera/record.py` get a new path; update every place that names them.
- No programming language in any folder name (user, 2026-11-17): `flight_cpp/` becomes `vision/flight/core_source/`
  (`flight_core/` was the first choice; it shadows the compiled module when the `.so` is missing, see Status).
- `flight_core`: `vision/flight/core_source/build.sh` builds into `vision/flight/`; imports become
  `from vision.flight import flight_core` (still optional, falling back as today). `.gitignore` for `*.so` covers the
  new place. `vision/deploy/setup_pi.sh` calls the new `build.sh` path.
- Update everything that names old paths: `vision/deploy/*.sh` and `*.py`, `vision/experiment/*`, `vision/helper/*`,
  `vision/README.md`, `CLAUDE.md` (layout table, pipeline section, commands, "new code goes at the `vision/` top level"
  rule → "in the matching `vision/<area>/` folder"), `.agent/roles/vision_builder.md`, `.claude/agents/vision-builder.md`,
  and the goal files under `.agent/plan/` where they give commands.
- The `vision-pipeline` branch: do not touch it now; note that it needs the same move (a merge, not cherry-picks).

## Checks (all must pass before calling it done)
1. Every module imports: `python -c "import vision.<each module>"` for all of them, on the Mac.
2. `bash vision/flight/core_source/build.sh` builds, and `from vision.flight import flight_core` loads the Kalman.
3. `pi_realtime_test.py` (ONNX, setup D + overlap) with `--landing-method fit` and `kalman` on all 20 test shots:
   landings identical to `data/recordings/clears_nov10/pipeline_tracks_setup_d_onnx_test.json` and
   `landing_test_score.json` row `kalman_filtered`. Restore any results files you overwrite.
4. `live_replay_test.py run` + `score` (ONNX) on `test/pi_live_test_package`: identical hits and landings to the current
   `live_results_onnx_Khangs-Air.json` (back it up first; restore afterwards).
5. `hit_from_rising_chain.py detect --package test/pi_hit_test_package --detector onnx --takes take_006 --redo` (with and
   without `--overlap`): cache identical to today's except timings (back up, restore).
6. `vision/robot/shot_landing.py 37` runs and prints the same landing as before; `vision/deploy/hailo_check.py --help`,
   `export_calibration_crops.py --help`, `motion_view_network.py --help` run.
7. `rsync -n` dry run of `push_to_pi.sh` copies the new tree (no `.so`, no caches).

## Status

- **In progress (waiting on the user):** the move is done on the Mac and all seven checks pass (vision builder,
  2026-11-17). Not committed; the Pi check and the commit wait for the user.
  - Files moved: as the table, with `git mv` for tracked files (renames staged) and plain `mv` for untracked ones
    (`hit_detection.py`, `live_mode.py`, `kalman_landing_model.py`, `kalman_landing_settings.json`,
    `final_landing.hpp`, `kalman_flight.hpp`, the two big test packages). New `__init__.py` in `vision/` and every
    subfolder (untracked: `git add` them with the commit; `vision/experiment/__init__.py` is re-included in
    `.gitignore`, `vision/helper/` stays ignored). Old `vision/flight_core*.so` deleted; rebuilt in `vision/flight/`.
  - Imports: every `vision/` script uses `vision.<area>.<module>`; sibling experiment imports use
    `vision.experiment.<module>`; optional `flight_core` imports are `from vision.flight import flight_core` inside the
    same try/except. Every entry script puts the repo root on `sys.path` (one line replacing the old one or two); the
    production entry scripts `calibration/calibrate_net.py`, `robot/shot_landing.py`, `tracking/yolo_tracker.py` got
    that line too (they used to rely on Python adding `vision/` as the script folder).
  - Paths: `REPO` in `shot_landing.py`, `DEFAULT_MODEL` in `yolo_tracker.py`, `RECORDINGS_ROOT` in `record.py` go one
    folder further up; `--package` / `--output` defaults and `CALIBRATION_PACKAGE` point at `test/…`; `build.sh`
    writes `vision/flight/flight_core*.so`; `setup_pi.sh` builds from the new path and checks
    `from vision.flight import flight_model`; `push_to_pi.sh` copies `test/…` with the same excludes; `.gitignore`.
    Docs: `CLAUDE.md` (layout table, package rule, pipeline file names, commands), `vision/README.md`,
    `.agent/roles/vision_builder.md`, `.claude/agents/vision-builder.md`, forward-looking paths in
    `landing_prediction.md`. Done reports in plan files and `docs/stats/` exports keep the old paths (history).
  - Why `core_source/`, not `flight_core/`: with `vision/flight/flight_core/` (no `__init__.py`) and no `.so` built,
    `from vision.flight import flight_core` imports the folder as a namespace package instead of raising
    `ImportError`, so `FLIGHT_ENGINE` said `c++` and the Python fallback (Windows, no compiler) broke. With the `.so`
    present the `.so` wins, but the fallback must keep working. With `core_source/`: no `.so` -> `python`, `.so` -> `c++`.
  - Checks (Mac M2, ONNX; every check run before the move and after it, outputs compared with timing keys removed):
    1. All 46 modules (production, deploy, experiment) and the 9 helpers import; every entry script's `--help` runs.
    2. `bash vision/flight/core_source/build.sh` builds `vision/flight/flight_core.cpython-313-darwin.so`;
       `from vision.flight import flight_core` loads it (`KalmanFlight`, `KalmanSettings`); engine `c++`.
    3. `pi_realtime_test.py` setup D + overlap (`--follow-model models/bird_yolo11n_v1_gray_stack_320.onnx
       --search-with-follow-model once-seen --overlap`), 20 shots: fit 23.6 / 20.4 cm, per-shot landings equal
       `pipeline_tracks_setup_d_onnx_test.json` (max difference 0.0 cm); Kalman 17.3 / 14.2 cm, equal
       `landing_test_score.json` `kalman_filtered` per shot (0.0 cm). Results files identical to the pre-move run.
    4. `live_replay_test.py run` + `score` (ONNX): results, log and score identical to the pre-move run and to the
       backed-up `live_*_onnx_Khangs-Air.json`: 20/20 hits (10 exact, 19 within 1, 20 within 2), 0 false triggers,
       landing 17.8 / 15.1 cm.
    5. `hit_from_rising_chain.py detect --package test/pi_hit_test_package --detector onnx --takes take_006 --redo`,
       with and without `--overlap`: caches identical to the pre-move run except timings (the backed-up cache without
       overlap lacks the newer `overlap: false` key; that comes from the earlier overlap change, not this move).
    6. `vision/robot/shot_landing.py 37`: landing (-271.4, 615.2) cm, miss 11.8 cm, first estimate 0.134 s, 33
       estimates, same as before; `hailo_check.py`, `export_calibration_crops.py`, `motion_view_network.py --help` run.
    7. `push_to_pi.sh` rsync dry run (to a local folder): copies the new `vision/` tree and the three `test/`
       packages; no `.so`, no `__pycache__`, no `helper/`, no hit caches or live results. As before, the Mac's
       `test/pi_test_package/results_*.json` are copied too (the script never excluded them).
    All overwritten results and caches restored from backups; the new `results_*overlap*.json` files removed.
  - `git status --short --ignored test`: tracked exactly the 20 clips, `manifest.json`, `net_calibration.json`;
    ignored `pi_hit_test_package/`, `pi_live_test_package/` and the results files.
  - Not logged in the Claude Docs: a pure move with identical results, no new measurement.
  - The `vision-pipeline` branch needs the same move (a merge from `main`, not cherry-picks).
- **Waiting on the user:** on the Pi the folders become `test/…` after the next push; the old
  `~/bird_catcher/pi_*_package` folders stay there as stale copies, and the old top-level `vision/*.py` files stay too
  (rsync does not delete; harmless, but the first Pi line removes them by name, keeping the new `vision/__init__.py`). Mac:
  ```bash
  vision/deploy/push_to_pi.sh
  ```
  Pi (`~/bird_catcher`):
  ```bash
  cd vision && rm -f bird_tracker.py calibrate_net.py calibrate_net.html court_3d.py crop_tracking.py detectors.py esp32_link.py flight_model.py frames.py hit_detection.py kalman_landing_model.py kalman_landing_settings.json landing_worker.py live_mode.py motion_view.py pipeline.py record.py shot_landing.py yolo_tracker.py flight_core*.so && rm -rf flight_cpp && cd ..
  bash vision/flight/core_source/build.sh
  .venv/bin/python -c "from vision.flight import flight_model; print(flight_model.FLIGHT_ENGINE)"
  .venv/bin/python vision/experiment/live_replay_test.py run --detector hailo
  .venv/bin/python vision/experiment/live_replay_test.py score --detector hailo
  ```
  The engine line must say `c++`. Compare with the last Pi run: 20/20 hits, 15.0 / 14.2 cm. Only after that works:
  `rm -rf ~/bird_catcher/pi_test_package ~/bird_catcher/pi_hit_test_package ~/bird_catcher/pi_live_test_package`.
  Then the commit (add the new `__init__.py` files and the untracked moved files) and push.

- **README for the firmware engineer: done on the Mac** (vision builder, 2026-11-17). Not committed. Waits on the
  user for: an ESP32-S3 compile and a bench test (no `pio` on the Mac), and the Pi command below.
  - Files: `vision/README.md` (rewritten: 1 what the robot gets + numbers with sources, 2 setup and tests on the Pi,
    3 driving the ESP32 with test data, 4 receiving on the ESP32, 5 coordinates for the motors, 6 troubleshooting,
    "How it works" with a file table; message table, CRC, wiring and the 20-shot table kept);
    `vision/experiment/live_replay_test.py` (`run --send-uart [PORT]`, default `DEFAULT_PORT`; `--baud`; `--stream`;
    `LandingSender` reuses `Esp32Link`; opened before the models load so a bad port fails at once; off = no object,
    one `if sender is not None` per frame); new `firmware/experiments/landing_receiver/landing_frame_parser.h`
    (plain C++ parser) and `landing_receiver.cpp` (Arduino: UART1 GPIO18/17 at 921600, `onLanding(xMm, yMm, final)`).
    `firmware/src/main.cpp` and `firmware/platformio.ini` untouched (the README shows the env to add).
  - `--send-uart` loopback (Mac, ONNX, `os.openpty()` as the port, `--baud 115200` because a Mac pty refuses 921600):
    `--stream` sent 736 frames / 8,832 bytes, seq 0..223 without gaps, 20 final, each equal to the run's results;
    without `--stream` 20 frames, all final. Shot 37: 37 frames, first `aa 55 01 05 02 bc f4 e7 13 00 22 07`
    (x −2884, y 5095 mm, in flight), last `aa 55 01 05 26 60 f5 b6 17 01 ed 4b` (x −2720, y 6070 mm, final, seq 38).
    Results of the sending runs and of a run without the option equal the pre-change results except timings
    (20/20 hits, 17.8 / 15.1 cm); results, log and score files restored from backups afterwards.
    `shot_landing.py 37 --send-uart <pty> --stream --landing-method kalman`: 37 frames, final (−273.0, 609.8) cm.
  - ESP32 example: not compiled for the ESP32-S3 (`pio` not installed). Host checks with Apple clang
    `-Wall -Wextra -Werror`: README frame decodes to x −2719, y 6138 mm, final, seq 1; CRC check value 0x29B1; a noisy
    stream gives 2 landings, 1 CRC error, 1 bad length, 2 lost; the captured 8,832 bytes give 736 landings, 20 final,
    0 lost, 0 CRC errors; `landing_receiver.cpp` compiles and runs against a stand-in `Arduino.h`.
  - Not in git: `vision/experiment/` is ignored except `pi_realtime_test.py`, so `live_replay_test.py`, its import
    `hit_from_rising_chain.py` and `make_live_test_package.py` (and the untracked `vision/pipeline/live_mode.py`,
    `vision/tracking/hit_detection.py`) would not reach the firmware engineer's clone; the README's live sections need
    them committed (`.gitignore` exceptions) or copied with `push_to_pi.sh`. Decision for the user / manager.
  - Not logged in the Claude Docs: no new speed or accuracy measurement (identical results).
  - Waiting on the user (Pi, after `vision/deploy/push_to_pi.sh`, UART wired, receiver flashed):
    ```bash
    .venv/bin/python vision/experiment/live_replay_test.py run --detector hailo --send-uart --stream
    ```
    The receiver should print one landing per estimate (736 on the Mac; the Hailo count may differ a little), 20 final, and its counters 0 lost, 0 CRC errors. Compile on a machine
    with PlatformIO: add the README's `[env:landing_receiver]` to `firmware/platformio.ini`, then
    `pio run -e landing_receiver`.

## Next: `vision/README.md` for the firmware engineer (user request, 2026-11-17; starts after the move's checks pass)

Reader: the firmware engineer, who works on the ESP32-S3 (`firmware/`) and the `vision-pipeline` branch, and needs to
test the vision side and connect it to the robot without knowing its internals. Rewrite `vision/README.md` around
their tasks, with the new layout's paths, and only commands that were run and work:
1. **What the vision side gives the robot**: a landing point in court cm (axes and robot home), in-flight estimates
   then a final one; current numbers (Pi, live, 20 test shots: 20/20 hits, 15.0 / 14.2 cm; first estimate about
   100 ms after the hit, final about 675 ms; Kalman), and what is not done yet (live ZED capture loop).
2. **Set up and test on the Pi**: `setup_pi.sh`, the C++ core build, `hailo_check.py`, `pi_realtime_test.py` on
   `test/pi_test_package/` (expected output), and the live replay test on `test/pi_live_test_package/` (how to make
   the package if it is missing).
3. **Drive the ESP32 from the Pi with test data**: `vision/robot/shot_landing.py` (`--send-uart`, `--stream`,
   `--pause`), and a live-mode equivalent: add a small `--send-uart [PORT]` (and `--stream`) option to
   `live_replay_test.py run` that sends each in-flight estimate and the final landing through `vision/robot/esp32_link.py`
   as they happen (off by default; reuse the existing frame format; seq increments per frame). Check it on the Mac
   against a loopback (for example `socat` pseudo-terminals, or a Python reader on the other end with
   `decode_frames`) and show what the ESP32 receives.
4. **Integrate on the ESP32**: wiring (Pi GPIO14/15 ↔ ESP32 GPIO18/17, GND, 921600 baud, enabling the Pi UART), the
   frame format (keep the existing table), and a complete receiving example for the ESP32 (Arduino / PlatformIO C++):
   read bytes from UART1, find `0xAA 0x55`, read type/len/seq/payload/CRC, check the CRC with the existing
   `crc16Ccitt`, decode int16 x / y mm and the final flag, detect lost frames by seq. Compile-check it with
   PlatformIO for the ESP32-S3 if `pio` is available, otherwise say it is not compiled. Do not change
   `firmware/src/main.cpp`; the example lives in the README (or `firmware/experiments/` if a file is clearer, say
   which).
5. **Coordinates for the motors**: court axes, robot home (−150, 330), how to turn a landing into a motor target
   (the vision side does not do it), mm units in the frame.
6. **Troubleshooting**: `catcher.local` not resolving (use the IP; restart `avahi-daemon`), Hailo load order, a
   half-written results file, pasting commands that lose spaces.

## User decisions before the commit (2026-11-17)
- Track `vision/experiment/live_replay_test.py`, `hit_from_rising_chain.py` and `make_live_test_package.py` in git
  (`.gitignore` exceptions added, CLAUDE.md layout row updated), so the firmware engineer can run the README's live
  sections from a clone. The live test video stays local; the README says how to make it.
- After committing on `pi-speedups`, merge `pi-speedups` into `vision-pipeline` and push both, so the engineer's
  branch has the new layout and matches the README.
