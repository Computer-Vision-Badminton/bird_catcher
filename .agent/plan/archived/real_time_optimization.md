# Goal: real time on the Pi

**Closed 2026-11-16 by the user.** Processing 50.5 → 11.0 ms per frame (budget 16.7 ms). The replay's wall time is
19.7 ms because the clip decode (8.9 ms) runs on the same thread; moving frame fetching to its own thread is the
first task of `live_camera_loop.md`. Remaining speed ideas are listed in the "Current status" section of `CLAUDE.md`.
Summary page: "Bird Catcher Pi Speed-Up" artifact (https://claude.ai/artifact/Y8h6N5zRVMudxckcHqVjtW). Full numbers:
the run log ("Processing vs wall time per frame" table) and the methods doc.

Real time at 60 fps on the Pi 5 + Hailo-10H: 16.7 ms per stereo frame.

Speed comes first (user decision, 2026-11-15): a speed-up may cost some landing accuracy. Every speed task still
reports the accuracy change (mean / median miss, per-shot shifts, estimate at hit +15), and the Kalman filter in
`landing_prediction.md` is meant to win it back. A change that loses the bird entirely on some shots is still a
failure.

## Where it stands (manager, 2026-11-16)

Pi with setup D (640 until the first detection, then 320) and the CPU work overlapped with the Hailo call: the CPU
stages take 11.0 ms per frame (10.9 median) while the Hailo call (10.6 ms) runs in parallel, so processing fits the
16.7 ms budget. The replay's wall time is 19.7 ms because it includes 8.9 ms of video decode on the same thread; a
live camera loop has to capture on its own thread or core to keep that out (see `live_camera_loop.md`). 71 of 840
frames still go over 16.7 ms (p95 17.4 ms): the 27 first-search frames at 640, plus flight-fit peaks.
Landing miss 25.0 / 24.9 cm (640 only: 23.0 / 21.9), accepted under the speed-first rule.

| Stage (ms, Pi) | 640 only | D + overlap |
| --- | --- | --- |
| gray | 1.5 | 1.5 |
| aim | 2.1 | 1.8 |
| YOLO (waiting for the Hailo) | 26.8 | 2.0 (Hailo busy 10.6 in parallel) |
| track | 0.7 | 0.7 |
| predict (flight fit) | 4.8 | 4.9 |
| **CPU stages, mean** | **35.9** | **11.0** |
| wall per frame in the replay (decode included) | | 19.7 |

| Date | Setup | Processing (ms) | Frame fetch: clip decode (ms) | Wall (ms) | Landing miss, mean / median |
| --- | --- | --- | --- | --- | --- |
| 2026-11-16 | D + CPU work overlapped with the Hailo | 11.0 | 8.9 | 19.7 (measured) | 25.0 / 24.9 cm |
| 2026-11-15 | 320 following, 640 searching | 27.1 | 8.9 | about 36.0 (sum) | 25.0 / 24.9 cm |
| 2026-11-15 | 640, both eyes in one call, blob search near the bird | 35.9 | 8.3 | about 44.2 (sum) | 23.0 / 21.9 cm |
| 2026-11-15 | motion view in the model, C++ flight fit | 50.5 | not recorded | – | 25.2 / 22.8 cm |
| 2026-11-15 | first Pi run: flight fit in Python (90.5 ms), motion view on the CPU (19.7 ms) | 147.1 | 9.1 | about 156.2 (sum) | 26.3 / 19.9 cm |

## Not done when the goal closed

1. **Cut the flight fit out of the frame**: it was the biggest CPU stage (4.9 ms, peaks 10.5). `--fit-background`
   already runs it in another process; measure it with D + overlap on the Pi, or fit every 2nd frame.
2. **Later candidates:**
   - YOLO on the left eye only, with the right-eye bird found by a small CPU search along the matching row (row
     offset 12.21 px, disparity 1–150 px). This halves the Hailo work per frame.
   - A lighter model with fewer contexts on the chip (each call loads 4 contexts now, at 640 and at 320).
   - The frame loop in C++: worth about 2–4 ms at most.
- Optional, for the report: setup D on the Pi without the overlap, all 20 shots.
- `vision/README.md` still shows the 640-only Pi status and landings.
- The code from this goal is uncommitted on `main` (as of closing).

## Done

- 2026-11-16, Pi: setup D + overlap. CPU stages 11.0 ms per frame, Hailo 10.6 ms in parallel, replay wall 19.7 ms
  (8.9 ms decode), estimate ready 30.8 ms after its newest frame. Landings identical to the Pi 320-follow run
  (25.0 / 24.9 cm), so D = B on the Pi and the overlap changes no landing. No hang with `run_async` after the load
  fix. Logged in the run log and the methods doc.
- 2026-11-16, load-order bug: with two `.hef` on one VDevice, `create_infer_model` timed out once the first model had
  run (its warm-up). Fixed in `pi_realtime_test.py` (create all, then warm up) with a guard in `HailoDetector`; noted
  in `CLAUDE.md`. Found with the user's plain-HailoRT and load-both-first tests on the Pi.
- 2026-11-15/16, overlap of CPU work with the Hailo (`--overlap`, `start_batch` / `wait` on every detector): Mac
  identity checks passed for A, B and D, Pi run above.
- 2026-11-15, 320 × 320 crop while following: Pi 9.7 ms per 2-crop call vs 24.0 at 640, still 4 contexts; accepted
  for speed with the INT8 accuracy loss (moved to `landing_prediction.md`).
- 2026-11-15, vision builder, Mac (ONNX): how often each crop mode finds the bird, and 320 while searching.
  - Code: `vision/pipeline.py` (`search_with_follow_detector`: `None` as before, `"always"`, `"once-seen"`),
    `vision/experiment/pi_realtime_test.py` (`--search-with-follow-model {always,once-seen}`), new
    `vision/experiment/crop_mode_detection_stats.py` (subclasses `ShotPipeline` to record mode, crops and
    detections per frame). A and B landings, stereo points and first estimates are identical to their references.
  - 320 everywhere (C) is not safe. Shot 0 gets its first estimate at +16 instead of +10, has no estimate at hit +15
    and keeps 13 of 19 stereo points. Missed-while-in-flight goes from 6 (A) and 8 (B) to 20, and 14 of those are
    searching frames where the bird is outside the 320 crop. Landing 24.2 / 20.9 cm vs B 23.6 / 20.4.
  - The bird is never lost inside the flight in any setup. All 20 losses happen after the flight and are never
    re-found. So only the first full-frame search matters: 27 frames over 20 shots at 640, 50 at 320.
  - D (`once-seen`: 640 only before the first detection) gives B's exact landings and counts, with 640 on only 27 of
    840 frames. Mac YOLO 29.5 ms per frame vs A 85.8 and B 64.8 (noisy; on fixed crops 78.7 / 23.0 ms per 2-crop call
    at 640 / 320).
  - Hand labels (val shots 6, 37, 56, 81) are full left-eye frames, 1280 × 720, named by take frame (same numbering
    as the package). 57 fall in tracked frames: 45 with the bird, 12 empty. Found by any detection 44 / 44 / 43 / 44
    (A / B / C / D), on the chain 42 / 40 / 41 / 40, no detections on the empty frames, chain error 1.1 px.
  - Logged in the methods doc (end of the 320 section: count table, landing table, recommendation) and the run log
    (rows for C and D).
- 2026-11-15, batch-1 fix check on the Pi: shots 37 and 62 land at (-272.3, 611.8) and (-26.0, 578.2), the same as
  the batch-2 run. Logged in the run log.
- 2026-11-15, 320 `.hef` compiled in Colab (exit code 0, 4 contexts, int8 16/16 check crops).
- 2026-11-15, Pi measurement of batch 2 + blob window: 50.5 → 35.9 ms per frame (27.9 fps); landing miss 25.2 / 22.8
  → 23.0 / 21.9 cm; estimate at hit +15: 26.2 / 24.2 cm. Split: aiming 7.9 → 2.1 ms (blob window, no right-eye
  search), YOLO 35.0 → 26.8 ms (batch 2; 35.1 ms at batch 1 with the same aiming). `hailo_check --batch 2`: 24.0 ms
  per 2-crop call. Logged in the run log; README status and 20-shot table and the `CLAUDE.md` status are updated.
- 2026-11-15, vision builder: both eyes in one detector call (`detect_batch`, Hailo batch 2 by default); right crop
  aimed from the left crop's aim, so no right-eye blob search; left blob search in a 640 × 640 window near the bird
  (full frame only before the first detection). Mac ONNX: aim 1.43 → 0.57 ms per frame; landing miss 23.9 / 19.6 →
  23.5 / 21.5 cm (landings moved 3.4 cm mean, 9.1 max). Logged in both Claude Docs.
