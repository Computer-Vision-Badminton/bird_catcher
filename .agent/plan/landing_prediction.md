# Goal: landing prediction within 15 cm

Predict where the shuttle lands within 15 cm mean miss on the 20 test shots, using a Kalman filter on the tracked
bird. Now: 23.5 / 21.5 cm mean / median on the Mac (ONNX), 25.2 / 22.8 cm on the Pi (Hailo), from a least-squares
fit of gravity + quadratic drag (terminal speed 6.5 m/s, RK4 at 25 ms) plus a linear correction learned on the 80
training shots (`vision/flight_model.py`).

## Why a Kalman filter (manager, 2026-11-15)

From the sensitivity section of the methods doc:

| Error source | Effect on landing |
| --- | --- |
| 0.5 px random pixel noise per frame | 4.9 cm sideways, 20.4 cm along the court |
| 1 px right-eye x error at 5.5 m | 3D point moves 39 cm (depth error ≈ Z² / (B·f) · Δd) |
| Terminal speed ± 0.25 m/s | landing y ± 30 cm |

Most of the remaining miss is along the court, and it matches depth noise from disparity. The current fit treats every
3D point as equally trustworthy. A Kalman filter can weight each frame by how noisy it really is. That only works if
it measures in the image (left-eye u, v and disparity in px), where the noise is about the same for every frame.
Triangulated 3D points are not: their depth noise grows with Z². The filter state can also carry the drag (terminal
speed), so each shot gets its own drag instead of one fixed 6.5 m/s.

Shots 25, 68 and 75 miss by 40–60 cm on both the Mac and the Pi, so their error comes from tracking or the flight
model, not the Hailo. Look at them first.

## INT8 accuracy at 320 (moved from the real-time plan, 2026-11-15)

On the Pi, 320 crops while following cost accuracy: 23.0 / 21.9 → 25.0 / 24.9 cm (14 of 20 shots worse; shot 62
5 → 16 cm, 12 18 → 31, 93 31 → 42). The Mac ONNX 320 model does not (23.6 / 20.4 cm), so the likely cause is INT8 at
320. 320 stays in the pipeline for speed (user decision); the filter should be tuned on 320-crop tracks. If it is
still short of 15 cm, measure the Hailo 320 detections against ONNX 320 frame by frame and try better calibration
crops (taken while following, bird near the centre).

## Rules for this goal

- Tune noise levels, drag and any correction on the 80 training shots only.
- The 20 test shots are scored once per finished candidate, not while tuning. Every test score goes in the methods
  doc, the bad ones too.
- The filter must also work frame by frame inside `ShotPipeline.push` (one update per frame, a landing estimate each
  frame). On the Pi it gets no more time than today's fit: 4.9 ms per frame.
- Write the filter in C++ from the start (user decision, 2026-11-16), as part of the flight core in `vision/flight_cpp/`
  (pybind11, the same RK4 flight model), built by `vision/flight_cpp/build.sh`. Python only calls it. The tuning and
  comparison harness stays in Python under `vision/experiment/`. Target: well under 1 ms per update on the Pi, so the
  4.9 ms least-squares fit (SciPy driving the C++ simulation, the biggest CPU stage left) leaves the frame.
- The current least-squares fit is not ported to C++: the filter replaces it.
- Build and compare it in `vision/experiment/` first. It replaces the fit in `ShotPipeline` only once it beats the
  current fit on the test shots.

## In progress

Nothing yet.

## Next

1. **Error breakdown on the training shots** (vision builder).
   For the 80 training shots: miss split into sideways (x) and along the court (y); pixel residuals of the current
   fit in left u, v and disparity; and what goes wrong on test shots 25, 68 and 75 (look at the tracks, not the
   landings).
   - Done when: one table of per-axis miss and pixel residual spread (px) on the training shots, plus a line per bad
     shot on its cause, in the methods doc. The residual spread sets the filter's measurement noise.
2. **Extended Kalman filter in C++, experiment** (vision builder). Filter in `vision/flight_cpp/` (exposed through
   `flight_core`), harness in `vision/experiment/kalman_landing.py`.
   - State: position and velocity in court cm (6 values), plus drag as a 7th value in a second variant.
   - Motion: the same gravity + quadratic drag and RK4 as `flight_model.py` (the C++ core can be reused).
   - Measurement: left u, v and disparity in px, through the calibration in `court_3d.py`; noise from task 1.
   - Start: the first few stereo points of the rising chain.
   - Landing: run the state forward to z = 0 after each update.
   - Done when: on the training shots, mean / median miss and estimate at hit +15 for the 6-value and 7-value
     variants are next to the current fit, plus ms per update on the Mac (and the current fit's ms per fit for
     comparison), all in the methods doc.
3. **Smoother for the final landing** (vision builder). Run a Rauch–Tung–Striebel smoother over the whole track once
   the shot ends, for `ShotPipeline.finish`. Also in C++, next to the filter.
   - Done when: training-shot miss with and without the smoother is in the methods doc.
4. **Correction and test score** (vision builder). Re-learn the linear correction on the training shots for the best
   variant (or show it is no longer needed), then score it once on the 20 test shots.
   - Done when: test mean / median miss and estimate at hit +15 are in the methods doc. The goal is met at ≤ 15 cm
     mean.
5. **Into the pipeline** (vision builder), only if task 4 meets the goal. Replace the fit in `ShotPipeline`, keep the
   old fit available in `vision/experiment/`, and measure on the Pi.
   - Done when: the Pi run (setup D + `--overlap`) shows the landing miss and ms per frame, the predict stage is well
     below today's 4.9 ms (target under 1 ms), the run log and `vision/README.md` are updated, and the C++ build still
     works on the Pi (`bash vision/flight_cpp/build.sh`).
6. **Landing miss against frame rate** (vision builder, low priority; moved from the real-time plan). Replay the
   test shots at 60, 30 and 20 fps (every 1st, 2nd, 3rd frame) with the best predictor. Report landing miss, estimate
   at hit +15 and time from hit to first estimate. This shows how much accuracy a dropped frame costs once the live
   camera loop starts dropping frames.

## Waiting on the user

- Confirm what "within 15 cm" means: mean miss on the 20 test shots ≤ 15 cm (assumed here), median ≤ 15 cm, or every
  shot ≤ 15 cm (much harder: three shots miss by 40–60 cm now).

## Done

Nothing yet.
