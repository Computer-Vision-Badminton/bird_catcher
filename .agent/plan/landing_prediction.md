# Goal: landing prediction within 15 cm

Predict where the shuttle lands within 15 cm mean miss on the 20 test shots, using a Kalman filter on the tracked
bird. Now: 23.5 / 21.5 cm mean / median on the Mac (ONNX), 25.2 / 22.8 cm on the Pi (Hailo), from a least-squares
fit of gravity + quadratic drag (terminal speed 6.5 m/s, RK4 at 25 ms) plus a linear correction learned on the 80
training shots (`vision/flight/flight_model.py`).

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
- Write the filter in C++ from the start (user decision, 2026-11-16), as part of the flight core in `vision/flight/core_source/`
  (pybind11, the same RK4 flight model), built by `vision/flight/core_source/build.sh`. Python only calls it. The tuning and
  comparison harness stays in Python under `vision/experiment/`. Target: well under 1 ms per update on the Pi, so the
  4.9 ms least-squares fit (SciPy driving the C++ simulation, the biggest CPU stage left) leaves the frame.
- The current least-squares fit is not ported to C++: the filter replaces it.
- Build and compare it in `vision/experiment/` first. It replaces the fit in `ShotPipeline` only once it beats the
  current fit on the test shots.

## In progress

Nothing. The overnight run (2026-11-16, tasks 1–5 one at a time, each checked by the manager) is finished; see
"Done" and "Waiting on the user".

## Where it stands (manager, 2026-11-16)

The goal (≤ 15 cm mean on the 20 test shots) is **not met**. The C++ Kalman filter (7 states, per-shot drag) beats the
current fit on the one-time test score, 17.3 / 14.2 cm against 23.6 / 20.4 cm (13 shots better, 7 worse; 95% interval
of the change [−13.2, +0.9] cm), and its predict stage is about 0.03 ms instead of 1.6 ms on the Mac. It is in the
pipeline as `--landing-method kalman`, off by default, because its mean estimate at hit +15 is worse (27.2 vs 24.9 cm,
mostly shot 0). The test shots are used up for these settings: any further tuning is judged on training-shot
cross-validation only, until new test data exists.

Most promising next steps (training shots only):
1. The restart logic: shot 0 restarts twice and is left with hit +18 to +23 (62 cm miss); two training shots (10, 36)
   show the same pattern.
2. The sideways error (x spread about 10 cm) is not pixel noise: check the camera calibration and the right-eye row
   offset (measured 9.6 px, calibrated 12.21 px).
3. Tune the filter on Pi INT8 tracks once the Pi run exists (the INT8 320 model shifts detections).
4. Record more shots (a new session) to get a fresh test set.

## Next

6. **Landing miss against frame rate** (vision builder, low priority; moved from the real-time plan). Replay the
   test shots at 60, 30 and 20 fps (every 1st, 2nd, 3rd frame) with the best predictor. Report landing miss, estimate
   at hit +15 and time from hit to first estimate. This shows how much accuracy a dropped frame costs once the live
   camera loop starts dropping frames.

## Waiting on the user

- Decide the default landing method. The Pi run (2026-11-16) is in: Kalman beats the fit on every measure there (see
  "Done"); the manager recommends switching the default to `kalman`.
- Review and commit the overnight work (uncommitted on `pi-speedups`): `vision/flight/core_source/` (filter, smoother),
  `vision/landing_prediction/kalman_landing_model.py`, `vision/landing_prediction/kalman_landing_settings.json`,
  `vision/pipeline/shot_pipeline.py`, `vision/calibration/court_3d.py`, `vision/robot/shot_landing.py`,
  `vision/experiment/pi_realtime_test.py`, this file (paths after the `vision/` layout move, see `vision_layout.md`).
- Confirm what "within 15 cm" means: mean miss on the 20 test shots ≤ 15 cm (assumed here), median ≤ 15 cm, or every
  shot ≤ 15 cm (much harder: three shots miss by 40–60 cm now).

## Done

- 2026-11-16, Pi run of task 5 (user ran it): setup D + overlap, 20 test shots, Pi INT8 tracks, nothing tuned.
  Kalman (per-shot drag) vs fit: final miss 16.6 / 13.6 vs 25.0 / 24.9 cm (14 shots better, 4 worse, 2 equal);
  hit +10 / +15 / +20 mean 39.8 / 25.5 / 15.8 vs 72.6 / 28.9 / 25.3 cm; first estimate hit +4 to +6 vs +7 to +10.
  Predict stage 0.07 vs 4.97 ms; CPU stages 6.21 vs 10.95 ms; wall per frame 15.15 vs 19.14 ms mean (14.24 median,
  22.68 p95; decode included), so the average frame now fits 16.7 ms; 24 vs 62 of 840 frames over budget. The C++
  core built on the Pi. Logged in the run log and the methods doc.

- **Task 5, into the pipeline as an option** (vision builder, 2026-11-16, MacBook Air M2, ONNX setup D). Off by default;
  the least-squares fit stays the default. Pi run waits for the user (see "Waiting on the user").
  - Files: `vision/kalman_landing_model.py` (new: `stereo_camera_from_calibration`, `kalman_settings_from`,
    `pixel_measurements`, `KalmanLandingPredictor`), `vision/kalman_landing_settings.json` (new: task 4's pre-registered
    `RECOMMENDED_SETTING` and `filtered_correction`, copied at full precision), `vision/pipeline.py`
    (`ShotPipeline(..., landing_method="fit" | "kalman")`, `LANDING_METHODS`; fit code moved unchanged into
    `update_fitted_landing`), `vision/court_3d.py` (`StereoCourt` keeps its `calibration`),
    `vision/experiment/pi_realtime_test.py` and `vision/shot_landing.py` (`--landing-method {fit,kalman}`, default fit;
    the realtime run name gets `_kalman`). No change to `flight_cpp`.
  - How it feeds the filter: each frame, the current chain's matches from hit +2 on become (time, left u, left v,
    disparity). If the already-fed measurements are still the start of that list, only the new ones are fed; if the
    chain was re-chosen, the filter is rebuilt from the whole current list. The estimate is the newest landing over
    the fed sequence, `[x, y, 1] @ C`; `finish` does the same on the final chain. That is exactly what task 2's
    `run_filter_over_track` does on the stored final chain.
  - Checks (all 20 test shots): option off, final landings identical to `pipeline_tracks_setup_d_onnx_test.json`
    with and without `--overlap` (0.0 cm; 23.6 / 20.4). Option on, final landings equal task 4 row 2 per shot
    (max gap 0.0 cm; 17.3 / 14.2), and the per-shot hit +10 / +15 / +20 misses equal row 2 too (gap 0.000 cm;
    36.2 / 27.2 / 15.1 mean), with and without `--overlap`. The new module on the stored final chains also gives
    row 2 exactly. `bash vision/flight_cpp/build.sh` builds. `shot_landing.py 6 --landing-method kalman` runs
    (640 only, 7.3 cm; fit 26.8 cm). Identity check only, not a new score.
  - Predict stage (Mac, mean / median / p95 ms per processed frame): fit 1.58 / 1.85 / 2.73, with `--overlap`
    1.91 / 2.20 / 3.43; Kalman 0.026 / 0.026 / 0.043, with `--overlap` 0.024 / 0.022 / 0.042 (about 70× less, and the
    Kalman figure includes building the measurements). Totals 22.6 / 20.1 (fit) vs 22.4 / 19.1 ms (Kalman), Mac noise.
  - The overwritten `pi_test_package/results_*.json` were restored from a copy and the new ones removed.
  - Push / Pi: `push_to_pi.sh` copies all of `vision/` (sources, the settings JSON) and skips `*.so`, so the Pi must
    rebuild `flight_core` (step 2 above; `setup_pi.sh` also builds it). Pi aarch64 build untested.
  - Not done: `vision/README.md` and `CLAUDE.md` unchanged (default unchanged); not cherry-picked to `vision-pipeline`.
  - Logged: run log (two Mac rows, Kalman and fit predict stage), methods doc (note at the end of "One-time test score
    of the Kalman filter (2026-11-16)").

- **Task 4, correction and test score** (vision builder, 2026-11-16, MacBook Air M2, ONNX setup D test tracks). The one-time,
  pre-registered test score.
  - Files: `vision/experiment/score_landing_test.py` (`--step preregister`, then `--step score`; the score step refuses to
    run without the pre-registration, if the code's settings differ from it, or if a score already exists). Outputs:
    `data/recordings/clears_nov10/landing_test_preregistration.json` (written 2026-11-16T01:17:23, test tracks not
    loaded; train-tracks sha256 recorded) and `data/recordings/clears_nov10/landing_test_score.json` (contains the
    pre-registration, all rows, per-shot misses, paired differences).
  - Fixed on the 80 training shots: filter `kalman_landing.RECOMMENDED_SETTING`; refinement
    `final_landing_smoother.RECOMMENDED_REFINE`; corrections `[x, y, 1] @ C`, filtered C =
    [[1.0795, 0.0641], [0.0969, 1.0780], [−73.05, −13.50]] (identical to task 2's `correction_on_all_80_train_shots`),
    refined C = [[1.0367, 0.0218], [0.1176, 1.0380], [−92.82, 10.13]]. In-flight estimates of rows 2 and 3 use the
    filtered C.
  - Test, 20 shots, cm (final mean / median / p90 / max; signed x, y mean ± std; hit +10 / +15 / +20 mean; first estimate):
    1. current, as deployed: 23.6 / 20.4 / 40.8 / 54.5; x −4.0 ± 7.9, y +1.6 ± 24.9; 71.7 / 24.9 / 23.8; frame 7 (7–10).
    2. Kalman B filtered + correction: **17.3 / 14.2** / 28.1 / 62.3; x −4.7 ± 8.8, y −4.6 ± 17.9; 36.2 / 27.2 / 15.1
       (+15 median 18.3 vs 21.1); frame 4 (4–6).
    3. Kalman B + refinement 3c + correction: 18.4 / 14.7 / 23.4 / 95.1; x −5.8 ± 11.4, y −5.4 ± 22.2; in-flight as row 2.
    No row misses a landing. Shots 25 / 68 / 75: current 54.5 / 40.4 / 43.8, filtered 28.0 / 16.3 / 7.8, refined
    9.8 / 22.3 / 10.2.
  - Paired final miss (bootstrap 95 %, 10 000 resamples): row 2 − row 1 −6.3 cm [−13.2, +0.9], 13 better / 7 worse,
    96 % of resamples below 0; row 3 − row 1 −5.2 cm [−13.8, +5.3], 14 / 6, 86 %; row 3 − row 2 +1.2 [−2.6, +5.6].
  - Shot 0 is the filter's failure: 2 restarts, 6 rejected points, last segment only hit +18 to +23, so 62 cm filtered
    and 95 cm refined (current 27). Diagnostic only, not a score: without shot 0 the means are 23.5 / 14.9 / 14.4.
  - Assessment: **the goal (≤ 15 cm mean) is not met** by row 2 or row 3. Row 2 beats the current method by 6.3 cm with
    about 96 % bootstrap confidence (the interval just touches zero), better median, y spread 24.9 → 17.9, first
    estimate 3 frames earlier, ~130× faster; but it is worse at hit +15 on the mean (27.2 vs 24.9) and has one 62 cm
    shot. Row 3 (the refinement) did not keep its training gain on test. For task 5: by the written rule it does not
    go ahead; if the manager integrates it as an improvement, use row 2 (filtered final + filtered C, no refinement),
    since the refinement is not better and adds 95 µs. The restart logic (shot 0) is the next thing to work on, on
    training shots only; these 20 test shots are now spent for these candidates.
  - Logged: methods doc, "One-time test score of the Kalman filter (2026-11-16)", before "Integrators". Nothing waits on
    the user.

- **Task 3, smoother for the final landing** (vision builder, 2026-11-16, MacBook Air M2, ONNX setup D training tracks).
  - Files: `vision/flight_cpp/final_landing.hpp` (new; included by `flight_core.cpp`, plain C++17, no warnings with
    -Wall -Wextra -pedantic), `vision/flight_cpp/kalman_flight.hpp` (keeps the segment's start points in
    `initial_measurements_`; makes `drag_of`, `landing_of`, `measurement_sigmas` public and adds read-only accessors),
    `vision/flight_cpp/flight_core.cpp` (include + `bind_final_landing`). Harness:
    `vision/experiment/final_landing_smoother.py` (`RECOMMENDED_REFINE`). Results:
    `data/recordings/clears_nov10/final_landing_smoother_train.json`.
  - Interface: `flight_core.smoothed_landing(kalman)` -> [x, y, z] m (RTS over `history()`), `flight_core.smoothed_history(kalman)`
    -> states, covariances; `flight_core.refine_final_landing(kalman, max_iterations=5, regate=False, sigma_v_px=-1,
    sigma_disparity_px=-1, terminal_speed_sigma_m_s=-1)` -> dict `landing` (m), `state` (7), `iterations`, `points_used`,
    `cost_start`, `cost_end`, or None before the filter starts (negative σ = use the filter's). It is a Levenberg–Marquardt
    MAP fit of the state at the last step over the latest segment (start points + all steps) in pixel space, using the
    filter's top-edge rule and drag prior mean; regate re-gates every point at Mahalanobis² 16.3 against the fit and refits.
  - Training, 5-fold CV, final landing mean / median cm (signed x, y mean ± std): B filtered (setting in folds) 16.2 / 14.0
    (−0.4 ± 10.5, +0.1 ± 15.7). 1. RTS smoother: identical (max difference 0.0 cm; at the last step RTS = filter by
    construction). 2. q 0–1 + drag walk 0–0.2, filter + RTS: 15.7 / 14.0 (paired −0.5 [−1.2, +0.1]); +15 18.4 vs 18.5, not
    worse. 3a. refinement with the filter's grid noise: 13.4 / 12.8 (−2.9 [−4.9, −0.9]), but the folds pick q = 1 filters,
    so +15 gets worse (21.6). Filter fixed at RECOMMENDED_SETTING: 14.7 / 13.6 (0.0 ± 10.6, +0.1 ± 12.9). 3b. refinement,
    the filter's own noise, regate: 14.0 / 12.8 (−0.8 [−2.2, +0.8]). **3c. own noise picked in folds, 3 it + regate:
    13.1 / 14.0 (−0.1 ± 10.3, −0.1 ± 10.7), paired −1.7 [−3.1, −0.3], 46 shots better / 34 worse**; 1 it + regate
    13.1 / 13.5. A modest gain: the mean and the y spread improve (12.9 → 10.7 cm), the median does not. Regating carries most
    of it.
  - Timing (Mac): filter landing 4.2 µs, RTS 12 µs, refinement 3c 95 µs median / 125 µs p95 / 161 µs max per call (once per
    shot); 1 it + regate 29 µs.
  - Checks: the filtered estimates are bit-identical at every frame of all 80 training shots for all 90 task-2 settings.
    Test shots 0, 6, 12 through `pi_realtime_test.py` (ONNX, setup D, follow 320, once-seen) give the same landings as
    `pipeline_tracks_setup_d_onnx_test.json`; the overwritten results file was restored. Test tracks were not used.
  - For task 4/5: the raw refined landing is still biased (x +28.7, y −17.7 cm), so it needs its own correction. It
    covers the latest segment only (after the last restart). RECOMMENDED_SETTING was chosen in task 2 with all 80 shots in
    view, so the fixed-filter baseline (14.7) is a little optimistic, for both rows alike. The Pi build is untested.
  - Logged: methods doc, "Smoother for the final landing on the training shots (2026-11-16)", before "Integrators".
    Nothing waits on the user.

- **Task 1, error breakdown on the training shots** (vision builder, 2026-11-16, MacBook Air, ONNX setup D).
  - Files: `vision/experiment/replay_pipeline_tracks.py` (replays shots through `ShotPipeline`, setup D; reuses
    `frames_needed` and `load_shots`, landing_x sign −1 as in `landing_experiment.py`), and
    `vision/experiment/track_error_breakdown.py` (per-axis miss, pixel residuals, projection helper).
    Data: `data/recordings/clears_nov10/pipeline_tracks_setup_d_onnx_{train,test}.json` plus
    `*_error_breakdown.json`. The replay reproduces all 20 Mac D test landings exactly (23.6 / 20.4 cm).
  - Training miss (80 shots, current fit + correction): 26.4 / 24.2 cm mean / median. x signed mean 0.2, std 10.7 cm;
    y signed mean 0.3, std 28.7 cm. At hit +15: 28.9 / 27.5 cm (x std 10.8, y std 31.6). The error is along the court.
  - Pixel residuals (robust / std): u 0.43 / 0.55, v 1.89 / 2.87, disparity 0.18 / 0.38 px. Disparity does not grow
    with depth or along the flight. There are outliers: 3 % of points beyond 4 robust σ, up to 4 px. Noise
    recommendation: written into task 2.
  - Shots 25, 68, 75: all three tracks are clean (no rejected points, disparity residual < 0.45 px, the same as
    the offline tracker within 0.3 px). Each track ends at the top of the image at hit +17–20, before the apex.
    The misses are all along the court. 25 (+54 cm) and 75 (+43 cm): a free-drag fit wants 5.8 m/s instead of
    6.5, so the extrapolation runs long. 68 (−38 cm): drag fits 6.5 m/s and it is the farthest start (8.3 m,
    10 px disparity), so a small disparity bias at range is the likely cause. Not proven.
  - Per-shot drag on train: median 6.38 m/s (IQR 6.18–6.59), correlates −0.69 with the y miss. In a plain 3D
    least-squares fit it only lowers the training miss to 24.6 cm (in-sample). The filter should do better,
    because it weights depth correctly.
  - Logged: methods doc, "Error breakdown on the training shots (2026-11-16)" under "Landing prediction".
    Nothing waits on the user.
- **Task 2, extended Kalman filter in C++** (vision builder, 2026-11-16, MacBook Air M2, ONNX setup D training tracks).
  - Files: `vision/flight_cpp/kalman_flight.hpp` (new, included by `flight_core.cpp`; built by `build.sh`, plain C++17,
    no Eigen) and `vision/flight_cpp/flight_core.cpp` (`landing_point` split into `landing_from` + wrapper, so the
    output is bit-identical). Harness: `vision/experiment/kalman_landing.py`. Results and per-shot raw estimates by frame:
    `data/recordings/clears_nov10/kalman_landing_train.json`.
  - Interface: `flight_core.StereoCamera(focal_px, principal_point_px, right_row_offset_px, distortion_k1_k2,
    baseline_m, camera_from_court, translation_m)` (`kalman_landing.stereo_camera_from_calibration(json)` builds it)
    with `measurement(court_m)` -> (u, v, disparity) and `court_point(u, v, d)`. `flight_core.KalmanSettings()` has
    read/write fields. `flight_core.KalmanFlight(camera, settings)` has `update(time_s, u, v, disparity)` -> landing
    [x, y, z] in court m (None until initialised), `state()` (x, y, z, vx, vy, vz, terminal speed; m, m/s),
    `covariance()` 7x7, `landing()`, `landing_covariance()` 2x2 m² (about 54 µs, on demand), `history()`,
    `rejected_count`, `reset_count`. `kalman_landing.run_filter_over_track(camera, entry, setting)` runs one shot.
    Variant A is the same filter with `terminal_speed_sigma_m_s = 0` (zero variance), which is exactly the 6-value filter.
  - Checks: build OK, no warnings with -Wall -Wextra -pedantic. The old core functions match the old build bit for bit on
    2000 random flights. Projection C++ vs Python under 1e-6 px on 24 track points. Test shots 0, 6, 12 replayed with
    `pi_realtime_test.py` (ONNX, setup D) give the same landings as `pipeline_tracks_setup_d_onnx_test.json`. The Pi
    build is untested (aarch64 GCC), but the code is plain C++17 with explicit includes.
  - Training shots, 5-fold cross-validated (setting and correction chosen inside the folds), mean / median cm:
    current fit + correction learned in folds 27.9 / 25.5 (+15: 30.4 / 27.2, y std 30.2; as deployed, in-sample
    26.4 / 24.2); A + correction 21.3 / 17.4 (+15: 19.6 / 18.9, y std 28.5); **B + correction 16.2 / 14.0
    (+15: 18.5 / 15.9, x std 10.5, y std 15.7)**. Raw (no correction): A 32.7 / 33.2, B 33.9 / 32.9 (bias x +25,
    y −11 cm). First estimate: frame 4 after the hit (median; range 4–7) vs 7 (7–10) for the current fit.
  - Timing (Mac): 4.9 µs per filter update including the landing (p95 7 µs), against 0.65 ms (median 0.52) per
    warm-started SciPy fit on the same machine, about 130× faster. Not yet measured on the Pi.
  - **Chosen settings for tasks 3–4** (`kalman_landing.RECOMMENDED_SETTING`): σ_u 0.5, σ_v 1.0, σ_disparity 0.2 px;
    acceleration noise 0; terminal speed prior 6.4 ± 0.15 m/s, no random walk; σ_v × 5 for v < 12 px; gate
    Mahalanobis² 16.3; points from hit +2; start from 3 points (triangulate the first and last, start at the last,
    velocity corrected for drag over the span, covariance = triangulation geometry × 3² plus 1 m/s on velocity); restart
    from the rejected points after 3 rejections in a row. Every fold picked the 0.15 prior and zero process noise;
    σ_v was 1.0 in 3 folds and 0.5 in 2. In-sample with a correction on all 80 shots: 14.2 / 13.4, +15 17.1 cm; that
    correction is in the results file (`recommended.correction_on_all_80_train_shots`). Final terminal speeds:
    median 6.31 m/s (IQR 6.21–6.38).
  - For task 3 (RTS smoother): `history()` returns per step `times_s`, `measurements`, `predicted_states` /
    `predicted_covariances` (prior), `states` / `covariances` (posterior), `transitions` (F from the previous step to
    this one, central differences; identity on the first step) and `accepted`. A restart clears the history, so it
    holds the latest segment only (7 restarts across 80 shots). Rejected steps keep posterior = prior. With zero
    process noise the RTS smoother mostly reproduces the final filtered state at the end, so the gain will come from
    re-linearising (an iterated smoother) or from smoothing with q > 0.
  - Limits: the filter still needs the linear correction (task 4). Its landing σ is right along the court
    (miss/σ std 1.18) but 6× too small sideways (median σ_x 1.8 cm against a 10.5 cm spread), so the x error is not
    pixel noise; calibration is the likely cause. Shots with mismatched early disparities (10, 36) restart and lose
    points. The structural choices (3 start points, restart after 3, start covariance scale 3) were set by looking at
    in-sample results on the default noise, not cross-validated.
  - Logged: methods doc, "Extended Kalman filter in C++ on the training shots (2026-11-16)", before "Integrators".
    The Mac µs timing is only in the methods doc; the Pi run log gets numbers once the filter runs on the Pi (task 5).
    Nothing waits on the user.
