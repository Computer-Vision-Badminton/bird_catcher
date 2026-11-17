# Goal: automatic hit detection

**Reopened 2026-11-17 by the user** (paused and archived earlier the same day) for the searching stage. Task 1 met the target on the 80 training shots: the rising chain finds
every hit, 78 within ±1 frame, 79 within ±2, 0 false triggers, about 3 frames (50 ms) of delay; no learned model was
needed. Not done: live mode with the one-time test score, a cheaper waiting state (the 640 search every frame is over
budget), Pi timing. Last step before archiving: make the hit script runnable on the Pi (Hailo option, a small hit test
package) so the user can test it there. Idea for later: tell the feeder's shot from the far player's hit by distance
from the camera (stereo disparity) instead of the image-row gate, which depends on this camera and feeder position.
Explanation page: "How the Rising Chain Finds the Hit" artifact (https://claude.ai/artifact/4KZu8XueJQpRSWy6hynBbg).


Find the frame where the shuttle is hit without a hand label, so a live robot can start tracking on its own. Test
shots now start at a hand-labelled hit frame.

Target: hit frame within ±2 frames of the label, no false triggers between shots, and test-shot landings (Kalman,
setup D) no worse than with the labelled hit by more than about 1 cm.

## Data (manager, 2026-11-16)

- Labels: `data/recordings/clears_nov10/shots.csv`, `hit_frame` for all 100 shots (80 train / 20 test, same split as
  the landing work). Clip windows put the hit 88–113 frames after `start_frame`.
- Negatives: the takes (`take_*.avi`, 20 takes) also hold the time between shots, median 172 frames (min 69): players
  walking, picking up shuttles. A detector must stay quiet there.
- `data/recordings/smash_nov10/` (3 takes) has no labels; later, a check that the detector also works on smashes.

## Scope (confirmed by the user, 2026-11-16)

- The label is exactly the racket–shuttle contact frame.
- Clears only for now; smashes and rallies later.
- Assumption (manager): the hit may be confirmed a few frames late: the pipeline keeps a short frame buffer and
  tracks from the detected hit frame, so a delay of a few frames costs no tracking.

## Rules for this goal

- Tune on the 80 training shots and the gaps around them only; score the 20 test shots once per finished candidate.
- Prefer the existing pipeline's own signals (YOLO detections, the rising chain) before training anything new; train a
  learned detector only if those miss the target. With 100 labelled hits, keep any learned model small.
- Watch the cost while waiting for a hit: before the first detection every frame runs the 640 full search (about
  27 ms of Hailo per frame on the Pi), which is over the 16.7 ms budget.

## In progress

Nothing.

## Follow-ups from the live run (manager, 2026-11-17)

1. Hit-frame sensitivity of the landing: one frame early moved a landing by up to 31 cm (Mac, shot 49). Make the
   Kalman start independent of a ±1 frame hit error; judge on training shots only (the test shots are used up).
   **Deferred by the user (2026-11-17): keep as a note, not worked on now.**
2. **Done 2026-11-17 (Pi: hit frame 106 → 12.3 ms).** The replay spike: when the hit fires, 8 buffered frames go through tracking at once (106 ms on the Pi), which a
   live camera would answer by skipping about 6 frames right after the hit. Feed tracking as frames arrive instead.
3. Searching is 0.8 ms over budget (17.5 ms): the gray and handoff around the 16.1 ms YOLO 640 call.
4. **Done 2026-11-17.** `score` on the Pi: store the reference numbers in the package manifest, so it does not need `data/`.
5. The last frame of each shot waits for its final YOLO call with nothing to overlap (about 23 ms, once per shot,
   after the landing is final). Harmless for now; at most one frame skipped while going back to searching.

## Next

1. **Live mode in the pipeline** (vision builder). Done on the Mac and the Pi 2026-11-17 (see "Done"). A pipeline entry that takes a continuous frame stream with no hit
   label: a short frame buffer, the task 1 rule (`vision/experiment/hit_from_rising_chain.py`, best rule in
   `data/recordings/clears_nov10/hit_detection_baseline_train.json`) moved into production code, then `ShotPipeline`
   from the detected hit. Score once on the 20 test shots: hit-frame error, false triggers, and the landing miss
   (Kalman, setup D) against the labelled-hit result (17.3 / 14.2 cm Mac).
2. **A cheaper waiting state** (vision builder). Waiting runs the 640 full search every frame (34 ms on the Mac, about
   27 ms of Hailo on the Pi), over the 16.7 ms budget. Ideas: search only the far hitter's region (the hits start at
   rows 470–510), run the 640 search every 2nd frame while nothing moves there, or 320 crops on the moving blobs there.
   Must keep task 1's hit accuracy and delay on the training shots.
3. **Pi timing of the waiting state** (user runs it): ms per frame while no shot is in the air.
4. Later: a learned detector only if live mode or smashes need it; check the row gate on smashes (`smash_nov10`).

## Ideas for the searching stage (manager, 2026-11-17, from the Pi run)

Searching costs 27.8 ms per frame on the Pi: YOLO 640 on the Hailo 17.7 ms + full-frame blob search 9.4 ms (CPU) + gray
0.8 ms, in sequence. Following before the hit costs 7.7 ms. In order (gains are estimates, not measured):
1. Faster blob search, same blobs: blur each frame once and keep it (now each frame is blurred 3 times), use
   `cv2.absdiff` / `cv2.min` on uint8 instead of NumPy int16, limit the search to where a shuttle can first appear.
   Check: identical blobs. Maybe 9.4 → 2–3 ms.
2. Overlap the blob search with the Hailo call while searching (aiming the next frame does not need this frame's
   YOLO result), as tracking already does. About max(17.7, blob search) per frame.
3. Search with the 320 model: 6.9 vs 17.7 ms. Setup C failed only because the test clips start 3 frames before the
   hit; in the live loop the search finds the feed about 100 frames earlier and the hit is caught while following.
   Check on the hit test package: all 12 hits unchanged, 0 false triggers, frames until the feed is found.
4. Search every 2nd or 3rd frame while nothing is found; check the same way.

## Waiting on the user

Nothing. (Every Pi run listed here earlier is done; results are under "Done".)

## Done

- 2026-11-17, Pi run of "both eyes while following before the hit" (user ran it, scored on the Pi): 20/20 hits, same
  signed errors as the previous run, 0 false triggers, 0 replay fallbacks, 5–7 frames seeded per hit. The hit frame
  costs 12.3 ms main (was a 106 ms replay), 0 hit frames over 16.7 ms. Following 11.7 ms wall (was 8.7), tracking
  11.7 ms, searching 17.6 ms. Landing 15.0 / 14.2 cm (was 16.4 / 14.6; shot 0 53.8 → 28.7 cm); hit +10 / +15 / +20
  mean 51.4 / 24.2 / 15.1 cm (the +10 mean is shot 0, 405 cm). 45.8 s for 3,520 frames. Accepted by the manager.
  Logged in the run log.

- **Both eyes while following before the hit, tracking seeded at the hit, Mac part** (vision builder, 2026-11-17).
  Mac M2, ONNX, overlap and reader thread on. Pi run waits on the user.
  - Files: `vision/hit_detection.py` (`WaitingDetector(..., right_row_offset=None)`: when given, 320 crops also cut
    the right crop at `right_crop_center` of the left aim and detect [left, right] in one `detect_batch` /
    `start_batch`; right grays converted only when needed; entry gains `right_crop`, `right_detections`,
    `stereo_detections`; `pending_stereo_frame`, `hand_over_pending`. Without the offset the path is unchanged:
    hit script caches for take_006 equal to the backed-up ones on all 1,388 frames, with and without `--overlap`).
    `vision/pipeline.py` (new only: `take_known_frame_time`, `take_known_grays`, `push_detected` (remembers the
    detections and updates the left follower as `push` does, then `track_and_predict`), `take_running_detection`;
    `push` and every other method unchanged). `vision/live_mode.py` (stage 2 on both eyes with the batch-2 320
    detector; buffer of stereo detections for the last 24 frames; at a hit, frames `hit - 2` .. `fired - 2` are
    seeded from the buffer, the running detection of `fired - 1` is handed to `ShotPipeline` as its pending
    detection, grays of the last 3 frames are handed over; replay only as fallback, counted in `replay_fallbacks`;
    hit events log `start`, `seeded_frames`, `replayed_frames`, `hit_frame_main_ms`; `--hailo-models four` renamed
    `search-batch-1`: the 640 `.hef` at batch 1 and 2, the 320 `.hef` at batch 2 only, three models).
    `vision/experiment/live_replay_test.py` (`--no-seed-tracking` = the old behaviour, results suffix
    `_replay_at_hit`; prints each hit; score reads the reference numbers from the manifest; "hit frames" line with
    seeded / replayed / fallbacks). `vision/experiment/make_live_test_package.py` (new command
    `update-scoring-only`; `scoring_only.shots[]` gains `labelled_hit_kalman_miss_hit_plus_{10,15,20}_cm`).
  - Check 1, hits: 20 hit events identical to the last Mac run (`fired_at`, `hit_estimate`, `fired_on_frame`,
    chains). Per frame, all 2,743 waiting detections (left) equal the old log; states and modes identical on all
    3,520 frames. 2,070 frames were detected on both eyes.
  - Check 2, landings: 17.8 / 15.1 cm mean / median (p90 28.5, max 41.9) vs the last live run 19.4 / 14.8 and the
    labelled-hit run 17.3 / 14.2. Shot 0 62.3 -> 29.1 cm (21 stereo points instead of 19: the stage-2 crops keep the
    bird near the hit where setup D's own start missed it); shot 37 has 17 points instead of 19 but moves 0.06 cm;
    the other 18 shots move at most 2.7 cm (87: 24.7 -> 27.3; 68, 93 -1.0; 25 -0.7). Hit +10 / +15 / +20: 68.7 /
    27.6 / 18.0 (last live 47.6 / 29.3 / 17.2); the +10 mean is driven by shot 0 (536 cm at +10). Nothing tuned.
    Old behaviour re-run in the same session (`--no-seed-tracking`): landings and all estimates identical to the
    last Mac run (19.4 / 14.8).
  - Check 3, spike: hit-frame main thread 21.3 ms median, max 30.3 (tracking frame 21.2); same session with the
    replay 208.7 median, max 434.7 (8 frames replayed). Replay fallbacks 0. Seeded frames per hit: 5-7, median 6.
    First landing estimate after the true hit: median 6 frames, identical per shot to the last run except shot 0
    (8 -> 6). A first try that waited for the running detection at the hit cost 39 ms (two YOLO waits); handing
    that detection over to `ShotPipeline` removed the second wait, same landings.
  - Check 4, stage 2 cost (Mac, same session, median main / wall): following 20.4 / 20.4 ms, YOLO call 19.5 (left
    only: 11.2 / 11.2, call 10.7); ONNX runs the two crops one after the other. Tracking 21.2 (21.9), searching 36.3
    (37.4). All frames median wall 21.1 (12.8); run 87.2 s (73.5). Mac CPU timing is noisy (the last session's run
    was twice as slow throughout).
  - Check 5, `--use-labelled-hits` with seeding: 15.8 / 14.3 cm vs `kalman_filtered` 17.3 / 14.2; 10 shots within
    0.1 cm, 7 within 0.6, shot 18 1.6, shot 87 2.7, shot 0 79.9 (62.3 -> 29.1, 21 stereo points instead of 19).
    Not within 0.1 cm because frames hit - 2 .. hit + 2 now come from the stage-2 crops (aimed by the waiting
    follower) instead of setup D's own first crops; the seeding was kept.
  - Check 6, other paths: `pi_realtime_test.py` setup D + overlap, shots 0, 6, 12: Kalman landings and +10/+15/+20
    misses equal `landing_test_score.json` exactly; fit landings equal `pipeline_tracks_setup_d_onnx_test.json`.
    New results files removed, `data/` files unchanged.
  - Follow-up 4: `make_live_test_package.py update-scoring-only` rewrote only `scoring_only` (other keys
    identical, old fields unchanged); `verify`: 3,520 compared, 0 different. Re-scoring the previous Pi run from the
    manifest gives the same hits and landing block as the score from `data/` (file restored).
  - Hailo note: following now uses the 320 `.hef` at batch 2 with [left, right] instead of batch 1 with the left
    crop; INT8 left detections may differ at the bit level, so the Pi hits need checking against the previous Pi log
    (`pi_live_test_package/live_*_hailo_catcher_before_seeding.json`).
  - Unmeasured: Pi timing, Pi hits and landings. Pipeline changes need cherry-picking to `vision-pipeline` when
    committed.
  - Logged: run log "Bird Catcher: Real-Time Performance on the Pi", one Mac row (2026-11-17).

- 2026-11-17, Pi run of the live mode (user ran it; scored on the Mac with `--machine catcher`, because `score` reads
  `data/recordings/…/landing_test_score.json`, which is not on the Pi). One-time test score of hit detection on the Pi:
  20/20 hits, 11 exact, 19 within ±1, 20 within ±2, 0 false triggers. Landing 16.4 / 14.6 cm (labelled hits, Mac:
  17.3 / 14.2). Wall per frame: searching 17.5 ms (656/735 over 16.7, YOLO 640 call 16.1), following 8.7 ms, tracking
  11.6 ms; 20 tracking frames over budget, one per shot, when the hit fires and 8 buffered frames are replayed at once
  (106 ms). Overall 11.6 ms per frame. Logged in the methods doc and the run log. Accepted by the manager.

- **Live mode, step (b): `LiveShotLoop` + replay runner, Mac part** (vision builder, 2026-11-17). Mac M2, ONNX,
  overlap on. Pi run waits on the user.
  - Package fix: `vision/experiment/make_live_test_package.py` windows now end at max(end_frame + 10, hit + 45)
    (19 of 20 windows grew by 4-12 frames; shot 0 unchanged). Remade: 3,520 frames (was 3,377), 175-176 per window,
    600 MB; `verify`: 3,520 compared, 0 different, container count = window sum = frame_times rows, no extra frame,
    times strictly increasing. No window clipped or overlapping (one test shot per take, none near a take edge).
  - Files: `vision/live_mode.py` (new, production: `LiveShotLoop` with searching / following / tracking, 24-frame
    ring buffer, `load_live_detectors` with Hailo `four` or `shared-batch-2`), `vision/experiment/live_replay_test.py`
    (new: `run` with reader thread, no labels, `--use-labelled-hits` debug; `score`, the only reader of
    `scoring_only`), `vision/experiment/make_live_test_package.py` (window end), `vision/deploy/push_to_pi.sh`
    (skips `live_score_*.json` too). `pipeline.py`, `hit_detection.py`, Kalman settings, hit rule and `flight_cpp/`
    unchanged.
  - Check 1, integration identity (`run --use-labelled-hits`, fires when frame hit + 3 arrives): all 20 final
    landings equal `landing_test_score.json` `kalman_filtered` within 0.06 cm (17.3 / 14.2 / p90 28.1 / max 62.3);
    hit +10 / +15 / +20 within 0.01 cm (36.2 / 27.2 / 15.1). Buffer replays 7 frames per shot, none missing.
  - Check 2, **one-time test score of hit detection** (no labels): 20 / 20 found, exact 10, within 1 19, within 2 20,
    worse 0; signed errors -1 x7 (25, 49, 50, 62, 68, 74, 93), +1 x2 (12, 99), +2 x1 (37). False triggers 0 (0 near
    a join). Rule delay median 3 frames (2-4); the loop acts one frame later (overlap): fired 4 frames (67 ms) after
    the true hit, median.
  - Landing (first end-to-end score): 19.4 / 14.8 cm mean / median, p90 30.3, max 62.3; signed x -5.4, y -7.4. Against
    the labelled hit (17.3 / 14.2): +2.1 cm mean, over the ~1 cm target. Exact hits move under 0.1 cm; one-frame
    errors move up to 48 cm: 49 11.4 -> 42.3, 50 16.2 -> 28.1, 68 16.2 -> 28.9, 25 28.0 -> 16.6 (better). Same stereo
    point counts; likely cause: the Kalman start gate (frame - hit >= 2) and the window move with the hit estimate.
    Hit +10 / +15 / +20 (from the true hit): 47.6 / 29.3 / 17.2 cm mean (labelled: 36.2 / 27.2 / 15.1). No shot
    without a landing. Nothing tuned afterwards.
  - End to end, camera time from the true hit: first landing estimate 6 frames (100 ms, 5-8), final landing 40
    frames (667 ms, 39-42). The 24-frame buffer replays 7-9 frames, never short.
  - Timing per state, Mac, medians (main / wall ms): searching (736 frames) 76.5 / 76.9, YOLO 640 call 78.3;
    following before the hit (2,050) 23.7 / 23.8, YOLO 320 call 23.0; tracking (714) 49.2 / 49.2, YOLO both eyes
    47.3, track 0.3, predict 0.1; hit frames 510 ms (replay). Over 16.7 ms by wall almost every frame (Mac CPU).
  - Check 3, reader thread: without it main thread drops (searching 66.0, following 13.3, tracking 37.1) but wall is
    the same (77.5 / 23.9 / 47.8; run 160.7 s both ways): onnxruntime uses every core, so on the Mac decoding on
    another thread only moves the cost. Same hits and landings.
  - Unmeasured: Pi timing and Pi hits / landings (commands under "Waiting on the user"); whether HailoRT accepts the
    same `.hef` at batch 1 and 2 on one VDevice. Pipeline-adjacent new file `vision/live_mode.py` needs
    cherry-picking to `vision-pipeline` when committed.
  - Logged: methods doc, "Live mode on the chained test video (2026-11-17)" at the end of the hit detection section;
    run log, two Mac rows (reader thread on / off).

- **Live mode, step (a): chained test video + `vision/hit_detection.py`** (vision builder, 2026-11-17). Mac M2, ONNX.
  No new measurement, nothing logged in the Claude Docs.
  - Files: `vision/hit_detection.py` (new, production), `vision/experiment/hit_from_rising_chain.py` (imports the
    waiting stage and hit rule from it; own copies removed), `vision/experiment/make_live_test_package.py` (new:
    `make`, `verify`), `.gitignore` (`pi_live_test_package/`), `vision/deploy/push_to_pi.sh` (copies
    `pi_live_test_package/`, skips `live_results_*.json`, `live_log_*.json`, `live_cache_*/`, `*.writing`).
  - Package `pi_live_test_package/` (549 MB): `live_test.avi` (MJPEG 2560 x 720, 3,377 frames, 20 windows of
    164-176 frames, take frames hit - 130 .. end_frame + 10, stream copy, ffmpeg concat with stream copy, shot order
    0, 6, ..., 99), `frame_times.csv`, `manifest.json`. No window clipped (one test shot per take, none at a take
    edge).
  - Bit identity: all 3,377 decoded frames equal their take frames by MD5 (0 different), container frame count and
    decoded count 3,377 = sum of windows, no extra frame. Frame times strictly increasing; join gap 16.663 ms (median
    in-window frame period), shift per window in the manifest (`time_shift_s`, window 0 unshifted).
  - Moved code identity: `detect --package pi_hit_test_package --detector onnx --redo` without and with `--overlap`:
    caches equal to the backed-up ones on all 4,785 frames (same frames, modes, crops, detections; only timings
    differ). `evaluate`: summary, per-shot and false triggers identical (12 found, 0 false triggers), both modes.
    Full training `evaluate` (whole rule sweep, incl. extrapolation rules) identical to
    `hit_detection_baseline_train.json` except the date. Original caches, results and baseline file restored.
  - `HitWatcher` fed frame by frame equals `fire_events` (same events, fire frames, estimates, chains) on the 3
    package takes (both caches) and on all 20 training takes of the full cache.
  - Unmeasured: everything of step (b).

- 2026-11-17, Pi run by the user: searching stage with the 640 crop kept. Faster blob search (same blobs) 9.4 → 3.3 ms;
  with the blob search overlapped with the YOLO 640 call, searching costs 9.2 ms of CPU per frame (was 27.8), the
  YOLO call (16.2 ms) runs in parallel; wall with the 8.4 ms clip decode 17.7 ms (was about 36). Following before the
  hit: 1.7 ms CPU, 10.1 ms wall. Same 12 hits, 0 false triggers, with and without overlap. Estimate for a live loop
  with the camera on its own thread: about 16.2 ms per searching frame, set by the YOLO 640 call, just under budget.
  Logged in the run log. Task accepted by the manager.

- **Searching stage: faster blob search + overlap with YOLO, 640 kept, Mac part** (vision builder, 2026-11-17). Mac M2,
  ONNX. Pi timing waits on the user.
  - Files: `vision/bird_tracker.py` (new `moving_blobs_in_uint8`: `cv2.absdiff` / `cv2.min` / `cv2.threshold` on
    uint8; old `moving_blobs` unchanged for its other users), `vision/pipeline.py` (`blurred_for_blobs`,
    `BlurredFrameCache` keeps each gray's blur for the next searches; `blobs_around` same name, same output, uint8
    path; `blobs_near` still blurs its window, since a window blur differs from a full-frame blur at the window
    edge; `ShotPipeline` uses the cache for its full-frame search), `vision/experiment/hit_from_rising_chain.py`
    (`--overlap` on detect and evaluate with its own cache and results file; speculative blob search for m+1 while
    YOLO runs on m via `start_batch` / `wait`; new per-frame timings `detector_call_ms`, `main_ms`, `decode_ms`,
    `frame_wall_ms`; every JSON written atomically: temporary file, then `os.replace`).
  - Blob identity: old against new on all 4,785 frame triples of the hit test package: 0 differences (92,297 blobs,
    centres and areas; 3,247 searching frames), `blobs_near` 0 differences on 4,785 windows.
  - Blob search alone, Mac: 1.26 -> 0.69 ms median (1.41 -> 0.79 mean; 0.94 without the blur cache). Search aim
    step in the hit script 1.89 -> 1.20 ms median. About half of what is left is `connectedComponentsWithStats`
    (other OpenCV CCL algorithms were not faster).
  - 20 test shots, setup D, ONNX: landings, stereo points and checkpoint misses identical to before on all 20, with
    fit (23.6 / 20.4 cm, = `pipeline_tracks_setup_d_onnx_test.json`) and kalman (17.3 / 14.2 cm, = row 2 of
    `landing_test_score.json`). `pi_test_package/results_*.json` restored.
  - Overlap identity: cache with `--overlap` equals the original run's cache on all 4,785 frames except timings
    (modes, crops, detections); the cache without overlap (new blob code) too. Same 12 hits, 0 false triggers.
    Original onnx cache and results file restored after the runs.
  - Searching frames, Mac, median, without -> with overlap (two pairs of runs, noisy): main thread excl. decode
    42.9 -> 46.9 and 51.1 -> 47.6 ms; detector call 40.6 -> 53.1 and 48.9 -> 53.9; wall incl. decode 48.3 -> 54.1
    and 57.5 -> 54.9. No gain on the Mac: onnxruntime uses every core, so the overlapped CPU work slows the
    detector. On the Pi the Hailo runs YOLO, so the overlap should hide the CPU work.
  - Hailo single crop: `HailoDetector.start_batch` with 1 crop and batch size 1 already takes the `run_async` path
    (the blocking fallback is only for more crops than the batch size). No change needed.
  - Unmeasured: Pi timings (commands under "Waiting on the user"). Pipeline changes need cherry-picking to
    `vision-pipeline` when committed.
  - Logged: two rows in "Bird Catcher: Real-Time Performance on the Pi" (2026-11-17, Mac).

- 2026-11-17, Pi run by the user (Hailo INT8, batch 1, hit test package: takes 000, 006, 011, 12 training shots):
  hit estimates identical to the Mac on all 12 shots (12 found, 6 exact, 10 within ±1, 11 within ±2; shot 33 +3,
  shot 57 −2), median delay 3 frames, 0 false triggers. Waiting with the 640 search: 27.8 ms median per frame (YOLO
  17.7, full-frame blob search 9.4 on the CPU, gray 0.8), over the 16.7 ms budget; following at 320: 7.7 ms. Two bugs
  hit on the way and fixed: Hailo float32 values in the JSON cache, and a half-written cache file left by that crash
  (re-run with --redo). Still to do: write cache files atomically. Logged in the run log and the methods doc.

- **Hit script runnable on the Pi** (vision builder, 2026-11-16). Mac M2, ONNX.
  - Files: `vision/experiment/hit_from_rising_chain.py` (`--detector {onnx,hailo}`, both Hailo models on one VDevice
    created before any inference, batch 1; `--package`; new commands `package` and `verify-package`),
    `vision/deploy/push_to_pi.sh` (copies `pi_hit_test_package/`, skips its caches and results), `.gitignore`
    (`pi_hit_test_package/`, 800 MB, stays out of git).
  - Package `pi_hit_test_package/`: take_000 1822-3496, take_006 127-563 and 733-1687, take_011 45-572 and
    743-1942 (exactly `processed_ranges`; 4,795 frames, 12 training shots incl. 1, 33, 57; 0.77 min of gaps). Clips cut
    by ffmpeg stream copy, frame times per clip, `manifest.json` with frame offsets, training hit labels, skipped test
    windows, gap ranges and the baseline rule. 799 MB.
  - Bit identity: all 4,795 decoded frames match the original takes by MD5 (0 different, no extra frames); packet MD5s
    of take_006 733-1687 identical too.
  - Result identity: the package run's per-frame cache equals the full-take cache for all 3 takes (same frames, crops,
    modes and detections, 0 differences); per-shot results identical to the baseline (12 found, 6 exact, 10 within 1,
    11 within 2, shot 33 +3, shot 57 -2, delay median 3, 0 false triggers).
  - Mac time: detect 3 min 3 s for the package (37-39 ms per frame including decoding), evaluate under 1 s.
  - Not logged in the Claude Docs (no new measurement). Pi waiting-state timing still unmeasured.

- **Task 1, baseline from the rising chain** (vision builder, 2026-11-16). Training takes only, Mac M2, ONNX, setup D
  waiting mode, left eye.
  - Files: `vision/experiment/hit_from_rising_chain.py` (`detect` caches per-frame detections in
    `data/recordings/clears_nov10/hit_detection_cache/`, `evaluate` sweeps rules, runs leave-one-take-out and writes the
    results); results in `data/recordings/clears_nov10/hit_detection_baseline_train.json`.
  - Coverage: every frame of each take from 150 frames before its first training shot to 150 after its last (29,899
    frames, all inter-shot gaps). The 20 test windows (start_frame to end_frame + 40) were skipped and the state reset.
  - Best rule: fire when a rising chain (bird_tracker rules) of >= 3 points, rise >= 20 px, confidence >= 0.25, ends on
    the newest frame and starts at image row >= 350. Hit = first chain point + 1. Looks back 40 frames, re-arms 40
    frames after firing.
  - 80 training shots: 80 found, 0 missed. Error: exact 34, within 1 78, within 2 79, worse 1. Mean / median absolute
    error 0.61 / 1 frame. Signed: -2: 1, -1: 21, 0: 34, +1: 23, +3: 1. Leave one take out: 34 / 76 / 78 / 2 worse,
    0.65 / 1 frame, 0 missed, 0 false.
  - False triggers: 0 in 4.55 min of gaps, and 0 in 6.36 min without a flight (0 per hour, but only 0.11 h measured).
  - Delay, true hit to firing: median 3 frames, range 2 to 5.
  - Mac idle cost: 34.1 ms per frame (YOLO 640 32.4, blob search 1.5, gray 0.16), measured on one process; 9.4 ms per
    frame while following at 320.
  - Failing cases: shot 33, +3 (YOLO first sees the bird 2 frames after contact). Shot 57, -2 (chain starts on a
    detection 3 frames before contact, then skips 2 frames).
  - Near-side feeds: about 100 frames before every clear, the feeder by the camera hits the shuttle over. Without the
    row gate this fires 46 times; feed chains start at rows 170 to 310, far hits at rows 470 to 510.
  - Limits: the gate assumes far-side clears and this camera. The gaps total only 4.6 min; YOLO finds anything in 37 of
    16,364 gap frames. Idle cost is the 640 full search every frame (about 27 ms of Hailo on the Pi, over budget), not
    the rule.
  - Task 2 (learned detector) does not look necessary for clears. Next: task 3 (live mode) and a cheaper waiting state.
  - Not measured: test shots (left for task 3), smashes, Pi idle timing (task 4).
  - Logged in "Bird Catcher: Methods Tested and Accuracy", section "Hit detection from the rising chain (2026-11-16)".
