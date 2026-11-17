# Goal: live camera capture loop

Run the pipeline on the live ZED camera instead of recorded clips: ZED over UVC at 2560 × 720, 60 fps, into
`ShotPipeline.push`, dropping frames when the pipeline falls behind and counting the drops.

## In progress

Nothing yet.

## Budget (manager, 2026-11-16, from the closed real-time goal)

Processing takes 11.0 ms per frame on the Pi (setup D + overlap); the 16.7 ms budget leaves about 5 ms. In the replay,
decoding each frame from the clip costs 8.9 ms on the same thread, so frames take 19.7 ms. Fetching a frame must not
sit on the processing thread.

## Next

1. **Fetch frames on their own thread, replay first** (vision builder). In `pi_realtime_test.py`, read and decode the
   next clip frame on a separate thread (a small queue) while the pipeline processes the current one, the way a camera
   thread would.
   - Done when: Pi wall time per frame for setup D + overlap is measured with and without the reader thread, with the
     same landings, in the run log. Target: wall close to the 11 ms processing time.
2. **Capture loop on the Pi** (vision builder).
   - Done when: it runs on the Pi for 60 s, logging fps processed and frames dropped.

## Waiting on the user

- Nothing yet.

## Done

Nothing yet.
