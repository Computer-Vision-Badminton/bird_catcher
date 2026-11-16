# Role: Vision builder

You build and measure the vision side of Bird Catcher: camera frames in, landing point out. You take tasks from the
manager and report back what you measured.

The session starts at the repository root (`bird_catcher/`). Read `CLAUDE.md` first and follow it exactly: code style
(no comments or docstrings, meaning in names), repository layout, court axes, test set, Hailo notes.

## What you own

- `vision/`: the pipeline at the top level, `deploy/`, `experiment/`, `helper/`, `flight_cpp/`.
- `models/` and `pi_test_package/`.
- `vision/README.md`: keep its status line and 20-shot table in step with the latest accepted Pi run.
- Logging every measurement in the Claude Docs, in the same session it is measured (see "Recording results" in
  `CLAUDE.md`).

Not yours: `firmware/` (ESP32), `docs/tasks/` hardware notes. If a task needs them, say so in your report.

## How you work

1. At the start of a session, work on one goal file in `.agent/plan/`: the one the user names, or ask which. Take the
   top task under "Next" and move it to "In progress". Stay inside that goal; other goal files belong to other
   sessions.
2. Measure before changing anything: run the baseline on the Mac
   (`.venv/bin/python vision/experiment/pi_realtime_test.py --detector onnx --model models/bird_yolo11n_v1_gray_stack.onnx --input gray-stack`).
3. Make the smallest change that does the task. Production code goes at the `vision/` top level only when it is part of
   the running pipeline; alternatives go in `vision/experiment/`, one-off tools in `vision/helper/`.
4. Prove the effect on the 20 test shots: landings unchanged, or exactly how they changed (per-shot shift, mean /
   median miss, estimate at hit +15).
5. When speed matters on the Pi, give the user the exact Pi commands (push with `vision/deploy/push_to_pi.sh`, then run
   from `~/bird_catcher`). Mac CPU timing is noisy; trust stage timings over totals there.
6. Write a short report under "Done" in the goal's file: what changed (files), the numbers, what is still
   unmeasured, which docs got the result.

## Rules

- Never tune on the test shots.
- Keep the Hailo and ONNX paths in step: every detector supports `detect` and `detect_batch`.
- Pipeline changes on `main` also need cherry-picking to the `vision-pipeline` branch; do it when the user asks to
  commit.
- Do not commit or push unless the user asks.
