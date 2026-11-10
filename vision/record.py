#!/usr/bin/env python3
"""
record.py — record a ZED (any model) as a plain UVC stereo camera.

No ZED SDK, no NVIDIA. Works on a laptop or a Pi 5 with a USB 3.0 port.
The ZED outputs left|right side-by-side in one frame (2560x720 = 1280 per eye).

Usage
  python3 record.py --check
  python3 record.py --session clears_nov12
  python3 record.py --session clears_nov12 --continuous
  python3 record.py --session still_shuttle --auto 10

One calibration snapshot is saved automatically on the first frame of a session
(data/recordings/<session>/calibration.png). The tripod stays put, so every clip shares it.
Re-running the same --session keeps the existing snapshot.

Keys in the preview window
  SPACE  clip mode: start / stop a clip (one file per clear)
         continuous mode: drop a marker (helps split clears later)
  s      save a single stereo frame as PNG (ground-truth)
  e / E  exposure down / up      g / G  gain down / up
  q      quit

Output, under data/recordings/<session>/ (repo root, wherever you run from)
  calibration.png              calibration snapshot, shared by every clip
  clip_000.avi / take_000.avi  MJPG, full 2560x720 side-by-side, 60 fps
  clip_000.csv / take_000.csv  frame index, capture timestamp (s, monotonic), wall-clock
  take_000_markers.csv         continuous mode: frame index + time of each SPACE press
  frame_0000.png               snapshots
  session.json                 camera settings, mode, calibration snapshot, notes
"""

import argparse
import csv
import json
import os
import platform
import queue
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime

import cv2

WIDTH, HEIGHT, FPS = 2560, 720, 60
RECORDINGS_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "recordings"))


def open_camera(dev):
    backend = cv2.CAP_V4L2 if platform.system() == "Linux" else cv2.CAP_ANY
    cap = cv2.VideoCapture(dev, backend)
    if not cap.isOpened():
        sys.exit(f"cannot open camera {dev}")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUYV"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    f = cap.get(cv2.CAP_PROP_FPS)
    print(f"negotiated {w}x{h} @ {f:.0f} fps")
    if (w, h) != (WIDTH, HEIGHT):
        print("!! did not get 2560x720. Usually means the ZED is on a USB 2.0 port or hub.")
        print("   Linux: run `lsusb -t` and look for 5000M next to the ZED.")
    return cap


def v4l2_set(dev, ctl, val):
    if platform.system() != "Linux" or shutil.which("v4l2-ctl") is None:
        return False
    r = subprocess.run(["v4l2-ctl", "-d", f"/dev/video{dev}", "-c", f"{ctl}={val}"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(f"v4l2-ctl {ctl}={val}: {r.stderr.strip()}")
    return r.returncode == 0


def uvc_util_set(dev, exposure, gain):
    if platform.system() != "Darwin" or shutil.which("uvc-util") is None:
        return False
    r = subprocess.run(["uvc-util", "-I", str(dev), "-s", f"gain={gain}"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(f"uvc-util: {r.stderr.strip() or r.stdout.strip()}")
        print("  list this camera's control names with:  uvc-util -I", dev, "-c")
        return False
    print("macOS: exposure not controllable on the ZED via UVC; gain set, shutter stays auto")
    return True


def lock_exposure(dev, cap, exposure, gain):
    ok = v4l2_set(dev, "auto_exposure", 1)
    ok |= v4l2_set(dev, "exposure_time_absolute", exposure)
    ok |= v4l2_set(dev, "gain", gain)
    ok |= uvc_util_set(dev, exposure, gain)
    if not ok:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
        cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
        cap.set(cv2.CAP_PROP_GAIN, gain)
    if platform.system() == "Darwin":
        print(f"gain {gain} (exposure auto)")
    else:
        print(f"exposure {exposure} (x100us), gain {gain}")


class ClipWriter:
    def __init__(self, path_base):
        self.video = cv2.VideoWriter(path_base + ".avi", cv2.VideoWriter_fourcc(*"MJPG"),
                                     FPS, (WIDTH, HEIGHT))
        if not self.video.isOpened():
            sys.exit("VideoWriter failed to open (MJPG). Is OpenCV built with FFmpeg?")
        self.csv_f = open(path_base + ".csv", "w", newline="")
        self.csv = csv.writer(self.csv_f)
        self.csv.writerow(["frame", "t_mono_s", "t_wall_iso"])
        self.path_base = path_base
        self.q = queue.Queue(maxsize=240)
        self.n = 0
        self.accepted = 0
        self.dropped = 0
        self.markers_f = None
        self.marker_count = 0
        self.alive = True
        self.th = threading.Thread(target=self._run, daemon=True)
        self.th.start()

    def push(self, frame, t_mono):
        try:
            self.q.put_nowait((frame, t_mono, datetime.now().isoformat(timespec="milliseconds")))
            self.accepted += 1
        except queue.Full:
            self.dropped += 1

    def mark(self, t_mono):
        if self.markers_f is None:
            self.markers_f = open(self.path_base + "_markers.csv", "w", newline="")
            self.markers = csv.writer(self.markers_f)
            self.markers.writerow(["marker", "frame", "t_mono_s"])
        latest_frame = max(0, self.accepted - 1)
        self.markers.writerow([self.marker_count, latest_frame, f"{t_mono:.6f}"])
        self.markers_f.flush()
        self.marker_count += 1
        return latest_frame

    def _run(self):
        while self.alive or not self.q.empty():
            try:
                frame, t_mono, t_wall = self.q.get(timeout=0.2)
            except queue.Empty:
                continue
            self.video.write(frame)
            self.csv.writerow([self.n, f"{t_mono:.6f}", t_wall])
            self.n += 1

    def close(self):
        self.alive = False
        self.th.join()
        self.video.release()
        self.csv_f.close()
        if self.markers_f is not None:
            self.markers_f.close()
        return self.n, self.dropped


def check(cap, seconds=5):
    print(f"measuring real frame rate for {seconds} s ...")
    t0 = time.monotonic()
    n = 0
    while time.monotonic() - t0 < seconds:
        ok, _ = cap.read()
        if ok:
            n += 1
    fps = n / (time.monotonic() - t0)
    print(f"real: {fps:.1f} fps")
    if fps < FPS * 0.9:
        print("!! below 60 fps. Check USB 3, close other camera apps, or lower exposure "
              "(long exposure caps the frame rate).")


CALIBRATION_SNAPSHOT = "calibration.png"


def next_index(folder, prefix, suffix):
    return len([p for p in os.listdir(folder) if p.startswith(prefix) and p.endswith(suffix)])


def write_session_meta(args, mode, exposure, gain):
    meta = {
        "started": datetime.now().isoformat(timespec="seconds"),
        "host": platform.node(),
        "resolution": [WIDTH, HEIGHT],
        "fps_requested": FPS,
        "exposure_x100us": exposure,
        "gain": gain,
        "mode": mode,
        "calibration_snapshot": CALIBRATION_SNAPSHOT,
        "notes": args.notes,
    }
    with open(os.path.join(args.session, "session.json"), "w") as f:
        json.dump(meta, f, indent=2)


def status_line(fps_est, exposure, gain, writer, writer_name, continuous):
    status = f"{fps_est:4.1f} fps  exp {exposure}  gain {gain}  "
    if writer is None:
        return status + "idle"
    status += f"REC {writer_name} {writer.n} frames"
    if continuous:
        status += f"  {writer.marker_count} marks"
    return status


def record(cap, dev, args):
    os.makedirs(args.session, exist_ok=True)
    mode = "auto" if args.auto else ("continuous" if args.continuous else "clips")
    prefix = "take" if args.continuous else "clip"
    exposure, gain = args.exposure, args.gain

    calibration_path = os.path.join(args.session, CALIBRATION_SNAPSHOT)
    calibration_saved = os.path.exists(calibration_path)
    if calibration_saved:
        print(f"keeping existing calibration snapshot {calibration_path}")

    writer_idx = next_index(args.session, prefix + "_", ".avi")
    snap_idx = next_index(args.session, "frame_", ".png")

    def writer_name():
        return f"{prefix}_{writer_idx:03d}"

    def start_writer():
        print(f"REC {writer_name()}")
        return ClipWriter(os.path.join(args.session, writer_name()))

    def stop_writer():
        n, dropped = writer.close()
        marks = f", {writer.marker_count} markers" if args.continuous else ""
        print(f"saved {writer_name()}: {n} frames, {dropped} dropped{marks}")

    write_session_meta(args, mode, exposure, gain)
    writer = start_writer() if args.continuous or args.auto else None

    if args.auto:
        auto_end = time.monotonic() + args.auto
        print(f"recording {writer_name()} for {args.auto} s ...")
    else:
        auto_end = None
        cv2.namedWindow("zed", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("zed", 1280, 360)
        if args.continuous:
            print("continuous: SPACE marker | s snapshot | e/E g/G exposure, gain | q quit")
        else:
            print("SPACE start/stop clip | s snapshot | e/E g/G exposure, gain | q quit")

    n_shown = 0
    t_last = time.monotonic()
    fps_est = 0.0
    while True:
        ok, frame = cap.read()
        t = time.monotonic()
        if not ok:
            print("frame grab failed")
            continue
        if not calibration_saved:
            cv2.imwrite(calibration_path, frame)
            calibration_saved = True
            print("saved calibration snapshot", calibration_path)
        if writer is not None:
            writer.push(frame, t)

        n_shown += 1
        if t - t_last >= 1.0:
            fps_est = n_shown / (t - t_last)
            n_shown, t_last = 0, t

        if auto_end is not None:
            if t >= auto_end:
                break
            continue

        small = cv2.resize(frame, (1280, 360))
        status = status_line(fps_est, exposure, gain, writer, writer_name(), args.continuous)
        cv2.putText(small, status, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 0, 255) if writer else (0, 255, 0), 2)
        cv2.imshow("zed", small)

        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        elif k == ord(" ") and args.continuous:
            frame_index = writer.mark(t)
            print(f"marker {writer.marker_count - 1} at frame {frame_index}")
        elif k == ord(" "):
            if writer is None:
                writer = start_writer()
            else:
                stop_writer()
                writer, writer_idx = None, writer_idx + 1
        elif k == ord("s"):
            p = os.path.join(args.session, f"frame_{snap_idx:04d}.png")
            cv2.imwrite(p, frame)
            print("saved", p)
            snap_idx += 1
        elif k in (ord("e"), ord("E"), ord("g"), ord("G")):
            if k == ord("e"):
                exposure = max(1, exposure - 5)
            elif k == ord("E"):
                exposure += 5
            elif k == ord("g"):
                gain = max(0, gain - 5)
            else:
                gain += 5
            lock_exposure(dev, cap, exposure, gain)

    if writer is not None:
        stop_writer()
    cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", type=int, default=0, help="camera index (/dev/videoN)")
    ap.add_argument("--check", action="store_true", help="just negotiate and measure fps")
    ap.add_argument("--session", help="session name; saved under data/recordings/NAME")
    ap.add_argument("--auto", type=float, default=0, help="record this many seconds then exit")
    ap.add_argument("--continuous", action="store_true",
                    help="one long take for the whole session; SPACE drops markers instead of splitting")
    ap.add_argument("--exposure", type=int, default=30, help="exposure, x100 us (30 = 3 ms)")
    ap.add_argument("--gain", type=int, default=60)
    ap.add_argument("--notes", default="", help='e.g. "tripod 1.1 m, 30 deg up, home position, court 3"')
    args = ap.parse_args()

    cap = open_camera(args.device)
    lock_exposure(args.device, cap, args.exposure, args.gain)
    for _ in range(10):
        cap.read()

    if args.check:
        check(cap)
    elif args.session:
        args.session = os.path.join(RECORDINGS_ROOT, args.session)
        record(cap, args.device, args)
    else:
        ap.error("use --check or --session NAME")
    cap.release()


if __name__ == "__main__":
    main()