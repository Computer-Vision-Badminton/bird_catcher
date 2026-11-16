#!/usr/bin/env python3
import argparse
import collections
import json
import os
import time

import cv2
import numpy as np

from court_3d import StereoCourt
from detectors import GRAY_STACK_INPUT, load_detector
from esp32_link import DEFAULT_BAUD, DEFAULT_PORT, Esp32Link
from pipeline import ShotPipeline

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_PACKAGE = os.path.join(REPO, "pi_test_package")
HAILO_MODEL = os.path.join(REPO, "models", "bird_yolo11n_v1_gray_stack.hef")
CPU_MODEL = os.path.join(REPO, "models", "bird_yolo11n_v1_gray_stack.onnx")
ROBOT_HOME_CM = (-150.0, 330.0)

LandingEstimate = collections.namedtuple("LandingEstimate", "frame seconds_after_hit x_cm y_cm")
ShotLanding = collections.namedtuple(
    "ShotLanding", "shot landing_cm measured_cm miss_cm offset_from_home_cm estimates ms_per_frame detector")


def detector_for_this_machine():
    try:
        import hailo_platform
        return load_detector("hailo", HAILO_MODEL, input_kind=GRAY_STACK_INPUT), "hailo"
    except ImportError:
        return load_detector("onnx", CPU_MODEL, input_kind=GRAY_STACK_INPUT), "onnx (cpu)"


def distance_cm(a, b):
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


class ShotReplay:
    def __init__(self, package=DEFAULT_PACKAGE, detector=None):
        self.package = package
        with open(os.path.join(package, "manifest.json")) as f:
            self.manifest = json.load(f)
        with open(os.path.join(package, "net_calibration.json")) as f:
            self.court = StereoCourt(json.load(f))
        self.shots = {entry["shot"]: entry for entry in self.manifest["shots"]}
        self.detector, self.detector_name = (detector, "custom") if detector else detector_for_this_machine()

    def shot_numbers(self):
        return sorted(self.shots)

    def landing(self, shot_number, on_estimate=None):
        entry = self.shots[shot_number]
        manifest = self.manifest
        pipeline = ShotPipeline(self.detector, self.court, manifest["right_row_offset_px"], manifest["terminal_speed_m_s"],
                                manifest["correction"], entry["hit_frame"], entry["end_frame"] + manifest["frames_after_hit"])
        hit_time = entry["frame_times_s"][entry["hit_frame"] - entry["first_frame"]]
        clip = cv2.VideoCapture(os.path.join(self.package, "clips", entry["clip"]))
        estimates, frame_ms = [], []
        for offset, time_s in enumerate(entry["frame_times_s"]):
            ok, stereo_frame = clip.read()
            if not ok:
                raise RuntimeError(f"{entry['clip']}: could not read frame {offset}")
            started = time.perf_counter()
            result = pipeline.push(entry["first_frame"] + offset, stereo_frame, time_s)
            if result is None:
                continue
            frame_ms.append((time.perf_counter() - started) * 1000)
            newest = result["landing_cm"]
            if newest is not None and (not estimates or estimates[-1].frame != result["landing_from_frame"]):
                frame = result["landing_from_frame"]
                estimate = LandingEstimate(frame, round(entry["frame_times_s"][frame - entry["first_frame"]] - hit_time, 3),
                                           round(newest[0], 1), round(newest[1], 1))
                estimates.append(estimate)
                if on_estimate:
                    on_estimate(estimate)
        final = pipeline.finish()
        landing = None if final is None else (round(float(final[0]), 1), round(float(final[1]), 1))
        measured = tuple(entry["actual_cm"]) if entry.get("actual_cm") else None
        return ShotLanding(
            shot=shot_number, landing_cm=landing, measured_cm=measured,
            miss_cm=None if landing is None or measured is None else round(distance_cm(landing, measured), 1),
            offset_from_home_cm=None if landing is None else (round(landing[0] - ROBOT_HOME_CM[0], 1), round(landing[1] - ROBOT_HOME_CM[1], 1)),
            estimates=estimates, ms_per_frame=round(float(np.mean(frame_ms)), 1) if frame_ms else None,
            detector=self.detector_name)

    def close(self):
        if hasattr(self.detector, "close"):
            self.detector.close()


_shared_replay = None


def landing_for_shot(shot_number, on_estimate=None):
    global _shared_replay
    if _shared_replay is None:
        _shared_replay = ShotReplay()
    return _shared_replay.landing(shot_number, on_estimate)


def main():
    parser = argparse.ArgumentParser(description="Replay recorded test shots through the pipeline and print (and optionally send) the predicted landing point in court centimetres.")
    parser.add_argument("shots", type=int, nargs="*", help="shot numbers; none = list the available shots")
    parser.add_argument("--all", action="store_true", help="replay every test shot")
    parser.add_argument("--send-uart", nargs="?", const=DEFAULT_PORT, metavar="PORT", help=f"send each landing to the ESP32 (default port {DEFAULT_PORT})")
    parser.add_argument("--baud", type=int, default=DEFAULT_BAUD)
    parser.add_argument("--stream", action="store_true", help="with --send-uart: also send every in-flight estimate, not only the final landing")
    parser.add_argument("--pause", type=float, default=0.0, help="seconds to wait between shots, e.g. to let the robot drive back home")
    args = parser.parse_args()

    replay = ShotReplay()
    shots = replay.shot_numbers() if args.all else args.shots
    if not shots:
        print("available test shots:", " ".join(str(n) for n in replay.shot_numbers()))
        return
    link = Esp32Link(args.send_uart, args.baud) if args.send_uart else None
    on_estimate = (lambda e: link.send_landing(e.x_cm, e.y_cm, final=False)) if link and args.stream else None
    print(f"detector: {replay.detector_name} | robot home: x {ROBOT_HOME_CM[0]:.0f} cm, y {ROBOT_HOME_CM[1]:.0f} cm")
    for index, shot_number in enumerate(shots):
        result = replay.landing(shot_number, on_estimate)
        if link and result.landing_cm:
            link.send_landing(*result.landing_cm, final=True)
        if result.landing_cm is None:
            print(f"shot {shot_number:>3}: not enough 3D points for a landing")
            continue
        first = result.estimates[0] if result.estimates else None
        print(f"shot {shot_number:>3}: landing x {result.landing_cm[0]:7.1f} cm, y {result.landing_cm[1]:6.1f} cm"
              f" | from home {result.offset_from_home_cm[0]:+6.1f}, {result.offset_from_home_cm[1]:+6.1f} cm"
              f" | measured {result.measured_cm} miss {result.miss_cm} cm"
              f" | first estimate {first.seconds_after_hit if first else '-'} s after hit, {len(result.estimates)} estimates"
              f" | {result.ms_per_frame} ms per frame{' | sent' if link else ''}")
        if args.pause and index < len(shots) - 1:
            time.sleep(args.pause)
    if link:
        link.close()
    replay.close()


if __name__ == "__main__":
    main()
