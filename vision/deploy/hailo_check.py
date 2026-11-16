#!/usr/bin/env python3
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crop_tracking import crop_origin
from detectors import GRAY_STACK_INPUT, INPUT_KINDS, MOTION_VIEW_INPUT, STEREO_BATCH, HailoDetector, UltralyticsDetector, describe_output
from frames import crop_eye, plain_gray
from motion_view import current_position_motion_view

CHECK_SHOT = 37
CHECK_FRAME = 826
BIRD_AT = (652.9, 266.1)
TIMING_RUNS = 50
CPU_WORK_MS = 8.0


def busy_cpu_for(milliseconds):
    until = time.perf_counter() + milliseconds / 1000
    while time.perf_counter() < until:
        pass


def mean_ms_of(action, runs=TIMING_RUNS):
    started = time.perf_counter()
    for _ in range(runs):
        action()
    return (time.perf_counter() - started) / runs * 1000


def check_async_run(detector, crop, batch):
    crops = [crop, np.zeros_like(crop)][:batch]
    same = detector.start_batch(crops).wait() == detector.detect_batch(crops)
    print(f"async: start_batch + wait finds the same as detect_batch: {same}")
    start_ms = []

    def start_then_wait():
        started = time.perf_counter()
        job = detector.start_batch(crops)
        start_ms.append((time.perf_counter() - started) * 1000)
        job.wait()

    def start_work_then_wait():
        job = detector.start_batch(crops)
        busy_cpu_for(CPU_WORK_MS)
        job.wait()

    sync_ms = mean_ms_of(lambda: detector.detect_batch(crops))
    async_ms = mean_ms_of(start_then_wait)
    overlapped_ms = mean_ms_of(start_work_then_wait)
    print(f"async: detect_batch {sync_ms:.2f} ms; start_batch returns in {np.mean(start_ms):.2f} ms, start + wait {async_ms:.2f} ms")
    print(f"async: start, {CPU_WORK_MS:.0f} ms of Python CPU work, wait: {overlapped_ms:.2f} ms "
          f"(overlapping works if this is near max({sync_ms:.1f}, {CPU_WORK_MS:.0f}), not {sync_ms + CPU_WORK_MS:.1f})")


def bird_crop(package_dir, input_kind, crop_size):
    with open(os.path.join(package_dir, "manifest.json")) as f:
        entry = next(shot for shot in json.load(f)["shots"] if shot["shot"] == CHECK_SHOT)
    clip = cv2.VideoCapture(os.path.join(package_dir, "clips", entry["clip"]))
    clip.set(cv2.CAP_PROP_POS_FRAMES, CHECK_FRAME - 1 - entry["first_frame"])
    grays = [plain_gray(crop_eye(clip.read()[1], "left")) for _ in range(3)]
    view = current_position_motion_view(*grays)
    left, top = crop_origin(*BIRD_AT, view.shape[1], view.shape[0], crop_size)
    window = (slice(top, top + crop_size), slice(left, left + crop_size))
    crop = np.dstack([gray[window] for gray in grays]) if input_kind == GRAY_STACK_INPUT else view[window]
    return np.ascontiguousarray(crop), np.ascontiguousarray(view[window]), (left, top)


def main():
    parser = argparse.ArgumentParser(description="Load the .hef on the AI HAT+ 2, show its inputs and outputs, and check it finds the bird in a known crop.")
    parser.add_argument("hef")
    parser.add_argument("--package", default="pi_test_package")
    parser.add_argument("--ncnn-model", help="also run the NCNN model on the same crop to compare")
    parser.add_argument("--hailo-input-bgr", action="store_true")
    parser.add_argument("--input", choices=INPUT_KINDS, default=MOTION_VIEW_INPUT, help="gray-stack for a .hef that builds the motion view itself")
    parser.add_argument("--batch", type=int, default=STEREO_BATCH, help="crops per Hailo call (2 = both eyes at once, as the pipeline runs)")
    parser.add_argument("--async-check", action="store_true", help="also check start_batch / wait (run_async) gives the same detections and lets CPU work overlap the call")
    args = parser.parse_args()

    detector = HailoDetector(args.hef, args.hailo_input_bgr, args.input, args.batch)
    print("model:", json.dumps(detector.describe()))
    crop, view_crop, (left, top) = bird_crop(args.package, args.input, detector.input_width)
    print("raw output:", describe_output(detector.raw_output(crop)))
    expected = (BIRD_AT[0] - left, BIRD_AT[1] - top)
    print(f"shot {CHECK_SHOT} frame {CHECK_FRAME}: bird expected at ({expected[0]:.1f}, {expected[1]:.1f}) in the crop")
    print("hailo finds:", [(round(x, 1), round(y, 1), round(c, 3)) for x, y, c in detector.detect(crop)])
    print("hailo on an empty crop:", detector.detect(np.zeros_like(crop)))
    started = time.perf_counter()
    for _ in range(TIMING_RUNS):
        detector.detect(crop)
    padded_to = f", padded to a call of {args.batch} crops, so it runs {args.batch} crops" if args.batch > 1 else ""
    print(f"hailo: {(time.perf_counter() - started) / TIMING_RUNS * 1000:.2f} ms per detect() of one crop{padded_to}, including colour conversion and decoding")
    batch_detections = detector.detect_batch([crop] * args.batch)
    print(f"batch of {args.batch}: every crop finds the bird the same way: {all(d == batch_detections[0] for d in batch_detections)}")
    started = time.perf_counter()
    for _ in range(TIMING_RUNS):
        detector.detect_batch([crop] * args.batch)
    call_ms = (time.perf_counter() - started) / TIMING_RUNS * 1000
    print(f"hailo: {call_ms:.2f} ms per call of {args.batch} crops, {call_ms / args.batch:.2f} ms per crop")
    if args.async_check:
        check_async_run(detector, crop, args.batch)
    if args.ncnn_model:
        print("ncnn finds:", [(round(x, 1), round(y, 1), round(c, 3)) for x, y, c in UltralyticsDetector(args.ncnn_model).detect(view_crop)])
    detector.close()


if __name__ == "__main__":
    main()
