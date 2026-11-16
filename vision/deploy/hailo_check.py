#!/usr/bin/env python3
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crop_tracking import CROP_SIZE, crop_origin
from detectors import GRAY_STACK_INPUT, INPUT_KINDS, MOTION_VIEW_INPUT, HailoDetector, UltralyticsDetector, describe_output
from frames import crop_eye, plain_gray
from motion_view import current_position_motion_view

CHECK_SHOT = 37
CHECK_FRAME = 826
BIRD_AT = (652.9, 266.1)
TIMING_RUNS = 50


def bird_crop(package_dir, input_kind=MOTION_VIEW_INPUT):
    with open(os.path.join(package_dir, "manifest.json")) as f:
        entry = next(shot for shot in json.load(f)["shots"] if shot["shot"] == CHECK_SHOT)
    clip = cv2.VideoCapture(os.path.join(package_dir, "clips", entry["clip"]))
    clip.set(cv2.CAP_PROP_POS_FRAMES, CHECK_FRAME - 1 - entry["first_frame"])
    grays = [plain_gray(crop_eye(clip.read()[1], "left")) for _ in range(3)]
    view = current_position_motion_view(*grays)
    left, top = crop_origin(*BIRD_AT, view.shape[1], view.shape[0])
    window = (slice(top, top + CROP_SIZE), slice(left, left + CROP_SIZE))
    crop = np.dstack([gray[window] for gray in grays]) if input_kind == GRAY_STACK_INPUT else view[window]
    return np.ascontiguousarray(crop), np.ascontiguousarray(view[window]), (left, top)


def main():
    parser = argparse.ArgumentParser(description="Load the .hef on the AI HAT+ 2, show its inputs and outputs, and check it finds the bird in a known crop.")
    parser.add_argument("hef")
    parser.add_argument("--package", default="pi_test_package")
    parser.add_argument("--ncnn-model", help="also run the NCNN model on the same crop to compare")
    parser.add_argument("--hailo-input-bgr", action="store_true")
    parser.add_argument("--input", choices=INPUT_KINDS, default=MOTION_VIEW_INPUT, help="gray-stack for a .hef that builds the motion view itself")
    args = parser.parse_args()

    detector = HailoDetector(args.hef, args.hailo_input_bgr, args.input)
    print("model:", json.dumps(detector.describe()))
    crop, view_crop, (left, top) = bird_crop(args.package, args.input)
    print("raw output:", describe_output(detector.raw_output(crop)))
    expected = (BIRD_AT[0] - left, BIRD_AT[1] - top)
    print(f"shot {CHECK_SHOT} frame {CHECK_FRAME}: bird expected at ({expected[0]:.1f}, {expected[1]:.1f}) in the crop")
    print("hailo finds:", [(round(x, 1), round(y, 1), round(c, 3)) for x, y, c in detector.detect(crop)])
    print("hailo on an empty crop:", detector.detect(np.zeros_like(crop)))
    started = time.perf_counter()
    for _ in range(TIMING_RUNS):
        detector.detect(crop)
    print(f"hailo: {(time.perf_counter() - started) / TIMING_RUNS * 1000:.2f} ms per crop, including colour conversion and decoding")
    if args.ncnn_model:
        print("ncnn finds:", [(round(x, 1), round(y, 1), round(c, 3)) for x, y, c in UltralyticsDetector(args.ncnn_model).detect(view_crop)])
    detector.close()


if __name__ == "__main__":
    main()
