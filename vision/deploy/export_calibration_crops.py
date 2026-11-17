#!/usr/bin/env python3
import argparse
import glob
import os
import random
import re
import shutil
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from vision.detection.crop_tracking import CROP_SIZE, crop_origin
from vision.detection.detectors import GRAY_STACK_INPUT, INPUT_KINDS, MOTION_VIEW_INPUT
from vision.camera.frames import FrameSource, crop_eye, plain_gray

FOLLOWER_AIM_ERROR_PX = 150
SHARE_WITH_BIRD = 0.8
JPEG_QUALITY = 95
TRAINING_IMAGE_NAME = re.compile(r"shot_\d+_(take_\d+)_f(\d+)\.jpg")


def bird_centers(label_path, width, height):
    if not os.path.exists(label_path):
        return []
    with open(label_path) as f:
        rows = [line.split() for line in f if line.strip()]
    return [(float(row[1]) * width, float(row[2]) * height) for row in rows]


def crop_like_pipeline(images, center, rng, crop_size=CROP_SIZE):
    height, width = images[0].shape[:2]
    aim_error_px = FOLLOWER_AIM_ERROR_PX * crop_size / CROP_SIZE
    aimed = (center[0] + rng.uniform(-aim_error_px, aim_error_px), center[1] + rng.uniform(-aim_error_px, aim_error_px))
    left, top = crop_origin(*aimed, width, height, crop_size)
    crops = [image[top:top + crop_size, left:left + crop_size] for image in images]
    return np.ascontiguousarray(np.dstack(crops) if len(crops) > 1 else crops[0])


class GrayFrames:
    def __init__(self, session_dir):
        self.session_dir = session_dir
        self.sources = {}

    def around(self, training_image_path):
        take, frame = TRAINING_IMAGE_NAME.fullmatch(os.path.basename(training_image_path)).groups()
        if take not in self.sources:
            self.sources[take] = FrameSource(os.path.join(self.session_dir, take + ".avi"))
        return [plain_gray(crop_eye(self.sources[take].read(int(frame) + offset), "left")) for offset in (-1, 0, 1)]


def save_crop(crop, folder, index, input_kind):
    if input_kind == GRAY_STACK_INPUT:
        cv2.imwrite(os.path.join(folder, f"crop_{index:04d}.png"), crop[..., ::-1])
    else:
        cv2.imwrite(os.path.join(folder, f"crop_{index:04d}.jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])


def main():
    parser = argparse.ArgumentParser(description="Cut square motion-view or gray-stack crops from the YOLO training images, aimed the way the pipeline aims, for Hailo INT8 calibration.")
    parser.add_argument("--dataset", default="data/annotation/clears_nov10_labeled")
    parser.add_argument("--count", type=int, default=512)
    parser.add_argument("--output", default="calibration_crops")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--input", choices=INPUT_KINDS, default=MOTION_VIEW_INPUT,
                        help="gray-stack: 3 gray frames (t-1, t, t+1) read from the session videos, saved as lossless PNG")
    parser.add_argument("--crop-size", type=int, default=CROP_SIZE, help="crop side in pixels; 320 for the model used while following the bird")
    parser.add_argument("--session", default="data/recordings/clears_nov10", help="videos for --input gray-stack")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    images = sorted(glob.glob(os.path.join(args.dataset, "images", "train", "*.jpg")))
    if not images:
        raise SystemExit(f"no training images under {args.dataset}/images/train")
    with_bird, without_bird = [], []
    for path in images:
        label_path = os.path.join(args.dataset, "labels", "train", os.path.basename(path)[:-4] + ".txt")
        has_bird = os.path.exists(label_path) and os.path.getsize(label_path) > 0
        (with_bird if has_bird else without_bird).append((path, label_path))
    bird_count = min(len(with_bird), round(args.count * SHARE_WITH_BIRD))
    empty_count = min(len(without_bird), args.count - bird_count)
    chosen = rng.sample(with_bird, bird_count) + rng.sample(without_bird, empty_count)

    if os.path.exists(args.output):
        shutil.rmtree(args.output)
    os.makedirs(args.output)
    grays = GrayFrames(args.session)
    for index, (path, label_path) in enumerate(chosen):
        frames = grays.around(path) if args.input == GRAY_STACK_INPUT else [cv2.imread(path)]
        height, width = frames[0].shape[:2]
        centers = bird_centers(label_path, width, height)
        center = rng.choice(centers) if centers else (rng.uniform(0, width), rng.uniform(0, height))
        save_crop(crop_like_pipeline(frames, center, rng, args.crop_size), args.output, index, args.input)
    archive = shutil.make_archive(args.output, "zip", args.output)
    print(f"{bird_count} crops with the bird, {empty_count} without, from {len(images)} training images")
    print("saved", archive)


if __name__ == "__main__":
    main()
