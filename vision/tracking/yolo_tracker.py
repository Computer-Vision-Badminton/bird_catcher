#!/usr/bin/env python3
import argparse
import csv
import json
import os
import sys
import time

import cv2
import numpy as np
from ultralytics import YOLO

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from vision.tracking.bird_tracker import EYES, FRAMES_AFTER_HIT, calibrated_right_row_offset, longest_rising_chain, match_in_right_eye, moving_blobs
from vision.detection.crop_tracking import CROP_SIZE, MIN_CONFIDENCE, CropFollower, crop_origin, right_eye_aim
from vision.detection.motion_view import current_position_motion_view
from vision.camera.frames import FrameSource, crop_eye, plain_gray

DEFAULT_MODEL = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "models", "bird_yolo11n_v1.pt")


class MotionViewFrames:
    def __init__(self, source):
        self.source = source
        self.grays = {}

    def gray(self, frame, eye):
        if (frame, eye) not in self.grays:
            stereo_frame = self.source.read(frame)
            for each_eye in EYES:
                self.grays[(frame, each_eye)] = plain_gray(crop_eye(stereo_frame, each_eye))
        return self.grays[(frame, eye)]

    def view(self, frame, eye):
        return current_position_motion_view(self.gray(frame - 1, eye), self.gray(frame, eye), self.gray(frame + 1, eye))

    def blobs(self, frame, eye):
        smoothed = [cv2.GaussianBlur(self.gray(index, eye), (3, 3), 0).astype(np.int16) for index in (frame - 1, frame, frame + 1)]
        return moving_blobs(*smoothed)


def detect_in_crops(model, views, frames, eye, device, left_found=None, right_row_offset=0.0):
    follower = CropFollower()
    found = {}
    centers = {}
    for frame in frames:
        view = views.view(frame, eye)
        height, width = view.shape[:2]
        aim_from_left = right_eye_aim(left_found.get(frame), right_row_offset) if left_found else None
        blobs = views.blobs(frame, eye) if not aim_from_left and follower.needs_blobs(frame) else []
        center = aim_from_left or follower.crop_center(frame, blobs, width, height)
        left, top = crop_origin(*center, width, height)
        crop = np.ascontiguousarray(view[top:top + CROP_SIZE, left:left + CROP_SIZE])
        result = model.predict(crop, imgsz=CROP_SIZE, conf=MIN_CONFIDENCE, device=device, verbose=False)[0]
        detections = [(float(x) + left, float(y) + top, float(c))
                      for (x, y, _, _), c in zip(result.boxes.xywh.cpu().numpy(), result.boxes.conf.cpu().numpy())]
        follower.remember(frame, detections)
        found[frame] = detections
        centers[str(frame)] = [left, top]
    return found, centers


def shot_frames(hit_frame, last_frame):
    return list(range(hit_frame - 2, min(last_frame, hit_frame + FRAMES_AFTER_HIT)))


def tracks_from_detections(found, frames, hit_frame, right_row_offset):
    left_chain = longest_rising_chain(found["left"], hit_frame)
    return {
        "left": [[frame, round(x, 2), round(y, 2)] for frame, x, y in left_chain],
        "stereo": match_in_right_eye(left_chain, found["right"], right_row_offset),
        "detections_per_frame": {eye: {str(frame): len(found[eye][frame]) for frame in frames} for eye in EYES},
    }


def track_shot(model, source, hit_frame, last_frame, right_row_offset, device):
    frames = shot_frames(hit_frame, last_frame)
    views = MotionViewFrames(source)
    found = {}
    crop_corners = {}
    for eye in EYES:
        left_found = found.get("left") if eye == "right" else None
        found[eye], crop_corners[eye] = detect_in_crops(model, views, frames, eye, device, left_found, right_row_offset)
    return {**tracks_from_detections(found, frames, hit_frame, right_row_offset), "crop_top_left": crop_corners}


def track_session(session_dir, model_path, device, track_shot_function=track_shot):
    model = YOLO(model_path)
    with open(os.path.join(session_dir, "shots.csv"), newline="") as f:
        shots = list(csv.DictReader(f))
    right_row_offset = calibrated_right_row_offset(session_dir)
    sources = {}
    tracks = []
    started = time.perf_counter()
    for shot in shots:
        take = shot["take"]
        if take not in sources:
            sources[take] = FrameSource(os.path.join(session_dir, take + ".avi"))
        hit_frame, last_frame = int(shot["hit_frame"]), int(shot["end_frame"])
        eyes = track_shot_function(model, sources[take], hit_frame, last_frame + FRAMES_AFTER_HIT, right_row_offset, device)
        tracks.append({"shot": int(shot["shot"]), "take": take, "hit_frame": hit_frame, **eyes})
        print(f"shot {shot['shot']:>3} {take} hit {hit_frame:>5}: left eye {len(eyes['left']):>2} frames, both eyes {len(eyes['stereo']):>2} frames")
    print(f"tracked {len(shots)} shots in {time.perf_counter() - started:.0f} s")
    return tracks


def main():
    parser = argparse.ArgumentParser(description=f"Find the bird after each labeled hit with the trained YOLO model on a {CROP_SIZE}x{CROP_SIZE} crop of the motion view in each eye.")
    parser.add_argument("session")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--output-name", default="bird_tracks_yolo11n_crop.json")
    args = parser.parse_args()
    tracks = track_session(args.session, args.model, args.device)
    output = os.path.join(args.session, args.output_name)
    with open(output, "w") as f:
        json.dump(tracks, f)
    print("saved", output)


if __name__ == "__main__":
    main()
