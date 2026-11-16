#!/usr/bin/env python3
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bird_tracker import EYES
from yolo_tracker import DEFAULT_MODEL, MIN_CONFIDENCE, MotionViewFrames, shot_frames, track_session, tracks_from_detections

IMAGE_SIZE = 1280
BATCH_SIZE = 16


def detect(model, images, device):
    detections = []
    for start in range(0, len(images), BATCH_SIZE):
        results = model.predict(images[start:start + BATCH_SIZE], imgsz=IMAGE_SIZE, conf=MIN_CONFIDENCE, device=device, verbose=False)
        for result in results:
            boxes = result.boxes
            centers = boxes.xywh.cpu().numpy()
            confidences = boxes.conf.cpu().numpy()
            detections.append([(float(x), float(y), float(c)) for (x, y, _, _), c in zip(centers, confidences)])
    return detections


def track_shot_full_frame(model, source, hit_frame, last_frame, right_row_offset, device):
    frames = shot_frames(hit_frame, last_frame)
    views = MotionViewFrames(source)
    found = {eye: dict(zip(frames, detect(model, [views.view(frame, eye) for frame in frames], device))) for eye in EYES}
    return tracks_from_detections(found, frames, hit_frame, right_row_offset)


def main():
    parser = argparse.ArgumentParser(description=f"Experiment: run YOLO on the whole {IMAGE_SIZE}-wide motion view of each eye instead of a crop.")
    parser.add_argument("session")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--output-name", default="bird_tracks_yolo11n_full.json")
    args = parser.parse_args()
    tracks = track_session(args.session, args.model, args.device, track_shot_full_frame)
    output = os.path.join(args.session, args.output_name)
    with open(output, "w") as f:
        json.dump(tracks, f)
    print("saved", output)


if __name__ == "__main__":
    main()
