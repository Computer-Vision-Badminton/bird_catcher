#!/usr/bin/env python3
import argparse
import csv
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bird_tracker import EYES, FRAMES_AFTER_HIT, calibrated_right_row_offset, longest_rising_chain, match_in_right_eye, moving_blobs
from frames import FrameSource, crop_eye, plain_gray


def smoothed_gray(stereo_frame, eye):
    return cv2.GaussianBlur(plain_gray(crop_eye(stereo_frame, eye)), (3, 3), 0).astype(np.int16)


def blobs_after_hit(source, hit_frame, last_frame):
    grays = {}

    def gray(index, eye):
        if (index, eye) not in grays:
            frame = source.read(index)
            for each_eye in EYES:
                grays[(index, each_eye)] = smoothed_gray(frame, each_eye)
        return grays[(index, eye)]

    found = {eye: {} for eye in EYES}
    for middle in range(hit_frame - 2, min(last_frame, hit_frame + FRAMES_AFTER_HIT)):
        for eye in EYES:
            found[eye][middle] = moving_blobs(gray(middle - 1, eye), gray(middle, eye), gray(middle + 1, eye))
    return found


def track_shot(source, hit_frame, last_frame, right_row_offset):
    found = blobs_after_hit(source, hit_frame, last_frame)
    left_chain = longest_rising_chain(found["left"], hit_frame)
    return {
        "left": [[frame, round(x, 2), round(y, 2)] for frame, x, y in left_chain],
        "stereo": match_in_right_eye(left_chain, found["right"], right_row_offset),
    }


def track_session(session_dir):
    with open(os.path.join(session_dir, "shots.csv"), newline="") as f:
        shots = list(csv.DictReader(f))
    right_row_offset = calibrated_right_row_offset(session_dir)
    sources = {}
    tracks = []
    for shot in shots:
        take = shot["take"]
        if take not in sources:
            sources[take] = FrameSource(os.path.join(session_dir, take + ".avi"))
        hit_frame, last_frame = int(shot["hit_frame"]), int(shot["end_frame"])
        eyes = track_shot(sources[take], hit_frame, last_frame + FRAMES_AFTER_HIT, right_row_offset)
        tracks.append({"shot": int(shot["shot"]), "take": take, "hit_frame": hit_frame, **eyes})
        print(f"shot {shot['shot']:>3} {take} hit {hit_frame:>5}: left eye {len(eyes['left']):>2} frames, both eyes {len(eyes['stereo']):>2} frames")
    return tracks


def main():
    parser = argparse.ArgumentParser(description="Find the bird after each labeled hit using motion only.")
    parser.add_argument("session", help="session folder with shots.csv and take_*.avi")
    args = parser.parse_args()
    tracks = track_session(args.session)
    output = os.path.join(args.session, "bird_tracks.json")
    with open(output, "w") as f:
        json.dump(tracks, f)
    print("saved", output)


if __name__ == "__main__":
    main()
