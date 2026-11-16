#!/usr/bin/env python3
import argparse
import csv
import json
import os
import shutil
import subprocess
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bird_tracker import FRAMES_AFTER_HIT, calibrated_right_row_offset
from frames import load_frame_times

CONTAINER_FPS = 60


def frames_needed(shot):
    hit_frame, end_frame = int(shot["hit_frame"]), int(shot["end_frame"])
    last_tracked_exclusive = min(end_frame + FRAMES_AFTER_HIT, hit_frame + FRAMES_AFTER_HIT)
    return hit_frame - 3, last_tracked_exclusive


def cut_frames(take_path, first_frame, last_frame, clip_path):
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{first_frame / CONTAINER_FPS:.6f}", "-i", take_path,
                    "-frames:v", str(last_frame - first_frame + 1), "-c", "copy", clip_path], check=True)


def clip_matches_take(take_path, first_frame, last_frame, clip_path):
    take, clip = cv2.VideoCapture(take_path), cv2.VideoCapture(clip_path)
    if int(clip.get(cv2.CAP_PROP_FRAME_COUNT)) != last_frame - first_frame + 1:
        return False
    for frame in (first_frame, last_frame):
        take.set(cv2.CAP_PROP_POS_FRAMES, frame)
        clip.set(cv2.CAP_PROP_POS_FRAMES, frame - first_frame)
        ok_take, original = take.read()
        ok_clip, copied = clip.read()
        if not (ok_take and ok_clip and np.array_equal(original, copied)):
            return False
    return True


def main():
    parser = argparse.ArgumentParser(description="Cut the frames the test shots need into a small folder to copy to the Pi.")
    parser.add_argument("session")
    parser.add_argument("--output", default="pi_test_package")
    parser.add_argument("--experiment", default="landing_experiment_yolo11n_crop.json", help="landing results with the tuned terminal speed and correction")
    parser.add_argument("--split", default="test")
    args = parser.parse_args()

    with open(os.path.join(args.session, "shots.csv"), newline="") as f:
        shots = [row for row in csv.DictReader(f) if row["split"] == args.split]
    with open(os.path.join(args.session, args.experiment)) as f:
        experiment = json.load(f)
    expected = {entry["shot"]: entry for entry in experiment["test_shots"]}

    os.makedirs(os.path.join(args.output, "clips"), exist_ok=True)
    shutil.copy2(os.path.join(args.session, "net_calibration.json"), os.path.join(args.output, "net_calibration.json"))
    take_times = {}
    package_shots = []
    for shot in shots:
        take_path = os.path.join(args.session, shot["take"] + ".avi")
        if shot["take"] not in take_times:
            take_times[shot["take"]] = load_frame_times(take_path)[0]
        first_frame, last_frame = frames_needed(shot)
        clip_name = f"shot_{int(shot['shot']):03d}.avi"
        clip_path = os.path.join(args.output, "clips", clip_name)
        cut_frames(take_path, first_frame, last_frame, clip_path)
        if not clip_matches_take(take_path, first_frame, last_frame, clip_path):
            raise SystemExit(f"{clip_name}: frames do not match {shot['take']} {first_frame}-{last_frame}")
        number = int(shot["shot"])
        package_shots.append({
            "shot": number, "take": shot["take"], "clip": clip_name,
            "first_frame": first_frame, "last_frame": last_frame,
            "hit_frame": int(shot["hit_frame"]), "end_frame": int(shot["end_frame"]),
            "frame_times_s": take_times[shot["take"]][first_frame:last_frame + 1],
            "actual_cm": expected[number]["actual_cm"] if number in expected else None,
            "mac_corrected_cm": expected[number]["corrected_cm"] if number in expected else None,
        })
        print(f"shot {number:>3}: frames {first_frame}-{last_frame} from {shot['take']} -> {clip_name}")

    manifest = {"session": os.path.basename(os.path.normpath(args.session)), "split": args.split,
                "right_row_offset_px": calibrated_right_row_offset(args.session),
                "terminal_speed_m_s": experiment["terminal_speed_m_s"], "correction": experiment["correction"],
                "frames_after_hit": FRAMES_AFTER_HIT, "shots": package_shots}
    with open(os.path.join(args.output, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    size_mb = sum(os.path.getsize(os.path.join(args.output, "clips", s["clip"])) for s in package_shots) / 1e6
    print(f"saved {len(package_shots)} shots, {size_mb:.0f} MB of clips, to {args.output}")


if __name__ == "__main__":
    main()
