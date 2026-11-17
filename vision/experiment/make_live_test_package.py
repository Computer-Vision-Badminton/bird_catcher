#!/usr/bin/env python3
import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from vision.experiment.hit_from_rising_chain import cut_clip_by_stream_copy, frame_count_of, frame_digest, read_shots, write_json_atomically

FRAMES_BEFORE_HIT = 130
FRAMES_AFTER_END = 10
FRAMES_AFTER_HIT_KEPT = 45
VIDEO_NAME = "live_test.avi"
FRAME_TIMES_NAME = "frame_times.csv"
MANIFEST_NAME = "manifest.json"
LANDING_SCORE_NAME = "landing_test_score.json"
FIT_TRACKS_NAME = "pipeline_tracks_setup_d_onnx_test.json"
CHECKPOINTS_AFTER_HIT = (10, 15, 20)
LABELS_NOTE = ("scoring_only holds hand labels and reference landings. Only the scoring step may read it; "
               "the live run must never read scoring_only, shots.csv or any other label.")


def test_shots_in_order(session):
    return sorted((shot for shot in read_shots(session) if shot["split"] == "test"), key=lambda shot: shot["shot"])


def take_times(session, take):
    with open(os.path.join(session, take + ".csv"), newline="") as f:
        return {int(row["frame"]): float(row["t_mono_s"]) for row in csv.DictReader(f)}


def wanted_window(shot):
    return shot["hit"] - FRAMES_BEFORE_HIT, max(shot["end"] + FRAMES_AFTER_END, shot["hit"] + FRAMES_AFTER_HIT_KEPT)


def clipped_window(shot, frame_count, other_test_windows):
    wanted_first, wanted_last = wanted_window(shot)
    first, last, notes = max(0, wanted_first), min(frame_count - 1, wanted_last), []
    if first != wanted_first:
        notes.append(f"start clipped to take start ({wanted_first} -> {first})")
    if last != wanted_last:
        notes.append(f"end clipped to take end ({wanted_last} -> {last})")
    for other_first, other_last in other_test_windows:
        if other_first <= last and other_last >= first:
            notes.append(f"overlaps another test window {other_first}..{other_last}")
    return first, last, notes


def planned_windows(session, shots):
    frame_counts = {take: frame_count_of(os.path.join(session, take + ".avi")) for take in {shot["take"] for shot in shots}}
    wanted = {shot["shot"]: wanted_window(shot) for shot in shots}
    windows = []
    for shot in shots:
        others = [wanted[other["shot"]] for other in shots if other["take"] == shot["take"] and other["shot"] != shot["shot"]]
        first, last, notes = clipped_window(shot, frame_counts[shot["take"]], others)
        windows.append({"shot": shot["shot"], "take": shot["take"], "first_take_frame": first, "last_take_frame": last,
                        "take_frame_count": frame_counts[shot["take"]], "clipping": notes})
    return windows


def concat_by_stream_copy(clip_paths, video_path):
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as listing:
        for clip_path in clip_paths:
            listing.write(f"file '{os.path.abspath(clip_path)}'\n")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", listing.name,
                    "-map", "0:v:0", "-c", "copy", video_path], check=True)
    os.remove(listing.name)


def median_frame_period(window_times):
    return statistics.median(later - earlier for times in window_times for earlier, later in zip(times, times[1:]))


def shifted_frame_times(windows, window_times, frame_period):
    rows, previous_end = [], None
    for window, times in zip(windows, window_times):
        shift = 0.0 if previous_end is None else previous_end + frame_period - times[0]
        window["time_shift_s"] = round(shift, 6)
        for take_frame, t in zip(range(window["first_take_frame"], window["last_take_frame"] + 1), times):
            rows.append({"frame": len(rows), "t_s": f"{t + shift:.6f}", "t_mono_s": f"{t:.6f}",
                         "take": window["take"], "take_frame": take_frame})
        previous_end = times[-1] + shift
    return rows


def write_frame_times(rows, path):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def number_video_frames(windows):
    next_frame = 0
    for window in windows:
        window["first_video_frame"] = next_frame
        window["last_video_frame"] = next_frame + window["last_take_frame"] - window["first_take_frame"]
        window["offset"] = window["first_video_frame"] - window["first_take_frame"]
        next_frame = window["last_video_frame"] + 1
    return next_frame


def scoring_only_labels(session, shots, windows):
    with open("test/pi_test_package/manifest.json") as f:
        actual_by_shot = {entry["shot"]: entry["actual_cm"] for entry in json.load(f)["shots"]}
    with open(os.path.join(session, LANDING_SCORE_NAME)) as f:
        kalman_by_shot = {int(shot): entry for shot, entry in json.load(f)["rows"]["kalman_filtered"]["per_shot"].items()}
    with open(os.path.join(session, FIT_TRACKS_NAME)) as f:
        fit_by_shot = {entry["shot"]: entry["final_landing_cm"] for entry in json.load(f)["shots"]}
    window_by_shot = {window["shot"]: window for window in windows}
    labels = []
    for shot in shots:
        offset = window_by_shot[shot["shot"]]["offset"]
        labels.append({"shot": shot["shot"], "take": shot["take"], "hit_take_frame": shot["hit"], "hit_video_frame": shot["hit"] + offset,
                       "end_take_frame": shot["end"], "end_video_frame": shot["end"] + offset,
                       "actual_cm": actual_by_shot[shot["shot"]],
                       "labelled_hit_kalman_landing_cm": kalman_by_shot[shot["shot"]]["landing_cm"],
                       "labelled_hit_fit_landing_cm": fit_by_shot[shot["shot"]],
                       **{f"labelled_hit_kalman_miss_hit_plus_{after_hit}_cm": kalman_by_shot[shot["shot"]][f"miss_hit_plus_{after_hit}_cm"]
                          for after_hit in CHECKPOINTS_AFTER_HIT}})
    return labels


def scoring_only(session, shots, windows):
    return {"source": {"hit": os.path.join(session, "shots.csv"), "actual_cm": "test/pi_test_package/manifest.json (signed court cm)",
                       "labelled_hit_kalman_landing_cm": os.path.join(session, LANDING_SCORE_NAME) + " rows.kalman_filtered.per_shot.landing_cm",
                       "labelled_hit_kalman_miss_hit_plus_cm": os.path.join(session, LANDING_SCORE_NAME)
                                                               + " rows.kalman_filtered.per_shot.miss_hit_plus_<10|15|20>_cm",
                       "labelled_hit_fit_landing_cm": os.path.join(session, FIT_TRACKS_NAME) + " final_landing_cm"},
            "shots": scoring_only_labels(session, shots, windows)}


def make_package(args):
    shots = test_shots_in_order(args.session)
    windows = planned_windows(args.session, shots)
    clip_folder = os.path.join(args.package, "windows_tmp")
    os.makedirs(clip_folder, exist_ok=True)
    clip_paths, window_times = [], []
    for window in windows:
        clip_path = os.path.join(clip_folder, f"shot_{window['shot']:03d}.avi")
        cut_clip_by_stream_copy(os.path.join(args.session, window["take"] + ".avi"), window["first_take_frame"], window["last_take_frame"], clip_path)
        clip_paths.append(clip_path)
        times = take_times(args.session, window["take"])
        window_times.append([times[frame] for frame in range(window["first_take_frame"], window["last_take_frame"] + 1)])
        print(f"shot {window['shot']}: {window['take']} {window['first_take_frame']}..{window['last_take_frame']}"
              + (f" ({'; '.join(window['clipping'])})" if window["clipping"] else ""), flush=True)
    concat_by_stream_copy(clip_paths, os.path.join(args.package, VIDEO_NAME))
    for clip_path in clip_paths:
        os.remove(clip_path)
    os.rmdir(clip_folder)
    frame_total = number_video_frames(windows)
    frame_period = median_frame_period(window_times)
    write_frame_times(shifted_frame_times(windows, window_times, frame_period), os.path.join(args.package, FRAME_TIMES_NAME))
    manifest = {
        "made": time.strftime("%Y-%m-%d"), "source_session": args.session, "video": VIDEO_NAME, "frame_times": FRAME_TIMES_NAME,
        "frame_count": frame_total, "camera_fps": 60,
        "frame_numbers": "video frame k (0-based) of window w is take frame k - offset; frame_times.csv t_s is the take's t_mono_s "
                         "plus the window's time_shift_s, so time keeps increasing across joins",
        "join_gap_s": round(frame_period, 6),
        "window_rule": f"per test shot, take frames hit - {FRAMES_BEFORE_HIT} .. max(end_frame + {FRAMES_AFTER_END}, hit + {FRAMES_AFTER_HIT_KEPT}), "
                       "stream copy, joined in shot order",
        "windows": windows,
        "labels_note": LABELS_NOTE,
        "scoring_only": scoring_only(args.session, shots, windows),
    }
    write_json_atomically(os.path.join(args.package, MANIFEST_NAME), manifest, indent=1)
    print(f"{frame_total} frames in {len(windows)} windows, join gap {frame_period * 1000:.3f} ms; saved {os.path.join(args.package, MANIFEST_NAME)}")


def update_scoring_only(args):
    path = os.path.join(args.package, MANIFEST_NAME)
    with open(path) as f:
        manifest = json.load(f)
    manifest["scoring_only"] = scoring_only(args.session, test_shots_in_order(args.session), manifest["windows"])
    write_json_atomically(path, manifest, indent=1)
    print(f"updated scoring_only in {path} ({len(manifest['scoring_only']['shots'])} shots); video and frame times unchanged")


def verify_package(args):
    with open(os.path.join(args.package, MANIFEST_NAME)) as f:
        manifest = json.load(f)
    joined = cv2.VideoCapture(os.path.join(args.package, manifest["video"]))
    reported_count = int(joined.get(cv2.CAP_PROP_FRAME_COUNT))
    compared, different = 0, []
    for window in manifest["windows"]:
        original = cv2.VideoCapture(os.path.join(manifest["source_session"], window["take"] + ".avi"))
        original.set(cv2.CAP_PROP_POS_FRAMES, window["first_take_frame"])
        window_different = 0
        for take_frame in range(window["first_take_frame"], window["last_take_frame"] + 1):
            ok_original, original_frame = original.read()
            ok_joined, joined_frame = joined.read()
            if not (ok_original and ok_joined) or frame_digest(original_frame) != frame_digest(joined_frame):
                different.append((window["shot"], take_frame))
                window_different += 1
            compared += 1
        original.release()
        print(f"shot {window['shot']}: {window['last_take_frame'] - window['first_take_frame'] + 1} frames, {window_different} different", flush=True)
    extra_frame, _ = joined.read()
    joined.release()
    with open(os.path.join(args.package, manifest["frame_times"]), newline="") as f:
        times = [float(row["t_s"]) for row in csv.DictReader(f)]
    window_sum = sum(window["last_take_frame"] - window["first_take_frame"] + 1 for window in manifest["windows"])
    print(f"compared {compared}, different {len(different)}, window sum {window_sum}, manifest {manifest['frame_count']}, "
          f"container frame count {reported_count}, extra frame after the end: {extra_frame}, frame_times rows {len(times)}, "
          f"times strictly increasing: {all(later > earlier for earlier, later in zip(times, times[1:]))}")


def main():
    parser = argparse.ArgumentParser(description="One continuous video of the 20 test shots for the live-mode replay.")
    parser.add_argument("command", choices=("make", "verify", "update-scoring-only"))
    parser.add_argument("--session", default="data/recordings/clears_nov10")
    parser.add_argument("--package", default="test/pi_live_test_package")
    args = parser.parse_args()
    {"make": make_package, "verify": verify_package, "update-scoring-only": update_scoring_only}[args.command](args)


if __name__ == "__main__":
    main()
